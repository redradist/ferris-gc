//! Thread-Local Allocation Buffer (TLAB) for fast bump-pointer allocation.
//!
//! Each thread gets a pre-allocated memory chunk. Allocations bump a pointer
//! within the chunk — no global allocator calls, no locks. When the chunk is
//! full, a new one is allocated. This dramatically speeds up the allocation
//! hot path for the thread-local GC.
//!
//! # Deallocation
//!
//! Individual objects within a TLAB block cannot be freed independently (they
//! are sub-allocations of a larger buffer). Instead, each block is reference-
//! counted via a non-atomic `ref_count`. When all sub-allocations in a block
//! have been collected (ref_count drops to 0), the entire block is freed.
//! This avoids the ~5ns overhead of `Arc` atomic operations on the hot path.

use std::alloc::{Layout, alloc, dealloc};

/// Default TLAB block size: 64 KiB.
const TLAB_DEFAULT_SIZE: usize = 64 * 1024;

/// A contiguous memory block backing TLAB sub-allocations.
///
/// The block owns its allocation and frees it on drop. Each sub-allocation
/// increments `ref_count`; deallocation decrements it. When `ref_count`
/// reaches 0, the block is freed via [`TlabBlock::release`].
pub(crate) struct TlabBlock {
    buffer: *mut u8,
    layout: Layout,
    /// Non-atomic reference count. Safe because TLAB is thread-local and
    /// all cross-thread access (background GC) is under a Mutex.
    ref_count: usize,
}

impl TlabBlock {
    /// Increment the reference count (one ref per sub-allocation).
    #[inline]
    pub(crate) unsafe fn add_ref(block: *mut TlabBlock) {
        unsafe {
            (*block).ref_count += 1;
        }
    }

    /// Decrement the reference count. If it reaches 0, frees the block.
    ///
    /// # Safety
    /// `block` must be a valid pointer from a previous `alloc_block` call.
    /// Must not be called more times than `add_ref` + 1 (the initial ref).
    #[inline]
    pub(crate) unsafe fn release(block: *mut TlabBlock) {
        unsafe {
            (*block).ref_count -= 1;
            if (*block).ref_count == 0 {
                // Free buffer, then free the TlabBlock struct itself.
                let b = Box::from_raw(block);
                // b's Drop frees the buffer
                drop(b);
            }
        }
    }
}

impl Drop for TlabBlock {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            // SAFETY: `buffer` was allocated with `self.layout` via `std::alloc::alloc`
            // and has not been freed yet (this is the only place that frees it).
            unsafe {
                dealloc(self.buffer, self.layout);
            }
        }
    }
}

// SAFETY: The buffer is only accessed via properly aligned sub-pointers.
// The TlabBlock owns the allocation and frees it on drop.
// No mutable aliasing occurs — once a sub-region is handed out, only the
// recipient uses it. The ref_count is only mutated under the gc_maps_lock
// Mutex or from the owning thread (thread-local GC).
unsafe impl Send for TlabBlock {}
unsafe impl Sync for TlabBlock {}

/// Thread-Local Allocation Buffer.
///
/// Manages a current `TlabBlock` and a bump-pointer (`cursor`) within it.
/// Allocations advance the cursor; when the block is exhausted, a new one
/// is allocated. Old blocks are kept alive via `ref_count` tracking in
/// the `TlabBlock` itself.
pub(crate) struct Tlab {
    /// The current block being allocated from. Holds one ref_count.
    current_block: *mut TlabBlock,
    /// Byte offset from the start of the current block's buffer.
    cursor: usize,
    /// Total capacity of the current block in bytes.
    capacity: usize,
}

impl Tlab {
    /// Create a new TLAB with a freshly allocated block of `TLAB_DEFAULT_SIZE`.
    /// Returns `None` if the system allocator fails.
    pub(crate) fn new() -> Option<Tlab> {
        Self::with_capacity(TLAB_DEFAULT_SIZE)
    }

    /// Create a TLAB with a specified block capacity.
    /// Returns `None` if the system allocator fails.
    pub(crate) fn with_capacity(capacity: usize) -> Option<Tlab> {
        let block = Self::alloc_block(capacity)?;
        Some(Tlab {
            current_block: block,
            cursor: 0,
            capacity,
        })
    }

    /// Allocate a new `TlabBlock` of the given size, returning a raw pointer.
    /// The block starts with ref_count = 1 (the Tlab's own reference).
    fn alloc_block(size: usize) -> Option<*mut TlabBlock> {
        let layout = Layout::from_size_align(size, 16).ok()?;
        // SAFETY: Layout is valid (non-zero size, power-of-two alignment).
        let buffer = unsafe { alloc(layout) };
        if buffer.is_null() {
            return None;
        }
        let block = Box::new(TlabBlock {
            buffer,
            layout,
            ref_count: 1, // Tlab owns one reference
        });
        Some(Box::into_raw(block))
    }

    /// Try to allocate `layout.size()` bytes with `layout.align()` alignment
    /// from the current TLAB block.
    ///
    /// Returns `Some((ptr, block_ptr))` on success, where `ptr` is the
    /// sub-allocation pointer and `block_ptr` is a raw pointer to the
    /// backing block (the caller must call `TlabBlock::release` when the
    /// object is collected).
    ///
    /// Returns `None` if the current block doesn't have enough space.
    #[inline]
    pub(crate) fn alloc(&mut self, layout: Layout) -> Option<(*mut u8, *mut TlabBlock)> {
        let align = layout.align();
        let size = layout.size();

        if size == 0 {
            return None;
        }

        let aligned_cursor = align_up(self.cursor, align);

        if aligned_cursor + size <= self.capacity {
            // SAFETY: `current_block.buffer` is valid for `self.capacity` bytes.
            let ptr = unsafe { (*self.current_block).buffer.add(aligned_cursor) };
            self.cursor = aligned_cursor + size;
            // SAFETY: current_block is valid; increment ref for this sub-allocation.
            unsafe { TlabBlock::add_ref(self.current_block) };
            Some((ptr, self.current_block))
        } else {
            None
        }
    }

    /// Allocate from a fresh TLAB block when the current one is exhausted.
    pub(crate) fn alloc_slow(&mut self, layout: Layout) -> Option<(*mut u8, *mut TlabBlock)> {
        let size = layout.size();
        let min_block_size = TLAB_DEFAULT_SIZE.max(size + layout.align());
        let new_block = Self::alloc_block(min_block_size)?;
        let capacity = min_block_size;

        // Release Tlab's reference on the old block.
        // SAFETY: current_block is valid; Tlab holds one ref.
        unsafe { TlabBlock::release(self.current_block) };

        self.current_block = new_block;
        self.cursor = 0;
        self.capacity = capacity;

        self.alloc(layout)
    }

    /// Convenience: try the fast path, then the slow path.
    #[inline]
    pub(crate) fn alloc_or_grow(&mut self, layout: Layout) -> Option<(*mut u8, *mut TlabBlock)> {
        self.alloc(layout).or_else(|| self.alloc_slow(layout))
    }

    /// Return the number of bytes remaining in the current block.
    #[allow(dead_code)]
    pub(crate) fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.cursor)
    }
}

impl Drop for Tlab {
    fn drop(&mut self) {
        // Release Tlab's reference on the current block.
        // SAFETY: current_block is valid; Tlab holds one ref.
        unsafe { TlabBlock::release(self.current_block) };
    }
}

/// Align `offset` up to the next multiple of `align`.
/// `align` must be a power of two.
#[inline]
fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tlab_basic_alloc() {
        let mut tlab = Tlab::new().expect("failed to create TLAB");
        let layout = Layout::new::<u64>();
        let result = tlab.alloc(layout);
        assert!(result.is_some());
        let (ptr, block) = result.unwrap();
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % std::mem::align_of::<u64>(), 0);
        // Release the sub-allocation ref
        unsafe { TlabBlock::release(block) };
    }

    #[test]
    fn tlab_multiple_allocs() {
        let mut tlab = Tlab::new().expect("failed to create TLAB");
        let layout = Layout::new::<u64>();
        let mut ptrs = Vec::new();
        let mut blocks = Vec::new();
        for _ in 0..100 {
            let (ptr, block) = tlab.alloc(layout).expect("alloc failed");
            ptrs.push(ptr);
            blocks.push(block);
        }
        for (i, &p) in ptrs.iter().enumerate() {
            assert_eq!(p as usize % std::mem::align_of::<u64>(), 0);
            for (j, &q) in ptrs.iter().enumerate() {
                if i != j {
                    assert_ne!(p, q);
                }
            }
        }
        // Release all refs
        for block in blocks {
            unsafe { TlabBlock::release(block) };
        }
    }

    #[test]
    fn tlab_exhaustion_returns_none() {
        let mut tlab = Tlab::with_capacity(32).expect("failed to create TLAB");
        let layout = Layout::from_size_align(64, 8).unwrap();
        let result = tlab.alloc(layout);
        assert!(result.is_none());
    }

    #[test]
    fn tlab_alloc_slow_grows() {
        let mut tlab = Tlab::with_capacity(32).expect("failed to create TLAB");
        let layout = Layout::from_size_align(64, 8).unwrap();
        assert!(tlab.alloc(layout).is_none());
        let result = tlab.alloc_slow(layout);
        assert!(result.is_some());
        let (_, block) = result.unwrap();
        unsafe { TlabBlock::release(block) };
    }

    #[test]
    fn tlab_alloc_or_grow() {
        let mut tlab = Tlab::with_capacity(32).expect("failed to create TLAB");
        let layout = Layout::from_size_align(64, 8).unwrap();
        let result = tlab.alloc_or_grow(layout);
        assert!(result.is_some());
        let (_, block) = result.unwrap();
        unsafe { TlabBlock::release(block) };
    }

    #[test]
    fn tlab_block_freed_when_all_refs_dropped() {
        let mut tlab = Tlab::with_capacity(256).expect("failed to create TLAB");
        let layout = Layout::new::<u64>();

        let (_ptr1, block1) = tlab.alloc(layout).unwrap();
        let (_ptr2, block2) = tlab.alloc(layout).unwrap();

        // Both point to the same block
        assert_eq!(block1, block2);
        // ref_count: 1 (tlab) + 2 (allocs) = 3
        assert_eq!(unsafe { (*block1).ref_count }, 3);

        // Force a new block
        tlab.alloc_slow(layout).unwrap();
        // Old block's tlab ref was released: ref_count = 2
        assert_eq!(unsafe { (*block1).ref_count }, 2);

        // Release the slow-alloc ref
        let (_, new_block) = tlab.alloc(Layout::new::<u8>()).unwrap();
        unsafe { TlabBlock::release(new_block) };

        unsafe { TlabBlock::release(block1) };
        assert_eq!(unsafe { (*block2).ref_count }, 1);
        // When block2 ref is released, the TlabBlock memory is freed
        unsafe { TlabBlock::release(block2) };
    }

    #[test]
    fn tlab_alignment_varied() {
        let mut tlab = Tlab::new().expect("failed to create TLAB");
        let mut blocks = Vec::new();

        for &align in &[1, 2, 4, 8, 16] {
            let layout = Layout::from_size_align(align, align).unwrap();
            let (ptr, block) = tlab.alloc(layout).expect("alloc failed");
            assert_eq!(ptr as usize % align, 0, "ptr not aligned to {}", align);
            blocks.push(block);
        }
        for block in blocks {
            unsafe { TlabBlock::release(block) };
        }
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(5, 1), 5);
    }
}
