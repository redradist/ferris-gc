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
//! counted via `Arc<TlabBlock>`. When all objects in a block have been
//! collected (all `Arc` clones dropped), the entire block is freed.

use std::alloc::{Layout, alloc, dealloc};
use std::sync::Arc;

/// Default TLAB block size: 64 KiB.
const TLAB_DEFAULT_SIZE: usize = 64 * 1024;

/// A contiguous memory block backing TLAB sub-allocations.
///
/// The block owns its allocation and frees it on drop. Individual objects
/// within the block hold `Arc<TlabBlock>` to keep the block alive until
/// all objects are collected.
pub(crate) struct TlabBlock {
    buffer: *mut u8,
    layout: Layout,
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
// recipient uses it. The TlabBlock itself is behind an Arc and is never
// mutated after creation.
unsafe impl Send for TlabBlock {}
unsafe impl Sync for TlabBlock {}

/// Thread-Local Allocation Buffer.
///
/// Manages a current `TlabBlock` and a bump-pointer (`cursor`) within it.
/// Allocations advance the cursor; when the block is exhausted, a new one
/// is allocated. Old blocks are kept alive via `Arc<TlabBlock>` references
/// held by the objects allocated from them.
pub(crate) struct Tlab {
    /// The current block being allocated from.
    current_block: Arc<TlabBlock>,
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
            current_block: Arc::new(block),
            cursor: 0,
            capacity,
        })
    }

    /// Allocate a new `TlabBlock` of the given size.
    fn alloc_block(size: usize) -> Option<TlabBlock> {
        // Use max alignment of 16 bytes to satisfy most type alignments.
        // Individual allocations will further align within the block.
        let layout = Layout::from_size_align(size, 16).ok()?;
        // SAFETY: Layout is valid (non-zero size, power-of-two alignment).
        let buffer = unsafe { alloc(layout) };
        if buffer.is_null() {
            return None;
        }
        Some(TlabBlock { buffer, layout })
    }

    /// Try to allocate `layout.size()` bytes with `layout.align()` alignment
    /// from the current TLAB block.
    ///
    /// Returns `Some((ptr, block_arc))` on success, where `ptr` is the
    /// sub-allocation pointer and `block_arc` is an `Arc` reference to the
    /// backing block (the caller must keep this alive until the object is
    /// collected).
    ///
    /// Returns `None` if the current block doesn't have enough space.
    /// The caller should call `alloc_slow` to get a new block and retry,
    /// or fall back to the system allocator.
    pub(crate) fn alloc(&mut self, layout: Layout) -> Option<(*mut u8, Arc<TlabBlock>)> {
        let align = layout.align();
        let size = layout.size();

        if size == 0 {
            // Zero-sized types don't need real memory, fall back to system allocator.
            return None;
        }

        // Align cursor up to the required alignment.
        let aligned_cursor = align_up(self.cursor, align);

        if aligned_cursor + size <= self.capacity {
            // SAFETY: `current_block.buffer` is valid for `self.capacity` bytes.
            // `aligned_cursor + size <= self.capacity` ensures the sub-region is
            // within bounds. The alignment is correct because `aligned_cursor` is
            // a multiple of `align`.
            let ptr = unsafe { self.current_block.buffer.add(aligned_cursor) };
            self.cursor = aligned_cursor + size;
            Some((ptr, Arc::clone(&self.current_block)))
        } else {
            None
        }
    }

    /// Allocate from a fresh TLAB block when the current one is exhausted.
    ///
    /// Allocates a new block (at least `TLAB_DEFAULT_SIZE` or large enough for
    /// the requested layout), replaces the current block, and performs the
    /// allocation from the new block.
    ///
    /// Returns `None` if the system allocator fails to provide a new block.
    pub(crate) fn alloc_slow(&mut self, layout: Layout) -> Option<(*mut u8, Arc<TlabBlock>)> {
        let size = layout.size();
        // New block must be at least TLAB_DEFAULT_SIZE, but also large enough
        // for this specific allocation (including worst-case alignment padding).
        let min_block_size = TLAB_DEFAULT_SIZE.max(size + layout.align());
        let new_block = Self::alloc_block(min_block_size)?;
        let capacity = min_block_size;

        self.current_block = Arc::new(new_block);
        self.cursor = 0;
        self.capacity = capacity;

        self.alloc(layout)
    }

    /// Convenience: try the fast path, then the slow path.
    pub(crate) fn alloc_or_grow(&mut self, layout: Layout) -> Option<(*mut u8, Arc<TlabBlock>)> {
        self.alloc(layout).or_else(|| self.alloc_slow(layout))
    }

    /// Return the number of bytes remaining in the current block.
    #[allow(dead_code)]
    pub(crate) fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.cursor)
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
        let (ptr, _block) = result.unwrap();
        assert!(!ptr.is_null());
        // Pointer should be aligned for u64
        assert_eq!(ptr as usize % std::mem::align_of::<u64>(), 0);
    }

    #[test]
    fn tlab_multiple_allocs() {
        let mut tlab = Tlab::new().expect("failed to create TLAB");
        let layout = Layout::new::<u64>();
        let mut ptrs = Vec::new();
        // Allocate many objects
        for _ in 0..100 {
            let (ptr, _block) = tlab.alloc(layout).expect("alloc failed");
            ptrs.push(ptr);
        }
        // All pointers should be unique and aligned
        for (i, &p) in ptrs.iter().enumerate() {
            assert_eq!(p as usize % std::mem::align_of::<u64>(), 0);
            for (j, &q) in ptrs.iter().enumerate() {
                if i != j {
                    assert_ne!(p, q);
                }
            }
        }
    }

    #[test]
    fn tlab_exhaustion_returns_none() {
        // Create a tiny TLAB
        let mut tlab = Tlab::with_capacity(32).expect("failed to create TLAB");
        let layout = Layout::from_size_align(64, 8).unwrap();
        // Should fail — 64 bytes doesn't fit in 32 byte block
        let result = tlab.alloc(layout);
        assert!(result.is_none());
    }

    #[test]
    fn tlab_alloc_slow_grows() {
        let mut tlab = Tlab::with_capacity(32).expect("failed to create TLAB");
        let layout = Layout::from_size_align(64, 8).unwrap();
        // Fast path fails
        assert!(tlab.alloc(layout).is_none());
        // Slow path allocates a new block
        let result = tlab.alloc_slow(layout);
        assert!(result.is_some());
    }

    #[test]
    fn tlab_alloc_or_grow() {
        let mut tlab = Tlab::with_capacity(32).expect("failed to create TLAB");
        let layout = Layout::from_size_align(64, 8).unwrap();
        // alloc_or_grow should succeed even though fast path fails
        let result = tlab.alloc_or_grow(layout);
        assert!(result.is_some());
    }

    #[test]
    fn tlab_block_freed_when_all_arcs_dropped() {
        let mut tlab = Tlab::with_capacity(256).expect("failed to create TLAB");
        let layout = Layout::new::<u64>();

        let (_ptr1, block1) = tlab.alloc(layout).unwrap();
        let (_ptr2, block2) = tlab.alloc(layout).unwrap();

        // Both Arcs point to the same block
        assert!(Arc::ptr_eq(&block1, &block2));
        // 1 (tlab.current_block) + 2 (block1, block2) = 3
        assert_eq!(Arc::strong_count(&block1), 3);

        // Force a new block by exhausting or replacing
        tlab.alloc_slow(layout).unwrap();
        // Now tlab.current_block is a different block, so block1/block2 refcount = 2
        assert_eq!(Arc::strong_count(&block1), 2);

        drop(block1);
        assert_eq!(Arc::strong_count(&block2), 1);
        // When block2 is dropped, the TlabBlock memory is freed
        // (no crash = success)
        drop(block2);
    }

    #[test]
    fn tlab_alignment_varied() {
        let mut tlab = Tlab::new().expect("failed to create TLAB");

        // Allocate with different alignments
        for &align in &[1, 2, 4, 8, 16] {
            let layout = Layout::from_size_align(align, align).unwrap();
            let (ptr, _block) = tlab.alloc(layout).expect("alloc failed");
            assert_eq!(ptr as usize % align, 0, "ptr not aligned to {}", align);
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
