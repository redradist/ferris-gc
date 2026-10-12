//! Per-object bookkeeping shared by both collectors: where an object's
//! memory came from (`GcObjMem`), its `ObjectEntry`, the sync object header
//! (`SyncGcInfo`), and compacted blocks.
use super::*;

/// Tracks the origin of a GC allocation so it can be correctly freed.
///
/// - `Mem` — allocated via `std::alloc::alloc`; freed with `std::alloc::dealloc`.
/// - `Tlab` — sub-allocated from a TLAB block; freed by releasing the block's
///   non-atomic ref_count, which frees the entire block once all refs are gone.
pub(crate) enum GcObjMem {
    /// System-allocated memory. The raw pointer is derived from the owning
    /// `ObjectEntry`'s fat `*const dyn Trace` thin pointer.
    Mem,
    /// TLAB sub-allocation. The `*mut TlabBlock` holds a ref_count reference
    /// to the backing block. Call `TlabBlock::release` to decrement.
    Tlab(*mut crate::tlab::TlabBlock),
    /// Compacted memory sub-allocation. The `Arc<CompactBlock>` keeps the
    /// contiguous buffer alive until all compacted objects are collected.
    Compact(Arc<CompactBlock>),
    /// Part of a combined allocation (e.g. an `ObjectEntry` co-allocated with
    /// the object). No separate deallocation needed — freed with the object.
    Inline,
}

impl GcObjMem {
    /// Deallocate this memory. For `Mem`, calls `std::alloc::dealloc`.
    /// For `Tlab`, releases the block's ref_count (block freed when count hits 0).
    ///
    /// # Safety
    /// For `Mem` variant, `ptr` must have been allocated with the given layout.
    /// The memory must not have been previously deallocated.
    pub(crate) unsafe fn dealloc_mem(self, ptr: *mut u8, layout: Layout) {
        match self {
            GcObjMem::Mem => {
                // SAFETY: Caller guarantees ptr was allocated with this layout.
                unsafe { dealloc(ptr, layout) };
            }
            GcObjMem::Tlab(block) => {
                // SAFETY: block pointer is valid; decrement ref_count.
                // When ref_count reaches 0, the block buffer is freed.
                unsafe { crate::tlab::TlabBlock::release(block) };
            }
            GcObjMem::Compact(_block) => {
                // Dropping the Arc<CompactBlock>. When all Arcs for this block
                // are dropped, the CompactBlock::drop frees the entire buffer.
            }
            GcObjMem::Inline => {
                // No-op: inline allocations are freed with their parent object.
            }
        }
    }
}

/// Allocate memory for the sync collector: a block of the per-thread pool
/// when the layout fits one of its size classes, the system allocator
/// otherwise. Recorded as `GcObjMem::Mem`; [`free_sync_mem`] tells the two
/// apart by the layout alone. Returns null if the allocator fails.
///
/// # Safety
/// `layout` must be non-zero-sized.
#[inline]
pub(crate) unsafe fn alloc_sync_mem(layout: Layout) -> *mut u8 {
    if crate::pool::serves(layout) {
        return crate::pool::alloc_block(layout).map_or(std::ptr::null_mut(), |p| p.as_ptr());
    }
    // SAFETY: Caller guarantees a non-zero-sized layout.
    unsafe { alloc(layout) }
}

/// Free memory from [`alloc_sync_mem`].
///
/// # Safety
/// `ptr` must come from `alloc_sync_mem(layout)` with this same `layout`,
/// and not have been freed already.
#[inline]
pub(crate) unsafe fn free_sync_mem(ptr: *mut u8, layout: Layout) {
    if crate::pool::serves(layout) {
        // SAFETY: a pool-served layout was allocated from the pool.
        unsafe { crate::pool::free_block(ptr) };
    } else {
        // SAFETY: otherwise it came from the system allocator.
        unsafe { dealloc(ptr, layout) };
    }
}

/// Compact layout: size as u32, alignment as log2.
/// Saves 8B vs std::alloc::Layout (8B vs 16B).
#[derive(Clone, Copy)]
pub(crate) struct CompactLayout {
    size: u32,
    align_shift: u8,
}

impl CompactLayout {
    #[inline]
    pub(crate) fn from_layout(layout: Layout) -> Self {
        CompactLayout {
            size: layout.size() as u32,
            align_shift: layout.align().trailing_zeros() as u8,
        }
    }
    #[inline]
    pub(crate) fn size(self) -> usize {
        self.size as usize
    }
    #[inline]
    pub(crate) fn to_layout(self) -> Layout {
        // SAFETY: Original Layout was valid; size fits in u32 (< 4GB),
        // alignment is a power of two reconstructed from trailing_zeros.
        unsafe { Layout::from_size_align_unchecked(self.size as usize, 1usize << self.align_shift) }
    }
}

/// `ObjectEntry::entry_block` value for an entry embedded in its object's own
/// allocation, so that freeing the object frees the entry too. A dangling (but
/// non-null, aligned) pointer that is only ever compared, never dereferenced.
pub(crate) const ENTRY_EMBEDDED: *mut crate::tlab::TlabBlock = std::ptr::dangling_mut();

/// Per-object metadata stored in a contiguous SlotMap.
pub(crate) struct ObjectEntry {
    pub(crate) ptr: *const dyn Trace,
    pub(crate) mem: GcObjMem,
    pub(crate) layout: CompactLayout,
    /// Packed generation (bits 31:30) + survive_count (bits 29:16, 14 bits) + region (bits 15:0).
    pub(crate) gen_survive_region: u32,
    /// Total number of live Gc<T>/GcCell<T> handles pointing to this object.
    /// RC hybrid: when handle_count reaches 0, the object is eagerly deallocated.
    /// Thread-local GC only: sync handles count themselves atomically in the
    /// object header (`SyncGcInfo::handle_count`).
    pub(crate) handle_count: u32,
    /// Number of handles that are currently roots (stack-owned, not yet reset by cascade).
    /// During collection: initialized to handle_count, then decremented by reset_root cascade.
    /// Objects with root_count > 0 after cascade are reachable from the stack.
    /// Both collectors (the sync one copies it from `SyncGcInfo::root_count`).
    pub(crate) root_count: u32,
    /// Offset from the thin pointer to the object's `root_ref_count` Cell inside LocalGcInfo.
    /// Allows inline mark-bit checks (is_traceable / clear_trace) without
    /// virtual dispatch through `*const dyn Trace`.
    pub(crate) root_ref_count_offset: u16,
    /// TLAB block that holds this ObjectEntry allocation (null = system-allocated).
    /// Released when the ObjectEntry is removed from the SlotMap and deallocated.
    ///
    /// [`ENTRY_EMBEDDED`] marks an entry that lives inside its own object's
    /// allocation (the sync collector's single `[object | entry]`
    /// block): it is freed together with the object, never on its own.
    pub(crate) entry_block: *mut crate::tlab::TlabBlock,
}

/// Header at the start of every sync collector object (`sync::GcPtr<T>`,
/// `sync::GcCellBox<T>`). `ObjectEntry::root_ref_count_offset` points at its
/// first field, so the collector reaches it without knowing `T`.
#[repr(C)]
pub(crate) struct SyncGcInfo {
    /// Mark bit and cascade visit guard, as in the thread-local `LocalGcInfo`.
    pub(crate) root_ref_count: AtomicUsize,
    /// Live handles (`sync::Gc`, `sync::GcCell`). A clone adds one, a drop
    /// subtracts one, and the drop that reaches zero frees the object - unless
    /// a collection freed it first (dead cycle), which the drop then finds
    /// gone from the object map.
    pub(crate) handle_count: AtomicU32,
    /// Collector only (root discovery, under the STW write lock): the handle
    /// count minus the handles found inside scanned objects. Above zero, some
    /// handle lives outside the heap - the object is a root.
    pub(crate) root_count: AtomicU32,
    /// The object's id. Written once at creation, under its shard's lock and
    /// before any handle exists; never written again.
    pub(crate) object_id: ObjectId,
}

impl SyncGcInfo {
    #[inline]
    pub(crate) fn new() -> SyncGcInfo {
        SyncGcInfo {
            root_ref_count: AtomicUsize::new(0),
            handle_count: AtomicU32::new(1),
            root_count: AtomicU32::new(0),
            // Replaced by the real id before the object is published.
            object_id: <ObjectId as crate::slot_map::SlotKey>::from_raw(u32::MAX, u32::MAX),
        }
    }
}

/// A pointer to an ObjectEntry allocated in the TLAB (or heap).
/// Stored in the SlotMap instead of the full ObjectEntry to reduce copy
/// overhead from 72B to 8B per insert/remove.
#[derive(Clone, Copy)]
pub(crate) struct ObjectEntryRef(pub(crate) *mut ObjectEntry);

impl Deref for ObjectEntryRef {
    type Target = ObjectEntry;
    #[inline]
    fn deref(&self) -> &ObjectEntry {
        unsafe { &*self.0 }
    }
}

impl DerefMut for ObjectEntryRef {
    #[inline]
    fn deref_mut(&mut self) -> &mut ObjectEntry {
        unsafe { &mut *self.0 }
    }
}

impl ObjectEntry {
    #[inline]
    pub(crate) fn generation(&self) -> Generation {
        match self.gen_survive_region >> 30 {
            0 => Generation::Gen0,
            1 => Generation::Gen1,
            _ => Generation::Gen2,
        }
    }

    #[inline]
    pub(crate) fn set_generation(&mut self, generation: Generation) {
        let bits = match generation {
            Generation::Gen0 => 0u32,
            Generation::Gen1 => 1u32,
            Generation::Gen2 => 2u32,
        };
        self.gen_survive_region = (bits << 30) | (self.gen_survive_region & 0x3FFF_FFFF);
    }

    #[inline]
    pub(crate) fn survive_count(&self) -> u32 {
        (self.gen_survive_region >> 16) & 0x3FFF
    }

    #[inline]
    pub(crate) fn set_survive_count(&mut self, count: u32) {
        let count = count.min(0x3FFF);
        self.gen_survive_region = (self.gen_survive_region & 0xC000_FFFF) | (count << 16);
    }

    #[inline]
    pub(crate) fn increment_survive_count(&mut self) {
        let current = self.survive_count();
        if current < 0x3FFF {
            self.set_survive_count(current + 1);
        }
    }

    #[inline]
    pub(crate) fn region(&self) -> RegionId {
        RegionId(self.gen_survive_region & 0xFFFF)
    }

    /// The sync collector's object header. Only for entries of the global
    /// (sharded) collector, whose objects all start their header with
    /// `root_ref_count`.
    /// SAFETY: The ObjectEntry's ptr must be valid.
    #[inline]
    pub(crate) unsafe fn sync_info(&self) -> &SyncGcInfo {
        unsafe {
            &*((self.ptr.get_thin_ptr() + self.root_ref_count_offset as usize) as *const SyncGcInfo)
        }
    }

    /// Get a reference to root_ref_count by deriving from the thin pointer + offset.
    /// SAFETY: The ObjectEntry's ptr must be valid.
    #[inline]
    pub(crate) unsafe fn root_ref_count(&self) -> &Cell<usize> {
        unsafe {
            &*((self.ptr.get_thin_ptr() + self.root_ref_count_offset as usize)
                as *const Cell<usize>)
        }
    }

    /// Get a reference to LocalGcInfo.root_count via thin pointer + known offset.
    /// The stored offset points at LocalGcInfo.root_ref_count; root_count's address
    /// is derived through `offset_of!` so it stays correct regardless of how
    /// the compiler orders LocalGcInfo's fields (repr(Rust) makes no ordering
    /// guarantee, and e.g. -Zrandomize-layout would break a hard-coded
    /// "right after root_ref_count" assumption).
    /// SAFETY: The ObjectEntry's ptr must be valid.
    #[inline]
    pub(crate) unsafe fn gc_info_root_count(&self) -> &Cell<u32> {
        unsafe {
            let offset = self.root_ref_count_offset as usize
                - std::mem::offset_of!(LocalGcInfo, root_ref_count)
                + std::mem::offset_of!(LocalGcInfo, root_count);
            &*((self.ptr.get_thin_ptr() + offset) as *const Cell<u32>)
        }
    }
}

/// Unified GC maps. Single Mutex protects all state to simplify locking.

/// RAII guard for a compacted memory block.
/// Frees the contiguous buffer when all objects in it have been collected.
pub(crate) struct CompactBlock {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: The block is only freed when all Arc references are dropped (during sweep),
// and the GC ensures no references exist at that point.
unsafe impl Send for CompactBlock {}
unsafe impl Sync for CompactBlock {}

impl CompactBlock {
    /// Own the buffer at `ptr`, allocated with `layout`.
    pub(crate) fn new(ptr: *mut u8, layout: Layout) -> CompactBlock {
        CompactBlock { ptr, layout }
    }
}

impl Drop for CompactBlock {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            // SAFETY: ptr was allocated with this layout via alloc().
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }
}
