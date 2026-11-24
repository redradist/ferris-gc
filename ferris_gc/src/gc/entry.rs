//! Per-object bookkeeping shared by both collectors: where an object's
//! memory came from (`GcObjMem`), its `ObjectEntry`, the sync object header
//! (`SyncGcInfo`).
use super::*;

/// Where an object's block came from, so it can be freed. Both collectors
/// allocate an object and its `ObjectEntry` as one block
/// ([`object_block_layout`]), so freeing the block frees both. Eight bytes:
/// the `NonNull` leaves room for `Mem`.
#[derive(Clone, Copy)]
pub(crate) enum GcObjMem {
    /// The sync collector's [`alloc_sync_mem`]: its pool or the system
    /// allocator.
    Mem,
    /// The thread-local collector's TLAB: releasing the block's non-atomic
    /// ref count frees it once all its objects are gone. A TLAB grows a block
    /// to fit any object, so it fails only when the system allocator does.
    Tlab(std::ptr::NonNull<crate::tlab::TlabBlock>),
}

/// Layout of one object's block `[A (object) | ObjectEntry]`, and the
/// entry's offset in it.
#[inline]
pub(crate) fn object_block_layout<A>() -> (Layout, usize) {
    let (block, off_e) = Layout::new::<A>()
        .extend(Layout::new::<ObjectEntry>())
        .expect("object block layout");
    (block.pad_to_align(), off_e)
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

/// Per-object metadata stored in a contiguous SlotMap. It lives in its
/// object's block (see [`GcObjMem`]).
pub(crate) struct ObjectEntry {
    pub(crate) ptr: *const dyn Trace,
    pub(crate) mem: GcObjMem,
    /// The thread-local collector records the object's layout, the sync
    /// collector the block's, which its pool frees by.
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
