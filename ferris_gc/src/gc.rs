use crate::card_table::CardTable;
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::generation::{
    CollectionPhase, CollectionStats, GcStats, Generation, MarkColor, PromotionConfig, RegionId,
};
use crate::sharded_map::{Counter, ShardedSlotMap};
use crate::slot_map::{ObjectId, SlotMap};
use std::alloc::{Layout, alloc, dealloc};
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::basic_strategy::basic_strategy_start;

mod entry;
mod local;
mod maps;
pub mod sync;
mod trace;

pub(crate) use entry::*;
pub use local::*;
pub(crate) use maps::*;
pub use trace::*;

thread_local! {
    /// Tagged context pointer for reset_root cascade.
    ///
    /// Encodes *both* a pointer to the active collector's `cascade_visited`
    /// buffer and the collecting flag in a single thread-local, eliminating one
    /// TLS read per node in the cascade hot path. It points at the buffer, not at
    /// `GcMaps`, because that buffer is the only state the cascade touches - so
    /// the context does not depend on which object-map type a collector uses.
    ///
    /// Encoding (bit 0 of the pointer, which is always free due to alignment):
    /// - `null`               → no cascade context
    /// - `ptr` (bit 0 clear)  → allocation-time cascade (only decrement if is_root)
    /// - `ptr | 1` (bit 0 set)→ collection cascade (always decrement root_count)
    pub(crate) static GC_MAPS_CTX: Cell<*mut CascadeVisited> = const { Cell::new(std::ptr::null_mut()) };

    /// The global collector's root-discovery cascade buffer while that
    /// collector discovers roots on this thread; null otherwise. Sync handles
    /// act only on this (`sync::cascade_handle`), so a thread-local
    /// collection that reaches a sync handle stored in a local object leaves
    /// the sync object's counts alone.
    pub(crate) static SYNC_CASCADE: Cell<*mut CascadeVisited> = const { Cell::new(std::ptr::null_mut()) };
}

/// Buffer of cascade guard cells set during root discovery (see
/// `GcMaps::cascade_visited`).
pub(crate) type CascadeVisited = Vec<*const Cell<usize>>;

/// Tag a cascade-context pointer as "collecting" by setting bit 0.
#[inline(always)]
fn gc_ctx_tag_collecting(ptr: *mut CascadeVisited) -> *mut CascadeVisited {
    (ptr as usize | 1) as *mut CascadeVisited
}

/// Extract the raw GcMaps pointer (strip tag bit).
#[inline(always)]
#[allow(dead_code)]
fn gc_ctx_ptr(tagged: *mut CascadeVisited) -> *mut CascadeVisited {
    (tagged as usize & !1usize) as *mut CascadeVisited
}

/// Check if the tagged pointer indicates collection mode.
#[inline(always)]
fn gc_ctx_is_collecting(tagged: *mut CascadeVisited) -> bool {
    (tagged as usize & 1) != 0
}

/// Error returned when a GC allocation fails due to memory exhaustion.
#[derive(Debug, Clone, Copy)]
pub struct GcAllocError;

impl std::fmt::Display for GcAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GC allocation failed")
    }
}

impl std::error::Error for GcAllocError {}

/// Sharded heap-size counters sample the peak each time a thread's shard
/// crosses a multiple of `1 << PEAK_SAMPLE_SHIFT` bytes (64 KiB).
const PEAK_SAMPLE_SHIFT: u32 = 16;

/// Shared GC bookkeeping used by both LocalGarbageCollector and GlobalGarbageCollector.
/// gc_maps is behind UnsafeCell for zero-cost thread-local access; gc_maps_lock
/// provides Mutex protection when cross-thread access is needed.
pub(crate) struct GarbageCollector<O: ObjectMap = LocalObjects> {
    pub(crate) gc_maps: UnsafeCell<GcMaps<O>>,
    /// Mutex for thread-safe gc_maps access (used by GlobalGC and background strategies).
    pub(crate) gc_maps_lock: Mutex<()>,
    /// One lock per object shard. Per-object work (allocate, clone a handle,
    /// RC-hybrid drop) takes only the relevant shard's lock; anything needing a
    /// consistent view of the whole map takes `gc_maps_lock` and then **every**
    /// shard lock in index order. That single fixed order (global, then shard
    /// 0..n) is the only one used, so the two paths cannot deadlock.
    pub(crate) shard_locks: Box<[crate::sharded_map::CachePadded<Mutex<()>>]>,
    /// Entry count of `weak_alive_map`, readable without any lock. While it is
    /// zero the weak map is definitely empty, so per-object hot paths can stay
    /// shard-scoped. Only written under the full-map lock (raised on weak
    /// insert, re-synced on the slow drop path); a stale non-zero costs a slower
    /// path, never correctness.
    pub(crate) weak_alive_len: AtomicUsize,
    /// Number of times marking could not scan a value because a
    /// `sync::GcCell` was mutably borrowed. A collection reads it when its
    /// marking starts; if it moved by the sweep, the marks are incomplete and
    /// the sweep frees nothing (see [`Self::keep_all_if_blocked_since`]). A
    /// counter rather than a flag, so an incremental cycle cannot lose a
    /// blocked scan to a full collection that runs between its steps.
    pub(crate) blocked_scans: AtomicUsize,
    /// Allocation counter for adaptive collection triggering.
    pub(crate) allocation_count: O::Counter,
    pub(crate) card_table: CardTable,
    pub(crate) stw_lock: crate::sharded_map::StwLock,
    pub(crate) incremental: Mutex<IncrementalState>,
    /// Cheap gate: true between `begin_collection` and `finish_collection` so
    /// the allocation hot path can skip locking `incremental` unless an
    /// incremental mark cycle is actually in progress. When set, newly created
    /// objects must be greyed into the incremental worklist (see
    /// `note_new_object_during_marking`) so the colors-based sweep in
    /// `finish_collection` does not reclaim live objects born mid-mark.
    pub(crate) marking: AtomicBool,
    /// When non-zero, the thread-local collector runs **on-thread incremental**
    /// Gen2 collection instead of a stop-the-world Gen0 collect: each allocation
    /// past the threshold advances the mark by this many objects (one bounded
    /// `mark_step`), so pauses stay short without a background collector thread
    /// racing the lockless mutator. Set by the local background/g1 strategies;
    /// zero (the default) keeps the classic full-collect behavior.
    pub(crate) local_incremental_budget: AtomicUsize,
    pub(crate) total_collections: AtomicUsize,
    pub(crate) last_collection: Mutex<Option<CollectionStats>>,
    /// Current region for new allocations.
    pub(crate) current_region: AtomicU32,
    /// Next region ID to assign.
    pub(crate) next_region_id: AtomicU32,
    /// Configurable promotion thresholds per generation.
    pub(crate) promotion_config: Mutex<PromotionConfig>,
    /// Current total heap size in bytes.
    pub(crate) current_heap_size: O::Counter,
    /// High-water mark of heap usage in bytes.
    pub(crate) peak_heap_size: AtomicUsize,
    /// Optional callback invoked after each collection cycle.
    #[allow(clippy::type_complexity)]
    pub(crate) on_collection: Mutex<Option<Box<dyn Fn(&CollectionStats) + Send + Sync>>>,
    /// Adaptive allocation threshold for maybe_collect.
    pub(crate) alloc_threshold: AtomicUsize,
    /// Thin addresses of dead objects whose destructors are currently running
    /// in a sweep dealloc phase. Interior `Gc`/`GcCell` handle drops consult
    /// this registry (see `is_dying`) so a handle pointing at a dead object of
    /// the same sweep batch — or at the object being dropped itself — becomes
    /// a no-op without ever dereferencing that object's memory.
    pub(crate) dying: Mutex<FxHashSet<usize>>,
    /// Fast-path gate for `is_dying`: number of entries in `dying`.
    /// Zero outside a sweep dealloc phase, so handle drops pay one relaxed
    /// atomic load in the common case.
    pub(crate) dying_len: AtomicUsize,
}

// SAFETY: GcMaps is behind UnsafeCell but protected by gc_maps_lock (Mutex<()>)
// for cross-thread access. Thread-local access bypasses the lock (single-threaded
// guarantee from thread_local!). All other mutable state is behind Mutex or atomics.
unsafe impl<O: ObjectMap> Sync for GarbageCollector<O> {}
unsafe impl<O: ObjectMap> Send for GarbageCollector<O> {}

#[allow(dead_code)]
impl<O: ObjectMap> GarbageCollector<O> {
    /// Thread-local collector: a single shard (one owning thread).
    pub(crate) fn new() -> Self {
        Self::with_shards(1)
    }

    /// Collector whose object map is split into `shards` thread-affine shards.
    /// Used by the global GC so concurrent allocation does not serialize.
    pub(crate) fn with_shards(shards: usize) -> Self {
        GarbageCollector {
            gc_maps: UnsafeCell::new(GcMaps {
                objects: O::with_shards(shards),
                ptr_to_object: FxHashMap::default(),
                weak_alive_map: FxHashMap::default(),
                gen0_ids: Vec::new(),
                cascade_visited: Vec::new(),
            }),
            gc_maps_lock: Mutex::new(()),
            shard_locks: (0..crate::sharded_map::shard_count_for(shards))
                .map(|_| crate::sharded_map::CachePadded(Mutex::new(())))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            weak_alive_len: AtomicUsize::new(0),
            blocked_scans: AtomicUsize::new(0),
            allocation_count: <O::Counter as Counter>::with_shards(shards),
            card_table: CardTable::new(),
            stw_lock: crate::sharded_map::StwLock::with_shards(shards),
            incremental: Mutex::new(IncrementalState::new()),
            marking: AtomicBool::new(false),
            local_incremental_budget: AtomicUsize::new(0),
            total_collections: AtomicUsize::new(0),
            last_collection: Mutex::new(None),
            current_region: AtomicU32::new(0),
            next_region_id: AtomicU32::new(1),
            promotion_config: Mutex::new(PromotionConfig::default()),
            current_heap_size: <O::Counter as Counter>::with_shards(shards),
            peak_heap_size: AtomicUsize::new(0),
            alloc_threshold: AtomicUsize::new(LOCAL_GC_ALLOC_THRESHOLD),
            on_collection: Mutex::new(None),
            dying: Mutex::new(FxHashSet::default()),
            dying_len: AtomicUsize::new(0),
        }
    }

    /// Is `addr` (thin pointer of a GC object allocation) part of the sweep
    /// batch whose destructors are currently running? Handle drops must call
    /// this BEFORE dereferencing their target: a `true` result means the
    /// object's entry is already gone from gc_maps and the collector owns all
    /// remaining bookkeeping, so the drop must do nothing at all.
    #[inline]
    pub(crate) fn is_dying(&self, addr: usize) -> bool {
        if self.dying_len.load(Ordering::Relaxed) == 0 {
            return false;
        }
        self.dying
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&addr)
    }

    /// Thread-safe locked access to gc_maps. Used by collection methods
    /// and GlobalGarbageCollector. Acquires gc_maps_lock Mutex.
    pub(crate) fn lock_gc_maps(&self) -> GcMapsGuard<'_, O> {
        let global = self.gc_maps_lock.lock().unwrap_or_else(|e| e.into_inner());
        // For a sharded map, acquire every shard lock in index order so this
        // guard also excludes the shard-scoped per-object paths. An unsharded
        // map has no such paths, so it takes only the global lock (and the
        // empty Vec does not allocate).
        let mut shards = Vec::new();
        if O::SHARDED {
            shards.reserve_exact(self.shard_locks.len());
            for l in self.shard_locks.iter() {
                shards.push(l.lock().unwrap_or_else(|e| e.into_inner()));
            }
        }
        // SAFETY: UnsafeCell::get never returns null; the global lock (plus
        // every shard lock when sharded) gives exclusive access to all of
        // gc_maps across threads.
        unsafe {
            let maps = std::ptr::NonNull::new_unchecked(self.gc_maps.get());
            GcMapsGuard::new(maps, global, shards)
        }
    }

    /// Unsynchronized zero-cost access to gc_maps for single-threaded use.
    ///
    /// # Safety
    /// Caller must guarantee no concurrent access (e.g., from a thread_local!
    /// context with no background collection thread accessing gc_maps).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn gc_maps_unsync(&self) -> &mut GcMaps<O> {
        unsafe { &mut *self.gc_maps.get() }
    }

    /// Set custom promotion thresholds.
    pub fn set_promotion_config(&self, config: PromotionConfig) {
        *self
            .promotion_config
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = config;
    }

    /// Get the current promotion config.
    pub fn promotion_config(&self) -> PromotionConfig {
        *self
            .promotion_config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> GcStats {
        let gc_maps = self.lock_gc_maps();
        let mut gen0 = 0usize;
        let mut gen1 = 0usize;
        let mut gen2 = 0usize;
        let mut heap_size = 0usize;
        for entry in gc_maps.objects.values() {
            match entry.generation() {
                Generation::Gen0 => gen0 += 1,
                Generation::Gen1 => gen1 += 1,
                Generation::Gen2 => gen2 += 1,
            }
            heap_size += entry.layout.size();
        }
        let live_handles = gc_maps
            .objects
            .values()
            .map(|e| {
                if O::SHARDED {
                    // SAFETY: registered objects are valid while the maps are locked.
                    unsafe { e.sync_info().handle_count.load(Ordering::Relaxed) as usize }
                } else {
                    e.handle_count as usize
                }
            })
            .sum::<usize>();
        #[allow(deprecated)]
        GcStats {
            heap_size,
            live_objects: gc_maps.objects.len(),
            live_handles,
            live_tracers: live_handles,
            gen0_objects: gen0,
            gen1_objects: gen1,
            gen2_objects: gen2,
            total_collections: self.total_collections.load(Ordering::Relaxed),
            last_collection: *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            // The sync collector samples its peak (see track_alloc); never
            // report it below the live heap.
            peak_heap_size: self
                .peak_heap_size
                .load(Ordering::Relaxed)
                .max(self.current_heap_size.load(Ordering::Relaxed)),
        }
    }

    /// Register a callback invoked after each collection cycle.
    pub fn set_on_collection(&self, callback: impl Fn(&CollectionStats) + Send + Sync + 'static) {
        *self.on_collection.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(callback));
    }

    /// Remove the collection callback.
    pub fn clear_on_collection(&self) {
        *self.on_collection.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Fire the on_collection callback (panic-safe).
    fn fire_on_collection(&self, stats: &CollectionStats) {
        let guard = self.on_collection.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cb) = guard.as_ref() {
            let cb: &(dyn Fn(&CollectionStats) + Send + Sync) = cb.as_ref();
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cb(stats)));
        }
    }

    /// Track heap growth after an allocation.
    #[inline]
    fn track_alloc(&self, size: usize) {
        let shard_now = self.current_heap_size.add(size);
        if <O::Counter as Counter>::SINGLE {
            // One shard: its value is the exact total, so the peak is exact.
            // A SingleCounter wraps a usize and is never negative, so the cast
            // needs no clamp.
            self.peak_heap_size
                .fetch_max(shard_now as usize, Ordering::Relaxed);
        } else if (shard_now - size as isize) >> PEAK_SAMPLE_SHIFT != shard_now >> PEAK_SAMPLE_SHIFT
        {
            // Sharded: an exact peak of the *sum* would need a shared write per
            // allocation - the contention this avoids. Sample it whenever this
            // thread's shard crosses a 64 KiB boundary, so the recorded peak is
            // within ~64 KiB per thread of the true one.
            self.peak_heap_size.fetch_max(
                self.current_heap_size.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
        }
    }

    /// Track heap shrinkage after a deallocation.
    #[inline]
    fn track_dealloc(&self, size: usize) {
        // Saturating for the single-shard (thread-local) counter; see
        // ShardedCounter::sub.
        self.current_heap_size.sub(size);
    }

    /// Finalize and drop a collected object in place. The object's memory is
    /// NOT freed — call `free_object_entry` afterwards.
    ///
    /// # Safety
    /// The `ObjectEntry` must refer to a valid, initialized object that has been
    /// removed from the GC maps and is no longer reachable.
    #[inline]
    unsafe fn drop_object_entry(entry_ref: ObjectEntryRef) {
        let fat_ptr = unsafe { (*entry_ref.0).ptr };
        // Finalize through vtable (GcPtr<T>::finalize delegates to T::finalize).
        unsafe { (&*fat_ptr).finalize() };
        // Drop through vtable drop glue (drops GcPtr<T> which drops T).
        unsafe { std::ptr::drop_in_place(fat_ptr as *mut dyn Trace) };
    }

    /// Free the memory of an already-dropped collected object and its
    /// `ObjectEntry`.
    ///
    /// # Safety
    /// `drop_object_entry` must have run for this entry, and no further access
    /// to the object or the entry may happen afterwards.
    // Forced for the same reason as create_gc: the RC-hybrid drop path.
    #[inline(always)]
    unsafe fn free_object_entry(entry_ref: ObjectEntryRef) {
        // Read all fields from the TLAB-allocated ObjectEntry before releasing memory.
        let oe_ptr = entry_ref.0;
        let fat_ptr = unsafe { (*oe_ptr).ptr };
        let layout = unsafe { (*oe_ptr).layout.to_layout() };
        let mem = unsafe { std::ptr::read(&(*oe_ptr).mem) };
        let entry_block = unsafe { (*oe_ptr).entry_block };

        let mem_ptr = fat_ptr.get_thin_ptr() as *mut u8;
        // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem or TLAB.
        unsafe { Self::dealloc_gc_mem(mem_ptr, mem, layout) };
        // Free the ObjectEntry memory itself.
        // Only the sync (sharded) collector embeds entries; the constant gate
        // compiles the check out of the thread-local drop path.
        if O::SHARDED && entry_block == ENTRY_EMBEDDED {
            // Lives inside the object's own block, which `dealloc_mem` above
            // just freed; every field was read before that.
        } else if !entry_block.is_null() {
            // TLAB-allocated: release the block reference.
            unsafe { crate::tlab::TlabBlock::release(entry_block) };
        } else {
            // Separately system-allocated (TLAB fallback): free via dealloc.
            unsafe { dealloc(oe_ptr as *mut u8, Layout::new::<ObjectEntry>()) };
        }
    }

    /// Finalize, drop, and deallocate a single collected object (RC-hybrid
    /// immediate-free path). A lone object freed because its handle count hit
    /// zero cannot reference itself, so no dying registration is needed:
    /// its interior handles point at other, still-valid allocations.
    ///
    /// # Safety
    /// The `ObjectEntry` must refer to a valid, initialized object that has been
    /// removed from the GC maps and is no longer reachable.
    #[inline]
    unsafe fn dealloc_object_entry(entry_ref: ObjectEntryRef) {
        unsafe { Self::drop_object_entry(entry_ref) };
        unsafe { Self::free_object_entry(entry_ref) };
    }

    /// Run the destructors of a sweep batch while EVERY dead allocation is
    /// still valid, with the batch registered in the dying registry. Interior
    /// `Gc`/`GcCell` handles dropped here check `is_dying` and no-op instead
    /// of dereferencing a dead (possibly already dropped) object — including
    /// the self-cycle case where the handle points at the object whose
    /// `drop_in_place` is currently running. Unregisters the batch before
    /// returning (memory frees run no user code, so the registry is only
    /// needed while destructors execute). Registration is add/remove of this
    /// batch's addresses only, so a nested emergency collection inside a
    /// finalizer keeps the outer batch registered.
    ///
    /// # Safety
    /// Every entry must be valid, removed from the GC maps, and unreachable.
    unsafe fn drop_collected(&self, object_deallocs: &[ObjectEntryRef]) {
        let addrs: Vec<usize> = object_deallocs
            .iter()
            .map(|entry| entry.ptr.get_thin_ptr())
            .collect();
        {
            let mut dying = self.dying.lock().unwrap_or_else(|e| e.into_inner());
            for &addr in &addrs {
                dying.insert(addr);
            }
            self.dying_len.store(dying.len(), Ordering::Relaxed);
        }
        for &entry in object_deallocs {
            // SAFETY: Object has been removed from gc_maps and is unreachable.
            unsafe { Self::drop_object_entry(entry) };
        }
        let mut dying = self.dying.lock().unwrap_or_else(|e| e.into_inner());
        for &addr in &addrs {
            dying.remove(&addr);
        }
        self.dying_len.store(dying.len(), Ordering::Relaxed);
    }

    /// Sweep dealloc phase: drop all dead objects first (two-phase, see
    /// `drop_collected`), then free their memory.
    ///
    /// # Safety
    /// Every entry must be valid, removed from the GC maps, and unreachable.
    unsafe fn dealloc_collected(&self, object_deallocs: Vec<ObjectEntryRef>) {
        if object_deallocs.is_empty() {
            return;
        }
        unsafe { self.drop_collected(&object_deallocs) };
        for entry in object_deallocs {
            // SAFETY: Object was dropped above; only its memory is freed here.
            unsafe { Self::free_object_entry(entry) };
        }
    }

    pub(crate) unsafe fn alloc_mem<T>(&self) -> (*mut T, (GcObjMem, Layout))
    where
        T: Sized,
    {
        let layout = Layout::new::<T>();
        // SAFETY: Layout is non-zero-sized and properly aligned for T; alloc returns a valid pointer or null.
        let mem = unsafe { alloc(layout) };
        if mem.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let type_ptr: *mut T = mem as *mut _;
        (type_ptr, (GcObjMem::Mem, layout))
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` instead of aborting on OOM.
    pub(crate) unsafe fn try_alloc_mem<T>(
        &self,
    ) -> Result<(*mut T, (GcObjMem, Layout)), GcAllocError>
    where
        T: Sized,
    {
        let layout = Layout::new::<T>();
        // SAFETY: Layout is non-zero-sized and properly aligned for T; alloc returns a valid pointer or null.
        let mem = unsafe { alloc(layout) };
        if mem.is_null() {
            return Err(GcAllocError);
        }
        let type_ptr: *mut T = mem as *mut _;
        Ok((type_ptr, (GcObjMem::Mem, layout)))
    }

    /// Fallible allocation with emergency GC collection on first failure.
    /// If the initial allocation fails, runs a full GC cycle to free dead objects,
    /// then retries once. Returns `Err(GcAllocError)` only if the retry also fails.
    pub(crate) unsafe fn try_alloc_mem_with_gc<T>(
        &self,
    ) -> Result<(*mut T, (GcObjMem, Layout)), GcAllocError>
    where
        T: Sized,
    {
        // SAFETY: Delegates to try_alloc_mem and collect, both of which uphold their own safety invariants.
        match unsafe { self.try_alloc_mem::<T>() } {
            Ok(result) => Ok(result),
            Err(_) => {
                // Emergency collection: free dead objects and retry
                unsafe { self.collect() };
                unsafe { self.try_alloc_mem::<T>() }
            }
        }
    }

    /// Get or create the alive flag for an object (used by weak references).
    pub(crate) fn get_or_create_weak_alive(&self, obj_id: ObjectId) -> Arc<AtomicBool> {
        let mut gc_maps = self.lock_gc_maps();
        if gc_maps.objects.get(obj_id).is_some() {
            let alive = gc_maps
                .weak_alive_map
                .entry(obj_id)
                .or_insert_with(|| Arc::new(AtomicBool::new(true)))
                .clone();
            // Raised under the full lock, before any shard-scoped drop of this
            // object can observe it: from here on drops take the full lock and
            // flip this flag.
            self.weak_alive_len
                .store(gc_maps.weak_alive_map.len(), Ordering::Release);
            return alive;
        }
        Arc::new(AtomicBool::new(false))
    }

    /// Write barrier: if the object is in Gen1+, mark its card dirty in the
    /// card table so that young-generation collections trace through it.
    /// During incremental marking, also re-grays Black objects to maintain
    /// the tri-color invariant (no Black->White edges).
    pub(crate) fn write_barrier(&self, obj_id: ObjectId, obj_ptr: *const dyn Trace) {
        let thin = obj_ptr.get_thin_ptr();
        let gc_maps = self.lock_gc_maps();
        if let Some(entry) = gc_maps.objects.get(obj_id) {
            if entry.generation() > Generation::Gen0 {
                self.card_table.mark_dirty(thin);
            }
        }
        drop(gc_maps);

        // During incremental marking, re-gray Black objects so their
        // new children are discovered in subsequent mark steps.
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        if incr.phase == CollectionPhase::Marking {
            if let Some(color) = incr.colors.get_mut(&obj_id) {
                if *color == MarkColor::Black {
                    *color = MarkColor::Gray;
                    incr.gray_stack.push(obj_id);
                }
            }
        }
    }

    /// Grey a freshly-allocated object into the incremental worklist when a
    /// mark cycle is in progress. Without this, an object born mid-mark is
    /// absent from `colors`, and the colors-based sweep in `finish_collection`
    /// would reclaim it (and never scan its children — which may include White
    /// objects reachable only through it). Greying it makes the sweep keep it
    /// and the drain scan its children. Cheap-gated on `self.marking` so the
    /// allocation hot path pays nothing outside a mark cycle.
    /// Current phase of the incremental collector (`Idle`/`Marking`/`Sweeping`).
    #[inline]
    pub(crate) fn incremental_phase(&self) -> CollectionPhase {
        self.incremental
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .phase
    }

    #[inline]
    pub(crate) fn note_new_object_during_marking(&self, obj_id: ObjectId) {
        if !self.marking.load(Ordering::Acquire) {
            return;
        }
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        if incr.phase == CollectionPhase::Marking {
            incr.colors.insert(obj_id, MarkColor::Gray);
            incr.gray_stack.push(obj_id);
        }
    }

    pub(crate) unsafe fn drop_handle(&self, object_id: ObjectId, is_root: bool) {
        let object_dealloc = {
            let mut gc_maps = self.lock_gc_maps();
            Self::drop_handle_inner(&mut gc_maps, &self.card_table, object_id, is_root)
        };
        if let Some(obj_entry) = object_dealloc {
            self.track_dealloc(obj_entry.layout.size());
            unsafe { Self::dealloc_object_entry(obj_entry) };
        }
    }

    #[inline]
    pub(crate) unsafe fn drop_handle_unsync(&self, object_id: ObjectId, is_root: bool) {
        let gc_maps = unsafe { self.gc_maps_unsync() };
        let object_dealloc = Self::drop_handle_inner(gc_maps, &self.card_table, object_id, is_root);
        if let Some(obj_entry) = object_dealloc {
            self.track_dealloc(obj_entry.layout.size());
            unsafe { Self::dealloc_object_entry(obj_entry) };
        }
    }

    #[inline]
    fn drop_handle_inner(
        gc_maps: &mut GcMaps<O>,
        card_table: &CardTable,
        object_id: ObjectId,
        is_root: bool,
    ) -> Option<ObjectEntryRef> {
        if let Some(obj_entry) = gc_maps.objects.get_mut(object_id) {
            if is_root {
                obj_entry.root_count = obj_entry.root_count.saturating_sub(1);
            }
            obj_entry.handle_count = obj_entry.handle_count.saturating_sub(1);
            if obj_entry.handle_count == 0 {
                let obj_gen = obj_entry.generation();
                // SAFETY: `object_id` was just found occupied.
                return Some(unsafe {
                    Self::unregister_object(gc_maps, card_table, object_id, obj_gen)
                });
            }
        }
        None
    }

    /// Full-lock variant of `release_object` (weak references present).
    fn release_object_inner(
        gc_maps: &mut GcMaps<O>,
        card_table: &CardTable,
        object_id: ObjectId,
    ) -> Option<ObjectEntryRef> {
        let obj_gen = gc_maps.objects.get(object_id)?.generation();
        // SAFETY: `object_id` was just found occupied.
        Some(unsafe { Self::unregister_object(gc_maps, card_table, object_id, obj_gen) })
    }

    /// Take an object that has no handles left out of the collector's books:
    /// its entry, its card-table registration and its weak flag. The caller
    /// frees it. Shared by both collectors' release paths (the sync shard
    /// fast path differs only in working on one shard with no weak refs).
    ///
    /// # Safety
    /// `object_id` must be occupied in `gc_maps`, with generation `obj_gen`.
    #[inline(always)]
    unsafe fn unregister_object(
        gc_maps: &mut GcMaps<O>,
        card_table: &CardTable,
        object_id: ObjectId,
        obj_gen: Generation,
    ) -> ObjectEntryRef {
        // SAFETY: the caller guarantees `object_id` is occupied.
        let entry = unsafe { gc_maps.objects.remove_unchecked(object_id) };
        if obj_gen != Generation::Gen0 {
            card_table.remove_object(entry.ptr.get_thin_ptr(), object_id);
        }
        if !gc_maps.weak_alive_map.is_empty() {
            if let Some(alive) = gc_maps.weak_alive_map.remove(&object_id) {
                alive.store(false, Ordering::Release);
            }
        }
        entry
    }

    /// Release object memory. The sync collector's `GcObjMem::Mem`
    /// came from [`alloc_sync_mem`] (possibly its pool); the constant gate
    /// keeps the thread-local paths exactly `dealloc_mem`.
    #[inline(always)]
    unsafe fn dealloc_gc_mem(ptr: *mut u8, mem: GcObjMem, layout: Layout) {
        if O::SHARDED && matches!(mem, GcObjMem::Mem) {
            unsafe { free_sync_mem(ptr, layout) };
        } else {
            unsafe { mem.dealloc_mem(ptr, layout) };
        }
    }

    /// Full collection (all generations). Backwards-compatible with existing callers.
    pub unsafe fn collect(&self) {
        // SAFETY: Caller upholds the safety contract; delegates to collect_generation with Gen2.
        unsafe {
            self.collect_generation(Generation::Gen2);
        }
    }

    /// Value of [`Self::blocked_scans`] when a collection's marking starts.
    #[inline]
    fn scan_epoch(&self) -> usize {
        self.blocked_scans.load(Ordering::Acquire)
    }

    /// If marking since `epoch` could not scan a mutably borrowed
    /// `sync::GcCell`, the marks are incomplete: mark every object live so
    /// this collection's sweep frees nothing; a later collection retries. The
    /// thread-local collector never blocks a scan.
    #[inline]
    fn keep_all_if_blocked_since(&self, gc_maps: &GcMaps<O>, epoch: usize) {
        if self.blocked_scans.load(Ordering::Acquire) != epoch {
            for entry in gc_maps.objects.values() {
                // SAFETY: every registered object is valid while the maps
                // are locked.
                unsafe { entry.root_ref_count().set(1) };
            }
        }
    }

    /// Handles a dead object still holds when a collection frees it (for
    /// `CollectionStats::handles_collected`): sync handles live inside other
    /// dead objects and are counted in the header.
    #[inline(always)]
    unsafe fn handles_held(entry: &ObjectEntry) -> usize {
        if O::SHARDED {
            unsafe { entry.sync_info().handle_count.load(Ordering::Relaxed) as usize }
        } else {
            // Thread-local handles die with their owner before the object
            // can be unreachable.
            0
        }
    }

    /// Root discovery, phase 1, for one object: its root count starts at its
    /// number of handles, and its cascade guard is cleared. Thread-local
    /// handles are counted in the entry; sync handles in the object header.
    #[inline(always)]
    unsafe fn init_root_count(entry: &mut ObjectEntry) {
        unsafe {
            if O::SHARDED {
                let info = entry.sync_info();
                let hc = info.handle_count.load(Ordering::Acquire);
                info.root_count.store(hc, Ordering::Relaxed);
                entry.root_count = hc;
            } else {
                let hc = entry.handle_count;
                if hc > 0 {
                    entry.gc_info_root_count().set(hc);
                }
                entry.root_count = hc;
            }
            entry.root_ref_count().set(0);
        }
    }

    /// Root discovery, phase 3, for one object: take back the root count the
    /// cascade left in the header (handles minus those found inside scanned
    /// objects) and clear the cascade guard.
    #[inline(always)]
    unsafe fn finish_root_count(entry: &mut ObjectEntry) {
        unsafe {
            if O::SHARDED {
                entry.root_count = entry.sync_info().root_count.load(Ordering::Relaxed);
            } else if entry.handle_count > 0 {
                entry.root_count = entry.gc_info_root_count().get();
            } else {
                entry.root_count = 0;
            }
            entry.root_ref_count().set(0);
        }
    }

    /// Root discovery: given gc_maps (as raw pointer), initialize root_count = handle_count
    /// for in-scope objects, then cascade reset_root through the object graph to decrement
    /// root_count for internal references. After this, objects with root_count > 0 are
    /// reachable from the stack (roots).
    ///
    /// Sets GC_COLLECTING=true so Gc::reset_root() always decrements (ignoring is_root).
    /// Uses root_ref_count as a cycle guard to prevent infinite recursion.
    ///
    /// `scope`: determines which objects are in scope.
    ///   - `None`: all objects (full collection)
    ///   - `Some(gen)`: only objects in generations <= gen
    ///
    /// `use_gen0_ids`: if true, use gen0_ids for O(Gen0) iteration.
    ///
    /// # Safety
    /// `gc_maps_raw` must be a valid mutable pointer to GcMaps<O>. No other references
    /// to GcMaps<O> may exist while this function runs.
    unsafe fn discover_roots(
        gc_maps_raw: *mut GcMaps<O>,
        scope: Option<Generation>,
        use_gen0_ids: bool,
    ) {
        unsafe {
            // Phase 1: Initialize root_count + LocalGcInfo.root_count + clear root_ref_count.
            // For thread-local objects (handle_count > 0): root_count = handle_count.
            // For sync objects: root_count = the header's handle count.
            // Also clear root_ref_count (used as cycle guard in cascade AND mark phase).
            match scope {
                None => {
                    for (_id, entry) in (*gc_maps_raw).objects.iter_mut() {
                        Self::init_root_count(entry);
                    }
                }
                Some(_max_gen) if use_gen0_ids => {
                    for &obj_id in &(*gc_maps_raw).gen0_ids {
                        if let Some(entry) = (*gc_maps_raw).objects.get_mut(obj_id) {
                            Self::init_root_count(entry);
                        }
                    }
                }
                Some(max_gen) => {
                    for (_id, entry) in (*gc_maps_raw).objects.iter_mut() {
                        if entry.generation() <= max_gen {
                            Self::init_root_count(entry);
                        }
                    }
                }
            }

            // Phase 2: Cascade reset_root through the object graph.
            // Tagged pointer with bit 0 set → collection mode (always decrement LocalGcInfo.root_count).
            // root_ref_count serves as cycle guard (0=unvisited, 1=visited).
            // Discard stale entries from a cascade that unwound mid-way (e.g.
            // a panicking user Trace impl): those pointers may reference
            // since-freed objects and must never be dereferenced.
            (*gc_maps_raw).cascade_visited.clear();

            // Each object's fields must be walked EXACTLY once: the guard is
            // checked-and-set here (outer loop) and in the cascade (handle
            // reset_root). Without the outer check, an object reached both by
            // this loop and by a cascade edge had its outgoing edges
            // decremented twice — a rooted self-referencing object (or a
            // rooted cycle) lost its stack-root status and was swept alive.
            GC_MAPS_CTX.set(gc_ctx_tag_collecting(
                &raw mut (*gc_maps_raw).cascade_visited,
            ));
            if O::SHARDED {
                SYNC_CASCADE.set(&raw mut (*gc_maps_raw).cascade_visited);
            }
            match scope {
                None => {
                    for entry in (*gc_maps_raw).objects.values() {
                        if entry.root_ref_count().get() == 0 {
                            entry.root_ref_count().set(1);
                            (&*entry.ptr).reset_root();
                        }
                    }
                }
                Some(_max_gen) if use_gen0_ids => {
                    for &obj_id in &(*gc_maps_raw).gen0_ids {
                        if let Some(entry) = (*gc_maps_raw).objects.get(obj_id) {
                            if entry.root_ref_count().get() == 0 {
                                entry.root_ref_count().set(1);
                                (&*entry.ptr).reset_root();
                            }
                        }
                    }
                }
                Some(max_gen) => {
                    for (_id, entry) in (*gc_maps_raw).objects.iter() {
                        if entry.generation() <= max_gen && entry.root_ref_count().get() == 0 {
                            entry.root_ref_count().set(1);
                            (&*entry.ptr).reset_root();
                        }
                    }
                }
            }
            GC_MAPS_CTX.set(std::ptr::null_mut());
            SYNC_CASCADE.set(std::ptr::null_mut());

            // Phase 3: Sync root_count from LocalGcInfo back + re-count sync roots + clear cycle guards.
            // Combined into one pass to avoid 3 extra iterations.
            match scope {
                None => {
                    for (_id, entry) in (*gc_maps_raw).objects.iter_mut() {
                        Self::finish_root_count(entry);
                    }
                }
                Some(_max_gen) if use_gen0_ids => {
                    for &obj_id in &(*gc_maps_raw).gen0_ids {
                        if let Some(entry) = (*gc_maps_raw).objects.get_mut(obj_id) {
                            Self::finish_root_count(entry);
                        }
                    }
                }
                Some(max_gen) => {
                    for (_id, entry) in (*gc_maps_raw).objects.iter_mut() {
                        if entry.generation() <= max_gen {
                            Self::finish_root_count(entry);
                        }
                    }
                }
            }

            // Clear cascade guards on every object the cascade visited —
            // including out-of-scope Gen1/Gen2 objects the per-scope passes
            // above never touch. root_ref_count doubles as the mark bit in
            // the trace phase, so a leaked guard would make the mark phase
            // treat the object as already marked, skip its children, and
            // sweep live young objects reachable only through it.
            for cell in (*gc_maps_raw).cascade_visited.drain(..) {
                (*cell).set(0);
            }
        }
    }

    /// Generational collection: collect objects in generations 0..=max_gen.
    /// For partial collections (< Gen2), traces from in-scope roots + dirty card table entries.
    /// For full collections (Gen2), traces from ALL roots and clears the card table.
    /// Sweep phase only collects objects in target generations.
    /// Surviving objects in gens < max_gen may be promoted.
    pub unsafe fn collect_generation(&self, max_gen: Generation) -> CollectionStats {
        unsafe {
            let start = std::time::Instant::now();

            // A full collection recomputes reachability from scratch, so any
            // in-flight on-thread incremental mark is stale — discard it and
            // clear the `marking` gate so grey-at-birth stops feeding a dead
            // cycle. (Only relevant when incremental mode is active.)
            if self.local_incremental_budget.load(Ordering::Relaxed) > 0 {
                let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
                if incr.phase != CollectionPhase::Idle {
                    incr.phase = CollectionPhase::Idle;
                    incr.colors.clear();
                    incr.gray_stack.clear();
                }
                drop(incr);
                self.marking.store(false, Ordering::Release);
            }

            let mut stats = CollectionStats::new(max_gen);

            let object_deallocs = {
                // STW: block all mutator operations during mark+sweep
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                let use_gen0_ids =
                    max_gen == Generation::Gen0 && gc_maps.gen0_ids.len() <= gc_maps.objects.len();
                let gc_maps_raw = &mut *gc_maps as *mut GcMaps<O>;
                let scope = if max_gen >= Generation::Gen2 {
                    None
                } else {
                    Some(max_gen)
                };
                let scan_epoch = self.scan_epoch();
                Self::discover_roots(gc_maps_raw, scope, use_gen0_ids);

                // Mark phase: trace from root objects (root_count > 0).
                if max_gen >= Generation::Gen2 {
                    for obj_entry in gc_maps.objects.values() {
                        if obj_entry.root_count > 0 {
                            (&*obj_entry.ptr).trace();
                        }
                    }
                } else if use_gen0_ids {
                    for &obj_id in &gc_maps.gen0_ids {
                        if let Some(obj_entry) = gc_maps.objects.get(obj_id) {
                            if obj_entry.root_count > 0 {
                                (&*obj_entry.ptr).trace();
                            }
                        }
                    }
                } else {
                    for (_id, obj_entry) in gc_maps.objects.iter() {
                        if obj_entry.generation() <= max_gen && obj_entry.root_count > 0 {
                            (&*obj_entry.ptr).trace();
                        }
                    }
                }

                // For partial collections, also trace from dirty card table entries.
                if max_gen < Generation::Gen2 {
                    let dirty_ids = self.card_table.dirty_objects();
                    for obj_id in &dirty_ids {
                        if let Some(entry) = gc_maps.objects.get(*obj_id) {
                            // SAFETY: Object pointer is valid while gc_maps lock is held.
                            (&*entry.ptr).trace();
                        }
                    }
                }

                // A mutably borrowed sync::GcCell left the mark incomplete.
                self.keep_all_if_blocked_since(&gc_maps, scan_epoch);

                // Combined pass: sweep + count + clear_trace + promotion in a single
                // iteration over objects. Avoids 3 extra full iterations and uses
                // inline root_ref_count access instead of vtable dispatch.
                let promo_cfg = self.promotion_config();
                let mut collected_objects: Vec<ObjectId> = Vec::new();
                if max_gen >= Generation::Gen2 {
                    // Full collection: iterate all objects. Rebuild gen0_ids afterward.
                    let mut new_gen0_ids = Vec::new();
                    for (id, entry) in gc_maps.objects.iter_mut() {
                        let marked = entry.root_ref_count().get() > 0;
                        stats.objects_scanned += 1;
                        if !marked {
                            collected_objects.push(id);
                            continue;
                        }
                        let cur_gen = entry.generation();
                        entry.increment_survive_count();
                        if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                            entry.set_survive_count(0);
                            if let Some(next_gen) = cur_gen.next() {
                                if cur_gen == Generation::Gen0 {
                                    let thin = entry.ptr.get_thin_ptr();
                                    self.card_table.register_object(thin, id);
                                }
                                entry.set_generation(next_gen);
                                stats.objects_promoted += 1;
                            }
                        } else if cur_gen == Generation::Gen0 {
                            new_gen0_ids.push(id);
                        }
                        entry.root_ref_count().set(0);
                    }
                    gc_maps.gen0_ids = new_gen0_ids;
                } else if use_gen0_ids {
                    // Gen0 partial collection: iterate only Gen0 objects via gen0_ids.
                    // Rebuild gen0_ids with surviving Gen0 objects (exclude dead and promoted).
                    let old_gen0_ids = std::mem::take(&mut gc_maps.gen0_ids);
                    for obj_id in old_gen0_ids {
                        let entry = match gc_maps.objects.get_mut(obj_id) {
                            Some(e) => e,
                            None => continue, // already removed by RC-hybrid
                        };
                        let marked = entry.root_ref_count().get() > 0;
                        stats.objects_scanned += 1;
                        if !marked {
                            collected_objects.push(obj_id);
                            continue;
                        }
                        let cur_gen = entry.generation();
                        entry.increment_survive_count();
                        if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                            entry.set_survive_count(0);
                            if let Some(next_gen) = cur_gen.next() {
                                if cur_gen == Generation::Gen0 {
                                    let thin = entry.ptr.get_thin_ptr();
                                    self.card_table.register_object(thin, obj_id);
                                }
                                entry.set_generation(next_gen);
                                stats.objects_promoted += 1;
                                // Promoted out of Gen0 — don't add back to gen0_ids
                                entry.root_ref_count().set(0);
                                continue;
                            }
                        }
                        entry.root_ref_count().set(0);
                        // Still Gen0 — keep in gen0_ids
                        gc_maps.gen0_ids.push(obj_id);
                    }
                } else {
                    // Partial collection with full iteration (Gen1, or Gen0 with bloated gen0_ids).
                    // Rebuild gen0_ids from live objects while sweeping.
                    let mut new_gen0_ids = Vec::new();
                    for (id, entry) in gc_maps.objects.iter_mut() {
                        let cur_gen = entry.generation();
                        if cur_gen > max_gen {
                            // Out of scope — but track Gen0 objects for gen0_ids
                            if cur_gen == Generation::Gen0 {
                                new_gen0_ids.push(id);
                            }
                            continue;
                        }
                        let marked = entry.root_ref_count().get() > 0;
                        stats.objects_scanned += 1;
                        if !marked {
                            collected_objects.push(id);
                            continue;
                        }
                        entry.increment_survive_count();
                        if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                            entry.set_survive_count(0);
                            if let Some(next_gen) = cur_gen.next() {
                                if cur_gen == Generation::Gen0 {
                                    let thin = entry.ptr.get_thin_ptr();
                                    self.card_table.register_object(thin, id);
                                }
                                entry.set_generation(next_gen);
                                stats.objects_promoted += 1;
                                entry.root_ref_count().set(0);
                                continue;
                            }
                        }
                        // Still in same generation — track Gen0 objects
                        if cur_gen == Generation::Gen0 {
                            new_gen0_ids.push(id);
                        }
                        entry.root_ref_count().set(0);
                    }
                    gc_maps.gen0_ids = new_gen0_ids;
                }
                stats.objects_collected = collected_objects.len();

                // Remove collected objects
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        stats.handles_collected += Self::handles_held(&entry);
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push(entry);
                    }
                }

                // On full collection, clear dirty flags in the card table
                if max_gen >= Generation::Gen2 {
                    self.card_table.clear_dirty();
                }

                object_deallocs
            };

            // Reset allocation counter after gen0 collection
            if max_gen >= Generation::Gen0 {
                self.allocation_count.store(0, Ordering::Relaxed);
            }

            // Dealloc phase: all locks released. Two-phase: drop every dead
            // object first (dying registry active), then free the memory.
            // SAFETY: Objects have been removed from gc_maps and are unreachable.
            self.dealloc_collected(object_deallocs);

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            // Record diagnostics
            stats.finish(start.elapsed());
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);

            stats
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn collect_all(&self) {
        unsafe {
            let object_deallocs = {
                // STW: block all mutator operations during cleanup
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();
                self.card_table.clear_all();

                let obj_entries = gc_maps.objects.drain();
                gc_maps.ptr_to_object.clear();
                // Signal all weak refs as dead
                for (_, alive) in gc_maps.weak_alive_map.drain() {
                    alive.store(false, Ordering::Release);
                }
                let object_deallocs: Vec<_> =
                    obj_entries.into_iter().map(|(_, entry)| entry).collect();
                object_deallocs
            };
            self.allocation_count.store(0, Ordering::Relaxed);
            // SAFETY: Objects have been removed from gc_maps and are unreachable.
            self.dealloc_collected(object_deallocs);
        }
    }

    /// Parallel collection: mark phase is serial, sweep/dealloc phase uses rayon for
    /// parallel finalization and deallocation. Collects objects in generations 0..=max_gen.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    #[cfg(feature = "parallel")]
    pub unsafe fn collect_parallel(&self, max_gen: Generation) -> CollectionStats {
        use rayon::prelude::*;

        unsafe {
            let start = std::time::Instant::now();
            let mut stats = CollectionStats::new(max_gen);

            let object_deallocs = {
                // STW: block all mutator operations during mark+sweep
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                let gc_maps_raw = &mut *gc_maps as *mut GcMaps<O>;
                let scope = if max_gen >= Generation::Gen2 {
                    None
                } else {
                    Some(max_gen)
                };
                let scan_epoch = self.scan_epoch();
                Self::discover_roots(gc_maps_raw, scope, false);

                // Mark phase: trace from root objects (root_count > 0).
                for obj_entry in gc_maps.objects.values() {
                    if obj_entry.root_count > 0 {
                        (&*obj_entry.ptr).trace();
                    }
                }

                // For partial collections, also trace from dirty card table entries.
                if max_gen < Generation::Gen2 {
                    let dirty_ids = self.card_table.dirty_objects();
                    for obj_id in &dirty_ids {
                        if let Some(entry) = gc_maps.objects.get(*obj_id) {
                            (&*entry.ptr).trace();
                        }
                    }
                }

                // A mutably borrowed sync::GcCell left the mark incomplete.
                self.keep_all_if_blocked_since(&gc_maps, scan_epoch);

                // Count in-scope objects directly (no HashSet allocation)
                stats.objects_scanned = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.generation() <= max_gen)
                    .count();

                // Sweep objects: collect unreachable in-scope objects (direct generation check)
                let collected_objects: Vec<ObjectId> = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.generation() <= max_gen && !(&*e.ptr).is_traceable())
                    .map(|(id, _)| id)
                    .collect();
                stats.objects_collected = collected_objects.len();

                // Unconditionally clear mark state for ALL objects.
                for entry in gc_maps.objects.values() {
                    (&*entry.ptr).clear_trace();
                }

                // Remove collected objects
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        stats.handles_collected += Self::handles_held(&entry);
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push(entry);
                    }
                }

                // On full collection, clear dirty flags in the card table
                if max_gen >= Generation::Gen2 {
                    self.card_table.clear_dirty();
                }

                // Promotion: surviving in-scope objects (already filtered by removal above)
                let promo_cfg = self.promotion_config();
                for (obj_id, entry) in gc_maps.objects.iter_mut() {
                    if entry.generation() > max_gen {
                        continue;
                    }
                    let cur_gen = entry.generation();
                    entry.increment_survive_count();
                    if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                        entry.set_survive_count(0);
                        if let Some(next_gen) = cur_gen.next() {
                            if cur_gen == Generation::Gen0 {
                                let thin = entry.ptr.get_thin_ptr();
                                self.card_table.register_object(thin, obj_id);
                            }
                            entry.set_generation(next_gen);
                            stats.objects_promoted += 1;
                        }
                    }
                }

                object_deallocs
            };

            // Reset allocation counter after gen0 collection
            if max_gen >= Generation::Gen0 {
                self.allocation_count.store(0, Ordering::Relaxed);
            }

            // Dealloc phase runs entirely on the collecting thread. Destructors
            // must be sequential (interior Gc/GcCell handle drops need THIS
            // thread's dying registry and thread-local collector), and the
            // memory frees must be too: objects from one TLAB block share a
            // NON-ATOMIC TlabBlock::ref_count, so concurrent free_object_entry
            // calls on rayon workers would be a data race. The parallel win of
            // this path is the parallel MARK phase, not the sweep. This also
            // frees the ObjectEntry allocations, which the previous parallel
            // path leaked.
            // SAFETY: Objects have been removed from gc_maps and are unreachable.
            self.dealloc_collected(object_deallocs);

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            // Record diagnostics
            stats.finish(start.elapsed());
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);

            stats
        }
    }

    /// Collection with parallel mark AND parallel sweep.
    ///
    /// Uses rayon to parallelize the mark phase (root tracing) and the sweep phase
    /// (deallocation). The mark phase partitions roots across worker threads;
    /// each worker traces its assigned roots independently. This works because
    /// `LocalGcInfo::root_ref_count` uses `Cell<usize>` for the thread-local GC.
    /// Parallel mark requires reverting to AtomicUsize or using UnsafeCell.
    ///
    /// Best for large heaps with many root objects. For small heaps, the serial
    /// `collect_generation()` is faster due to lower overhead.
    ///
    /// # Safety
    /// Caller must ensure no references to GC-managed objects are being
    /// actively dereferenced. STW write lock is acquired internally.
    #[cfg(feature = "parallel")]
    pub unsafe fn collect_parallel_mark(&self, max_gen: Generation) -> CollectionStats {
        use rayon::prelude::*;

        unsafe {
            let start = std::time::Instant::now();
            let mut stats = CollectionStats::new(max_gen);

            let object_deallocs = {
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                let gc_maps_raw = &mut *gc_maps as *mut GcMaps<O>;
                let scope = if max_gen >= Generation::Gen2 {
                    None
                } else {
                    Some(max_gen)
                };
                let scan_epoch = self.scan_epoch();
                Self::discover_roots(gc_maps_raw, scope, false);

                // Parallel mark phase: collect root objects, then trace in parallel.
                struct SendTracePtr(*const dyn Trace);
                unsafe impl Send for SendTracePtr {}
                unsafe impl Sync for SendTracePtr {}

                let root_objects: Vec<SendTracePtr> = gc_maps
                    .objects
                    .values()
                    .filter(|e| e.root_count > 0)
                    .map(|e| SendTracePtr(e.ptr))
                    .collect();

                root_objects.par_iter().for_each(|stp| {
                    (&*stp.0).trace();
                });

                // Card table entries (serial, typically few entries)
                if max_gen < Generation::Gen2 {
                    let dirty_ids = self.card_table.dirty_objects();
                    for obj_id in &dirty_ids {
                        if let Some(entry) = gc_maps.objects.get(*obj_id) {
                            (&*entry.ptr).trace();
                        }
                    }
                }

                // A mutably borrowed sync::GcCell left the mark incomplete.
                self.keep_all_if_blocked_since(&gc_maps, scan_epoch);

                // Count in-scope objects directly
                stats.objects_scanned = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.generation() <= max_gen)
                    .count();

                // Sweep objects: collect unreachable in-scope objects
                let collected_objects: Vec<ObjectId> = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.generation() <= max_gen && !(&*e.ptr).is_traceable())
                    .map(|(id, _)| id)
                    .collect();
                stats.objects_collected = collected_objects.len();

                // Clear marks
                for entry in gc_maps.objects.values() {
                    (&*entry.ptr).clear_trace();
                }

                // Remove collected objects
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        stats.handles_collected += Self::handles_held(&entry);
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push(entry);
                    }
                }

                if max_gen >= Generation::Gen2 {
                    self.card_table.clear_dirty();
                }

                // Promotion: surviving in-scope objects
                let promo_cfg = self.promotion_config();
                for (obj_id, entry) in gc_maps.objects.iter_mut() {
                    if entry.generation() > max_gen {
                        continue;
                    }
                    let cur_gen = entry.generation();
                    entry.increment_survive_count();
                    if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                        entry.set_survive_count(0);
                        if let Some(next_gen) = cur_gen.next() {
                            if cur_gen == Generation::Gen0 {
                                let thin = entry.ptr.get_thin_ptr();
                                self.card_table.register_object(thin, obj_id);
                            }
                            entry.set_generation(next_gen);
                            stats.objects_promoted += 1;
                        }
                    }
                }

                object_deallocs
            };

            if max_gen >= Generation::Gen0 {
                self.allocation_count.store(0, Ordering::Relaxed);
            }

            // Dealloc phase: fully sequential — see collect_parallel for why
            // neither destructors nor frees may run on rayon workers
            // (dying registry / thread-local collector; non-atomic
            // TlabBlock::ref_count).
            // SAFETY: Objects have been removed from gc_maps and are unreachable.
            self.dealloc_collected(object_deallocs);

            self.track_dealloc(stats.bytes_freed);
            stats.finish(start.elapsed());
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);

            stats
        }
    }

    /// Begin an incremental collection cycle.
    /// Short STW: snapshots roots, initializes tri-color marks (all in-scope objects White,
    /// root-reachable objects Gray), and sets phase to Marking.
    pub unsafe fn begin_collection(&self, max_gen: Generation) {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let mut gc_maps = self.lock_gc_maps();

        // Lazily rebuild ptr_to_object for mark_step's trace_children resolution.
        gc_maps.rebuild_ptr_to_object();

        incr.phase = CollectionPhase::Marking;
        incr.max_gen = max_gen;
        incr.scan_epoch = self.scan_epoch();
        incr.colors.clear();
        incr.gray_stack.clear();
        // Gate for the allocation hot path: objects born from here until
        // finish_collection must be greyed (see note_new_object_during_marking).
        self.marking.store(true, Ordering::Release);

        let gc_maps_raw = &mut *gc_maps as *mut GcMaps<O>;
        unsafe {
            Self::discover_roots(gc_maps_raw, None, false);
        }

        // Initialize all in-scope objects as White
        for (obj_id, entry) in gc_maps.objects.iter() {
            if entry.generation() <= max_gen {
                incr.colors.insert(obj_id, MarkColor::White);
            }
        }

        // Gray objects reachable from roots (root_count > 0)
        for (obj_id, obj_entry) in gc_maps.objects.iter() {
            if obj_entry.root_count > 0 {
                if let Some(color) = incr.colors.get_mut(&obj_id) {
                    if *color == MarkColor::White {
                        *color = MarkColor::Gray;
                        incr.gray_stack.push(obj_id);
                    }
                }
            }
        }
    }

    /// Process a batch of gray objects from the worklist.
    /// Short STW per batch. Returns `true` when the gray stack is empty (marking complete).
    /// Each step discovers immediate children of `budget` objects via `trace_children`.
    pub unsafe fn mark_step(&self, budget: usize) -> bool {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let gc_maps = self.lock_gc_maps();

        let mut processed = 0;
        let mut children_buf = Vec::new();

        while processed < budget {
            let obj_id = match incr.gray_stack.pop() {
                Some(id) => id,
                None => break,
            };

            // Discover immediate children via trace_children
            children_buf.clear();
            if let Some(entry) = gc_maps.objects.get(obj_id) {
                unsafe {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).trace_children(&mut children_buf);
                }
            }

            // Convert child raw ptrs to ObjectIds and gray any White children
            for &child_ptr in &children_buf {
                let thin = child_ptr.get_thin_ptr();
                if let Some(&child_id) = gc_maps.ptr_to_object.get(&thin) {
                    if let Some(color) = incr.colors.get_mut(&child_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(child_id);
                        }
                    }
                }
            }

            // This object is now fully scanned
            incr.colors.insert(obj_id, MarkColor::Black);
            processed += 1;
        }

        incr.gray_stack.is_empty()
    }

    /// Finish an incremental collection: re-mark from card table dirty entries, sweep White objects,
    /// promote survivors, and reset state.
    /// Short STW.
    pub unsafe fn finish_collection(&self) -> CollectionStats {
        unsafe {
            let start = std::time::Instant::now();
            let max_gen;
            let (object_deallocs, mut stats) = {
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                max_gen = incr.max_gen;
                incr.phase = CollectionPhase::Sweeping;
                // From here the snapshot is final under STW; stop greying new
                // allocations (there won't be any while the world is stopped).
                self.marking.store(false, Ordering::Release);

                // Finish marking by draining the worklist to a fixpoint instead
                // of re-marking the whole heap from scratch. The concurrent mark
                // steps + the write barrier (which re-greys mutated objects) and
                // grey-at-birth for objects allocated mid-mark mean `colors`
                // already reflects everything except the still-grey remainder;
                // draining it completes the tri-color mark. This is the pause
                // reduction: work is proportional to the leftover grey set, not
                // to the live heap.

                // For partial collections, old->young references originate from
                // out-of-scope old objects; seed the worklist from dirty cards so
                // the young objects they keep alive are marked.
                if max_gen < Generation::Gen2 {
                    let dirty_ids = self.card_table.dirty_objects();
                    for &obj_id in &dirty_ids {
                        if let Some(color) = incr.colors.get_mut(&obj_id) {
                            if *color == MarkColor::White {
                                *color = MarkColor::Gray;
                                incr.gray_stack.push(obj_id);
                            }
                        }
                    }
                }

                let mut children_buf: Vec<*const dyn Trace> = Vec::new();
                while let Some(obj_id) = incr.gray_stack.pop() {
                    children_buf.clear();
                    if let Some(entry) = gc_maps.objects.get(obj_id) {
                        // SAFETY: object pointer valid while gc_maps lock held.
                        (&*entry.ptr).trace_children(&mut children_buf);
                    }
                    for &child_ptr in &children_buf {
                        let thin = child_ptr.get_thin_ptr();
                        if let Some(&child_id) = gc_maps.ptr_to_object.get(&thin) {
                            if let Some(color) = incr.colors.get_mut(&child_id) {
                                if *color == MarkColor::White {
                                    *color = MarkColor::Gray;
                                    incr.gray_stack.push(child_id);
                                }
                            }
                        }
                    }
                    incr.colors.insert(obj_id, MarkColor::Black);
                }

                // Project the tri-color result onto root_ref_count, the mark bit
                // the sweep below reads: Black => reachable, White => garbage.
                // Every in-scope object is present in `colors` (begin_collection
                // seeds all of them White, and mid-mark allocations are greyed),
                // so absence can only mean out-of-scope, which the sweep skips.
                for (id, entry) in gc_maps.objects.iter() {
                    if entry.generation() <= max_gen {
                        let live = matches!(incr.colors.get(&id), Some(MarkColor::Black));
                        entry.root_ref_count().set(usize::from(live));
                    }
                }

                // A mutably borrowed sync::GcCell left the mark incomplete.
                self.keep_all_if_blocked_since(&gc_maps, incr.scan_epoch);

                let mut stats = CollectionStats::new(max_gen);

                // Combined pass: sweep + count + clear_trace + promotion.
                let promo_cfg = self.promotion_config();
                let mut collected_objects: Vec<ObjectId> = Vec::new();
                for (id, entry) in gc_maps.objects.iter_mut() {
                    let in_scope = entry.generation() <= max_gen;
                    let marked = entry.root_ref_count().get() > 0;

                    if in_scope {
                        stats.objects_scanned += 1;
                        if !marked {
                            collected_objects.push(id);
                            continue;
                        }
                        let cur_gen = entry.generation();
                        entry.increment_survive_count();
                        if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                            entry.set_survive_count(0);
                            if let Some(next_gen) = cur_gen.next() {
                                if cur_gen == Generation::Gen0 {
                                    let thin = entry.ptr.get_thin_ptr();
                                    self.card_table.register_object(thin, id);
                                }
                                entry.set_generation(next_gen);
                                stats.objects_promoted += 1;
                            }
                        }
                    }
                    entry.root_ref_count().set(0);
                }
                stats.objects_collected = collected_objects.len();

                // Remove collected objects
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        stats.handles_collected += Self::handles_held(&entry);
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push(entry);
                    }
                }

                // Clear card table dirty flags on full collection
                if max_gen >= Generation::Gen2 {
                    self.card_table.clear_dirty();
                }

                // Reset incremental state
                incr.phase = CollectionPhase::Idle;
                incr.colors.clear();
                incr.gray_stack.clear();

                (object_deallocs, stats)
            };

            // Reset allocation counter
            if max_gen >= Generation::Gen0 {
                self.allocation_count.store(0, Ordering::Relaxed);
            }

            // Dealloc phase: all locks released. Two-phase: drop every dead
            // object first (dying registry active), then free the memory.
            // SAFETY: Objects have been removed from gc_maps and are unreachable.
            self.dealloc_collected(object_deallocs);

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            // Record diagnostics
            stats.finish(start.elapsed());
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);

            stats
        }
    }

    /// Convenience: run a complete incremental collection.
    /// Breaks the mark phase into steps of `step_budget` objects each,
    /// releasing the STW lock between steps to reduce pause times.
    pub unsafe fn collect_incremental(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to begin_collection, mark_step, finish_collection.
        unsafe {
            self.begin_collection(max_gen);
            loop {
                let done = self.mark_step(step_budget);
                if done {
                    break;
                }
            }
            self.finish_collection()
        }
    }

    /// Time-budgeted incremental mark step.
    /// Processes gray objects until `max_duration` elapses or the gray stack is empty.
    /// Returns `true` when marking is complete.
    ///
    /// Unlike `mark_step` which limits by object count, this limits by wall-clock
    /// time, making pause times predictable regardless of object graph shape.
    pub unsafe fn mark_step_timed(&self, max_duration: Duration) -> bool {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let gc_maps = self.lock_gc_maps();

        let deadline = Instant::now() + max_duration;
        let mut children_buf = Vec::new();

        loop {
            if Instant::now() >= deadline {
                break;
            }

            let obj_id = match incr.gray_stack.pop() {
                Some(id) => id,
                None => break,
            };

            children_buf.clear();
            if let Some(entry) = gc_maps.objects.get(obj_id) {
                // SAFETY: entry.ptr is valid for the lifetime of the GcMaps<O> lock.
                unsafe {
                    (&*entry.ptr).trace_children(&mut children_buf);
                }
            }

            for &child_ptr in &children_buf {
                let thin = child_ptr.get_thin_ptr();
                if let Some(&child_id) = gc_maps.ptr_to_object.get(&thin) {
                    if let Some(color) = incr.colors.get_mut(&child_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(child_id);
                        }
                    }
                }
            }

            incr.colors.insert(obj_id, MarkColor::Black);
        }

        incr.gray_stack.is_empty()
    }

    /// Run a complete incremental collection with time-budgeted mark steps.
    /// Each mark step runs for at most `max_step_duration`, then releases
    /// the STW lock to allow mutator progress.
    pub unsafe fn collect_incremental_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to begin_collection, mark_step_timed, finish_collection.
        unsafe {
            self.begin_collection(max_gen);
            loop {
                let done = self.mark_step_timed(max_step_duration);
                if done {
                    break;
                }
            }
            self.finish_collection()
        }
    }

    // ---- Concurrent marking (Strategy 21) ----

    /// Begin a concurrent collection cycle.
    /// Short STW: snapshots roots AND object graph edges so that subsequent
    /// `concurrent_mark_step` calls can traverse the graph without STW.
    pub unsafe fn begin_concurrent_collection(&self, max_gen: Generation) {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let mut gc_maps = self.lock_gc_maps();

        // Lazily rebuild ptr_to_object for edge snapshot resolution.
        gc_maps.rebuild_ptr_to_object();

        incr.phase = CollectionPhase::Marking;
        incr.max_gen = max_gen;
        incr.scan_epoch = self.scan_epoch();
        incr.colors.clear();
        incr.gray_stack.clear();
        incr.edges.clear();

        let gc_maps_raw = &mut *gc_maps as *mut GcMaps<O>;
        unsafe {
            Self::discover_roots(gc_maps_raw, None, false);
        }

        // Initialize all in-scope objects as White and snapshot their edges
        let mut children_buf = Vec::new();
        for (obj_id, entry) in gc_maps.objects.iter() {
            if entry.generation() <= max_gen {
                incr.colors.insert(obj_id, MarkColor::White);
                children_buf.clear();
                unsafe {
                    (&*entry.ptr).trace_children(&mut children_buf);
                }
                let child_ids: Vec<ObjectId> = children_buf
                    .iter()
                    .filter_map(|&ptr| gc_maps.ptr_to_object.get(&ptr.get_thin_ptr()).copied())
                    .collect();
                incr.edges.insert(obj_id, child_ids);
            }
        }

        // Gray objects reachable from roots (root_count > 0)
        for (obj_id, obj_entry) in gc_maps.objects.iter() {
            if obj_entry.root_count > 0 {
                if let Some(color) = incr.colors.get_mut(&obj_id) {
                    if *color == MarkColor::White {
                        *color = MarkColor::Gray;
                        incr.gray_stack.push(obj_id);
                    }
                }
            }
        }
    }

    /// Process gray objects using the edge snapshot. NO STW lock required.
    /// Safe to call concurrently with mutators because edges were snapshotted.
    /// Returns `true` when marking is complete (gray stack empty).
    pub fn concurrent_mark_step(&self, budget: usize) -> bool {
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let mut processed = 0;

        while processed < budget {
            let obj_id = match incr.gray_stack.pop() {
                Some(id) => id,
                None => break,
            };

            // Use snapshotted edges instead of following live pointers
            if let Some(children) = incr.edges.get(&obj_id).cloned() {
                for child_id in children {
                    if let Some(color) = incr.colors.get_mut(&child_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(child_id);
                        }
                    }
                }
            }

            incr.colors.insert(obj_id, MarkColor::Black);
            processed += 1;
        }

        incr.gray_stack.is_empty()
    }

    /// Convenience: run a complete concurrent collection.
    /// begin = short STW (snapshot), mark = no STW, finish = short STW (re-mark + sweep).
    pub unsafe fn collect_concurrent(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to begin_concurrent_collection, concurrent_mark_step, finish_collection.
        unsafe {
            self.begin_concurrent_collection(max_gen);
            loop {
                let done = self.concurrent_mark_step(step_budget);
                if done {
                    break;
                }
            }
            // finish_collection handles re-graying from card table + new roots
            self.finish_collection()
        }
    }

    /// Time-budgeted concurrent mark step. NO STW lock required.
    /// Processes gray objects from the edge snapshot until `max_duration` elapses
    /// or the gray stack is empty. Returns `true` when marking is complete.
    pub fn concurrent_mark_step_timed(&self, max_duration: Duration) -> bool {
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Instant::now() + max_duration;

        loop {
            if Instant::now() >= deadline {
                break;
            }

            let obj_id = match incr.gray_stack.pop() {
                Some(id) => id,
                None => break,
            };

            if let Some(children) = incr.edges.get(&obj_id).cloned() {
                for child_id in children {
                    if let Some(color) = incr.colors.get_mut(&child_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(child_id);
                        }
                    }
                }
            }

            incr.colors.insert(obj_id, MarkColor::Black);
        }

        incr.gray_stack.is_empty()
    }

    /// Run a complete concurrent collection with time-budgeted mark steps.
    /// Each mark step runs for at most `max_step_duration` with no STW overhead.
    pub unsafe fn collect_concurrent_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to begin_concurrent_collection, concurrent_mark_step_timed, finish_collection.
        unsafe {
            self.begin_concurrent_collection(max_gen);
            loop {
                let done = self.concurrent_mark_step_timed(max_step_duration);
                if done {
                    break;
                }
            }
            self.finish_collection()
        }
    }

    // ---- Region-based collection (Strategy 22) ----

    /// Create a new region. Future allocations go into this region.
    pub fn new_region(&self) -> RegionId {
        let id = self.next_region_id.fetch_add(1, Ordering::Relaxed);
        self.current_region.store(id, Ordering::Relaxed);
        RegionId(id)
    }

    /// Get the current allocation region.
    pub fn current_region(&self) -> RegionId {
        RegionId(self.current_region.load(Ordering::Relaxed))
    }

    /// Collect only objects in the specified region.
    /// Traces from all roots (needed for correctness), but sweeps only objects in the target region.
    pub unsafe fn collect_region(&self, region: RegionId) -> CollectionStats {
        unsafe {
            let start = std::time::Instant::now();
            let mut stats = CollectionStats::new(Generation::Gen2);

            let object_deallocs = {
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                let gc_maps_raw = &mut *gc_maps as *mut GcMaps<O>;
                let scan_epoch = self.scan_epoch();
                Self::discover_roots(gc_maps_raw, None, false);

                // Mark phase: trace from root objects (root_count > 0).
                for obj_entry in gc_maps.objects.values() {
                    if obj_entry.root_count > 0 {
                        (&*obj_entry.ptr).trace();
                    }
                }

                // Also trace from dirty card table entries
                let dirty_ids = self.card_table.dirty_objects();
                for obj_id in &dirty_ids {
                    if let Some(entry) = gc_maps.objects.get(*obj_id) {
                        (&*entry.ptr).trace();
                    }
                }

                // A mutably borrowed sync::GcCell left the mark incomplete.
                self.keep_all_if_blocked_since(&gc_maps, scan_epoch);

                // Combined pass: sweep + count + clear_trace in a single iteration.
                let mut collected_objects: Vec<ObjectId> = Vec::new();
                for (id, entry) in gc_maps.objects.iter_mut() {
                    let in_region = entry.region() == region;
                    let marked = entry.root_ref_count().get() > 0;

                    if in_region {
                        stats.objects_scanned += 1;
                        if !marked {
                            collected_objects.push(id);
                            continue;
                        }
                    }
                    entry.root_ref_count().set(0);
                }
                stats.objects_collected = collected_objects.len();

                // Remove collected objects
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        stats.handles_collected += Self::handles_held(&entry);
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push(entry);
                    }
                }

                object_deallocs
            };

            // Dealloc phase: all locks released. Two-phase: drop every dead
            // object first (dying registry active), then free the memory.
            // SAFETY: Objects have been removed from gc_maps and are unreachable.
            self.dealloc_collected(object_deallocs);

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            stats.finish(start.elapsed());
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);
            stats
        }
    }

    /// Return per-region liveness statistics without running a collection.
    pub fn region_stats(&self) -> Vec<crate::generation::RegionStats> {
        let gc_maps = self.lock_gc_maps();
        let (total_bytes, object_count) = gc_maps.compute_region_stats();
        let mut region_ids: Vec<RegionId> = total_bytes.keys().copied().collect();
        for key in object_count.keys() {
            if !region_ids.contains(key) {
                region_ids.push(*key);
            }
        }
        region_ids
            .into_iter()
            .map(|rid| crate::generation::RegionStats {
                region_id: rid,
                total_bytes: total_bytes.get(&rid).copied().unwrap_or(0),
                object_count: object_count.get(&rid).copied().unwrap_or(0),
                estimated_garbage_ratio: 0.0,
            })
            .collect()
    }

    /// G1-style "Garbage First" collection. Runs a mark phase, computes per-region
    /// garbage ratios, then sweeps regions with the highest garbage ratio first,
    /// stopping when the elapsed time reaches `pause_target`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are held
    /// across this call (stop-the-world requirement).
    pub unsafe fn collect_garbage_first(&self, pause_target: Duration) -> CollectionStats {
        unsafe {
            let start = Instant::now();
            let mut stats = CollectionStats::new(Generation::Gen2);

            let object_deallocs = {
                // STW: block all mutator operations during mark+sweep
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                // ---- Mark phase (same as collect_generation Gen2) ----
                let gc_maps_raw = &mut *gc_maps as *mut GcMaps<O>;
                let scan_epoch = self.scan_epoch();
                Self::discover_roots(gc_maps_raw, None, false);

                // Mark from root objects (root_count > 0).
                for obj_entry in gc_maps.objects.values() {
                    if obj_entry.root_count > 0 {
                        (&*obj_entry.ptr).trace();
                    }
                }

                // A mutably borrowed sync::GcCell left the mark incomplete.
                self.keep_all_if_blocked_since(&gc_maps, scan_epoch);

                stats.objects_scanned = gc_maps.objects.len();

                // ---- Compute per-region garbage ratio using inline mark bits ----
                let mut region_total: HashMap<RegionId, usize> = HashMap::new();
                let mut region_marked: HashMap<RegionId, usize> = HashMap::new();

                for (_id, entry) in gc_maps.objects.iter() {
                    *region_total.entry(entry.region()).or_insert(0) += 1;
                    if entry.root_ref_count().get() > 0 {
                        *region_marked.entry(entry.region()).or_insert(0) += 1;
                    }
                }

                // Build sorted list: highest garbage ratio first
                let mut regions_by_garbage: Vec<(RegionId, f64)> = region_total
                    .iter()
                    .map(|(&rid, &total)| {
                        let marked = region_marked.get(&rid).copied().unwrap_or(0);
                        let ratio = if total > 0 {
                            (total - marked) as f64 / total as f64
                        } else {
                            0.0
                        };
                        (rid, ratio)
                    })
                    .collect();
                regions_by_garbage
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // ---- Sweep regions one by one, highest garbage first ----
                let mut object_deallocs = Vec::new();

                for (region, _ratio) in &regions_by_garbage {
                    if start.elapsed() >= pause_target && stats.objects_collected > 0 {
                        break;
                    }

                    let dead_in_region: Vec<ObjectId> = gc_maps
                        .objects
                        .iter()
                        .filter(|(_id, e)| e.region() == *region && e.root_ref_count().get() == 0)
                        .map(|(id, _)| id)
                        .collect();

                    if dead_in_region.is_empty() {
                        continue;
                    }

                    for obj_id in &dead_in_region {
                        if let Some(entry) = gc_maps.objects.remove(*obj_id) {
                            let thin = entry.ptr.get_thin_ptr();
                            gc_maps.ptr_to_object.remove(&thin);
                            self.card_table.remove_object(thin, *obj_id);
                            stats.bytes_freed += entry.layout.size();
                            stats.handles_collected += Self::handles_held(&entry);
                            if let Some(alive) = gc_maps.weak_alive_map.remove(obj_id) {
                                alive.store(false, Ordering::Release);
                            }
                            object_deallocs.push(entry);
                        }
                    }
                    stats.objects_collected += dead_in_region.len();
                }

                // Clear mark state for ALL surviving objects using inline access.
                for entry in gc_maps.objects.values() {
                    entry.root_ref_count().set(0);
                }

                object_deallocs
            };

            // Dealloc phase: all locks released. Two-phase: drop every dead
            // object first (dying registry active), then free the memory.
            // SAFETY: Objects have been removed from gc_maps and are unreachable.
            self.dealloc_collected(object_deallocs);

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            stats.finish(start.elapsed());
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);
            stats
        }
    }

    /// Compact the heap by copying all live objects into a contiguous buffer.
    ///
    /// Performs a full collection first (to free dead objects), then:
    /// 1. Allocates a single contiguous buffer for all live objects.
    /// 2. Copies each live object (GcPtr) into the buffer at aligned offsets.
    /// 3. Updates handle pointers via `Trace::relocate()`.
    /// 4. Updates ObjectEntry metadata.
    /// 5. Frees old individual allocations.
    ///
    /// Returns the number of objects compacted.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    /// For thread-local GC this is trivially safe (single-threaded).
    /// For global GC, STW write lock is acquired internally.
    pub unsafe fn compact(&self) -> usize {
        // COMPACTION IS CURRENTLY DISABLED — this intentionally returns 0
        // without doing any work (asserted by compact_is_disabled_returns_zero).
        //
        // The GcInternal elimination made handles direct tagged pointers, so
        // the tracer registry that used to make EVERY handle reachable for
        // relocation is gone. Re-enabling requires:
        //   (a) a relocate() cascade through all Trace impls (derive macro +
        //       default_trace) to patch interior Gc/GcCell handles, and
        //   (b) pinning rooted objects — stack handles cannot be discovered,
        //       so objects with root_count > 0 must not move.
        // Until then, callers of _compact() get 0 and no objects are moved;
        // all objects stay valid.
        return 0;
        #[allow(unreachable_code)]
        unsafe {
            // Full collection first to free dead objects
            self.collect();

            let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
            let mut gc_maps = self.lock_gc_maps();

            let num_objects = gc_maps.objects.len();
            if num_objects == 0 {
                return 0;
            }

            // Calculate total size needed with proper alignment
            let mut total_size: usize = 0;
            let mut obj_layouts: Vec<(ObjectId, Layout)> = Vec::with_capacity(num_objects);
            for (id, entry) in gc_maps.objects.iter() {
                let layout = entry.layout.to_layout();
                // Align up
                let align_offset = total_size % layout.align();
                if align_offset != 0 {
                    total_size += layout.align() - align_offset;
                }
                obj_layouts.push((id, layout));
                total_size += layout.size();
            }

            if total_size == 0 {
                return 0;
            }

            // Allocate contiguous buffer with max alignment
            let max_align = obj_layouts
                .iter()
                .map(|(_, l)| l.align())
                .max()
                .unwrap_or(1);
            let compact_layout = match Layout::from_size_align(total_size, max_align) {
                Ok(l) => l,
                Err(_) => return 0,
            };
            // SAFETY: compact_layout has non-zero size and valid alignment.
            let compact_buf = alloc(compact_layout);
            if compact_buf.is_null() {
                return 0; // allocation failed, skip compaction
            }

            // Copy each object into the contiguous buffer
            let mut offset: usize = 0;
            let mut relocations: Vec<(ObjectId, *const u8, *mut u8, Layout)> =
                Vec::with_capacity(num_objects);

            for (id, layout) in &obj_layouts {
                // Align offset
                let align_offset = offset % layout.align();
                if align_offset != 0 {
                    offset += layout.align() - align_offset;
                }

                let entry = gc_maps
                    .objects
                    .get(*id)
                    .expect("compact: object entry not found for id in obj_layouts");
                let old_ptr = entry.ptr.get_thin_ptr() as *mut u8;
                // SAFETY: offset is within compact_buf bounds (calculated from accumulated layout sizes).
                let new_ptr = compact_buf.add(offset);

                // SAFETY: old_ptr and new_ptr point to valid, non-overlapping memory of at least layout.size() bytes.
                std::ptr::copy_nonoverlapping(old_ptr, new_ptr, layout.size());

                relocations.push((*id, old_ptr as *const u8, new_ptr, *layout));

                offset += layout.size();
            }

            // Update ObjectEntry pointers and create new GcObjMem for each object.
            // All objects now live in the compact buffer. We use Mem variant pointing
            // into the buffer. Only the LAST object "owns" the buffer for dealloc;
            // others use a Tlab-like scheme. For simplicity, we track the buffer
            // separately and never dealloc individual objects from it.
            // Instead, we store a shared Arc that owns the compact buffer.
            let compact_block = Arc::new(CompactBlock::new(compact_buf, compact_layout));

            let mut old_mems: Vec<(*mut u8, GcObjMem, Layout)> = Vec::with_capacity(num_objects);

            for (id, _old_raw, new_raw, layout) in &relocations {
                // Two-phase update: first update ObjectEntry (needs &mut objects),
                // then update ptr_to_object (needs &mut ptr_to_object).
                // We collect the old/new thin ptrs to avoid overlapping borrows.
                let (old_thin, new_thin) = {
                    let entry = match gc_maps.objects.get_mut(*id) {
                        Some(e) => e,
                        None => continue,
                    };
                    let old_thin = entry.ptr.get_thin_ptr();

                    // Reconstruct fat pointer: preserve vtable, update data pointer.
                    // SAFETY: new_raw points to a valid copy of the original object
                    // within the compact buffer. The object was fully copied above.
                    let mut fat_ptr_bytes = [0u8; std::mem::size_of::<*const dyn Trace>()];
                    std::ptr::copy_nonoverlapping(
                        &entry.ptr as *const *const dyn Trace as *const u8,
                        fat_ptr_bytes.as_mut_ptr(),
                        fat_ptr_bytes.len(),
                    );
                    // Overwrite data pointer (first pointer-sized field)
                    std::ptr::copy_nonoverlapping(
                        &(*new_raw as *const u8) as *const *const u8 as *const u8,
                        fat_ptr_bytes.as_mut_ptr(),
                        std::mem::size_of::<*const u8>(),
                    );
                    // SAFETY: read_unaligned because fat_ptr_bytes is a [u8] (align 1)
                    // but *const dyn Trace requires pointer alignment.
                    let new_fat_ptr: *const dyn Trace =
                        std::ptr::read_unaligned(fat_ptr_bytes.as_ptr() as *const *const dyn Trace);
                    entry.ptr = new_fat_ptr;
                    let new_thin = entry.ptr.get_thin_ptr();

                    // root_ref_count_offset is relative — no update needed after relocation.

                    // Replace memory tracking with compact block reference.
                    // Use std::mem::replace to properly move the old GcObjMem out,
                    // avoiding double-drop of Arc refs in Compact/Tlab variants.
                    let old_mem =
                        std::mem::replace(&mut entry.mem, GcObjMem::Compact(compact_block.clone()));
                    old_mems.push((old_thin as *mut u8, old_mem, *layout));
                    entry.layout = CompactLayout::from_layout(*layout);

                    (old_thin, new_thin)
                };

                // Update ptr_to_object mapping (borrows gc_maps, not gc_maps.objects)
                gc_maps.ptr_to_object.remove(&old_thin);
                gc_maps.ptr_to_object.insert(new_thin, *id);

                // Update card table
                self.card_table.remove_object(old_thin, *id);
                self.card_table.register_object(new_thin, *id);
            }

            // TODO: Relocate Gc/GcCell handle pointers after compaction.
            // Old approach used tracers.for_each_tracer to call relocate() on each GcInternal.
            // New approach: iterate object graph and update Gc::ptr / GcCell::ptr cells.
            // For now, compact() returns 0 early so this code is unreachable.

            drop(gc_maps);

            for (ptr, mem, layout) in old_mems {
                // SAFETY: Old allocation is no longer in use; objects have been relocated to the compact buffer.
                Self::dealloc_gc_mem(ptr, mem, layout);
            }

            num_objects
        }
    }
}

/// Shard-scoped access for the global collector.
///
/// These paths run concurrently on different threads, so they must never
/// create `&mut GcMaps` (two live `&mut` to one value is undefined behaviour
/// even when the threads touch disjoint shards - Miri reports the reference
/// retags as a data race). They take a **shared** `&GcMaps` and reach only
/// their own shard through that shard's `UnsafeCell`, under that shard's lock.
/// `lock_gc_maps` holds every shard lock, so it excludes all of them.
impl GarbageCollector<SyncObjects> {
    /// Run `f` with exclusive access to **only the calling thread's shard**.
    /// The allocation fast path: each thread owns a distinct shard, so this
    /// lock is uncontended and concurrent allocation scales.
    ///
    /// `f` receives the shard's own `SlotMap` and the shard index; turn a key it
    /// returns into a map-wide id with `SyncObjects::tag_key`.
    #[inline]
    pub(crate) fn with_own_shard<R>(&self, f: impl FnOnce(&mut LocalObjects, usize) -> R) -> R {
        let shard = crate::sharded_map::current_thread_shard(self.shard_locks.len());
        self.with_shard(shard, |m| f(m, shard))
    }

    /// Run `f` with exclusive access to **only the shard owning `id`**, for the
    /// per-object hot paths (handle clone, RC-hybrid drop). `f` receives that
    /// shard's `SlotMap`, which is keyed by `id.strip_shard()`.
    #[inline]
    pub(crate) fn with_shard_of<R>(
        &self,
        id: ObjectId,
        f: impl FnOnce(&mut LocalObjects) -> R,
    ) -> R {
        use crate::slot_map::SlotKey;
        self.with_shard(id.shard(), f)
    }

    #[inline]
    fn with_shard<R>(&self, shard: usize, f: impl FnOnce(&mut LocalObjects) -> R) -> R {
        let shard = shard.min(self.shard_locks.len() - 1);
        let _g = self.shard_locks[shard]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // SAFETY: only a shared reference to GcMaps is created. Holding this
        // shard's lock excludes `lock_gc_maps` (which takes every shard lock,
        // and is the only place `&mut GcMaps` or `&self` map readers occur
        // for the global collector) and every other user of this shard, so the
        // `&mut` to this shard's SlotMap is unique.
        let map = unsafe { &mut *(*self.gc_maps.get()).objects.shard_ptr(shard) };
        f(map)
    }

    /// Free an object whose last handle was just dropped (the header's
    /// handle count reached zero). Takes only the object's shard lock in the
    /// common case; keeping this on the full-map lock made concurrent
    /// workloads ~18x slower once that lock had to take every shard.
    ///
    /// Finding the object already gone is normal: a collection that ran
    /// after the count reached zero freed it as unreachable.
    ///
    /// `unique` is the header count when the caller's handle is the only one
    /// and was not decremented. With weak references present an upgrade may
    /// have added a handle since; that is re-checked under the full lock
    /// (upgrades hold the object's shard lock), and then the caller's handle
    /// is just taken off the count. Without weak references nothing can add
    /// one, so the count is left as it is - the object is freed.
    pub(crate) unsafe fn release_object(&self, object_id: ObjectId, unique: Option<&AtomicU32>) {
        use crate::slot_map::SlotKey;
        let fast = self.with_shard_of(object_id, |shard| {
            // Re-check the weak hint *under* the shard lock: a weak ref is
            // created under the full lock, which holds this shard's lock, so
            // this read is ordered after any such creation. With weak refs
            // present the weak map (unsharded) must be updated too, so defer
            // to the full-lock path before touching anything.
            if self.weak_alive_len.load(Ordering::Acquire) != 0 {
                return None;
            }
            let key = object_id.strip_shard();
            let Some(entry) = shard.get(key) else {
                return Some(None);
            };
            let obj_gen = entry.generation();
            // SAFETY: `key` was just found occupied in this shard.
            let entry = unsafe { shard.remove_unchecked(key) };
            if obj_gen != Generation::Gen0 {
                // CardTable synchronizes itself.
                self.card_table
                    .remove_object(entry.ptr.get_thin_ptr(), object_id);
            }
            Some(Some(entry))
        });
        let released = match fast {
            Some(r) => r,
            None => {
                let mut gc_maps = self.lock_gc_maps();
                if let Some(count) = unique {
                    if count
                        .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire)
                        .is_err()
                    {
                        // A weak upgrade added a handle: this one only leaves.
                        count.fetch_sub(1, Ordering::Release);
                        return;
                    }
                }
                let r = Self::release_object_inner(&mut gc_maps, &self.card_table, object_id);
                // Re-sync the hint while every shard is held, so a program that
                // used weak refs only briefly returns to the fast path.
                self.weak_alive_len
                    .store(gc_maps.weak_alive_map.len(), Ordering::Release);
                r
            }
        };
        if let Some(entry) = released {
            self.track_dealloc(entry.layout.size());
            unsafe { Self::dealloc_object_entry(entry) };
        }
    }
}

#[cfg(test)]
mod tests;
