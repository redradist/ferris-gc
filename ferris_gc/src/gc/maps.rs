//! The object maps a collector is generic over (`ObjectMap`), and the
//! collector's bookkeeping (`GcMaps`) with its lock guard.
use super::*;

/// Object-map storage a collector is generic over. The thread-local collector
/// uses a plain [`SlotMap`]; the global collector uses a thread-affine
/// [`ShardedSlotMap`] so concurrent allocation does not serialize. Static
/// dispatch keeps the thread-local path free of any sharding cost (key
/// decoding, tag masking, the shard indirection) and keeps its full 32-bit
/// index space.
pub(crate) trait ObjectMap: Sized {
    /// Whether per-object work may run under a single shard lock while other
    /// shards are being mutated. When false, `lock_gc_maps` skips shard locks.
    const SHARDED: bool;
    /// Counter type for the allocation / heap-size statistics.
    type Counter: Counter;
    fn with_shards(n: usize) -> Self;
    fn get(&self, k: ObjectId) -> Option<&ObjectEntryRef>;
    fn get_mut(&mut self, k: ObjectId) -> Option<&mut ObjectEntryRef>;
    fn remove(&mut self, k: ObjectId) -> Option<ObjectEntryRef>;
    /// # Safety
    /// `k` must refer to a valid, currently occupied slot.
    unsafe fn remove_unchecked(&mut self, k: ObjectId) -> ObjectEntryRef;
    fn len(&self) -> usize;
    fn iter(&self) -> impl Iterator<Item = (ObjectId, &ObjectEntryRef)>;
    fn iter_mut(&mut self) -> impl Iterator<Item = (ObjectId, &mut ObjectEntryRef)>;
    fn values(&self) -> impl Iterator<Item = &ObjectEntryRef>;
    fn drain(&mut self) -> Vec<(ObjectId, ObjectEntryRef)>;
}

/// Object map of the thread-local collector.
pub(crate) type LocalObjects = SlotMap<ObjectId, ObjectEntryRef>;
/// Object map of the global collector.
pub(crate) type SyncObjects = ShardedSlotMap<ObjectId, ObjectEntryRef>;

macro_rules! forward_object_map {
    ($ty:ty, $sharded:expr, $counter:ty, $new:expr) => {
        impl ObjectMap for $ty {
            const SHARDED: bool = $sharded;
            type Counter = $counter;
            #[inline(always)]
            fn with_shards(n: usize) -> Self {
                ($new)(n)
            }
            #[inline(always)]
            fn get(&self, k: ObjectId) -> Option<&ObjectEntryRef> {
                <$ty>::get(self, k)
            }
            #[inline(always)]
            fn get_mut(&mut self, k: ObjectId) -> Option<&mut ObjectEntryRef> {
                <$ty>::get_mut(self, k)
            }
            #[inline(always)]
            fn remove(&mut self, k: ObjectId) -> Option<ObjectEntryRef> {
                <$ty>::remove(self, k)
            }
            #[inline(always)]
            unsafe fn remove_unchecked(&mut self, k: ObjectId) -> ObjectEntryRef {
                unsafe { <$ty>::remove_unchecked(self, k) }
            }
            #[inline(always)]
            fn len(&self) -> usize {
                <$ty>::len(self)
            }
            #[inline(always)]
            fn iter(&self) -> impl Iterator<Item = (ObjectId, &ObjectEntryRef)> {
                <$ty>::iter(self)
            }
            #[inline(always)]
            fn iter_mut(&mut self) -> impl Iterator<Item = (ObjectId, &mut ObjectEntryRef)> {
                <$ty>::iter_mut(self)
            }
            #[inline(always)]
            fn values(&self) -> impl Iterator<Item = &ObjectEntryRef> {
                <$ty>::values(self)
            }
            #[inline(always)]
            fn drain(&mut self) -> Vec<(ObjectId, ObjectEntryRef)> {
                <$ty>::drain(self)
            }
        }
    };
}
forward_object_map!(
    LocalObjects,
    false,
    crate::sharded_map::SingleCounter,
    |_n: usize| SlotMap::new()
);
forward_object_map!(
    SyncObjects,
    true,
    crate::sharded_map::ShardedCounter,
    ShardedSlotMap::with_shards
);

pub(crate) struct GcMaps<O: ObjectMap = LocalObjects> {
    pub(crate) objects: O,
    /// Maps thin object pointer → ObjectId for trace_children resolution.
    pub(crate) ptr_to_object: FxHashMap<usize, ObjectId>,
    /// Weak-reference alive flags. Only populated for objects that have
    /// weak references — most objects never need this, so keeping it
    /// out of ObjectEntry saves 8B per object on the hot path.
    pub(crate) weak_alive_map: FxHashMap<ObjectId, Arc<AtomicBool>>,
    /// Fast Gen0 object list for O(Gen0) partial collections.
    /// Avoids iterating all objects during Gen0 mark/sweep.
    pub(crate) gen0_ids: Vec<ObjectId>,
    /// Guard cells set by the phase-2 root-discovery cascade, cleared in
    /// phase 3. Tracked explicitly because a partial (Gen0/Gen1) cascade can
    /// cross into out-of-scope old-generation objects that the per-scope
    /// clearing passes never touch — and `root_ref_count` doubles as the mark
    /// bit during trace, so a leaked guard makes the mark phase skip the
    /// object's children and sweep live objects. Buffer is reused across
    /// collections.
    pub(crate) cascade_visited: CascadeVisited,
}

impl<O: ObjectMap> GcMaps<O> {
    /// Total number of handles across all objects.
    #[cfg(test)]
    pub(crate) fn total_handles(&self) -> usize {
        self.objects.values().map(|e| e.handle_count as usize).sum()
    }

    /// Compute per-region total bytes and object counts on demand from the objects SlotMap.
    pub(crate) fn compute_region_stats(
        &self,
    ) -> (HashMap<RegionId, usize>, HashMap<RegionId, usize>) {
        let mut total_bytes: HashMap<RegionId, usize> = HashMap::new();
        let mut object_count: HashMap<RegionId, usize> = HashMap::new();
        for (_id, entry) in self.objects.iter() {
            *total_bytes.entry(entry.region()).or_insert(0) += entry.layout.size();
            *object_count.entry(entry.region()).or_insert(0) += 1;
        }
        (total_bytes, object_count)
    }

    /// Rebuild ptr_to_object from the current objects SlotMap.
    /// Called lazily before incremental/concurrent marking that needs
    /// pointer-to-ObjectId resolution.
    pub(crate) fn rebuild_ptr_to_object(&mut self) {
        self.ptr_to_object.clear();
        for (id, entry) in self.objects.iter() {
            let thin = entry.ptr.get_thin_ptr();
            self.ptr_to_object.insert(thin, id);
        }
    }
}

/// State for incremental tri-color marking.
pub(crate) struct IncrementalState {
    pub(crate) phase: CollectionPhase,
    pub(crate) max_gen: Generation,
    pub(crate) colors: FxHashMap<ObjectId, MarkColor>,
    pub(crate) gray_stack: Vec<ObjectId>,
    /// Snapshot of object graph edges for concurrent marking.
    /// Populated during begin_concurrent_collection so that mark steps
    /// can traverse the graph without following live pointers.
    pub(crate) edges: FxHashMap<ObjectId, Vec<ObjectId>>,
    /// `blocked_scans` when this cycle's marking began.
    pub(crate) scan_epoch: usize,
}

impl IncrementalState {
    pub(crate) fn new() -> IncrementalState {
        IncrementalState {
            phase: CollectionPhase::Idle,
            max_gen: Generation::Gen0,
            colors: FxHashMap::default(),
            gray_stack: Vec::new(),
            edges: FxHashMap::default(),
            scan_epoch: 0,
        }
    }
}

/// RAII guard for accessing GcMaps. Optionally holds a MutexGuard when
/// thread-safe access is needed (GlobalGarbageCollector, background strategies).
/// For thread-local access, the lock field is None and access is zero-cost.
pub(crate) struct GcMapsGuard<'a, O: ObjectMap = LocalObjects> {
    /// A raw pointer, not `&'a mut GcMaps`: a reference field would stay
    /// protected (in Stacked Borrows terms) for the whole of the guard's drop,
    /// while the drop releases the shard locks - so another thread could
    /// lock a shard and access it under a still-live `&mut` of the whole map.
    /// References are created per use in `Deref`/`DerefMut` instead.
    maps: std::ptr::NonNull<GcMaps<O>>,
    _marker: std::marker::PhantomData<&'a mut GcMaps<O>>,
    _global: std::sync::MutexGuard<'a, ()>,
    /// Shard guards; empty (no allocation) for an unsharded map.
    _shards: Vec<std::sync::MutexGuard<'a, ()>>,
}

impl<'a, O: ObjectMap> GcMapsGuard<'a, O> {
    /// A guard over `maps`, holding the global lock and (sharded) every shard
    /// lock.
    ///
    /// # Safety
    /// The locks must give exclusive access to `*maps` for `'a`.
    pub(crate) unsafe fn new(
        maps: std::ptr::NonNull<GcMaps<O>>,
        global: std::sync::MutexGuard<'a, ()>,
        shards: Vec<std::sync::MutexGuard<'a, ()>>,
    ) -> Self {
        GcMapsGuard {
            maps,
            _marker: std::marker::PhantomData,
            _global: global,
            _shards: shards,
        }
    }
}

impl<'a, O: ObjectMap> Deref for GcMapsGuard<'a, O> {
    type Target = GcMaps<O>;
    fn deref(&self) -> &GcMaps<O> {
        // SAFETY: the guard's locks give exclusive access to the maps.
        unsafe { self.maps.as_ref() }
    }
}

impl<'a, O: ObjectMap> DerefMut for GcMapsGuard<'a, O> {
    fn deref_mut(&mut self) -> &mut GcMaps<O> {
        // SAFETY: the guard's locks give exclusive access to the maps.
        unsafe { self.maps.as_mut() }
    }
}
