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
    /// See `SlotMap::park_vacant_runs`.
    fn park_vacant_runs(&mut self);
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
            #[inline(always)]
            fn park_vacant_runs(&mut self) {
                <$ty>::park_vacant_runs(self)
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
    /// Avoids iterating all objects during Gen0 mark/sweep. Thread-local
    /// collector only: sync allocations cannot append to it (see
    /// `collect_generation`), so it stays empty there.
    pub(crate) gen0_ids: Vec<ObjectId>,
    /// Which collections could free anything (thread-local collector): bit
    /// `g` is set if a collection of generations `0..=g` could.
    ///
    /// The last handle's drop frees an object at once, so a collection only
    /// frees unreachable objects that still have handles (cycles). Such
    /// garbage can only appear when a handle is dropped and its object keeps
    /// other handles: a handle moved into another object stays reachable
    /// through it. Every such drop sets all bits; a collection of
    /// generations `0..=g` clears bits `0..=g` when it starts, as it frees
    /// all garbage in that scope (later drops set them again). An incomplete
    /// scan or an abandoned incremental cycle sets all bits again. The
    /// allocation triggers skip a collection whose bit is clear. The sync
    /// collector has no allocation trigger and never sets them.
    pub(crate) garbage_possible: u8,
}

/// `GcMaps::garbage_possible` with every generation's bit set.
pub(crate) const GARBAGE_POSSIBLE_ALL: u8 = 0b111;

impl<O: ObjectMap> GcMaps<O> {
    /// A handle was dropped while its object kept others, or a collection
    /// left garbage it may not have seen: any collection could free some.
    #[inline(always)]
    pub(crate) fn note_garbage_possible(&mut self) {
        self.garbage_possible = GARBAGE_POSSIBLE_ALL;
    }

    /// A collection of generations `0..=max_gen` starts: it frees all the
    /// garbage that scope holds now. Its passes over the object map skip
    /// the long runs of vacant slots a large free left behind.
    #[inline]
    pub(crate) fn start_collecting(&mut self, max_gen: Generation) {
        self.garbage_possible &= !((2u8 << max_gen as u8) - 1);
        self.objects.park_vacant_runs();
    }

    /// Whether a collection of generations `0..=max_gen` could free anything.
    #[inline]
    pub(crate) fn garbage_possible(&self, max_gen: Generation) -> bool {
        self.garbage_possible & (1 << max_gen as u8) != 0
    }

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
        self.ptr_to_object.reserve(self.objects.len());
        for (id, entry) in self.objects.iter() {
            let thin = entry.ptr.get_thin_ptr();
            self.ptr_to_object.insert(thin, id);
        }
    }
}

/// An in-scope object's mark: its color, and where its snapshot children
/// are in [`IncrementalState::children`].
#[derive(Clone, Copy)]
pub(crate) struct Mark {
    pub(crate) color: MarkColor,
    first_child: u32,
    child_count: u32,
}

/// State for incremental tri-color marking.
///
/// Marking is snapshot-at-the-beginning: `begin_collection` records each
/// in-scope object's children as they are then, and the mark follows only
/// that snapshot. Mutations made while the mark runs therefore cannot hide an
/// object from it, so no write barrier is needed: every object reachable when
/// marking began is marked, and an object born since is greyed, so it is
/// live (`note_new_object_during_marking`). A handle a weak upgrade creates
/// is the one way to reach an object that was not reachable then, so the
/// upgrade greys it (`note_handle_during_marking`).
pub(crate) struct IncrementalState {
    pub(crate) phase: CollectionPhase,
    pub(crate) max_gen: Generation,
    /// The mark of every in-scope object and every object born mid-mark.
    pub(crate) marks: FxHashMap<ObjectId, Mark>,
    pub(crate) gray_stack: Vec<ObjectId>,
    /// The snapshot's children of all objects, one run per object (one
    /// buffer rather than a `Vec` per object).
    children: Vec<ObjectId>,
    /// `blocked_scans` when this cycle's marking began.
    pub(crate) scan_epoch: usize,
}

impl IncrementalState {
    pub(crate) fn new() -> IncrementalState {
        IncrementalState {
            phase: CollectionPhase::Idle,
            max_gen: Generation::Gen0,
            marks: FxHashMap::default(),
            gray_stack: Vec::new(),
            children: Vec::new(),
            scan_epoch: 0,
        }
    }

    /// Forget the cycle's marks, worklist and snapshot.
    pub(crate) fn clear(&mut self) {
        self.marks.clear();
        self.gray_stack.clear();
        self.children.clear();
    }

    /// Record an in-scope object White, with its children now. Children out
    /// of scope get no mark, so `grey` skips them.
    pub(crate) fn snapshot(&mut self, id: ObjectId, children: impl Iterator<Item = ObjectId>) {
        let first = self.children.len();
        self.children.extend(children);
        let mark = Mark {
            color: MarkColor::White,
            first_child: u32::try_from(first).expect("snapshot over 2^32 edges"),
            child_count: (self.children.len() - first) as u32,
        };
        self.marks.insert(id, mark);
    }

    /// Grey an object born mid-mark. It has no snapshot, so its step only
    /// blackens it. Out of line: inlined, it changed the code of the
    /// allocation fast path that only checks the `marking` gate.
    #[inline(never)]
    pub(crate) fn grey_born(&mut self, id: ObjectId) {
        let mark = Mark {
            color: MarkColor::Gray,
            first_child: 0,
            child_count: 0,
        };
        self.marks.insert(id, mark);
        self.gray_stack.push(id);
    }

    /// Grey `id` if it is in scope and not yet reached.
    #[inline]
    pub(crate) fn grey(&mut self, id: ObjectId) {
        if let Some(mark) = self.marks.get_mut(&id) {
            if mark.color == MarkColor::White {
                mark.color = MarkColor::Gray;
                self.gray_stack.push(id);
            }
        }
    }

    /// Whether marking reached `id`.
    #[inline]
    pub(crate) fn is_black(&self, id: ObjectId) -> bool {
        matches!(self.marks.get(&id), Some(mark) if mark.color == MarkColor::Black)
    }

    /// Blacken grey objects while `keep_going()` allows, greying their
    /// snapshot children. Returns `true` when no grey object is left.
    pub(crate) fn mark(&mut self, mut keep_going: impl FnMut() -> bool) -> bool {
        while keep_going() {
            let Some(id) = self.gray_stack.pop() else {
                break;
            };
            let Some(mark) = self.marks.get_mut(&id) else {
                continue;
            };
            mark.color = MarkColor::Black;
            let first = mark.first_child as usize;
            for i in first..first + mark.child_count as usize {
                let child = self.children[i];
                self.grey(child);
            }
        }
        self.gray_stack.is_empty()
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
