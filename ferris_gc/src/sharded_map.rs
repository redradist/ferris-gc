//! Thread-affine sharded object map for the **sync** GC.
//!
//! The global GC's single `objects` map was the dominant scalability wall:
//! every allocation on every thread funnelled through one `Mutex`, so aggregate
//! allocation throughput *fell* as threads were added (measured 2.0 M/s at one
//! thread down to 1.0 M/s at four).
//!
//! Splitting the map only helps when each shard is **owned by a thread**.
//! Hash/round-robin sharding was measured to give nothing (14.9 M/s vs 15.8 M/s
//! for a single mutex) because every thread still touches every shard and the
//! cache lines keep bouncing. With thread affinity the same microbenchmark
//! reaches 56 M/s.
//!
//! So each thread inserts only into *its own* shard. Cross-thread operations
//! (a handle dropped on another thread, collection) address a shard by the
//! shard index encoded in the key, so they stay correct while the common
//! allocate-on-my-own-thread path stays uncontended.
//!
//! Locking lives *outside* this type (in `GarbageCollector`): the global
//! collector locks one shard for per-object work and every shard to collect.
//! The thread-local collector does not use this type at all - it keeps a plain
//! `SlotMap` (see `gc::ObjectMap`), so it pays nothing for sharding.

use crate::slot_map::{SlotKey, SlotMap};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::{LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Stop-the-world lock split per thread.
///
/// Every sync mutator operation (allocate, clone, `GcCell::borrow_mut`, weak
/// upgrade) takes a *read* lock so a collection cannot run underneath it. With
/// one `RwLock` that read is still an atomic write to a shared reader count, so
/// the line bounced between cores on every allocation. Here each thread
/// read-locks only **its own** padded `RwLock` (same index as its object
/// shard), and a collection write-locks **all** of them in index order.
///
/// Semantics are unchanged: a writer still excludes every reader, and readers
/// still never exclude each other. Lock order is always "STW lock(s) first,
/// then object shard lock(s)", writers always in index order, so it adds no
/// deadlock. `read`/`write` return `LockResult` like `RwLock`, so callers keep
/// their `.unwrap_or_else(|e| e.into_inner())`.
pub(crate) struct StwLock {
    shards: Box<[CachePadded<RwLock<()>>]>,
}

/// Write guard holding every shard of a [`StwLock`]. The first guard is kept
/// inline so a single-shard (thread-local) collector never allocates.
pub(crate) struct StwWriteGuard<'a> {
    _first: RwLockWriteGuard<'a, ()>,
    _rest: Vec<RwLockWriteGuard<'a, ()>>,
}

impl StwLock {
    pub(crate) fn with_shards(n: usize) -> Self {
        let n = n.clamp(1, MAX_SHARDS);
        StwLock {
            shards: (0..n)
                .map(|_| CachePadded(RwLock::new(())))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    /// Read-lock the calling thread's shard.
    #[inline]
    pub(crate) fn read(&self) -> LockResult<RwLockReadGuard<'_, ()>> {
        self.shards[current_thread_shard(self.shards.len())].read()
    }

    /// Write-lock every shard, in index order. Poisoning is ignored, as every
    /// caller already does with `.unwrap_or_else(|e| e.into_inner())`.
    pub(crate) fn write(&self) -> LockResult<StwWriteGuard<'_>> {
        let first = self.shards[0].write().unwrap_or_else(|e| e.into_inner());
        let rest = self.shards[1..]
            .iter()
            .map(|l| l.write().unwrap_or_else(|e| e.into_inner()))
            .collect();
        Ok(StwWriteGuard {
            _first: first,
            _rest: rest,
        })
    }
}

/// Counter interface a collector is generic over (see `gc::ObjectMap`): the
/// thread-local collector uses [`SingleCounter`] - a plain atomic, exactly the
/// old code - and the global collector uses [`ShardedCounter`]. Dispatch is
/// static, so the thread-local allocation path pays nothing for sharding
/// (routing it through a runtime-checked sharded counter cost ~24 extra
/// instructions per allocation).
pub(crate) trait Counter: Send + Sync {
    /// True when the value of the calling thread's slot is the exact total.
    const SINGLE: bool;
    fn with_shards(n: usize) -> Self;
    /// Add `n`; returns the calling thread's slot value afterwards (the exact
    /// total when [`Self::SINGLE`]).
    fn add(&self, n: usize) -> isize;
    fn sub(&self, n: usize);
    fn load(&self, order: Ordering) -> usize;
    fn store(&self, v: usize, order: Ordering);
}

/// A plain atomic counter (the thread-local collector).
pub(crate) struct SingleCounter(AtomicUsize);

impl Counter for SingleCounter {
    const SINGLE: bool = true;
    #[inline]
    fn with_shards(_n: usize) -> Self {
        SingleCounter(AtomicUsize::new(0))
    }
    #[inline]
    fn add(&self, n: usize) -> isize {
        (self.0.fetch_add(n, Ordering::Relaxed) + n) as isize
    }
    /// Saturating: never underflows when a release is tracked without a
    /// matching add.
    #[inline]
    fn sub(&self, n: usize) {
        let mut cur = self.0.load(Ordering::Relaxed);
        loop {
            match self.0.compare_exchange_weak(
                cur,
                cur.saturating_sub(n),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
    #[inline]
    fn load(&self, order: Ordering) -> usize {
        self.0.load(order)
    }
    #[inline]
    fn store(&self, v: usize, order: Ordering) {
        self.0.store(v, order)
    }
}

/// A counter split per thread, so the hot path writes only its own cache line
/// (the global collector).
///
/// The sync collector bumped `allocation_count` and `current_heap_size` on
/// every allocation and drop; as single atomics those lines bounced between
/// cores just like the old shared reader count. Each thread now updates its own
/// padded shard and readers sum the shards.
///
/// Shards are signed: an object allocated on one thread and freed on another
/// moves one shard down and another up; only the sum is meaningful, and it is
/// clamped at zero when read.
pub(crate) struct ShardedCounter {
    shards: Box<[CachePadded<AtomicIsize>]>,
}

impl Counter for ShardedCounter {
    const SINGLE: bool = false;
    fn with_shards(n: usize) -> Self {
        let n = n.clamp(1, MAX_SHARDS);
        ShardedCounter {
            shards: (0..n)
                .map(|_| CachePadded(AtomicIsize::new(0)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
    #[inline]
    fn add(&self, n: usize) -> isize {
        let mine = &self.shards[current_thread_shard(self.shards.len())];
        mine.fetch_add(n as isize, Ordering::Relaxed) + n as isize
    }
    /// A shard may go negative; the clamp happens on read.
    #[inline]
    fn sub(&self, n: usize) {
        let mine = &self.shards[current_thread_shard(self.shards.len())];
        mine.fetch_sub(n as isize, Ordering::Relaxed);
    }
    /// Sum of all shards, clamped at zero (shards are read `Relaxed`).
    #[inline]
    fn load(&self, _order: Ordering) -> usize {
        let sum: isize = self.shards.iter().map(|s| s.load(Ordering::Relaxed)).sum();
        sum.max(0) as usize
    }
    /// Reset the total to `v` (shard 0 holds it, the rest are zeroed).
    fn store(&self, v: usize, _order: Ordering) {
        self.shards[0].store(v as isize, Ordering::Relaxed);
        for s in self.shards[1..].iter() {
            s.store(0, Ordering::Relaxed);
        }
    }
}

/// Maximum shards, matching the shard bits reserved in a key's index.
/// Aligns `T` to its own 128-byte slot so that per-thread state written on the
/// hot path never shares a cache line with another thread's.
///
/// Sharding by thread separates the data *logically*, but adjacent shards (and
/// their 8-byte `Mutex<()>` locks - eight to a line) still shared cache lines,
/// so every lock/insert bounced the line between cores: four threads allocated
/// no faster than two. 128 rather than 64 because Intel's adjacent-line
/// prefetcher pulls lines in pairs.
#[repr(align(128))]
pub(crate) struct CachePadded<T>(pub(crate) T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for CachePadded<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

pub(crate) const MAX_SHARDS: usize = 1 << crate::slot_map::SHARD_BITS;

/// Hands each thread a stable, distinct ordinal. Threads keep it for life,
/// which is what gives shard **affinity** — the property that actually removes
/// the contention.
static NEXT_THREAD_ORDINAL: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_ORDINAL: usize = NEXT_THREAD_ORDINAL.fetch_add(1, Ordering::Relaxed);
}

/// Shard this thread owns in a map of `shard_count` shards.
#[inline]
pub(crate) fn current_thread_shard(shard_count: usize) -> usize {
    if shard_count <= 1 {
        return 0;
    }
    THREAD_ORDINAL.try_with(|t| *t % shard_count).unwrap_or(0)
}

#[cold]
#[inline(never)]
fn shard_exhausted() -> ! {
    panic!(
        "global GC object shard exhausted ({} slots)",
        crate::slot_map::MAX_SLOTS_PER_SHARD
    )
}

/// One shard: its own `SlotMap`, behind its own cell, on its own cache line.
type Shard<K, V> = CachePadded<UnsafeCell<SlotMap<K, V>>>;

/// A `SlotMap` split into per-thread shards, exposing a `SlotMap`-compatible
/// surface so collector code iterates it unchanged.
///
/// Each shard lives in its own `UnsafeCell`. That is what makes concurrent
/// per-shard mutation sound: a thread working on its shard obtains `&mut` to
/// **that shard only** (via [`Self::shard_ptr`]), never to the map or to the
/// enclosing `GcMaps`. Two threads each holding `&mut GcMaps` - even while
/// touching disjoint shards - is undefined behaviour (Miri reports the
/// reference retags as a data race), so the shard-scoped paths must reach their
/// shard through a shared reference plus this per-shard cell.
///
/// Shard 0 is stored inline; only shards 1.. live behind a `Box`, so the thread
/// owning shard 0 pays no extra dependent load.
///
/// # Concurrency contract
/// `&mut self` methods need exclusive access to the whole map (the global
/// collector's `lock_gc_maps`, which holds every shard lock). `&self` readers
/// (`get`, `iter`, `values`, `len`) need the same exclusion of shard writers:
/// they are only called under `lock_gc_maps`. Concurrent per-shard work goes
/// through [`Self::shard_ptr`] under that shard's lock.
pub(crate) struct ShardedSlotMap<K: SlotKey, V> {
    first: Shard<K, V>,
    rest: Box<[Shard<K, V>]>,
}

impl<K: SlotKey, V> ShardedSlotMap<K, V> {
    /// Create a map with `n` shards (clamped to `1..=MAX_SHARDS`).
    pub(crate) fn with_shards(n: usize) -> Self {
        let n = n.clamp(1, MAX_SHARDS);
        let rest = (1..n)
            .map(|_| CachePadded(UnsafeCell::new(SlotMap::new())))
            .collect::<Vec<_>>();
        ShardedSlotMap {
            first: CachePadded(UnsafeCell::new(SlotMap::new())),
            rest: rest.into_boxed_slice(),
        }
    }

    #[inline]
    pub(crate) fn shard_count(&self) -> usize {
        1 + self.rest.len()
    }

    /// Cell of shard `s`, or `None` if out of range.
    #[inline]
    fn cell(&self, s: usize) -> Option<&UnsafeCell<SlotMap<K, V>>> {
        if s == 0 {
            Some(&self.first.0)
        } else {
            self.rest.get(s - 1).map(|p| &p.0)
        }
    }

    /// Raw pointer to shard `s` (clamped into range), for shard-scoped work
    /// through a **shared** reference to the map.
    ///
    /// # Safety
    /// Dereferencing it mutably requires holding that shard's lock and no
    /// concurrent `&mut self` / `&self`-reader use of the map (both happen only
    /// under the full lock, which also holds this shard's lock).
    #[inline]
    pub(crate) fn shard_ptr(&self, s: usize) -> *mut SlotMap<K, V> {
        match self.cell(s) {
            Some(c) => c.get(),
            None => self.first.0.get(),
        }
    }

    /// Turn a key produced by shard `shard`'s own `SlotMap` into a map-wide
    /// key by tagging it with the shard index.
    ///
    /// A shard's slot index must stay below `MAX_SLOTS_PER_SHARD`, or it would
    /// spill into the tag bits and alias another shard's key; that is checked
    /// here, deterministically, rather than left to wrap silently.
    #[inline]
    pub(crate) fn tag_key(raw: K, shard: usize) -> K {
        if raw.index() >= crate::slot_map::MAX_SLOTS_PER_SHARD {
            shard_exhausted();
        }
        if shard == 0 {
            raw
        } else {
            raw.with_shard(shard)
        }
    }

    #[inline]
    fn shard_mut(&mut self, s: usize) -> Option<&mut SlotMap<K, V>> {
        if s == 0 {
            Some(self.first.get_mut())
        } else {
            self.rest.get_mut(s - 1).map(|p| p.0.get_mut())
        }
    }

    /// Shared view of shard `s`. See the type-level concurrency contract.
    #[inline]
    fn shard_ref(&self, s: usize) -> Option<&SlotMap<K, V>> {
        // SAFETY: `&self` readers run only under the full lock, which excludes
        // every shard-scoped writer, so no `&mut` to this shard is live.
        self.cell(s).map(|c| unsafe { &*c.get() })
    }

    /// Insert into `shard`, returning a key tagged with that shard index.
    /// (Tests only; the collector inserts via `shard_ptr` + `tag_key`.)
    #[cfg(test)]
    pub(crate) fn insert_in(&mut self, shard: usize, value: V) -> K {
        let shard = if shard < self.shard_count() { shard } else { 0 };
        let raw = self.shard_mut(shard).expect("shard in range").insert(value);
        Self::tag_key(raw, shard)
    }

    /// A key whose shard tag is out of range addresses nothing (`None`).
    #[inline]
    pub(crate) fn get(&self, key: K) -> Option<&V> {
        self.shard_ref(key.shard())?.get(key.strip_shard())
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.shard_mut(key.shard())?.get_mut(key.strip_shard())
    }

    #[inline]
    pub(crate) fn remove(&mut self, key: K) -> Option<V> {
        self.shard_mut(key.shard())?.remove(key.strip_shard())
    }

    /// Remove without bounds or generation checks, from the key's own shard.
    ///
    /// # Safety
    /// The key must refer to a valid, currently occupied slot in that shard.
    #[inline(always)]
    pub(crate) unsafe fn remove_unchecked(&mut self, key: K) -> V {
        let s = key.shard();
        let m = if s == 0 {
            self.first.get_mut()
        } else {
            unsafe { self.rest.get_unchecked_mut(s - 1).0.get_mut() }
        };
        unsafe { m.remove_unchecked(key.strip_shard()) }
    }

    pub(crate) fn len(&self) -> usize {
        (0..self.shard_count())
            .filter_map(|s| self.shard_ref(s))
            .map(|m| m.len())
            .sum()
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate every occupied entry across all shards. Keys are re-tagged with
    /// their shard index so callers get the same ids they were handed.
    ///
    /// Hand-rolled rather than `flat_map`: collection walks this in its hottest
    /// loops.
    pub(crate) fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            map: self,
            shard: 0,
            cur: self.shard_ref(0).map(|m| m.iter()),
        }
    }

    pub(crate) fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            rest: &mut self.rest,
            shard: 0,
            cur: Some(self.first.get_mut().iter_mut()),
        }
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    #[allow(dead_code)]
    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }

    #[allow(dead_code)]
    pub(crate) fn clear(&mut self) {
        self.first.get_mut().clear();
        for s in self.rest.iter_mut() {
            s.0.get_mut().clear();
        }
    }

    pub(crate) fn drain(&mut self) -> Vec<(K, V)> {
        let mut out = self.first.get_mut().drain();
        for (i, s) in self.rest.iter_mut().enumerate() {
            out.extend(
                s.0.get_mut()
                    .drain()
                    .into_iter()
                    .map(|(k, v)| (k.with_shard(i + 1), v)),
            );
        }
        out
    }
}

/// Iterator over every occupied entry of a [`ShardedSlotMap`].
pub(crate) struct Iter<'a, K: SlotKey, V> {
    map: &'a ShardedSlotMap<K, V>,
    shard: usize,
    cur: Option<crate::slot_map::Iter<'a, K, V>>,
}

impl<'a, K: SlotKey, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.cur.as_mut()?.next() {
                Some((k, v)) if self.shard == 0 => return Some((k, v)),
                Some((k, v)) => return Some((k.with_shard(self.shard), v)),
                None => {
                    self.shard += 1;
                    self.cur = self.map.shard_ref(self.shard).map(|m| m.iter());
                }
            }
        }
    }
}

/// Mutable iterator over every occupied entry of a [`ShardedSlotMap`].
pub(crate) struct IterMut<'a, K: SlotKey, V> {
    rest: &'a mut [Shard<K, V>],
    shard: usize,
    cur: Option<crate::slot_map::IterMut<'a, K, V>>,
}

impl<'a, K: SlotKey, V> Iterator for IterMut<'a, K, V> {
    type Item = (K, &'a mut V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.cur.as_mut()?.next() {
                Some((k, v)) if self.shard == 0 => return Some((k, v)),
                Some((k, v)) => return Some((k.with_shard(self.shard), v)),
                None => {
                    // Slice-pattern match moves the `&'a mut [_]` so head/tail
                    // keep 'a (split_first_mut would reborrow for this call).
                    match std::mem::take(&mut self.rest) {
                        [] => self.cur = None,
                        [head, tail @ ..] => {
                            self.shard += 1;
                            self.cur = Some(head.0.get_mut().iter_mut());
                            self.rest = tail;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot_map::ObjectId;

    #[test]
    fn insert_get_remove_across_shards() {
        let mut m: ShardedSlotMap<ObjectId, u32> = ShardedSlotMap::with_shards(4);
        let a = m.insert_in(0, 10);
        let b = m.insert_in(3, 20);
        assert_eq!(a.shard(), 0);
        assert_eq!(b.shard(), 3);
        assert_eq!(m.get(a), Some(&10));
        assert_eq!(m.get(b), Some(&20));
        assert_eq!(m.len(), 2);
        assert_eq!(m.remove(a), Some(10));
        assert_eq!(m.get(a), None);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn keys_from_different_shards_do_not_collide() {
        let mut m: ShardedSlotMap<ObjectId, u32> = ShardedSlotMap::with_shards(4);
        // Same slot index in different shards must be distinct keys.
        let a = m.insert_in(1, 1);
        let b = m.insert_in(2, 2);
        assert_ne!(a, b);
        assert_eq!(m.get(a), Some(&1));
        assert_eq!(m.get(b), Some(&2));
    }

    #[test]
    fn iter_and_iter_mut_cover_every_shard_with_tagged_keys() {
        let mut m: ShardedSlotMap<ObjectId, u32> = ShardedSlotMap::with_shards(4);
        // Leave shard 1 empty to exercise skipping an empty shard.
        let ids: Vec<_> = [0usize, 2, 2, 3]
            .iter()
            .enumerate()
            .map(|(v, &sh)| m.insert_in(sh, v as u32))
            .collect();
        let seen: Vec<_> = m.iter().map(|(k, v)| (k, *v)).collect();
        assert_eq!(seen.len(), ids.len());
        for (k, v) in &seen {
            assert_eq!(m.get(*k), Some(v), "iter key must address its entry");
        }
        for (_, v) in m.iter_mut() {
            *v += 100;
        }
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(m.get(*id), Some(&(i as u32 + 100)));
        }
    }

    #[test]
    fn sharded_counter_sums_across_threads_and_clamps() {
        let c = std::sync::Arc::new(ShardedCounter::with_shards(8));
        // Allocate on four threads, free everything on one other thread: the
        // per-thread slots go negative/positive but the sum is exact.
        let adders: Vec<_> = (0..4)
            .map(|_| {
                let c = c.clone();
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        c.add(3);
                    }
                })
            })
            .collect();
        for h in adders {
            h.join().unwrap();
        }
        assert_eq!(c.load(Ordering::Relaxed), 12_000);
        let c2 = c.clone();
        std::thread::spawn(move || {
            for _ in 0..1000 {
                c2.sub(9);
            }
        })
        .join()
        .unwrap();
        assert_eq!(c.load(Ordering::Relaxed), 3_000);
        // More released than added: the total clamps at zero.
        c.sub(10_000);
        assert_eq!(c.load(Ordering::Relaxed), 0);
        c.store(5, Ordering::Relaxed);
        assert_eq!(c.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn single_counter_saturates() {
        let c = SingleCounter::with_shards(1);
        assert_eq!(c.add(4), 4);
        c.sub(10);
        assert_eq!(c.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn stw_write_excludes_a_reader_on_another_threads_shard() {
        use std::sync::atomic::AtomicBool;
        use std::sync::{Arc, mpsc};
        let lock = Arc::new(StwLock::with_shards(8));
        let acquired = Arc::new(AtomicBool::new(false));
        let (held_tx, held_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        // A reader on its own thread (hence its own shard) holds the lock...
        let l = lock.clone();
        let reader = std::thread::spawn(move || {
            let _g = l.read().unwrap_or_else(|e| e.into_inner());
            held_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        held_rx.recv().unwrap();
        // ...so a writer on yet another thread must not get through.
        let (l, a) = (lock.clone(), acquired.clone());
        let writer = std::thread::spawn(move || {
            let _g = l.write().unwrap_or_else(|e| e.into_inner());
            a.store(true, Ordering::SeqCst);
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            !acquired.load(Ordering::SeqCst),
            "a STW write must exclude readers on every shard"
        );
        release_tx.send(()).unwrap();
        reader.join().unwrap();
        writer.join().unwrap();
        assert!(acquired.load(Ordering::SeqCst));
    }

    #[test]
    fn single_shard_behaves_like_slotmap() {
        let mut m: ShardedSlotMap<ObjectId, u32> = ShardedSlotMap::with_shards(1);
        let a = m.insert_in(0, 7);
        assert_eq!(a.shard(), 0);
        assert_eq!(m.get(a), Some(&7));
        assert_eq!(m.drain().len(), 1);
        assert!(m.is_empty());
    }
}
