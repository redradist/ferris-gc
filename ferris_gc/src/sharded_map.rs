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
use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum shards, matching the shard bits reserved in a key's index.
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
    first: UnsafeCell<SlotMap<K, V>>,
    rest: Box<[UnsafeCell<SlotMap<K, V>>]>,
}

impl<K: SlotKey, V> ShardedSlotMap<K, V> {
    /// Create a map with `n` shards (clamped to `1..=MAX_SHARDS`).
    pub(crate) fn with_shards(n: usize) -> Self {
        let n = n.clamp(1, MAX_SHARDS);
        let rest = (1..n)
            .map(|_| UnsafeCell::new(SlotMap::new()))
            .collect::<Vec<_>>();
        ShardedSlotMap {
            first: UnsafeCell::new(SlotMap::new()),
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
            Some(&self.first)
        } else {
            self.rest.get(s - 1)
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
            None => self.first.get(),
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
            self.rest.get_mut(s - 1).map(UnsafeCell::get_mut)
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
            unsafe { self.rest.get_unchecked_mut(s - 1).get_mut() }
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
            s.get_mut().clear();
        }
    }

    pub(crate) fn drain(&mut self) -> Vec<(K, V)> {
        let mut out = self.first.get_mut().drain();
        for (i, s) in self.rest.iter_mut().enumerate() {
            out.extend(
                s.get_mut()
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
    rest: &'a mut [UnsafeCell<SlotMap<K, V>>],
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
                            self.cur = Some(head.get_mut().iter_mut());
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
    fn single_shard_behaves_like_slotmap() {
        let mut m: ShardedSlotMap<ObjectId, u32> = ShardedSlotMap::with_shards(1);
        let a = m.insert_in(0, 7);
        assert_eq!(a.shard(), 0);
        assert_eq!(m.get(a), Some(&7));
        assert_eq!(m.drain().len(), 1);
        assert!(m.is_empty());
    }
}
