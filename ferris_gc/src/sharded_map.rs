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
//! Locking lives *outside* this type (in `GarbageCollector`), mirroring the
//! existing `gc_maps` / `gc_maps_lock` split: the thread-local collector takes
//! no locks at all, while the global collector locks one shard for per-object
//! work and every shard to collect.

use crate::slot_map::{SlotKey, SlotMap};
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

/// A `SlotMap` split into per-thread shards, exposing a `SlotMap`-compatible
/// surface so collector code iterates it unchanged.
pub(crate) struct ShardedSlotMap<K: SlotKey, V> {
    shards: Box<[SlotMap<K, V>]>,
}

impl<K: SlotKey, V> ShardedSlotMap<K, V> {
    /// Create a map with `n` shards (clamped to `1..=MAX_SHARDS`).
    pub(crate) fn with_shards(n: usize) -> Self {
        let n = n.clamp(1, MAX_SHARDS);
        let shards = (0..n).map(|_| SlotMap::new()).collect::<Vec<_>>();
        ShardedSlotMap {
            shards: shards.into_boxed_slice(),
        }
    }

    /// Index of the shard owning `key`, clamped so a malformed key cannot
    /// index out of bounds.
    #[inline]
    fn shard_of(&self, key: K) -> usize {
        let s = key.shard();
        if s < self.shards.len() { s } else { 0 }
    }

    /// Insert into `shard`, returning a key tagged with that shard index.
    #[inline]
    pub(crate) fn insert_in(&mut self, shard: usize, value: V) -> K {
        let shard = if shard < self.shards.len() { shard } else { 0 };
        let raw = self.shards[shard].insert(value);
        raw.with_shard(shard)
    }

    /// Insert into the **calling thread's** shard. Mirrors `SlotMap::insert`
    /// so existing call sites are unchanged.
    #[inline]
    pub(crate) fn insert(&mut self, value: V) -> K {
        let shard = current_thread_shard(self.shards.len());
        self.insert_in(shard, value)
    }

    #[inline]
    pub(crate) fn get(&self, key: K) -> Option<&V> {
        self.shards[self.shard_of(key)].get(key.strip_shard())
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, key: K) -> Option<&mut V> {
        let s = self.shard_of(key);
        self.shards[s].get_mut(key.strip_shard())
    }

    #[inline]
    pub(crate) fn remove(&mut self, key: K) -> Option<V> {
        let s = self.shard_of(key);
        self.shards[s].remove(key.strip_shard())
    }

    /// Remove without bounds or generation checks, from the key's own shard.
    ///
    /// # Safety
    /// The key must refer to a valid, currently occupied slot in that shard.
    #[inline(always)]
    pub(crate) unsafe fn remove_unchecked(&mut self, key: K) -> V {
        let s = self.shard_of(key);
        unsafe { self.shards[s].remove_unchecked(key.strip_shard()) }
    }

    pub(crate) fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Iterate every occupied entry across all shards. Keys are re-tagged with
    /// their shard index so callers get the same ids they were handed.
    ///
    /// Hand-rolled rather than `flat_map`: collection walks this in its hottest
    /// loops, and the nested-closure version measurably slowed the thread-local
    /// collector (one shard, which should pay nothing).
    pub(crate) fn iter(&self) -> Iter<'_, K, V> {
        let mut it = Iter {
            shards: &self.shards,
            next_shard: 0,
            shard: 0,
            cur: None,
        };
        it.advance();
        it
    }

    pub(crate) fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let mut it = IterMut {
            rest: &mut self.shards,
            shard: 0,
            cur: None,
        };
        it.advance();
        it
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
        for s in self.shards.iter_mut() {
            s.clear();
        }
    }

    pub(crate) fn drain(&mut self) -> Vec<(K, V)> {
        let mut out = Vec::with_capacity(self.len());
        for (i, s) in self.shards.iter_mut().enumerate() {
            out.extend(s.drain().into_iter().map(|(k, v)| (k.with_shard(i), v)));
        }
        out
    }
}

/// Iterator over every occupied entry of a [`ShardedSlotMap`].
pub(crate) struct Iter<'a, K: SlotKey, V> {
    shards: &'a [SlotMap<K, V>],
    next_shard: usize,
    shard: usize,
    cur: Option<crate::slot_map::Iter<'a, K, V>>,
}

impl<'a, K: SlotKey, V> Iter<'a, K, V> {
    fn advance(&mut self) {
        if self.next_shard < self.shards.len() {
            let i = self.next_shard;
            self.next_shard += 1;
            self.shard = i;
            self.cur = Some(self.shards[i].iter());
        } else {
            self.cur = None;
        }
    }
}

impl<'a, K: SlotKey, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.cur.as_mut()?.next() {
                Some((k, v)) => return Some((k.with_shard(self.shard), v)),
                None => {
                    if self.next_shard >= self.shards.len() {
                        self.cur = None;
                        return None;
                    }
                    self.advance();
                }
            }
        }
    }
}

/// Mutable iterator over every occupied entry of a [`ShardedSlotMap`].
pub(crate) struct IterMut<'a, K: SlotKey, V> {
    rest: &'a mut [SlotMap<K, V>],
    shard: usize,
    cur: Option<crate::slot_map::IterMut<'a, K, V>>,
}

impl<'a, K: SlotKey, V> IterMut<'a, K, V> {
    fn advance(&mut self) {
        // Slice-pattern match moves the `&'a mut [_]` so head/tail keep 'a
        // (a reborrow via split_first_mut would shorten it to this call).
        match std::mem::take(&mut self.rest) {
            [] => self.cur = None,
            [head, tail @ ..] => {
                self.cur = Some(head.iter_mut());
                self.rest = tail;
            }
        }
    }
}

impl<'a, K: SlotKey, V> Iterator for IterMut<'a, K, V> {
    type Item = (K, &'a mut V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.cur.as_mut()?.next() {
                Some((k, v)) => return Some((k.with_shard(self.shard), v)),
                None => {
                    if self.rest.is_empty() {
                        self.cur = None;
                        return None;
                    }
                    self.shard += 1;
                    self.advance();
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
        let a = m.insert(7);
        assert_eq!(a.shard(), 0);
        assert_eq!(m.get(a), Some(&7));
        assert_eq!(m.drain().len(), 1);
        assert!(m.is_empty());
    }
}
