use std::marker::PhantomData;

/// Trait for slot-map key types with generation-counted u64 encoding.
/// Upper 32 bits = generation, lower 32 bits = index.
///
/// The `shard` accessors are used only by `crate::sharded_map` (the global
/// collector). There the top `SHARD_BITS` of the index carry the owning shard,
/// and the sharded map itself guarantees a shard never hands out an index that
/// would reach those bits. The thread-local collector uses a plain `SlotMap`, so
/// its keys keep the full 32-bit index space and pay no masking.
pub trait SlotKey: Copy + Eq {
    fn from_raw(index: u32, ver: u32) -> Self;
    fn index(self) -> u32;
    fn slot_generation(self) -> u32;

    /// Index of the shard owning this key (see `crate::sharded_map`).
    fn shard(self) -> usize;
    /// Same key tagged with `shard`.
    fn with_shard(self, shard: usize) -> Self;
    /// Same key with the shard tag cleared, as the owning shard stores it.
    fn strip_shard(self) -> Self;

    #[allow(dead_code)]
    fn raw(self) -> u64 {
        ((self.slot_generation() as u64) << 32) | (self.index() as u64)
    }
}

/// Identifies an object slot in the GC's object arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ObjectId(u64);

impl ObjectId {
    /// Sentinel value used as a placeholder before the real id is assigned.
    pub const ZERO: ObjectId = ObjectId(0);
}

/// Bits of the index reserved for the shard tag in **sharded** keys.
pub const SHARD_BITS: u32 = 6;
/// Bits left for the per-shard slot index in sharded keys.
pub const INDEX_BITS: u32 = 32 - SHARD_BITS;
/// Maximum slots one shard of a sharded map may hold (67,108,864). Total
/// sharded capacity (64 x 67.1M) equals the plain 32-bit index space.
pub const MAX_SLOTS_PER_SHARD: u32 = 1 << INDEX_BITS;
const SHARD_MASK: u64 = ((1u64 << SHARD_BITS) - 1) << INDEX_BITS;

impl SlotKey for ObjectId {
    fn from_raw(index: u32, slot_gen: u32) -> Self {
        ObjectId(((slot_gen as u64) << 32) | (index as u64))
    }
    fn index(self) -> u32 {
        self.0 as u32
    }
    fn slot_generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
    fn shard(self) -> usize {
        ((self.0 & SHARD_MASK) >> INDEX_BITS) as usize
    }
    fn with_shard(self, shard: usize) -> Self {
        ObjectId((self.0 & !SHARD_MASK) | (((shard as u64) << INDEX_BITS) & SHARD_MASK))
    }
    fn strip_shard(self) -> Self {
        ObjectId(self.0 & !SHARD_MASK)
    }
}

const SENTINEL: u32 = u32::MAX;

/// Shorter runs of vacant slots stay on the free list.
const MIN_PARKED_RUN: u32 = 64;

enum SlotValue<V> {
    Occupied(V),
    Vacant { next_free: u32 },
}

struct Slot<V> {
    ver: u32,
    value: SlotValue<V>,
}

/// A slot-map: O(1) insert/remove with generation-checked keys and
/// cache-friendly sequential iteration over a dense `Vec`.
///
/// Iteration stops after the last occupied slot and jumps over *parked* runs
/// of vacant slots. Without that, a map that once held many more values than
/// it does now (a large structure built, then dropped) makes every pass walk
/// every vacant slot, and since freed slots are reused last-freed first, new
/// values land at the high indices. `park_vacant_runs` takes long runs of
/// vacant slots off the free list. Once the free list runs out, inserts take
/// the lowest parked run's slots in order, before growing the map; removes
/// never touch the parked runs.
pub struct SlotMap<K: SlotKey, V> {
    slots: Vec<Slot<V>>,
    free_head: u32,
    len: usize,
    /// Parked runs of vacant slots, as `start..end`: disjoint, not on the
    /// free list, in descending order (the lowest run is unparked first).
    parked: Vec<(u32, u32)>,
    /// Slots in `parked`.
    parked_slots: usize,
    _marker: PhantomData<K>,
}

impl<K: SlotKey, V> SlotMap<K, V> {
    pub fn new() -> Self {
        SlotMap {
            slots: Vec::new(),
            free_head: SENTINEL,
            len: 0,
            parked: Vec::new(),
            parked_slots: 0,
            _marker: PhantomData,
        }
    }

    /// Number of occupied slots.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of slots, occupied or vacant.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Every vacant slot is on the free list or in exactly one parked run;
    /// the runs are disjoint, apart (adjacent ones are merged), all vacant
    /// and in descending order.
    #[cfg(test)]
    fn check_invariants(&self) {
        let mut on_free_list = vec![false; self.slots.len()];
        let mut i = self.free_head;
        while i != SENTINEL {
            assert!(!on_free_list[i as usize], "free list cycles");
            on_free_list[i as usize] = true;
            match self.slots[i as usize].value {
                SlotValue::Vacant { next_free } => i = next_free,
                SlotValue::Occupied(_) => panic!("an occupied slot is on the free list"),
            }
        }
        let mut parked = vec![false; self.slots.len()];
        let mut prev_start = u32::MAX;
        for &(start, end) in &self.parked {
            assert!(
                start < end && end < prev_start,
                "runs not descending, disjoint and apart"
            );
            prev_start = start;
            for j in start..end {
                parked[j as usize] = true;
            }
        }
        assert_eq!(parked.iter().filter(|&&p| p).count(), self.parked_slots);
        for (j, slot) in self.slots.iter().enumerate() {
            let vacant = matches!(slot.value, SlotValue::Vacant { .. });
            assert_eq!(
                vacant,
                on_free_list[j] ^ parked[j],
                "slot {j}: vacant {vacant}, free {}, parked {}",
                on_free_list[j],
                parked[j]
            );
        }
    }

    /// A vacant slot for an insert: the head of the free list, else the
    /// first slot of the lowest parked run. `None` if neither has one.
    #[inline(always)]
    fn take_vacant_slot(&mut self) -> Option<u32> {
        if self.free_head != SENTINEL {
            let idx = self.free_head;
            // SAFETY: free_head always points to a valid Vacant slot.
            let slot = unsafe { self.slots.get_unchecked_mut(idx as usize) };
            self.free_head = match slot.value {
                SlotValue::Vacant { next_free } => next_free,
                // SAFETY: free_head invariant guarantees this is Vacant.
                _ => unsafe { std::hint::unreachable_unchecked() },
            };
            return Some(idx);
        }
        let run = self.parked.last_mut()?;
        let idx = run.0;
        run.0 += 1;
        if run.0 == run.1 {
            self.parked.pop();
        }
        self.parked_slots -= 1;
        Some(idx)
    }

    /// Park the runs of at least `MIN_PARKED_RUN` vacant slots, so passes
    /// over the map jump over them. The pass costs O(len + vacant slots not
    /// parked yet), so it runs only once those outnumber the occupied ones
    /// (by 1024): amortized O(1) per freed slot.
    #[inline]
    pub fn park_vacant_runs(&mut self) {
        let unparked_vacant = self.slots.len() - self.len - self.parked_slots;
        if unparked_vacant > self.len + 1024 {
            self.park_vacant_runs_now();
        }
    }

    #[cold]
    #[inline(never)]
    fn park_vacant_runs_now(&mut self) {
        let n = self.slots.len();
        let occupied =
            |slots: &[Slot<V>], i: usize| matches!(slots[i].value, SlotValue::Occupied(_));
        let old = std::mem::take(&mut self.parked);
        let mut old_runs = old.iter().rev().copied();
        let mut next_old = old_runs.next();
        let mut runs: Vec<(u32, u32)> = Vec::new(); // ascending
        // The free list is rebuilt from the vacant slots not parked, lowest
        // first.
        let (mut head, mut tail) = (SENTINEL, SENTINEL);
        let mut i = 0;
        while i < n {
            // The slots up to the next old run, a vacant run at a time.
            let seg_end = next_old.map_or(n, |(start, _)| start as usize);
            while i < seg_end {
                while i < seg_end && occupied(&self.slots, i) {
                    i += 1;
                }
                let start = i;
                while i < seg_end && !occupied(&self.slots, i) {
                    i += 1;
                }
                if i - start >= MIN_PARKED_RUN as usize {
                    runs.push((start as u32, i as u32));
                    continue;
                }
                for j in start as u32..i as u32 {
                    self.slots[j as usize].value = SlotValue::Vacant {
                        next_free: SENTINEL,
                    };
                    if tail == SENTINEL {
                        head = j;
                    } else if let SlotValue::Vacant { next_free } =
                        &mut self.slots[tail as usize].value
                    {
                        *next_free = j;
                    }
                    tail = j;
                }
            }
            if let Some((start, end)) = next_old {
                runs.push((start, end));
                i = end as usize;
                next_old = old_runs.next();
            }
        }
        // Adjacent runs become one.
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(runs.len());
        for (start, end) in runs {
            match merged.last_mut() {
                Some(last) if last.1 == start => last.1 = end,
                _ => merged.push((start, end)),
            }
        }
        self.free_head = head;
        self.parked_slots = merged.iter().map(|&(s, e)| (e - s) as usize).sum();
        merged.reverse();
        self.parked = merged;
    }

    /// Returns `true` if there are no occupied slots.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a value and return its key.
    #[inline]
    pub fn insert(&mut self, value: V) -> K {
        self.len += 1;
        if let Some(idx) = self.take_vacant_slot() {
            // SAFETY: `take_vacant_slot` returns a vacant slot of this map.
            let slot = unsafe { self.slots.get_unchecked_mut(idx as usize) };
            let key = K::from_raw(idx, slot.ver);
            slot.value = SlotValue::Occupied(value);
            key
        } else {
            let idx = self.slots.len() as u32;
            let ver = 0u32;
            self.slots.push(Slot {
                ver,
                value: SlotValue::Occupied(value),
            });
            K::from_raw(idx, ver)
        }
    }

    /// Insert with access to the key during construction.
    /// The closure receives the key that will be assigned, and must return the value.
    #[allow(dead_code)]
    pub fn insert_with_key(&mut self, f: impl FnOnce(K) -> V) -> K {
        self.len += 1;
        if let Some(idx) = self.take_vacant_slot() {
            let slot = &mut self.slots[idx as usize];
            let key = K::from_raw(idx, slot.ver);
            slot.value = SlotValue::Occupied(f(key));
            key
        } else {
            let idx = self.slots.len() as u32;
            let ver = 0u32;
            let key = K::from_raw(idx, ver);
            self.slots.push(Slot {
                ver,
                value: SlotValue::Occupied(f(key)),
            });
            key
        }
    }

    /// Remove the value at `key`. Returns `None` if the key is stale or invalid.
    #[inline]
    pub fn remove(&mut self, key: K) -> Option<V> {
        let idx = key.index() as usize;
        if idx >= self.slots.len() {
            return None;
        }
        let slot = &mut self.slots[idx];
        if slot.ver != key.slot_generation() {
            return None;
        }
        match &slot.value {
            SlotValue::Vacant { .. } => None,
            SlotValue::Occupied(_) => {
                // Bump generation so old keys are invalidated
                slot.ver = slot.ver.wrapping_add(1);
                let old = std::mem::replace(
                    &mut slot.value,
                    SlotValue::Vacant {
                        next_free: self.free_head,
                    },
                );
                self.free_head = idx as u32;
                self.len -= 1;
                match old {
                    SlotValue::Occupied(v) => Some(v),
                    SlotValue::Vacant { .. } => unreachable!(),
                }
            }
        }
    }

    /// Look up a value by key. Returns `None` if the key is stale or invalid.
    #[inline]
    pub fn get(&self, key: K) -> Option<&V> {
        let idx = key.index() as usize;
        if idx >= self.slots.len() {
            return None;
        }
        let slot = &self.slots[idx];
        if slot.ver != key.slot_generation() {
            return None;
        }
        match &slot.value {
            SlotValue::Occupied(v) => Some(v),
            SlotValue::Vacant { .. } => None,
        }
    }

    /// Mutable look up by key. Returns `None` if the key is stale or invalid.
    #[inline]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        let idx = key.index() as usize;
        if idx >= self.slots.len() {
            return None;
        }
        let slot = &mut self.slots[idx];
        if slot.ver != key.slot_generation() {
            return None;
        }
        match &mut slot.value {
            SlotValue::Occupied(v) => Some(v),
            SlotValue::Vacant { .. } => None,
        }
    }

    /// Mutable look up by key without bounds or generation checks.
    ///
    /// # Safety
    /// The key must refer to a valid, currently occupied slot.
    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, key: K) -> &mut V {
        let slot = unsafe { self.slots.get_unchecked_mut(key.index() as usize) };
        match &mut slot.value {
            SlotValue::Occupied(v) => v,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    /// Remove the value at `key` without bounds or generation checks.
    ///
    /// # Safety
    /// The key must refer to a valid, currently occupied slot.
    #[inline(always)]
    pub unsafe fn remove_unchecked(&mut self, key: K) -> V {
        let idx = key.index() as usize;
        let slot = unsafe { self.slots.get_unchecked_mut(idx) };
        slot.ver = slot.ver.wrapping_add(1);
        let old = std::mem::replace(
            &mut slot.value,
            SlotValue::Vacant {
                next_free: self.free_head,
            },
        );
        self.free_head = idx as u32;
        self.len -= 1;
        match old {
            SlotValue::Occupied(v) => v,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    /// Returns `true` if the key refers to a currently occupied slot.
    #[allow(dead_code)]
    pub fn contains_key(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// Iterate over all occupied `(key, &value)` pairs.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(&self.slots, &self.parked)
    }

    /// Iterate over all occupied `(key, &mut value)` pairs.
    #[allow(dead_code)]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut::new(&mut self.slots, &self.parked)
    }

    /// Iterate over all occupied values.
    #[allow(dead_code)]
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate over all occupied values mutably.
    #[allow(dead_code)]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }

    /// Remove all entries. Frees all slots.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = SENTINEL;
        self.len = 0;
        self.parked.clear();
        self.parked_slots = 0;
    }

    /// Drain all occupied entries, returning (key, value) pairs.
    /// Leaves the SlotMap empty.
    pub fn drain(&mut self) -> Vec<(K, V)> {
        let mut result = Vec::with_capacity(self.len);
        for (idx, slot) in self.slots.drain(..).enumerate() {
            if let SlotValue::Occupied(v) = slot.value {
                result.push((K::from_raw(idx as u32, slot.ver), v));
            }
        }
        self.free_head = SENTINEL;
        self.len = 0;
        self.parked.clear();
        self.parked_slots = 0;
        result
    }
}

impl<K: SlotKey, V> Default for SlotMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Iterators ---

/// Where the segment of slots before the next parked run (the last of
/// `runs`) ends.
#[inline]
fn segment_end(runs: &[(u32, u32)], slot_count: usize) -> usize {
    runs.last().map_or(slot_count, |&(start, _)| start as usize)
}

/// Immutable iterator over occupied `(key, &value)` pairs in a `SlotMap`.
///
/// It walks the slots a segment at a time - the slots between parked runs -
/// as a plain slice, and jumps over each run.
pub struct Iter<'a, K: SlotKey, V> {
    slots: &'a [Slot<V>],
    /// The current segment, and the index of its first slot.
    seg: std::iter::Enumerate<std::slice::Iter<'a, Slot<V>>>,
    base: usize,
    /// Parked runs not yet jumped over, lowest last.
    skip: &'a [(u32, u32)],
    _marker: PhantomData<K>,
}

impl<'a, K: SlotKey, V> Iter<'a, K, V> {
    fn new(slots: &'a [Slot<V>], parked: &'a [(u32, u32)]) -> Self {
        Iter {
            seg: slots[..segment_end(parked, slots.len())].iter().enumerate(),
            base: 0,
            slots,
            skip: parked,
            _marker: PhantomData,
        }
    }
}

impl<'a, K: SlotKey, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            for (i, slot) in self.seg.by_ref() {
                if let SlotValue::Occupied(ref v) = slot.value {
                    return Some((K::from_raw((self.base + i) as u32, slot.ver), v));
                }
            }
            // The segment is done: jump over the next parked run.
            let (&(_, end), rest) = self.skip.split_last()?;
            self.skip = rest;
            self.base = end as usize;
            self.seg = self.slots[self.base..segment_end(rest, self.slots.len())]
                .iter()
                .enumerate();
        }
    }
}

/// Mutable iterator over occupied `(key, &mut value)` pairs in a `SlotMap`,
/// walking the slots as [`Iter`] does.
#[allow(dead_code)]
pub struct IterMut<'a, K: SlotKey, V> {
    /// The slots after the current segment (the parked runs among them are
    /// jumped over).
    rest: &'a mut [Slot<V>],
    seg: std::iter::Enumerate<std::slice::IterMut<'a, Slot<V>>>,
    base: usize,
    /// Index of `rest`'s first slot.
    rest_base: usize,
    skip: &'a [(u32, u32)],
    _marker: PhantomData<K>,
}

impl<'a, K: SlotKey, V> IterMut<'a, K, V> {
    fn new(slots: &'a mut [Slot<V>], parked: &'a [(u32, u32)]) -> Self {
        let end = segment_end(parked, slots.len());
        let (seg, rest) = slots.split_at_mut(end);
        IterMut {
            rest,
            seg: seg.iter_mut().enumerate(),
            base: 0,
            rest_base: end,
            skip: parked,
            _marker: PhantomData,
        }
    }
}

impl<'a, K: SlotKey, V> Iterator for IterMut<'a, K, V> {
    type Item = (K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            for (i, slot) in self.seg.by_ref() {
                if let SlotValue::Occupied(ref mut v) = slot.value {
                    return Some((K::from_raw((self.base + i) as u32, slot.ver), v));
                }
            }
            // The segment is done: jump over the next parked run.
            let (&(start, end), rest_runs) = self.skip.split_last()?;
            self.skip = rest_runs;
            let rest = std::mem::take(&mut self.rest);
            let (_, after_run) = rest.split_at_mut(end as usize - self.rest_base);
            let seg_len = segment_end(rest_runs, end as usize + after_run.len()) - end as usize;
            let (seg, rest) = after_run.split_at_mut(seg_len);
            debug_assert!(start as usize >= self.rest_base);
            self.base = end as usize;
            self.rest_base = end as usize + seg_len;
            self.rest = rest;
            self.seg = seg.iter_mut().enumerate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut map = SlotMap::<ObjectId, i32>::new();
        let k1 = map.insert(10);
        let k2 = map.insert(20);
        assert_eq!(map.get(k1), Some(&10));
        assert_eq!(map.get(k2), Some(&20));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn remove_and_reuse() {
        let mut map = SlotMap::<ObjectId, i32>::new();
        let k1 = map.insert(10);
        let k2 = map.insert(20);
        assert_eq!(map.remove(k1), Some(10));
        assert_eq!(map.len(), 1);
        // k1 is now stale
        assert_eq!(map.get(k1), None);
        // Insert reuses the freed slot
        let k3 = map.insert(30);
        assert_eq!(k3.index(), k1.index());
        // But generation differs, so k1 still doesn't work
        assert_eq!(map.get(k1), None);
        assert_eq!(map.get(k3), Some(&30));
        assert_eq!(map.get(k2), Some(&20));
    }

    #[test]
    fn insert_with_key_works() {
        let mut map = SlotMap::<ObjectId, String>::new();
        let k = map.insert_with_key(|key| format!("id={}", key.index()));
        assert_eq!(map.get(k), Some(&"id=0".to_string()));
    }

    #[test]
    fn iter_yields_all_occupied() {
        let mut map = SlotMap::<ObjectId, i32>::new();
        let _k1 = map.insert(1);
        let k2 = map.insert(2);
        let _k3 = map.insert(3);
        map.remove(k2);
        let items: Vec<_> = map.iter().map(|(_, v)| *v).collect();
        assert_eq!(items.len(), 2);
        assert!(items.contains(&1));
        assert!(items.contains(&3));
    }

    #[test]
    fn drain_empties_map() {
        let mut map = SlotMap::<ObjectId, i32>::new();
        map.insert(1);
        map.insert(2);
        let drained: Vec<_> = map.drain();
        assert_eq!(drained.len(), 2);
        assert!(map.is_empty());
    }

    #[test]
    fn double_remove_returns_none() {
        let mut map = SlotMap::<ObjectId, i32>::new();
        let k = map.insert(42);
        assert_eq!(map.remove(k), Some(42));
        assert_eq!(map.remove(k), None);
    }

    #[test]
    fn clear_resets_everything() {
        let mut map = SlotMap::<ObjectId, i32>::new();
        map.insert(1);
        map.insert(2);
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn get_mut_modifies_value() {
        let mut map = SlotMap::<ObjectId, i32>::new();
        let k = map.insert(10);
        *map.get_mut(k).unwrap() = 99;
        assert_eq!(map.get(k), Some(&99));
    }

    #[test]
    fn parked_runs_are_skipped_and_reused_lowest_first() {
        let mut map = SlotMap::<ObjectId, u32>::new();
        let keys: Vec<ObjectId> = (0..5000).map(|i| map.insert(i)).collect();
        // Keep a few, scattered; free the rest in index order, so the free
        // list's head is the highest freed slot.
        let kept = [3usize, 700, 2500, 4999];
        for (i, &k) in keys.iter().enumerate() {
            if !kept.contains(&i) {
                map.remove(k);
            }
        }
        map.park_vacant_runs();
        map.check_invariants();
        assert_eq!(map.parked, vec![(2501, 4999), (701, 2500), (4, 700)]);
        // A pass jumps over the parked runs.
        let mut it = map.iter();
        assert_eq!(it.next().map(|(_, &v)| v), Some(3));
        assert_eq!(it.next().map(|(_, &v)| v), Some(700));
        assert_eq!((it.base, it.skip.len()), (700, 2));
        let got: Vec<u32> = map.values().copied().collect();
        assert_eq!(got, vec![3, 700, 2500, 4999]);
        assert_eq!(map.iter_mut().count(), 4);
        // Stale keys into parked slots stay invalid.
        assert!(map.get(keys[10]).is_none());
        assert!(map.remove(keys[10]).is_none());
        // New values reuse the lowest slots first.
        let a = map.insert(100);
        let b = map.insert(101);
        assert_eq!((a.index(), b.index()), (0, 1));
        assert_eq!(map.get(a), Some(&100));
        let got: Vec<u32> = map.values().copied().collect();
        assert_eq!(got, vec![100, 101, 3, 700, 2500, 4999]);
        // Once the free list (slot 2) runs out, the lowest run comes back.
        assert_eq!(map.insert(102).index(), 2);
        assert_eq!(map.insert(103).index(), 4);
        assert_eq!(map.insert(104).index(), 5);
        // The rest of the run stays parked.
        assert_eq!(map.parked.last(), Some(&(6, 700)));
        map.check_invariants();
        assert_eq!(map.slot_count(), 5000, "parking frees no slot");
    }

    #[test]
    fn slot_map_matches_a_model_under_random_parking() {
        use std::collections::BTreeMap;
        let mut map = SlotMap::<ObjectId, u64>::new();
        let mut model: BTreeMap<u32, (ObjectId, u64)> = BTreeMap::new();
        let mut dead: Vec<ObjectId> = Vec::new();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let rounds = if cfg!(miri) { 3_000 } else { 200_000 };
        for step in 0..rounds {
            match rand(100) {
                // Bursts of inserts and of removes make long vacant runs.
                0..=44 => {
                    let v = step as u64;
                    let k = map.insert(v);
                    assert!(
                        model.insert(k.index(), (k, v)).is_none(),
                        "a live slot was reused"
                    );
                }
                45..=89 if !model.is_empty() => {
                    let idx = *model.keys().nth(rand(model.len() as u64) as usize).unwrap();
                    let (k, v) = model.remove(&idx).unwrap();
                    assert_eq!(map.remove(k), Some(v));
                    dead.push(k);
                }
                90..=94 => map.park_vacant_runs(),
                _ => map.park_vacant_runs_now(),
            }
            if step % 997 == 0 {
                map.check_invariants();
                let got: Vec<(ObjectId, u64)> = map.iter().map(|(k, &v)| (k, v)).collect();
                let want: Vec<(ObjectId, u64)> = model.values().copied().collect();
                assert_eq!(got, want);
                assert_eq!(map.len(), model.len());
                for &k in dead.iter().rev().take(50) {
                    assert!(map.get(k).is_none(), "a stale key resolved");
                }
            }
        }
    }
}
