use std::marker::PhantomData;

/// Trait for slot-map key types with generation-counted u64 encoding.
/// Upper 32 bits = generation, lower 32 bits = index.
pub trait SlotKey: Copy + Eq {
    fn from_raw(index: u32, ver: u32) -> Self;
    fn index(self) -> u32;
    fn slot_generation(self) -> u32;

    #[allow(dead_code)]
    fn raw(self) -> u64 {
        ((self.slot_generation() as u64) << 32) | (self.index() as u64)
    }
}

/// Identifies an object slot in the GC's object arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ObjectId(u64);

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
}

const SENTINEL: u32 = u32::MAX;

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
pub struct SlotMap<K: SlotKey, V> {
    slots: Vec<Slot<V>>,
    free_head: u32,
    len: usize,
    _marker: PhantomData<K>,
}

impl<K: SlotKey, V> SlotMap<K, V> {
    pub fn new() -> Self {
        SlotMap {
            slots: Vec::new(),
            free_head: SENTINEL,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Number of occupied slots.
    pub fn len(&self) -> usize {
        self.len
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
        if self.free_head != SENTINEL {
            let idx = self.free_head;
            // SAFETY: free_head always points to a valid Vacant slot.
            let slot = unsafe { self.slots.get_unchecked_mut(idx as usize) };
            // SAFETY: Slots on the freelist are always Vacant.
            self.free_head = match slot.value {
                SlotValue::Vacant { next_free } => next_free,
                // SAFETY: free_head invariant guarantees this is Vacant.
                _ => unsafe { std::hint::unreachable_unchecked() },
            };
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
        if self.free_head != SENTINEL {
            let idx = self.free_head;
            let slot = &mut self.slots[idx as usize];
            match slot.value {
                SlotValue::Vacant { next_free } => {
                    self.free_head = next_free;
                }
                SlotValue::Occupied(_) => unreachable!("free slot is occupied"),
            }
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
        Iter {
            slots: &self.slots,
            idx: 0,
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// Iterate over all occupied `(key, &mut value)` pairs.
    #[allow(dead_code)]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            slots: self.slots.as_mut_slice(),
            idx: 0,
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// Iterate over all occupied values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.slots.iter().filter_map(|slot| match &slot.value {
            SlotValue::Occupied(v) => Some(v),
            SlotValue::Vacant { .. } => None,
        })
    }

    /// Iterate over all occupied values mutably.
    #[allow(dead_code)]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.slots
            .iter_mut()
            .filter_map(|slot| match &mut slot.value {
                SlotValue::Occupied(v) => Some(v),
                SlotValue::Vacant { .. } => None,
            })
    }

    /// Remove all entries. Frees all slots.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = SENTINEL;
        self.len = 0;
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
        result
    }
}

impl<K: SlotKey, V> Default for SlotMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Iterators ---

/// Immutable iterator over occupied `(key, &value)` pairs in a `SlotMap`.
pub struct Iter<'a, K: SlotKey, V> {
    slots: &'a [Slot<V>],
    idx: usize,
    remaining: usize,
    _marker: PhantomData<K>,
}

impl<'a, K: SlotKey, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 && self.idx < self.slots.len() {
            let slot = &self.slots[self.idx];
            let idx = self.idx;
            self.idx += 1;
            if let SlotValue::Occupied(ref v) = slot.value {
                self.remaining -= 1;
                return Some((K::from_raw(idx as u32, slot.ver), v));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

/// Mutable iterator over occupied `(key, &mut value)` pairs in a `SlotMap`.
#[allow(dead_code)]
pub struct IterMut<'a, K: SlotKey, V> {
    slots: &'a mut [Slot<V>],
    idx: usize,
    remaining: usize,
    _marker: PhantomData<K>,
}

impl<'a, K: SlotKey, V> Iterator for IterMut<'a, K, V> {
    type Item = (K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 && self.idx < self.slots.len() {
            let idx = self.idx;
            self.idx += 1;
            let slot = &mut self.slots[idx];
            if let SlotValue::Occupied(ref mut v) = slot.value {
                self.remaining -= 1;
                // SAFETY: We only hand out one &mut per slot, and we advance idx each time
                let v_ref = unsafe { &mut *(v as *mut V) };
                return Some((K::from_raw(idx as u32, slot.ver), v_ref));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
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
}
