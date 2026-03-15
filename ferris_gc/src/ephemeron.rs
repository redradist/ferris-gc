//! Ephemeron tables — weak key→value associations.
//!
//! An [`EphemeronTable`] maps GC-managed keys to values. Entries are
//! automatically removed when their key is collected, similar to
//! `WeakMap` in JavaScript or `ConditionalWeakTable` in .NET.
//!
//! Unlike a `HashMap<Gc<K>, V>`, entries in an `EphemeronTable` do **not**
//! prevent their keys from being collected by the GC.

use crate::gc::{Gc, GcWeak, Trace};

/// A table mapping GC-managed keys to values via weak references.
///
/// When a key object is collected by the GC, its entry is automatically
/// removed on the next [`cleanup`](EphemeronTable::cleanup) call (which
/// the collector invokes during sweep).
///
/// # Examples
///
/// ```ignore
/// let mut table = EphemeronTable::new();
/// let key = Gc::new(42);
/// table.insert(&key, "hello");
/// assert_eq!(table.get(&key), Some(&"hello"));
/// drop(key);
/// // After GC collects the key:
/// table.cleanup();
/// assert_eq!(table.len(), 0);
/// ```
pub struct EphemeronTable<K: 'static + Sized + Trace, V> {
    entries: Vec<Option<(GcWeak<K>, V)>>,
}

impl<K: 'static + Sized + Trace, V> EphemeronTable<K, V> {
    /// Create an empty ephemeron table.
    pub fn new() -> Self {
        EphemeronTable {
            entries: Vec::new(),
        }
    }

    /// Insert a key-value pair. If the key already exists, the value is updated.
    pub fn insert(&mut self, key: &Gc<K>, value: V)
    where
        K: PartialEq,
    {
        // Check if key already exists
        let weak = Gc::downgrade(key);
        for entry in self.entries.iter_mut().flatten() {
            if let Some(existing_key) = entry.0.upgrade() {
                if **existing_key == ***key {
                    entry.1 = value;
                    return;
                }
            }
        }
        self.entries.push(Some((weak, value)));
    }

    /// Insert a key-value pair without checking for duplicates.
    /// Faster than `insert` when you know the key is not already present.
    pub fn insert_unique(&mut self, key: &Gc<K>, value: V) {
        let weak = Gc::downgrade(key);
        // Try to reuse a vacant slot
        for slot in &mut self.entries {
            if slot.is_none() {
                *slot = Some((weak, value));
                return;
            }
        }
        self.entries.push(Some((weak, value)));
    }

    /// Look up the value associated with a key.
    /// Returns `None` if the key is not in the table or has been collected.
    pub fn get(&self, key: &Gc<K>) -> Option<&V>
    where
        K: PartialEq,
    {
        for entry in self.entries.iter().flatten() {
            if let Some(existing_key) = entry.0.upgrade() {
                if **existing_key == ***key {
                    return Some(&entry.1);
                }
            }
        }
        None
    }

    /// Look up the value associated with a key, returning a mutable reference.
    pub fn get_mut(&mut self, key: &Gc<K>) -> Option<&mut V>
    where
        K: PartialEq,
    {
        for entry in self.entries.iter_mut().flatten() {
            if let Some(existing_key) = entry.0.upgrade() {
                if **existing_key == ***key {
                    return Some(&mut entry.1);
                }
            }
        }
        None
    }

    /// Remove the entry for a key. Returns the value if the key was present.
    pub fn remove(&mut self, key: &Gc<K>) -> Option<V>
    where
        K: PartialEq,
    {
        for slot in &mut self.entries {
            if let Some(entry) = slot {
                if let Some(existing_key) = entry.0.upgrade() {
                    if **existing_key == ***key {
                        return slot.take().map(|(_, v)| v);
                    }
                }
            }
        }
        None
    }

    /// Remove all entries whose keys have been collected.
    /// Called automatically during GC sweep, but can also be called manually.
    pub fn cleanup(&mut self) {
        for slot in &mut self.entries {
            if let Some(entry) = slot {
                if !entry.0.is_alive() {
                    *slot = None;
                }
            }
        }
    }

    /// Number of live entries (keys not yet collected).
    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    /// Returns `true` if there are no live entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all live (key, &value) pairs.
    /// Entries whose keys have been collected are skipped.
    pub fn iter(&self) -> impl Iterator<Item = (Gc<K>, &V)> {
        self.entries.iter().filter_map(|slot| {
            slot.as_ref()
                .and_then(|(weak, val)| weak.upgrade().map(|key| (key, val)))
        })
    }

    /// Iterate over all live values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().filter_map(|slot| {
            slot.as_ref()
                .and_then(|(weak, val)| if weak.is_alive() { Some(val) } else { None })
        })
    }

    /// Shrink internal storage by removing vacant slots.
    pub fn shrink(&mut self) {
        self.entries.retain(|slot| slot.is_some());
    }
}

impl<K: 'static + Sized + Trace, V> Default for EphemeronTable<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: 'static + Sized + Trace + std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug
    for EphemeronTable<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (weak, val) in self.entries.iter().flatten() {
            if let Some(key) = weak.upgrade() {
                map.entry(&format_args!("{:?}", **key), val);
            }
        }
        map.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::{Finalize, LOCAL_GC};

    struct TestVal(i32);

    impl Trace for TestVal {
        fn is_root(&self) -> bool {
            false
        }
        fn reset_root(&self) {}
        fn trace(&self) {}
        fn reset(&self) {}
        fn is_traceable(&self) -> bool {
            false
        }
    }
    impl Finalize for TestVal {
        fn finalize(&self) {}
    }
    impl PartialEq for TestVal {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    #[test]
    fn ephemeron_insert_and_get() {
        let key = Gc::new(TestVal(42));
        let mut table = EphemeronTable::new();
        table.insert(&key, "hello");
        assert_eq!(table.get(&key), Some(&"hello"));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn ephemeron_insert_updates_existing() {
        let key = Gc::new(TestVal(1));
        let mut table = EphemeronTable::new();
        table.insert(&key, 10);
        table.insert(&key, 20);
        assert_eq!(table.get(&key), Some(&20));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn ephemeron_remove() {
        let key = Gc::new(TestVal(1));
        let mut table = EphemeronTable::new();
        table.insert(&key, 100);
        assert_eq!(table.remove(&key), Some(100));
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn ephemeron_cleanup_removes_dead_keys() {
        let mut table = EphemeronTable::<TestVal, i32>::new();
        {
            let key = Gc::new(TestVal(99));
            table.insert_unique(&key, 42);
            assert_eq!(table.len(), 1);
        }
        // Key dropped, collect to free it
        LOCAL_GC.with(|gc| unsafe { gc.borrow().collect() });
        table.cleanup();
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn ephemeron_iter_skips_dead() {
        let key1 = Gc::new(TestVal(1));
        let mut table = EphemeronTable::new();
        table.insert_unique(&key1, "alive");
        {
            let key2 = Gc::new(TestVal(2));
            table.insert_unique(&key2, "will_die");
        }
        LOCAL_GC.with(|gc| unsafe { gc.borrow().collect() });
        let live: Vec<_> = table.values().collect();
        assert_eq!(live.len(), 1);
        assert_eq!(*live[0], "alive");
    }

    #[test]
    fn ephemeron_default_is_empty() {
        let table: EphemeronTable<TestVal, i32> = EphemeronTable::default();
        assert!(table.is_empty());
    }
}
