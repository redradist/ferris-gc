//! Sparse card table for generational write barriers.
//!
//! Each "card" covers a `CARD_SIZE`-byte aligned address range. When a GC-managed
//! object is mutated, the write barrier computes the card index from the object
//! address (`addr >> CARD_SHIFT`) and marks the card dirty.
//!
//! During collection the GC iterates dirty cards and resolves them back to
//! [`ObjectId`]s via a reverse map maintained during allocation/deallocation.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::slot_map::ObjectId;

/// Number of low-order address bits covered by a single card.
/// Each card covers `1 << CARD_SHIFT` = 512 bytes.
const CARD_SHIFT: usize = 9;

/// Byte value for a clean card.
const CARD_CLEAN: u8 = 0;
/// Byte value for a dirty card.
const CARD_DIRTY: u8 = 1;

/// Compute the card index for a given raw address.
#[inline]
fn card_index(addr: usize) -> usize {
    addr >> CARD_SHIFT
}

/// A sparse card table for tracking generational write barrier dirty state.
///
/// # Design
///
/// The card table replaces the old `Mutex<HashSet<ObjectId>>` remembered set.
/// It maps object addresses to dirty flags, removing the need to acquire the
/// `gc_maps` Mutex in the write barrier hot path.
///
/// **Old write barrier** (2 Mutex acquisitions + HashMap lookup + HashSet insert):
/// ```text
/// lock gc_maps -> lookup ptr_to_object -> check generation -> lock remembered_set -> HashSet::insert
/// ```
///
/// **New write barrier** (1 Mutex acquisition + HashMap insert):
/// ```text
/// lock card_table -> compute card index -> set dirty flag
/// ```
///
/// During collection (where we already hold `gc_maps`), we resolve dirty cards
/// back to `ObjectId`s via the `card_objects` reverse map.
///
/// Because system-allocated objects are scattered across address space, a flat
/// array would be enormous. Instead we use a hash map keyed by card index
/// (sparse representation). A second map (`card_objects`) records which
/// [`ObjectId`]s reside in each card so that the collector can resolve dirty
/// cards back to objects quickly.
pub(crate) struct CardTable {
    /// Card index -> dirty flag.
    cards: Mutex<HashMap<usize, u8>>,
    /// Card index -> set of ObjectIds whose address falls in that card.
    /// Updated during allocation/deallocation, NOT in the write barrier.
    card_objects: Mutex<HashMap<usize, Vec<ObjectId>>>,
}

impl CardTable {
    /// Create a new empty card table.
    pub(crate) fn new() -> Self {
        CardTable {
            cards: Mutex::new(HashMap::new()),
            card_objects: Mutex::new(HashMap::new()),
        }
    }

    /// Register an object's address with its card.
    /// Called during allocation (not on the hot path).
    pub(crate) fn register_object(&self, thin_ptr: usize, obj_id: ObjectId) {
        let cidx = card_index(thin_ptr);
        let mut co = self.card_objects.lock().unwrap_or_else(|e| e.into_inner());
        co.entry(cidx).or_default().push(obj_id);
    }

    /// Unregister an object from its card.
    /// Called during deallocation / sweep.
    pub(crate) fn unregister_object(&self, thin_ptr: usize, obj_id: ObjectId) {
        let cidx = card_index(thin_ptr);
        let mut co = self.card_objects.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ids) = co.get_mut(&cidx) {
            ids.retain(|id| *id != obj_id);
            if ids.is_empty() {
                co.remove(&cidx);
            }
        }
    }

    /// Mark the card containing `addr` as dirty.
    /// This is the write-barrier hot path — only one Mutex acquisition.
    #[inline]
    pub(crate) fn mark_dirty(&self, addr: usize) {
        let cidx = card_index(addr);
        let mut cards = self.cards.lock().unwrap_or_else(|e| e.into_inner());
        cards.insert(cidx, CARD_DIRTY);
    }

    /// Collect all [`ObjectId`]s from dirty cards, then clean those cards.
    /// Called during collection while the caller already holds `gc_maps`.
    pub(crate) fn drain_dirty_objects(&self) -> Vec<ObjectId> {
        let mut cards = self.cards.lock().unwrap_or_else(|e| e.into_inner());
        let co = self.card_objects.lock().unwrap_or_else(|e| e.into_inner());

        let mut result = Vec::new();
        for (&cidx, flag) in cards.iter() {
            if *flag == CARD_DIRTY {
                if let Some(ids) = co.get(&cidx) {
                    result.extend_from_slice(ids);
                }
            }
        }
        // Clean all cards
        for flag in cards.values_mut() {
            *flag = CARD_CLEAN;
        }
        result
    }

    /// Return all [`ObjectId`]s from dirty cards WITHOUT clearing the dirty flags.
    /// Used for partial (generational) collections that should not clear the
    /// remembered set.
    pub(crate) fn dirty_objects(&self) -> Vec<ObjectId> {
        let cards = self.cards.lock().unwrap_or_else(|e| e.into_inner());
        let co = self.card_objects.lock().unwrap_or_else(|e| e.into_inner());

        let mut result = Vec::new();
        for (&cidx, flag) in cards.iter() {
            if *flag == CARD_DIRTY {
                if let Some(ids) = co.get(&cidx) {
                    result.extend_from_slice(ids);
                }
            }
        }
        result
    }

    /// Remove a specific object from the card table (e.g., when it's collected).
    /// Also cleans the card's dirty flag if no objects remain in that card.
    pub(crate) fn remove_object(&self, thin_ptr: usize, obj_id: ObjectId) {
        self.unregister_object(thin_ptr, obj_id);
        // Also clean the card if no objects remain in it.
        let cidx = card_index(thin_ptr);
        let co = self.card_objects.lock().unwrap_or_else(|e| e.into_inner());
        if !co.contains_key(&cidx) {
            let mut cards = self.cards.lock().unwrap_or_else(|e| e.into_inner());
            cards.remove(&cidx);
        }
    }

    /// Clear all dirty flags (but keep object registrations).
    /// Used on full collection (Gen2).
    pub(crate) fn clear_dirty(&self) {
        let mut cards = self.cards.lock().unwrap_or_else(|e| e.into_inner());
        cards.clear();
    }

    /// Returns true if there are any dirty cards.
    pub(crate) fn has_dirty(&self) -> bool {
        let cards = self.cards.lock().unwrap_or_else(|e| e.into_inner());
        cards.values().any(|&f| f == CARD_DIRTY)
    }

    /// Returns true if there are no dirty cards.
    pub(crate) fn is_clean(&self) -> bool {
        !self.has_dirty()
    }

    /// Clear all state including object registrations.
    /// Used during collect_all / full cleanup.
    pub(crate) fn clear_all(&self) {
        self.cards.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.card_objects
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot_map::{ObjectId, SlotKey};

    fn make_obj_id(idx: u32) -> ObjectId {
        ObjectId::from_raw(idx, 0)
    }

    #[test]
    fn mark_dirty_and_drain() {
        let ct = CardTable::new();
        let addr1: usize = 0x1000;
        let addr2: usize = 0x2000;
        let id1 = make_obj_id(0);
        let id2 = make_obj_id(1);

        ct.register_object(addr1, id1);
        ct.register_object(addr2, id2);

        // Nothing dirty yet
        assert!(ct.is_clean());
        assert!(ct.drain_dirty_objects().is_empty());

        // Mark addr1 dirty
        ct.mark_dirty(addr1);
        assert!(ct.has_dirty());

        let dirty = ct.drain_dirty_objects();
        assert_eq!(dirty.len(), 1);
        assert!(dirty.contains(&id1));

        // After drain, should be clean
        assert!(ct.is_clean());
    }

    #[test]
    fn multiple_objects_same_card() {
        let ct = CardTable::new();
        // Two addresses in the same 512-byte card
        let addr1: usize = 0x1000;
        let addr2: usize = 0x1100; // same card (0x1000 >> 9 == 0x1100 >> 9 == 8)
        let id1 = make_obj_id(0);
        let id2 = make_obj_id(1);

        ct.register_object(addr1, id1);
        ct.register_object(addr2, id2);

        ct.mark_dirty(addr1);
        let dirty = ct.dirty_objects();
        assert_eq!(dirty.len(), 2); // Both objects in the dirty card
    }

    #[test]
    fn remove_object_cleans_up() {
        let ct = CardTable::new();
        let addr: usize = 0x3000;
        let id = make_obj_id(5);

        ct.register_object(addr, id);
        ct.mark_dirty(addr);
        assert!(ct.has_dirty());

        ct.remove_object(addr, id);
        // Card should be removed since no objects remain
        assert!(ct.is_clean());
    }

    #[test]
    fn clear_dirty_resets_flags() {
        let ct = CardTable::new();
        let addr: usize = 0x4000;
        let id = make_obj_id(0);

        ct.register_object(addr, id);
        ct.mark_dirty(addr);
        assert!(ct.has_dirty());

        ct.clear_dirty();
        assert!(ct.is_clean());
    }

    #[test]
    fn clear_all_removes_everything() {
        let ct = CardTable::new();
        let addr: usize = 0x5000;
        let id = make_obj_id(0);

        ct.register_object(addr, id);
        ct.mark_dirty(addr);

        ct.clear_all();
        assert!(ct.is_clean());
        assert!(ct.drain_dirty_objects().is_empty());
    }

    #[test]
    fn dirty_objects_does_not_clear() {
        let ct = CardTable::new();
        let addr: usize = 0x6000;
        let id = make_obj_id(0);

        ct.register_object(addr, id);
        ct.mark_dirty(addr);

        // dirty_objects should return the object but NOT clear
        let d1 = ct.dirty_objects();
        assert_eq!(d1.len(), 1);
        let d2 = ct.dirty_objects();
        assert_eq!(d2.len(), 1); // still dirty
    }
}
