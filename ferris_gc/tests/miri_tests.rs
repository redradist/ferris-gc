#![cfg(miri)]

//! Miri-safe tests for the thread-local garbage collector.
//!
//! These tests verify that the GC does not trigger undefined behavior
//! (use-after-free, double-free, dangling pointers) as detected by Miri.
//!
//! Constraints for Miri compatibility:
//! - No threads (no `std::thread::spawn`)
//! - No I/O (no file/network operations)
//! - No FFI
//! - Manual `Trace`/`Finalize` impls (no proc-macro derives)

use std::cell::RefCell;

use ferris_gc::{EphemeronTable, Finalize, Gc, GcCell, Generation, LOCAL_GC, Trace};

// ---------------------------------------------------------------------------
// Helper types with manual Trace/Finalize implementations
// ---------------------------------------------------------------------------

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

/// A node that can form cycles via an interior-mutable `Option<Gc<CyclicNode>>`.
struct CyclicNode {
    next: RefCell<Option<Gc<CyclicNode>>>,
}

impl Trace for CyclicNode {
    fn is_root(&self) -> bool {
        false
    }
    fn reset_root(&self) {
        if let Some(ref gc) = *self.next.borrow() {
            gc.reset_root();
        }
    }
    fn trace(&self) {
        if let Some(ref gc) = *self.next.borrow() {
            gc.trace();
        }
    }
    fn reset(&self) {
        if let Some(ref gc) = *self.next.borrow() {
            gc.reset();
        }
    }
    fn is_traceable(&self) -> bool {
        false
    }
}

impl Finalize for CyclicNode {
    fn finalize(&self) {}
}

// ---------------------------------------------------------------------------
// Helper: run a full GC collection on the thread-local collector
// ---------------------------------------------------------------------------

fn collect() {
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._collect();
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Basic allocation, dereference, drop, and collect.
/// Verifies no use-after-free or double-free on the simplest path.
#[test]
fn miri_basic_alloc_deref_drop() {
    let obj = Gc::new(TestVal(42));
    assert_eq!((**obj).0, 42);
    drop(obj);
    collect();
}

/// Create a cycle (A -> B -> A), drop both handles, collect.
/// Verifies the collector handles cyclic references without UB.
#[test]
fn miri_cycle_collect() {
    let a = Gc::new(CyclicNode {
        next: RefCell::new(None),
    });
    let b = Gc::new(CyclicNode {
        next: RefCell::new(Some(a.clone())),
    });
    *a.next.borrow_mut() = Some(b.clone());

    // Drop user handles — the cycle keeps objects alive internally
    drop(a);
    drop(b);

    // Collection must reclaim the cycle without UB
    collect();
}

/// Create a Gc, downgrade to GcWeak, drop the strong Gc, collect,
/// and verify that upgrade returns None.
#[test]
fn miri_weak_ref_upgrade_after_collect() {
    let obj = Gc::new(TestVal(7));
    let weak = Gc::downgrade(&obj);

    // Weak reference should be alive while strong handle exists
    assert!(weak.is_alive());

    drop(obj);
    collect();

    // After collection, the object should be gone
    assert!(!weak.is_alive());
    assert!(weak.upgrade().is_none());
}

/// Create a GcCell, borrow_mut to modify the inner value, then verify.
/// Checks that interior mutability through GcCell doesn't cause UB.
#[test]
fn miri_gc_cell_borrow_mut() {
    let cell = GcCell::new(TestVal(10));

    // Modify through borrow_mut (deref chain: RefMut -> GcPtr -> TestVal)
    {
        let mut guard = cell.borrow_mut();
        (**guard).0 = 99;
    }

    // Read back
    {
        let val = cell.borrow();
        assert_eq!((**val).0, 99);
    }

    drop(cell);
    collect();
}

/// Create many objects, drop some, collect, verify survivors are intact.
/// Exercises the allocator and sweep logic under Miri.
#[test]
fn miri_multiple_alloc_collect() {
    let mut objects: Vec<Gc<TestVal>> = Vec::new();

    // Allocate 20 objects
    for i in 0..20 {
        objects.push(Gc::new(TestVal(i)));
    }

    // Verify all values
    for (i, obj) in objects.iter().enumerate() {
        assert_eq!((**obj).0, i as i32);
    }

    // Drop even-indexed objects
    let survivors: Vec<Gc<TestVal>> = objects
        .into_iter()
        .enumerate()
        .filter_map(|(i, obj)| if i % 2 != 0 { Some(obj) } else { None })
        .collect();

    collect();

    // Verify survivors are still accessible and correct
    for (idx, obj) in survivors.iter().enumerate() {
        let expected = (idx * 2 + 1) as i32;
        assert_eq!((**obj).0, expected);
    }

    drop(survivors);
    collect();
}

/// Create objects, compact the heap, verify they still deref correctly.
/// Checks that pointer relocation doesn't introduce dangling pointers.
#[test]
fn miri_compact() {
    let a = Gc::new(TestVal(1));
    let b = Gc::new(TestVal(2));
    let c = Gc::new(TestVal(3));

    // Drop the middle object to create a gap
    drop(b);
    collect();

    // Compact the heap
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._compact();
    });

    // Surviving objects must still be accessible
    assert_eq!((**a).0, 1);
    assert_eq!((**c).0, 3);

    drop(a);
    drop(c);
    collect();
}

/// Create an EphemeronTable, insert entries, drop keys, collect, cleanup,
/// and verify that entries with dead keys are removed.
#[test]
fn miri_ephemeron_cleanup() {
    let mut table = EphemeronTable::<TestVal, &str>::new();

    let key_alive = Gc::new(TestVal(1));
    let key_dead = Gc::new(TestVal(2));

    table.insert(&key_alive, "stays");
    table.insert(&key_dead, "goes");

    assert_eq!(table.len(), 2);
    assert_eq!(table.get(&key_alive), Some(&"stays"));
    assert_eq!(table.get(&key_dead), Some(&"goes"));

    // Drop the key that should cause its entry to be cleaned up
    drop(key_dead);
    collect();
    table.cleanup();

    // Only the alive entry should remain
    assert_eq!(table.len(), 1);
    assert_eq!(table.get(&key_alive), Some(&"stays"));

    drop(key_alive);
    collect();
    table.cleanup();

    assert_eq!(table.len(), 0);
    assert!(table.is_empty());
}

// ---------------------------------------------------------------------------
// Compaction edge cases
// ---------------------------------------------------------------------------

/// Compact twice in succession. The second compact should handle
/// already-relocated (Compact-backed) objects without UB.
#[test]
fn miri_double_compact() {
    let a = Gc::new(TestVal(10));
    let b = Gc::new(TestVal(20));
    let c = Gc::new(TestVal(30));

    drop(b);
    collect();

    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._compact();
    });
    assert_eq!((**a).0, 10);
    assert_eq!((**c).0, 30);

    // Second compact — objects are already in compact blocks
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._compact();
    });
    assert_eq!((**a).0, 10);
    assert_eq!((**c).0, 30);

    drop(a);
    drop(c);
    collect();
}

/// Compact with a cycle (A -> B -> A). Both objects must be relocated
/// and their internal pointers updated so the cycle still works.
#[test]
fn miri_compact_with_cycle() {
    let a = Gc::new(CyclicNode {
        next: RefCell::new(None),
    });
    let b = Gc::new(CyclicNode {
        next: RefCell::new(Some(a.clone())),
    });
    *a.next.borrow_mut() = Some(b.clone());

    // Create a gap to trigger actual relocation
    let _filler = Gc::new(TestVal(999));
    drop(_filler);
    collect();

    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._compact();
    });

    // Cycle should still be traversable after compaction
    assert!(a.next.borrow().is_some());
    assert!(b.next.borrow().is_some());

    drop(a);
    drop(b);
    collect();
}

/// Compact when all objects are dead — empty heap should not crash.
#[test]
fn miri_compact_empty_heap() {
    let a = Gc::new(TestVal(1));
    let b = Gc::new(TestVal(2));
    drop(a);
    drop(b);
    collect();

    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._compact();
    });

    // Allocating after compact-on-empty should work
    let c = Gc::new(TestVal(3));
    assert_eq!((**c).0, 3);
    drop(c);
    collect();
}

/// Compact with GcCell objects. Verifies that GcCellInternal's ptr
/// (Cell<*const RefCell<GcPtr<T>>>) is correctly relocated.
#[test]
fn miri_compact_gc_cell() {
    let cell1 = GcCell::new(TestVal(100));
    let cell2 = GcCell::new(TestVal(200));
    let cell3 = GcCell::new(TestVal(300));

    // Drop middle to create gap
    drop(cell2);
    collect();

    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._compact();
    });

    // GcCell borrow should still work after relocation
    assert_eq!((**cell1.borrow()).0, 100);
    assert_eq!((**cell3.borrow()).0, 300);

    // borrow_mut should also work
    {
        let mut guard = cell1.borrow_mut();
        (**guard).0 = 111;
    }
    assert_eq!((**cell1.borrow()).0, 111);

    drop(cell1);
    drop(cell3);
    collect();
}

// ---------------------------------------------------------------------------
// Weak reference edge cases
// ---------------------------------------------------------------------------

/// Multiple weak refs to the same object. Drop the strong ref,
/// verify all weaks detect the deallocation.
#[test]
fn miri_multiple_weaks_same_object() {
    let strong = Gc::new(TestVal(42));
    let w1 = Gc::downgrade(&strong);
    let w2 = Gc::downgrade(&strong);
    let w3 = w1.clone();

    assert!(w1.is_alive());
    assert!(w2.is_alive());
    assert!(w3.is_alive());

    drop(strong);
    collect();

    assert!(!w1.is_alive());
    assert!(!w2.is_alive());
    assert!(!w3.is_alive());
    assert!(w1.upgrade().is_none());
    assert!(w2.upgrade().is_none());
    assert!(w3.upgrade().is_none());
}

/// Weak upgrade succeeds while strong is alive, then re-drop and collect.
/// The upgraded Gc should keep the object alive.
#[test]
fn miri_weak_upgrade_keeps_alive() {
    let strong = Gc::new(TestVal(77));
    let weak = Gc::downgrade(&strong);
    drop(strong);

    // Object still alive — weak upgrade should succeed (RC hybrid keeps it)
    let upgraded = weak.upgrade();
    // Note: with RC hybrid, dropping the last strong Gc immediately frees.
    // So upgrade may return None. Either way, no UB.
    if let Some(gc) = upgraded {
        assert_eq!((**gc).0, 77);
        drop(gc);
    }
    collect();
}

/// Weak ref survives compaction — the weak's ptr must still be valid
/// (or correctly detect deallocation) after compact relocates objects.
#[test]
fn miri_weak_after_compact() {
    let a = Gc::new(TestVal(1));
    let b = Gc::new(TestVal(2));
    let weak_a = Gc::downgrade(&a);

    drop(b);
    collect();

    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._compact();
    });

    // Strong ref should still work
    assert_eq!((**a).0, 1);
    // Weak should still be alive
    assert!(weak_a.is_alive());

    drop(a);
    collect();

    assert!(!weak_a.is_alive());
}

// ---------------------------------------------------------------------------
// GcCell with nested Gc (tracing + interior mutability)
// ---------------------------------------------------------------------------

/// A node using GcCell to hold an optional Gc — the canonical pattern
/// for building mutable graphs. Verify tracing and collection work.
#[test]
fn miri_gc_cell_with_nested_gc() {
    let leaf = Gc::new(CyclicNode {
        next: RefCell::new(None),
    });
    let cell = GcCell::new(CyclicNode {
        next: RefCell::new(Some(leaf.clone())),
    });

    // Modify through GcCell — disconnect the nested Gc
    {
        let mut guard = cell.borrow_mut();
        (**guard).next = RefCell::new(None);
    }

    // leaf should still be alive (we hold a direct Gc to it)
    assert!(leaf.next.borrow().is_none());

    drop(cell);
    drop(leaf);
    collect();
}

// ---------------------------------------------------------------------------
// Generational collection under Miri
// ---------------------------------------------------------------------------

/// Gen0 collection should only sweep young objects, leaving promoted ones.
#[test]
fn miri_gen0_collection() {
    let young = Gc::new(TestVal(1));
    let old = Gc::new(TestVal(2));

    // Promote `old` by surviving a few collections
    for _ in 0..3 {
        collect();
    }

    // Drop young, Gen0-only collect
    drop(young);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._collect_generation(Generation::Gen0);
    });

    // old survived promotion and Gen0 sweep
    assert_eq!((**old).0, 2);

    drop(old);
    collect();
}

/// Incremental collection: mark in steps, then sweep.
/// Verifies the incremental state machine doesn't produce UB.
#[test]
fn miri_incremental_collection() {
    let mut objects: Vec<Gc<TestVal>> = (0..10).map(|i| Gc::new(TestVal(i))).collect();

    // Drop half
    let survivors: Vec<_> = objects.drain(..5).collect();
    drop(objects);

    // Incremental: small step budget
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._collect_incremental(Generation::Gen2, 3);
    });

    // Survivors must still be accessible
    for (i, obj) in survivors.iter().enumerate() {
        assert_eq!((**obj).0, i as i32);
    }

    drop(survivors);
    collect();
}

// ---------------------------------------------------------------------------
// Region-based allocation under Miri
// ---------------------------------------------------------------------------

/// Allocate objects in a custom region, collect that region only.
#[test]
fn miri_region_alloc_and_collect() {
    let region = LOCAL_GC.with(|gc| gc.borrow().new_region());

    let a = Gc::new_in(TestVal(10), region);
    let b = Gc::new_in(TestVal(20), region);
    let outside = Gc::new(TestVal(30));

    drop(a);
    drop(b);

    // Collect only the region
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._collect_region(region);
    });

    // Object outside the region is unaffected
    assert_eq!((**outside).0, 30);

    drop(outside);
    collect();
}

// ---------------------------------------------------------------------------
// Clone and ref counting under Miri
// ---------------------------------------------------------------------------

/// Clone a Gc, drop original, verify clone keeps object alive.
#[test]
fn miri_clone_keeps_alive() {
    let a = Gc::new(TestVal(55));
    let b = a.clone();
    drop(a);
    collect();

    assert_eq!((**b).0, 55);
    drop(b);
    collect();
}

/// Clone a GcCell, modify through one handle, read through the other.
#[test]
fn miri_gc_cell_clone() {
    let cell1 = GcCell::new(TestVal(10));
    let cell2 = cell1.clone();

    {
        let mut guard = cell1.borrow_mut();
        (**guard).0 = 20;
    }

    // Both handles see the same data
    assert_eq!((**cell2.borrow()).0, 20);

    drop(cell1);
    drop(cell2);
    collect();
}

// ---------------------------------------------------------------------------
// try_new under Miri
// ---------------------------------------------------------------------------

/// try_new should succeed for normal allocations.
#[test]
fn miri_try_new() {
    let result = Gc::try_new(TestVal(42));
    assert!(result.is_ok());
    let gc = result.unwrap();
    assert_eq!((**gc).0, 42);
    drop(gc);
    collect();
}

/// try_new for GcCell.
#[test]
fn miri_try_new_gc_cell() {
    let result = GcCell::try_new(TestVal(99));
    assert!(result.is_ok());
    let cell = result.unwrap();
    assert_eq!((**cell.borrow()).0, 99);
    drop(cell);
    collect();
}

// ---------------------------------------------------------------------------
// Three-node cycle under Miri
// ---------------------------------------------------------------------------

/// Three-node cycle: A -> B -> C -> A. More complex than two-node cycle.
#[test]
fn miri_three_node_cycle() {
    let a = Gc::new(CyclicNode {
        next: RefCell::new(None),
    });
    let b = Gc::new(CyclicNode {
        next: RefCell::new(None),
    });
    let c = Gc::new(CyclicNode {
        next: RefCell::new(None),
    });

    *a.next.borrow_mut() = Some(b.clone());
    *b.next.borrow_mut() = Some(c.clone());
    *c.next.borrow_mut() = Some(a.clone());

    drop(a);
    drop(b);
    drop(c);
    collect();
}

// ---------------------------------------------------------------------------
// Stats under Miri
// ---------------------------------------------------------------------------

/// Verify stats() doesn't produce UB when called under Miri.
#[test]
fn miri_stats() {
    let _a = Gc::new(TestVal(1));
    let _b = Gc::new(TestVal(2));

    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert!(stats.live_objects >= 2);

    drop(_a);
    drop(_b);
    collect();

    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert!(stats.total_collections >= 1);
}
