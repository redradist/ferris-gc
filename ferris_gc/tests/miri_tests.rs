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

use ferris_gc::{EphemeronTable, Finalize, Gc, GcCell, LOCAL_GC, Trace};

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

    // Modify through borrow_mut
    {
        let mut guard = cell.borrow_mut();
        guard.t = TestVal(99);
    }

    // Read back
    let val = cell.borrow();
    assert_eq!(val.t.0, 99);

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
