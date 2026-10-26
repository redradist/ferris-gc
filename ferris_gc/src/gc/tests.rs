use crate::gc::{Gc, GcCell, LOCAL_GC};
use crate::sharded_map::Counter;
use crate::{Finalize, Trace};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Clean residual state from previous tests that may have run on this thread.
fn clean_gc_state() {
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
}

#[test]
fn object_entry_size() {
    // Track ObjectEntry size to catch regressions: 64 bytes since the
    // per-handle tracer list was removed (sync handles count themselves
    // in the object header, thread-local ones in handle_count).
    let size = std::mem::size_of::<super::ObjectEntry>();
    assert!(size <= 64, "ObjectEntry grew unexpectedly to {size}B");
}

#[test]
fn size_gc_is_compact() {
    assert_eq!(std::mem::size_of::<Gc<i32>>(), 8);
    assert_eq!(std::mem::size_of::<GcCell<i32>>(), 8);
    // Option adds discriminant (Cell kills NonZeroUsize niche)
    assert_eq!(std::mem::size_of::<Option<Gc<i32>>>(), 16);
}

#[test]
fn one_object() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().total_handles());
    let _one = Gc::new(1);
    LOCAL_GC.with(move |gc| unsafe {
        gc.borrow_mut().collect();
        assert_eq!(
            gc.borrow().core.lock_gc_maps().total_handles() - baseline,
            1
        );
    });
}

#[test]
fn gc_collect_one_from_one() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().total_handles());
    {
        let _one = Gc::new(1);
    }
    LOCAL_GC.with(move |gc| unsafe {
        gc.borrow_mut().collect();
        assert_eq!(
            gc.borrow().core.lock_gc_maps().total_handles() - baseline,
            0
        );
    });
}

#[test]
#[allow(unused_assignments)]
fn two_objects_reassign() {
    // Reassigning drops the old Gc (drop_handle decrements handle_count),
    // so only 1 handle remains for the surviving Gc.
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().total_handles());
    let mut one = Gc::new(1);
    one = Gc::new(2);
    LOCAL_GC.with(move |gc| {
        assert_eq!(
            gc.borrow().core.lock_gc_maps().total_handles() - baseline,
            1
        );
    });
    drop(one);
}

#[test]
#[allow(unused_assignments)]
fn gc_collect_after_reassign() {
    // After reassign, one live Gc remains. collect() keeps live objects.
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().total_handles());
    let mut one = Gc::new(1);
    one = Gc::new(2);
    LOCAL_GC.with(move |gc| unsafe {
        gc.borrow_mut().collect();
        assert_eq!(
            gc.borrow().core.lock_gc_maps().total_handles() - baseline,
            1
        );
    });
    drop(one);
}

#[test]
#[allow(unused_assignments)]
fn gc_collect_two_from_two() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().total_handles());
    {
        let mut one = Gc::new(1);
        one = Gc::new(2);
        drop(one);
    }
    LOCAL_GC.with(move |gc| unsafe {
        gc.borrow_mut().collect();
        assert_eq!(
            gc.borrow().core.lock_gc_maps().total_handles() - baseline,
            0
        );
    });
}

#[test]
fn ptr_to_object_cleaned_on_collect() {
    clean_gc_state();
    let baseline_handles = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().total_handles());
    let baseline_p2o = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().ptr_to_object.len());
    {
        let _one = Gc::new(1);
    }
    LOCAL_GC.with(move |gc| unsafe {
        gc.borrow_mut().collect();
        assert_eq!(
            gc.borrow().core.lock_gc_maps().total_handles() - baseline_handles,
            0
        );
        assert_eq!(
            gc.borrow().core.lock_gc_maps().ptr_to_object.len() - baseline_p2o,
            0
        );
    });
}

static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

struct DropCounter {
    _value: String,
}

impl Trace for DropCounter {
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

impl Finalize for DropCounter {
    fn finalize(&self) {}
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn collect_calls_drop_on_gc_objects() {
    clean_gc_state();
    DROP_COUNT.store(0, Ordering::SeqCst);
    {
        let _obj = Gc::new(DropCounter {
            _value: String::from("hello"),
        });
    }
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    assert_eq!(
        DROP_COUNT.load(Ordering::SeqCst),
        1,
        "Drop should be called during collect"
    );
}

#[test]
fn clone_from_registers_with_gc() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().total_handles());
    let source = Gc::new(42);
    let mut target = Gc::new(99);
    target.clone_from(&source);
    LOCAL_GC.with(|gc| {
        let delta = gc.borrow().core.lock_gc_maps().total_handles() - baseline;
        // source + target + the new clone = at least 2 handles alive
        assert!(
            delta >= 2,
            "clone_from should register a new handle with GC, got {delta}"
        );
    });
}

struct CyclicNode {
    next: std::cell::RefCell<Option<Gc<CyclicNode>>>,
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
    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        if let Some(ref gc) = *self.next.borrow() {
            gc.trace_children(children);
        }
    }
}

impl Finalize for CyclicNode {
    fn finalize(&self) {}
}

#[test]
fn objects_start_in_gen0() {
    clean_gc_state();
    let _obj = Gc::new(42);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let gc_maps = gc_ref.core.lock_gc_maps();
        assert!(
            gc_maps
                .objects
                .values()
                .any(|e| e.generation() == crate::generation::Generation::Gen0),
            "newly allocated objects should be in Gen0"
        );
    });
}

#[test]
fn gen0_collection_does_not_collect_promoted_objects() {
    clean_gc_state();
    let _obj = Gc::new(100);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        // Survive 3 Gen0 collections to promote to Gen1
        for _ in 0..3 {
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
        }
        // Verify object was promoted to Gen1
        {
            let gc_maps = gc_ref.core.lock_gc_maps();
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation() == crate::generation::Generation::Gen1),
                "object should be promoted to Gen1 after surviving 3 Gen0 collections"
            );
        }

        let baseline_objs = gc_ref.core.lock_gc_maps().objects.len();
        // Gen0-only collection should not touch Gen1 objects
        unsafe {
            gc_ref
                .core
                .collect_generation(crate::generation::Generation::Gen0);
        }
        let after_objs = gc_ref.core.lock_gc_maps().objects.len();
        assert_eq!(
            baseline_objs, after_objs,
            "Gen0 collection must not collect Gen1 objects"
        );
    });
}

#[test]
fn promotion_gen0_to_gen1() {
    clean_gc_state();
    let _obj = Gc::new(77);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        // Gen0 promotion threshold is 3
        for _ in 0..2 {
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
        }
        // Still Gen0 after 2 survivals
        {
            let gc_maps = gc_ref.core.lock_gc_maps();
            assert!(
                gc_maps
                    .objects
                    .values()
                    .all(|e| e.generation() == crate::generation::Generation::Gen0),
                "object should still be in Gen0 after 2 survivals"
            );
        }

        // Third survival triggers promotion
        unsafe {
            gc_ref
                .core
                .collect_generation(crate::generation::Generation::Gen0);
        }
        let gc_maps = gc_ref.core.lock_gc_maps();
        assert!(
            gc_maps
                .objects
                .values()
                .any(|e| e.generation() == crate::generation::Generation::Gen1),
            "object should be promoted to Gen1 after 3 survivals"
        );
    });
}

#[test]
fn promotion_gen1_to_gen2() {
    clean_gc_state();
    let _obj = Gc::new(55);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        // Promote to Gen1 first (3 Gen0 collections)
        for _ in 0..3 {
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
        }
        // Now survive 5 Gen1 collections to promote to Gen2
        for _ in 0..5 {
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen1);
            }
        }
        let gc_maps = gc_ref.core.lock_gc_maps();
        assert!(
            gc_maps
                .objects
                .values()
                .any(|e| e.generation() == crate::generation::Generation::Gen2),
            "object should be promoted to Gen2 after surviving Gen1 threshold"
        );
    });
}

#[test]
fn dead_object_collected_by_gen0() {
    // RC hybrid: object is freed immediately when last handle drops.
    // Verify the object is gone without needing an explicit Gen0 collection.
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    {
        let _obj = Gc::new(99);
    }
    let after = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert_eq!(
        after, baseline,
        "RC hybrid should free object immediately when last handle drops"
    );
}

#[test]
fn allocation_count_tracks_new_objects() {
    clean_gc_state();
    let before = LOCAL_GC.with(|gc| gc.borrow().core.allocation_count.load(Ordering::Relaxed));
    let _a = Gc::new(1);
    let _b = Gc::new(2);
    let after = LOCAL_GC.with(|gc| gc.borrow().core.allocation_count.load(Ordering::Relaxed));
    assert_eq!(
        after - before,
        2,
        "allocation_count should increment per new object"
    );
}

#[test]
fn write_barrier_marks_old_gen_object_card_dirty() {
    clean_gc_state();
    use crate::gc::GcCell;
    let cell = GcCell::new(Option::<Gc<i32>>::None);

    // Promote cell's object to Gen1 (survive 3 Gen0 collections)
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        for _ in 0..3 {
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
        }
        // Verify it's in Gen1 (check by value to avoid fat pointer comparison issues)
        let gc_maps = gc_ref.core.lock_gc_maps();
        assert!(
            gc_maps
                .objects
                .values()
                .any(|e| e.generation() == crate::generation::Generation::Gen1),
            "cell should be promoted to Gen1 after 3 Gen0 collections"
        );
    });

    // Card table should be clean before mutation
    LOCAL_GC.with(|gc| {
        assert!(
            gc.borrow().core.card_table.is_clean(),
            "card table should be clean before write barrier"
        );
    });

    // Mutate via borrow_mut() — triggers write barrier
    // (must be outside LOCAL_GC.with borrow to avoid RefCell conflict)
    {
        let young = Gc::new(42);
        **cell.borrow_mut() = Some(young);
    }

    // Verify cell's card is now dirty in the card table
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        assert!(
            gc_ref.core.card_table.has_dirty(),
            "write barrier should mark card dirty in card table"
        );
    });
}

#[test]
fn card_table_cleared_on_full_collection() {
    clean_gc_state();
    use crate::gc::GcCell;
    let cell = GcCell::new(Option::<Gc<i32>>::None);

    // Promote to Gen1
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        for _ in 0..3 {
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
        }
    });

    // Trigger write barrier (outside LOCAL_GC borrow)
    **cell.borrow_mut() = Some(Gc::new(99));

    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        assert!(
            gc_ref.core.card_table.has_dirty(),
            "card table should have dirty cards after write barrier"
        );

        // Full collection (Gen2) should clear card table dirty flags
        unsafe {
            gc_ref
                .core
                .collect_generation(crate::generation::Generation::Gen2);
        }
        assert!(
            gc_ref.core.card_table.is_clean(),
            "card table should be clean after full collection"
        );
    });
}

#[test]
fn write_barrier_preserves_young_object_during_gen0_collection() {
    // Key correctness test: an old-gen object holds a reference to a young
    // object. Without write barrier, the young object would be collected.
    clean_gc_state();
    use crate::gc::GcCell;
    let cell = GcCell::new(Option::<Gc<i32>>::None);

    // Promote cell to Gen1
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        for _ in 0..3 {
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
        }
    });

    // Create young object and store reference in old-gen cell
    let young = Gc::new(777);
    **cell.borrow_mut() = Some(young.clone());
    drop(young); // Drop user handle — only reference is from old-gen cell

    // Gen0 collection — the young object must survive because
    // it's referenced by an old-gen object tracked via the card table
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let objs_before = gc_ref.core.lock_gc_maps().objects.len();
        unsafe {
            gc_ref
                .core
                .collect_generation(crate::generation::Generation::Gen0);
        }
        let objs_after = gc_ref.core.lock_gc_maps().objects.len();
        assert_eq!(
            objs_before, objs_after,
            "young object referenced from old-gen (via card table) must survive Gen0 collection"
        );
    });
}

#[test]
fn compact_preserves_live_objects() {
    clean_gc_state();
    let n = 100;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(Gc::new(i as i32));
    }
    // Drop half to create fragmentation
    for i in (0..n).step_by(2) {
        v[i] = Gc::new(0);
    }
    LOCAL_GC.with(|gc| unsafe { gc.borrow().compact() });
    // Verify values are still accessible after compaction
    assert_eq!(***v.get(1).unwrap(), 1);
    assert_eq!(***v.get(3).unwrap(), 3);
    drop(v);
    LOCAL_GC.with(|gc| unsafe { gc.borrow().collect() });
}

#[test]
fn compact_repeated_iterations() {
    clean_gc_state();
    for _ in 0..20 {
        let n = 1000;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(Gc::new(i as i32));
        }
        for i in (0..n).step_by(2) {
            v[i] = Gc::new(0);
        }
        LOCAL_GC.with(|gc| unsafe { gc.borrow().compact() });
        assert_eq!(***v.get(1).unwrap(), 1);
        drop(v);
        LOCAL_GC.with(|gc| unsafe { gc.borrow().collect() });
    }
}

#[test]
fn no_write_barrier_for_gen0_objects() {
    clean_gc_state();
    use crate::gc::GcCell;
    let cell = GcCell::new(Option::<Gc<i32>>::None);

    // Cell is still in Gen0 — borrow_mut should NOT mark card dirty
    **cell.borrow_mut() = Some(Gc::new(1));

    LOCAL_GC.with(|gc| {
        assert!(
            gc.borrow().core.card_table.is_clean(),
            "Gen0 objects should not mark cards dirty in card table"
        );
    });
}

#[test]
fn cyclic_gc_collect_does_not_overflow() {
    // Create a cycle: a → b → a. Without cycle protection in trace()/reset(),
    // collect() will recurse infinitely and overflow the stack.
    let result = std::thread::Builder::new()
        .stack_size(256 * 1024) // small stack to detect infinite recursion quickly
        .spawn(|| {
            let a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(Some(a.clone())),
            });
            *a.next.borrow_mut() = Some(b.clone());
            // Drop user handles — cycle keeps objects alive internally
            drop(a);
            drop(b);
            LOCAL_GC.with(|gc| unsafe {
                gc.borrow_mut().collect();
            });
        })
        .unwrap()
        .join();
    assert!(
        result.is_ok(),
        "collect() with cyclic references must not stack overflow"
    );
}

#[test]
fn weak_upgrade_returns_some_while_alive() {
    clean_gc_state();
    let strong = Gc::new(42);
    let weak = Gc::downgrade(&strong);
    let upgraded = weak.upgrade();
    assert!(
        upgraded.is_some(),
        "upgrade should succeed while object is alive"
    );
    assert_eq!(**upgraded.unwrap(), 42);
}

#[test]
fn weak_upgrade_returns_none_after_collection() {
    clean_gc_state();
    let weak = {
        let strong = Gc::new(99);
        Gc::downgrade(&strong)
    }; // strong dropped here
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    assert!(
        weak.upgrade().is_none(),
        "upgrade should return None after object is collected"
    );
}

#[test]
fn weak_clone_shares_alive_flag() {
    clean_gc_state();
    let strong = Gc::new(7);
    let weak1 = Gc::downgrade(&strong);
    let weak2 = weak1.clone();
    drop(strong);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    assert!(weak1.upgrade().is_none());
    assert!(weak2.upgrade().is_none());
}

#[test]
fn weak_does_not_prevent_collection() {
    // RC hybrid: object is freed immediately when last strong handle drops.
    // A weak reference is not a handle, so it doesn't affect RC.
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    let weak = {
        let strong = Gc::new(123);
        Gc::downgrade(&strong)
    };
    // RC should have freed the object immediately
    let after = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert_eq!(
        after, baseline,
        "RC hybrid should free object immediately, weak doesn't prevent it"
    );
    assert!(
        weak.upgrade().is_none(),
        "weak upgrade must return None after strong dropped"
    );
}

// --- Incremental tri-color collection tests ---

#[test]
fn incremental_collects_dead_objects() {
    // RC hybrid frees non-cyclic objects immediately. Verify object is gone.
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    {
        let _obj = Gc::new(42);
    }
    // Object already freed by RC
    let after = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert_eq!(
        after, baseline,
        "RC hybrid should free dead object immediately"
    );
    // Incremental collection still works correctly (just nothing to collect)
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let _stats = unsafe {
            gc_ref
                .core
                .collect_incremental(crate::generation::Generation::Gen2, 10)
        };
    });
}

#[test]
fn incremental_preserves_live_objects() {
    clean_gc_state();
    let _live = Gc::new(99);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let objs_before = gc_ref.core.lock_gc_maps().objects.len();
        let stats = unsafe {
            gc_ref
                .core
                .collect_incremental(crate::generation::Generation::Gen2, 10)
        };
        assert_eq!(
            stats.objects_collected, 0,
            "incremental must not collect live objects"
        );
        assert_eq!(gc_ref.core.lock_gc_maps().objects.len(), objs_before);
    });
}

#[test]
fn incremental_step_by_step() {
    // Verify that begin_collection + mark_step + finish_collection works.
    // With RC hybrid, dead non-cyclic objects are already freed, so test with live objects.
    clean_gc_state();
    let _live = Gc::new(2);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let objs_before = gc_ref.core.lock_gc_maps().objects.len();
        unsafe {
            gc_ref
                .core
                .begin_collection(crate::generation::Generation::Gen2);
            while !gc_ref.core.mark_step(1) {}
            let stats = gc_ref.core.finish_collection();
            assert_eq!(
                stats.objects_collected, 0,
                "live object must not be collected"
            );
        }
        let objs_after = gc_ref.core.lock_gc_maps().objects.len();
        assert_eq!(objs_before, objs_after, "live object count must not change");
    });
}

#[test]
fn incremental_with_child_references() {
    // Test that trace_children discovers object graph correctly
    clean_gc_state();
    use crate::gc::GcCell;
    let parent = GcCell::new(Option::<Gc<i32>>::None);
    let child = Gc::new(42);
    **parent.borrow_mut() = Some(child.clone());
    drop(child); // only reference is from parent

    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let objs_before = gc_ref.core.lock_gc_maps().objects.len();
        unsafe {
            gc_ref
                .core
                .collect_incremental(crate::generation::Generation::Gen2, 1);
        }
        let objs_after = gc_ref.core.lock_gc_maps().objects.len();
        assert_eq!(
            objs_before, objs_after,
            "child referenced from parent must survive incremental collection"
        );
    });
}

#[test]
fn incremental_promotes_survivors() {
    clean_gc_state();
    let _obj = Gc::new(55);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        // 3 Gen0 incremental collections should promote to Gen1
        for _ in 0..3 {
            unsafe {
                gc_ref
                    .core
                    .collect_incremental(crate::generation::Generation::Gen0, 10);
            }
        }
        let gc_maps = gc_ref.core.lock_gc_maps();
        assert!(
            gc_maps
                .objects
                .values()
                .any(|e| e.generation() == crate::generation::Generation::Gen1),
            "incremental collection should promote survivors"
        );
    });
}

#[test]
fn incremental_phase_tracking() {
    clean_gc_state();
    let _obj = Gc::new(1);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        // Idle before collection
        assert_eq!(
            gc_ref
                .core
                .incremental
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .phase,
            crate::generation::CollectionPhase::Idle
        );

        unsafe {
            gc_ref
                .core
                .begin_collection(crate::generation::Generation::Gen2);
        }
        assert_eq!(
            gc_ref
                .core
                .incremental
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .phase,
            crate::generation::CollectionPhase::Marking
        );

        while unsafe { !gc_ref.core.mark_step(1) } {}
        // Still marking until finish_collection
        assert_eq!(
            gc_ref
                .core
                .incremental
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .phase,
            crate::generation::CollectionPhase::Marking
        );

        unsafe {
            gc_ref.core.finish_collection();
        }
        assert_eq!(
            gc_ref
                .core
                .incremental
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .phase,
            crate::generation::CollectionPhase::Idle
        );
    });
}

// --- OOM handling tests ---

#[test]
fn try_new_succeeds_for_normal_allocation() {
    clean_gc_state();
    let result = Gc::try_new(42);
    assert!(
        result.is_ok(),
        "try_new should succeed for normal allocation"
    );
    assert_eq!(**result.unwrap(), 42);
}

#[test]
fn try_new_gc_ref_cell_succeeds() {
    clean_gc_state();
    use crate::gc::GcCell;
    let result = GcCell::try_new(100);
    assert!(result.is_ok(), "try_new should succeed for GcCell");
    let cell = result.unwrap();
    assert_eq!(**cell.borrow(), 100);
}

#[test]
fn try_new_object_participates_in_gc() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    {
        let _obj = Gc::try_new(77).unwrap();
    }
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    let after = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert_eq!(
        after, baseline,
        "try_new objects should be collected when dead"
    );
}

#[test]
fn gc_alloc_error_is_display() {
    use crate::gc::GcAllocError;
    let err = GcAllocError;
    assert_eq!(format!("{}", err), "GC allocation failed");
}

// --- Diagnostics API tests ---

#[test]
fn stats_reports_live_objects() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    let _a = Gc::new(1);
    let _b = Gc::new(2);
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert_eq!(
        stats.live_objects - baseline,
        2,
        "stats should report 2 new live objects"
    );
}

#[test]
fn stats_reports_live_handles() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().stats().live_handles);
    let _a = Gc::new(1);
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert_eq!(
        stats.live_handles - baseline,
        1,
        "stats should report 1 new handle"
    );
}

#[test]
fn stats_reports_heap_size() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().stats().heap_size);
    let _a = Gc::new(42i32);
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert!(
        stats.heap_size > baseline,
        "heap_size should increase after allocation"
    );
}

#[test]
fn stats_reports_generation_counts() {
    clean_gc_state();
    let _obj = Gc::new(1);
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert!(stats.gen0_objects >= 1, "new object should be in Gen0");

    // Promote to Gen1
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        for _ in 0..3 {
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
        }
    });
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert!(stats.gen1_objects >= 1, "promoted object should be in Gen1");
}

#[test]
fn stats_tracks_total_collections() {
    clean_gc_state();
    let before = LOCAL_GC.with(|gc| gc.borrow().stats().total_collections);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    let after = LOCAL_GC.with(|gc| gc.borrow().stats().total_collections);
    assert_eq!(
        after - before,
        2,
        "total_collections should increment per collect call"
    );
}

#[test]
fn stats_tracks_last_collection() {
    clean_gc_state();
    // With RC hybrid, dead objects are already freed. Just verify stats recording works.
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert!(
        stats.last_collection.is_some(),
        "last_collection should be Some after collect"
    );
}

#[test]
fn stats_reports_allocation_count() {
    clean_gc_state();
    // After clean_gc_state (which calls collect), allocation_count is reset
    let _a = Gc::new(1);
    let _b = Gc::new(2);
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert!(
        stats.allocation_count >= 2,
        "allocation_count should track new objects"
    );
}

// --- Debug impl tests ---

#[test]
fn debug_gc_prints_value() {
    clean_gc_state();
    let gc = Gc::new(42);
    let s = format!("{:?}", gc);
    assert_eq!(s, "Gc(42)");
}

#[test]
fn debug_gc_ref_cell_prints_value() {
    clean_gc_state();
    let cell = GcCell::new(7);
    let s = format!("{:?}", cell);
    assert_eq!(s, "GcCell(7)");
}

#[test]
fn debug_gc_weak_alive() {
    clean_gc_state();
    let strong = Gc::new(1);
    let weak = Gc::downgrade(&strong);
    assert_eq!(format!("{:?}", weak), "GcWeak(alive)");
}

#[test]
fn debug_gc_weak_dead() {
    clean_gc_state();
    let weak = {
        let strong = Gc::new(1);
        Gc::downgrade(&strong)
    };
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    assert_eq!(format!("{:?}", weak), "GcWeak(dead)");
}

// --- Cycle tests ---

#[test]
fn self_cycle_collected() {
    // A single node pointing to itself should be collected without overflow.
    let result = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(a.clone());
            drop(a);
            LOCAL_GC.with(|gc| unsafe {
                gc.borrow_mut().collect();
            });
        })
        .unwrap()
        .join();
    assert!(result.is_ok(), "self-cycle must not stack overflow");
}

#[test]
fn three_node_cycle_no_overflow() {
    // a → b → c → a — verify no stack overflow on collect
    let result = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let c = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(c.clone());
            *c.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
            drop(c);
            LOCAL_GC.with(|gc| unsafe {
                gc.borrow_mut().collect();
            });
        })
        .unwrap()
        .join();
    assert!(result.is_ok(), "3-node cycle must not stack overflow");
}

struct VecNode {
    children: std::cell::RefCell<Vec<Gc<VecNode>>>,
}
impl Trace for VecNode {
    fn is_root(&self) -> bool {
        false
    }
    fn reset_root(&self) {
        for gc in self.children.borrow().iter() {
            gc.reset_root();
        }
    }
    fn trace(&self) {
        for gc in self.children.borrow().iter() {
            gc.trace();
        }
    }
    fn reset(&self) {
        for gc in self.children.borrow().iter() {
            gc.reset();
        }
    }
    fn is_traceable(&self) -> bool {
        false
    }
    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        for gc in self.children.borrow().iter() {
            gc.trace_children(children);
        }
    }
}
impl Finalize for VecNode {
    fn finalize(&self) {}
}

#[test]
fn cycle_in_vec_no_overflow() {
    // Two nodes each holding the other in their Vec of children.
    let result = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(|| {
            let a = Gc::new(VecNode {
                children: std::cell::RefCell::new(vec![]),
            });
            let b = Gc::new(VecNode {
                children: std::cell::RefCell::new(vec![]),
            });
            let b_clone = b.clone();
            a.children.borrow_mut().push(b_clone);
            let a_clone = a.clone();
            b.children.borrow_mut().push(a_clone);
            drop(a);
            drop(b);
            LOCAL_GC.with(|gc| unsafe {
                gc.borrow_mut().collect();
            });
        })
        .unwrap()
        .join();
    assert!(result.is_ok(), "cycle in Vec must not stack overflow");
}

// --- Root-discovery regression tests (branch-review logic bugs) ---

#[test]
fn rooted_self_cycle_survives_collect() {
    clean_gc_state();
    let a = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    *a.next.borrow_mut() = Some(a.clone());
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    // The stack handle must keep the self-referencing object alive.
    // Regression: the phase-2 outer loop walked fields without the
    // visited guard, so the self edge was decremented twice and the
    // object lost its root status and was swept while still referenced.
    let live = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    assert_eq!(live, 1, "rooted self-cycle must survive collection");
    assert!(a.next.borrow().is_some());
    *a.next.borrow_mut() = None; // break the cycle for clean teardown
}

#[test]
fn rooted_two_node_cycle_survives_collect() {
    clean_gc_state();
    let a = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    let b = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    *a.next.borrow_mut() = Some(b.clone());
    *b.next.borrow_mut() = Some(a.clone());
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    // Both stack handles must keep the cycle alive (same double-decrement
    // regression as rooted_self_cycle_survives_collect, via a 2-cycle).
    let live = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    assert_eq!(live, 2, "rooted 2-cycle must survive collection");
    assert!(a.next.borrow().is_some());
    assert!(b.next.borrow().is_some());
    *a.next.borrow_mut() = None; // break the cycle for clean teardown
}

#[test]
fn gen0_collect_clears_cascade_guard_on_old_gen() {
    clean_gc_state();
    LOCAL_GC.with(|gc| {
        gc.borrow()
            .set_promotion_config(crate::generation::PromotionConfig {
                gen0_threshold: 1,
                gen1_threshold: 5,
            })
    });
    // M survives one full collection -> promoted to Gen1.
    let m = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    // Chain: R (Gen0, stack handle) -> M (Gen1) -> Y (Gen0, only handle
    // is M's edge).
    let y = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    *m.next.borrow_mut() = Some(y.clone());
    drop(y);
    let r = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(Some(m.clone())),
    });
    // Gen0 collection: the phase-2 cascade R -> M sets M's visit guard.
    // Regression: phase 3 cleared guards only for in-scope (Gen0) objects,
    // and root_ref_count doubles as the mark bit — with M's guard leaked,
    // the mark phase skipped M's children and swept live Y.
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut()
            .collect_generation(crate::generation::Generation::Gen0);
    });
    let live = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    assert_eq!(live, 3, "Y must survive: it is reachable via R -> M -> Y");
    assert!(m.next.borrow().is_some());
    // A second Gen0 collection must not lose anything either.
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut()
            .collect_generation(crate::generation::Generation::Gen0);
    });
    let live = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    assert_eq!(live, 3);
    let _ = r;
}

#[test]
fn compact_is_disabled_returns_zero() {
    clean_gc_state();
    let a = Gc::new(42);
    let compacted = LOCAL_GC.with(|gc| unsafe { gc.borrow().compact() });
    assert_eq!(
        compacted, 0,
        "compaction is documented as disabled; when re-enabling it, \
         replace this test with real relocation coverage"
    );
    assert_eq!(**a, 42);
}

// --- Concurrent marking tests (Strategy 21) ---

#[test]
fn concurrent_collects_dead_objects_via_cycle() {
    // RC hybrid handles simple drops, but cycles need tracing.
    // Test concurrent marking on a cycle.
    clean_gc_state();
    {
        let a = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        let b = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        *a.next.borrow_mut() = Some(b.clone());
        *b.next.borrow_mut() = Some(a.clone());
        drop(a);
        drop(b);
    }
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let objs_before = gc_ref.core.lock_gc_maps().objects.len();
        let _stats = unsafe {
            gc_ref
                .core
                .collect_concurrent(crate::generation::Generation::Gen2, 10)
        };
        let objs_after = gc_ref.core.lock_gc_maps().objects.len();
        assert!(
            objs_after < objs_before,
            "concurrent collection should collect cyclic garbage"
        );
    });
}

#[test]
fn concurrent_preserves_live_objects() {
    clean_gc_state();
    let _live = Gc::new(42);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let objs_before = gc_ref.core.lock_gc_maps().objects.len();
        let stats = unsafe {
            gc_ref
                .core
                .collect_concurrent(crate::generation::Generation::Gen2, 10)
        };
        assert_eq!(
            stats.objects_collected, 0,
            "concurrent must not collect live objects"
        );
        let objs_after = gc_ref.core.lock_gc_maps().objects.len();
        assert_eq!(objs_before, objs_after);
    });
}

#[test]
fn concurrent_step_by_step() {
    clean_gc_state();
    let _live = Gc::new(99);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        unsafe {
            gc_ref
                .core
                .begin_concurrent_collection(crate::generation::Generation::Gen2);
        }
        // concurrent_mark_step does NOT require STW
        while !gc_ref.core.concurrent_mark_step(1) {}
        let stats = unsafe { gc_ref.core.finish_collection() };
        assert_eq!(
            stats.objects_collected, 0,
            "live objects must survive concurrent collection"
        );
    });
}

// --- Region-based collection tests (Strategy 22) ---

#[test]
fn objects_assigned_to_current_region() {
    clean_gc_state();
    let _obj = Gc::new(42);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let gc_maps = gc_ref.core.lock_gc_maps();
        assert!(
            gc_maps.objects.values().any(|e| e.region().0 == 0),
            "new objects should be in region 0 (default)"
        );
    });
}

#[test]
fn new_region_changes_allocation_target() {
    clean_gc_state();
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let region1 = gc_ref.core.new_region();
        assert!(region1.0 > 0, "new region should have non-zero ID");
        assert_eq!(gc_ref.core.current_region(), region1);
    });
    let _obj = Gc::new(77);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let region_id = gc_ref.core.current_region();
        let gc_maps = gc_ref.core.lock_gc_maps();
        assert!(
            gc_maps.objects.values().any(|e| e.region() == region_id),
            "object should be in the new region"
        );
    });
}

#[test]
fn collect_region_sweeps_only_target_region() {
    clean_gc_state();
    // Create objects in region 0 (default)
    let _keep = Gc::new(1);

    // Switch to a new region and create + drop objects there
    LOCAL_GC.with(|gc| {
        gc.borrow().core.new_region();
    });
    // Objects in new region will be RC-freed when dropped, so create a cycle
    let region_id = LOCAL_GC.with(|gc| gc.borrow().core.current_region());
    {
        let a = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        let b = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        *a.next.borrow_mut() = Some(b.clone());
        *b.next.borrow_mut() = Some(a.clone());
        drop(a);
        drop(b);
    }

    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let stats = unsafe { gc_ref.core.collect_region(region_id) };
        assert!(
            stats.objects_collected >= 2,
            "should collect cycle in target region"
        );
    });

    // Object in region 0 should still be alive
    assert_eq!(**_keep, 1, "object in other region must survive");
}

// --- RC hybrid tests (Strategy 23) ---

#[test]
fn rc_immediate_dealloc_on_last_drop() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    {
        let _obj = Gc::new(42);
        let during = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
        assert_eq!(during - baseline, 1, "object should be alive during scope");
    }
    let after = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert_eq!(
        after, baseline,
        "RC should free object immediately when last handle drops"
    );
}

#[test]
fn rc_clone_keeps_object_alive() {
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    let a = Gc::new(99);
    let b = a.clone();
    drop(a);
    // Object should still be alive (b holds a reference)
    let after_drop_a = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert_eq!(
        after_drop_a - baseline,
        1,
        "object should survive when clone exists"
    );
    assert_eq!(**b, 99);
    drop(b);
    // Now object should be freed
    let after_drop_b = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert_eq!(
        after_drop_b, baseline,
        "RC should free object when last clone drops"
    );
}

#[test]
fn rc_does_not_free_cyclic_objects() {
    // Cycles keep ref_count > 0, so RC doesn't free them.
    // Tracing GC is needed.
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    {
        let a = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        let b = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        *a.next.borrow_mut() = Some(b.clone());
        *b.next.borrow_mut() = Some(a.clone());
        drop(a);
        drop(b);
    }
    // Cycle objects should still be alive (ref_count > 0 due to internal refs)
    let after = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert!(after > baseline, "cyclic objects should not be freed by RC");

    // Tracing GC should collect them
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    let final_count = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    assert_eq!(
        final_count, baseline,
        "tracing GC should collect cycles that RC cannot"
    );
}

#[test]
fn rc_ref_count_tracks_correctly() {
    // Verify RC behavior: object stays alive while any handle exists,
    // freed when last handle drops.
    clean_gc_state();
    let baseline = LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len());
    let a = Gc::new(42);
    assert_eq!(
        LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len()),
        baseline + 1,
        "one object after creation"
    );

    let b = a.clone();
    // Clone shares the same object — count unchanged
    assert_eq!(
        LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len()),
        baseline + 1,
        "still one object after clone"
    );

    drop(b);
    // Object survives (a still holds it)
    assert_eq!(
        LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len()),
        baseline + 1,
        "object alive while handle exists"
    );
    assert_eq!(**a, 42);

    drop(a);
    // Last handle dropped — RC frees object
    assert_eq!(
        LOCAL_GC.with(|gc| gc.borrow().core.lock_gc_maps().objects.len()),
        baseline,
        "object freed after last handle dropped"
    );
}

#[test]
fn on_thread_incremental_preserves_live_linked_structure() {
    // On-thread incremental Gen2 collection (local background/g1 strategies)
    // advances a bounded mark step per allocation, interleaved with mutation.
    // A live linked structure built while collection is in progress must
    // survive: this exercises grey-at-birth + the marks-based drain across
    // many begin/mark_step/finish cycles. Regression for the class of bug
    // where interleaved incremental marking freed live nodes (which also
    // required VecNode to implement trace_children — the incremental
    // collector traverses via trace_children, not trace()).
    clean_gc_state();
    LOCAL_GC.with(|gc| {
        let g = gc.borrow();
        // Aggressive incremental: collect early and step in tiny batches so
        // many cycles interleave with the build below.
        g.core.alloc_threshold.store(16, Ordering::Relaxed);
        g.core.local_incremental_budget.store(4, Ordering::Relaxed);
    });

    // Build a chain of 200 nodes bottom-up; the head keeps the whole chain
    // reachable. Interleave throwaway allocations to drive collection.
    const CHAIN: usize = 200;
    let mut head = Gc::new(VecNode {
        children: std::cell::RefCell::new(Vec::new()),
    });
    for _ in 0..CHAIN {
        for _ in 0..8 {
            // Garbage that should be reclaimed, driving mark steps/finishes.
            let _g = Gc::new(VecNode {
                children: std::cell::RefCell::new(Vec::new()),
            });
        }
        let parent = Gc::new(VecNode {
            children: std::cell::RefCell::new(vec![head]),
        });
        head = parent;
    }

    // Traverse the chain from the head — a freed live node would UAF/segfault
    // or shorten the count.
    fn depth(n: &Gc<VecNode>) -> usize {
        let kids = n.children.borrow();
        1 + kids.iter().map(depth).sum::<usize>()
    }
    assert_eq!(
        depth(&head),
        CHAIN + 1,
        "live linked structure lost nodes under on-thread incremental collection"
    );

    LOCAL_GC.with(|gc| {
        gc.borrow()
            .core
            .local_incremental_budget
            .store(0, Ordering::Relaxed);
    });
    drop(head);
}

#[test]
fn object_allocated_during_incremental_mark_survives() {
    // Regression: the marks-based sweep in finish_collection reclaims any
    // object not marked Black. An object allocated *after* begin_collection
    // is absent from the initial `marks` snapshot, so without grey-at-birth
    // (note_new_object_during_marking) it would be swept while still live.
    clean_gc_state();
    let _root: Vec<Gc<i32>> = (0..10).map(|i| Gc::new(i)).collect();

    // Open an incremental mark cycle (phase = Marking) without finishing it.
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()
            .core
            .begin_collection(crate::generation::Generation::Gen2);
    });

    // Allocate DURING the mark cycle. Must be greyed at birth and survive.
    let born_mid_mark = Gc::new(999i32);

    // Drain + marks-based sweep.
    let stats = LOCAL_GC.with(|gc| unsafe { gc.borrow().core.finish_collection() });

    // The mid-mark object is still rooted and must not have been collected.
    assert_eq!(
        **born_mid_mark, 999,
        "mid-mark allocation was corrupted/freed"
    );
    assert_eq!(
        stats.objects_collected, 0,
        "no live object should be collected"
    );
    drop(born_mid_mark);
    drop(_root);
}

#[test]
fn timed_incremental_collection_works() {
    use std::time::Duration;
    clean_gc_state();
    // Keep objects alive during collection to verify timed marking scans them
    let v: Vec<Gc<i32>> = (0..100).map(|i| Gc::new(i)).collect();

    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let objs_before = gc_ref.core.lock_gc_maps().objects.len();
        unsafe {
            let stats = gc_ref.core.collect_incremental_timed(
                crate::generation::Generation::Gen2,
                Duration::from_secs(1),
            );
            // All live objects should be scanned, none collected
            assert_eq!(stats.objects_collected, 0);
            assert!(stats.objects_scanned > 0);
        }
        let objs_after = gc_ref.core.lock_gc_maps().objects.len();
        assert_eq!(objs_before, objs_after, "no objects should be collected");
    });
    drop(v);
}

#[test]
fn timed_concurrent_collection_works() {
    use std::time::Duration;
    clean_gc_state();
    let v: Vec<Gc<i32>> = (0..100).map(|i| Gc::new(i)).collect();

    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let objs_before = gc_ref.core.lock_gc_maps().objects.len();
        unsafe {
            let stats = gc_ref.core.collect_concurrent_timed(
                crate::generation::Generation::Gen2,
                Duration::from_secs(1),
            );
            assert_eq!(stats.objects_collected, 0);
            assert!(stats.objects_scanned > 0);
        }
        let objs_after = gc_ref.core.lock_gc_maps().objects.len();
        assert_eq!(objs_before, objs_after, "no objects should be collected");
    });
    drop(v);
}

#[test]
fn stats_tracks_collection_duration() {
    clean_gc_state();
    let _obj = Gc::new(42);
    drop(_obj);
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        unsafe {
            gc_ref.core.collect();
        }
        let stats = gc_ref.stats();
        let last = stats
            .last_collection
            .expect("should have a last collection");
        assert!(
            last.duration > std::time::Duration::ZERO,
            "duration should be > 0"
        );
    });
}

#[test]
fn stats_tracks_bytes_freed() {
    clean_gc_state();
    // Create a self-cycle so RC hybrid cannot eagerly free it.
    // Only mark-sweep collection can reclaim cycles.
    let a = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    *a.next.borrow_mut() = Some(a.clone());
    drop(a);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow().core.collect();
    });
    LOCAL_GC.with(|gc| {
        let stats = gc.borrow().stats();
        let last = stats
            .last_collection
            .expect("should have a last collection");
        assert!(last.bytes_freed > 0, "bytes_freed should be > 0");
    });
}

#[test]
fn stats_tracks_peak_heap_size() {
    clean_gc_state();
    let _obj1 = Gc::new(1_i64);
    let _obj2 = Gc::new(2_i64);
    let peak_with_objects = LOCAL_GC.with(|gc| gc.borrow().stats().peak_heap_size);
    assert!(peak_with_objects > 0, "peak should be > 0 after alloc");
    drop(_obj1);
    drop(_obj2);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow().core.collect();
    });
    let peak_after = LOCAL_GC.with(|gc| gc.borrow().stats().peak_heap_size);
    assert!(
        peak_after >= peak_with_objects,
        "peak should not decrease after collection"
    );
}

#[test]
fn on_collection_callback_fires() {
    clean_gc_state();
    let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let called_clone = called.clone();
    LOCAL_GC.with(|gc| {
        gc.borrow().set_on_collection(move |_stats| {
            called_clone.store(true, std::sync::atomic::Ordering::Release);
        });
    });
    let _obj = Gc::new(42);
    drop(_obj);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow().core.collect();
    });
    assert!(
        called.load(std::sync::atomic::Ordering::Acquire),
        "callback should have been called"
    );
    LOCAL_GC.with(|gc| {
        gc.borrow().clear_on_collection();
    });
}

#[test]
fn on_collection_callback_panic_does_not_break_gc() {
    clean_gc_state();
    LOCAL_GC.with(|gc| {
        gc.borrow().set_on_collection(|_stats| {
            panic!("intentional panic in callback");
        });
    });
    let _obj = Gc::new(42);
    drop(_obj);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow().core.collect();
    });
    // GC should still work after panicking callback
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert!(stats.total_collections > 0);
    LOCAL_GC.with(|gc| {
        gc.borrow().clear_on_collection();
    });
}

// ---- G1 Garbage-First tests ----

#[test]
fn region_stats_tracks_allocations() {
    clean_gc_state();
    // Allocate objects in default region (0)
    let _a = Gc::new(1u64);
    let _b = Gc::new(2u64);

    // Switch to a new region and allocate there
    let region1 = LOCAL_GC.with(|gc| gc.borrow().new_region());

    let _c = Gc::new_in(3u64, region1);

    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let stats = gc_ref.region_stats();
        // Should have at least 2 regions (default 0 + region1)
        assert!(
            stats.len() >= 2,
            "expected at least 2 regions, got {}",
            stats.len()
        );

        let r0_stats = stats
            .iter()
            .find(|s| s.region_id == crate::generation::RegionId(0));
        assert!(r0_stats.is_some(), "default region stats should exist");
        let r0 = r0_stats.unwrap();
        assert!(
            r0.object_count >= 2,
            "default region should have at least 2 objects, got {}",
            r0.object_count
        );
        assert!(r0.total_bytes > 0, "default region should have > 0 bytes");

        let r1_stats = stats
            .iter()
            .find(|s| s.region_id == crate::generation::RegionId(region1.id()));
        assert!(r1_stats.is_some(), "region1 stats should exist");
        let r1 = r1_stats.unwrap();
        assert_eq!(r1.object_count, 1, "region1 should have 1 object");
        assert!(r1.total_bytes > 0, "region1 should have > 0 bytes");
    });
}

#[test]
fn collect_garbage_first_collects_dirtiest_region() {
    clean_gc_state();

    // Region 0: keep objects alive
    let _keep1 = Gc::new(100u64);
    let _keep2 = Gc::new(200u64);

    // Create a new region with garbage (cyclic references that become unreachable)
    let dirty_region = LOCAL_GC.with(|gc| gc.borrow().new_region());
    {
        let a = Gc::new_in(
            CyclicNode {
                next: std::cell::RefCell::new(None),
            },
            dirty_region,
        );
        let b = Gc::new_in(
            CyclicNode {
                next: std::cell::RefCell::new(None),
            },
            dirty_region,
        );
        *a.next.borrow_mut() = Some(b.clone());
        *b.next.borrow_mut() = Some(a.clone());
        // Drop all handles => cycles become garbage
        drop(a);
        drop(b);
    }

    // Run G1 collection with a generous pause target
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let stats = unsafe { gc_ref.collect_garbage_first(std::time::Duration::from_secs(10)) };
        assert!(
            stats.objects_collected >= 2,
            "G1 should collect at least the 2 cyclic garbage objects, got {}",
            stats.objects_collected
        );
    });

    // Objects in region 0 should still be alive
    assert_eq!(**_keep1, 100u64);
    assert_eq!(**_keep2, 200u64);
}

#[test]
fn collect_garbage_first_respects_pause_target() {
    clean_gc_state();

    // Create some objects
    let _keep = Gc::new(42u64);

    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let start = std::time::Instant::now();
        let stats = unsafe { gc_ref.collect_garbage_first(std::time::Duration::from_millis(100)) };
        let elapsed = start.elapsed();
        // The method should return in reasonable time (within 2x the target + mark overhead)
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "G1 collection took too long: {:?}",
            elapsed
        );
        // Stats should be valid
        assert_eq!(stats.generation, crate::generation::Generation::Gen2);
        // Duration should be recorded
        assert!(stats.duration > std::time::Duration::ZERO);
    });
}

// --- Parallel collection tests ---

#[cfg(feature = "parallel")]
#[test]
fn parallel_collect_basic() {
    clean_gc_state();
    // RC hybrid immediately frees non-cyclic objects on drop, so we need
    // cycles to ensure objects survive until the collector runs.
    {
        let a = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        let b = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        let c = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        // a → b → c → a
        *a.next.borrow_mut() = Some(b.clone());
        *b.next.borrow_mut() = Some(c.clone());
        *c.next.borrow_mut() = Some(a.clone());
    }
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let stats = unsafe {
            gc_ref
                .core
                .collect_parallel(crate::generation::Generation::Gen2)
        };
        // All 3 cyclic objects should have been collected
        assert!(
            stats.objects_collected >= 3,
            "expected at least 3 objects collected, got {}",
            stats.objects_collected,
        );
        // bytes_freed should be positive
        assert!(
            stats.bytes_freed > 0,
            "expected bytes_freed > 0, got {}",
            stats.bytes_freed,
        );
        // No objects should remain from this batch
        let maps = gc_ref.core.lock_gc_maps();
        assert_eq!(
            maps.objects.len(),
            0,
            "all objects should be collected after parallel collect"
        );
    });
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_collect_preserves_live() {
    clean_gc_state();
    // Keep one object alive on the stack
    let live = Gc::new(42);
    // Create a dead cycle that the collector should reclaim
    {
        let dead_a = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        let dead_b = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        *dead_a.next.borrow_mut() = Some(dead_b.clone());
        *dead_b.next.borrow_mut() = Some(dead_a.clone());
    }
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let stats = unsafe {
            gc_ref
                .core
                .collect_parallel(crate::generation::Generation::Gen2)
        };
        // The 2 dead cyclic objects should have been collected
        assert!(
            stats.objects_collected >= 2,
            "expected at least 2 objects collected, got {}",
            stats.objects_collected,
        );
        // The live object should still be accessible
        assert_eq!(**live, 42);
        // There should be exactly 1 object remaining (the live one)
        let maps = gc_ref.core.lock_gc_maps();
        assert_eq!(
            maps.objects.len(),
            1,
            "live object should survive parallel collect"
        );
    });
    drop(live);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_collect_cycles() {
    clean_gc_state();
    // Create a cycle: a → b → a
    {
        let a = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        let b = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(None),
        });
        *a.next.borrow_mut() = Some(b.clone());
        *b.next.borrow_mut() = Some(a.clone());
        // Drop user handles — cycle keeps objects alive internally
    }
    LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();
        let stats = unsafe {
            gc_ref
                .core
                .collect_parallel(crate::generation::Generation::Gen2)
        };
        // Both cyclic objects should have been collected
        assert!(
            stats.objects_collected >= 2,
            "expected at least 2 cyclic objects collected, got {}",
            stats.objects_collected,
        );
        let maps = gc_ref.core.lock_gc_maps();
        assert_eq!(
            maps.objects.len(),
            0,
            "cyclic objects should be collected by parallel collect"
        );
    });
}

// ---- TLAB integration tests ----

#[test]
fn tlab_basic_allocation() {
    // Objects allocated via the TLAB should work identically to system-allocated objects.
    clean_gc_state();
    let a = Gc::new(42i32);
    let b = Gc::new(100i32);
    assert_eq!(**a, 42);
    assert_eq!(**b, 100);
    // Verify the objects are tracked by the GC
    LOCAL_GC.with(|gc| {
        let gc = gc.borrow();
        let maps = gc.core.lock_gc_maps();
        assert!(maps.objects.len() >= 2);
    });
    drop(a);
    drop(b);
    // After collection, all should be cleaned up
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
}

#[test]
fn tlab_many_allocations() {
    // Allocate enough objects to exhaust one TLAB block and require a new one.
    clean_gc_state();
    // Use String which is ~24 bytes on stack + heap allocation.
    // With GcPtr overhead, we need many allocations to exhaust a 64KB TLAB block.
    let mut gcs: Vec<Gc<String>> = Vec::new();
    for i in 0..2000 {
        gcs.push(Gc::new(format!("item_{}", i)));
    }
    // All objects should be valid
    for (i, gc) in gcs.iter().enumerate() {
        assert_eq!(&***gc, &format!("item_{}", i));
    }
    // TLAB should have been initialized
    LOCAL_GC.with(|gc| {
        let gc = gc.borrow();
        assert!(
            gc.tlab.is_some(),
            "TLAB should be initialized after allocations"
        );
    });
    drop(gcs);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
}

#[test]
fn tlab_fallback_to_system() {
    // When TLAB cannot service a request, it should fall back to the system allocator.
    // We test by allocating a very large String to stress the allocator path.
    // Even if the TLAB grows to accommodate, the fallback is always available.
    clean_gc_state();
    let big = Gc::new("x".repeat(256 * 1024));
    assert!(big.len() == 256 * 1024);
    drop(big);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
}

#[test]
fn tlab_dealloc_frees_block() {
    // When all objects in a TLAB block are collected, the block should be freed.
    // We verify this indirectly: allocate objects, force a new block, drop all
    // objects from the first block, collect, and check that no crash occurs.
    clean_gc_state();
    // Allocate some objects into the initial TLAB block
    let a = Gc::new(1u64);
    let b = Gc::new(2u64);
    let c = Gc::new(3u64);
    assert_eq!(**a, 1);
    assert_eq!(**b, 2);
    assert_eq!(**c, 3);
    // Drop all and collect — the TLAB block's ref_count should reach 0
    drop(a);
    drop(b);
    drop(c);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
        // After collection, all objects should be gone
        let gc_ref = gc.borrow();
        let maps = gc_ref.core.lock_gc_maps();
        assert_eq!(
            maps.objects.len(),
            0,
            "all TLAB-allocated objects should be collected"
        );
    });
}

#[test]
fn tlab_gc_ref_cell_allocation() {
    // GcCell should also use the TLAB for allocation.
    clean_gc_state();
    let cell = GcCell::new(String::from("hello"));
    {
        let borrowed = cell.borrow();
        assert_eq!(&***borrowed, "hello");
    }
    {
        let mut borrowed = cell.borrow_mut();
        **borrowed = String::from("world");
    }
    {
        let borrowed = cell.borrow();
        assert_eq!(&***borrowed, "world");
    }
    drop(cell);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
}

#[test]
fn tlab_mixed_alloc_and_collect() {
    // Interleave TLAB allocations with collections to ensure correctness.
    clean_gc_state();
    for _ in 0..10 {
        let _a = Gc::new(42i32);
        let _b = Gc::new(String::from("test"));
        let _c = GcCell::new(vec![1, 2, 3]);
        // Drop _a, _b, _c at end of iteration
    }
    // Force collection
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
        let gc_ref = gc.borrow();
        let maps = gc_ref.core.lock_gc_maps();
        assert_eq!(maps.objects.len(), 0, "all objects should be collected");
    });
}

#[test]
fn tlab_try_new_allocation() {
    // Fallible allocation should also use TLAB.
    clean_gc_state();
    let result = Gc::try_new(99u32);
    assert!(result.is_ok());
    let gc = result.unwrap();
    assert_eq!(**gc, 99);
    drop(gc);
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
}

// --- Allocation trigger skips collections that cannot free anything ---

/// Whether a collection of generations `0..=max_gen` could free anything.
fn garbage_possible(max_gen: crate::generation::Generation) -> bool {
    LOCAL_GC.with(|gc| {
        // SAFETY: thread-local collector, no collection in progress.
        unsafe { gc.borrow().core.gc_maps_unsync().garbage_possible(max_gen) }
    })
}

/// Collections so far, and whether a Gen0 collection could free anything.
fn trigger_state() -> (usize, bool) {
    let total = LOCAL_GC.with(|gc| gc.borrow().stats().total_collections);
    (total, garbage_possible(crate::generation::Generation::Gen0))
}

fn self_cycle() -> Gc<CyclicNode> {
    let a = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    *a.next.borrow_mut() = Some(a.clone());
    a
}

#[test]
fn allocation_trigger_skips_collection_without_garbage() {
    clean_gc_state();
    LOCAL_GC.with(|gc| {
        gc.borrow()
            .core
            .alloc_threshold
            .store(16, Ordering::Relaxed)
    });
    let (before, possible) = trigger_state();
    assert!(!possible, "a collection leaves no garbage behind");

    // Objects kept, moved into other objects, or freed by their last
    // handle's drop never make garbage: every trigger is skipped.
    let mut kept = Vec::new();
    for i in 0..20 {
        kept.push(Gc::new(i));
    }
    let mut chain = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    for _ in 0..50 {
        chain = Gc::new(CyclicNode {
            next: std::cell::RefCell::new(Some(chain)),
        });
    }
    for i in 0..1000 {
        drop(Gc::new(i));
    }
    let (after, possible) = trigger_state();
    assert!(!possible);
    assert_eq!(after, before, "no collection could free anything");

    // Skipped triggers still trim the ids of objects freed by their last
    // handle's drop, which collections otherwise compact: at most twice the
    // (here constant) objects after the last trigger, plus one id per
    // allocation since.
    LOCAL_GC.with(|gc| {
        let g = gc.borrow();
        let since_trigger = g.core.allocation_count.load(Ordering::Relaxed);
        // SAFETY: thread-local collector, no collection in progress.
        let maps = unsafe { g.core.gc_maps_unsync() };
        assert!(
            maps.gen0_ids.len() <= 2 * maps.objects.len() + since_trigger,
            "gen0_ids grew to {} for {} objects",
            maps.gen0_ids.len(),
            maps.objects.len()
        );
    });
    drop(kept);
    drop(chain);
}

#[test]
fn allocation_trigger_collects_cycle_after_handle_drop() {
    clean_gc_state();
    LOCAL_GC.with(|gc| {
        gc.borrow()
            .core
            .alloc_threshold
            .store(16, Ordering::Relaxed)
    });
    let live_before = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    let (before, _) = trigger_state();

    // The stack handle is dropped while the self edge remains: the cycle
    // is garbage only a collection can free.
    drop(self_cycle());
    let (_, possible) = trigger_state();
    assert!(
        possible,
        "dropping a handle its object outlives allows garbage"
    );

    for i in 0..16 {
        drop(Gc::new(i));
    }
    let (after, possible) = trigger_state();
    assert!(after > before, "the trigger collected");
    assert!(!possible);
    let live = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    assert_eq!(live, live_before, "the dropped cycle was freed");
}

#[test]
fn incremental_trigger_skips_collection_without_garbage() {
    use crate::generation::CollectionPhase;
    clean_gc_state();
    LOCAL_GC.with(|gc| {
        let g = gc.borrow();
        g.core.alloc_threshold.store(16, Ordering::Relaxed);
        g.core.local_incremental_budget.store(4, Ordering::Relaxed);
    });
    let phase = || LOCAL_GC.with(|gc| gc.borrow().core.incremental_phase());
    let live_before = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);

    let kept: Vec<_> = (0..100).map(Gc::new).collect();
    assert_eq!(
        phase(),
        CollectionPhase::Idle,
        "no mark cycle without garbage"
    );

    // A dropped cycle starts a mark cycle at the next trigger, which frees
    // it. (Skipped triggers raise the threshold to the GOGC floor; bring
    // the next trigger back within reach once the cycle exists.)
    let cycle = self_cycle();
    LOCAL_GC.with(|gc| {
        gc.borrow()
            .core
            .alloc_threshold
            .store(16, Ordering::Relaxed)
    });
    drop(cycle);
    for i in 0..200 {
        drop(Gc::new(i));
    }
    assert_eq!(phase(), CollectionPhase::Idle);
    let live = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    assert_eq!(
        live,
        live_before + kept.len(),
        "the dropped cycle was freed"
    );

    LOCAL_GC.with(|gc| {
        gc.borrow()
            .core
            .local_incremental_budget
            .store(0, Ordering::Relaxed);
    });
    drop(kept);
}

/// A graph node whose edges are mutated through `GcCell` (the write
/// barrier keeps old-to-young edges visible to partial collections).
struct CellNode {
    children: Vec<GcCell<CellNode>>,
}

impl Trace for CellNode {
    fn is_root(&self) -> bool {
        false
    }
    fn reset_root(&self) {
        self.children.iter().for_each(|c| c.reset_root());
    }
    fn trace(&self) {
        self.children.iter().for_each(|c| c.trace());
    }
    fn reset(&self) {
        self.children.iter().for_each(|c| c.reset());
    }
    fn is_traceable(&self) -> bool {
        false
    }
    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.children
            .iter()
            .for_each(|c| c.trace_children(children));
    }
}

impl Finalize for CellNode {
    fn finalize(&self) {}
}

#[test]
fn no_garbage_while_flag_clear_under_random_mutation() {
    // Random graph mutation (edges, moves, drops, cycles) with the
    // allocation trigger firing often and collections of every scope (so
    // survivors are promoted, and each bit gets cleared alone). Whenever a
    // scope's bit says no garbage is possible, a collection of that scope
    // must indeed free nothing. Any live object freed by a partial
    // collection (see CASCADE_PENDING) shows up as a use-after-free.
    use crate::generation::Generation;
    clean_gc_state();
    LOCAL_GC.with(|gc| gc.borrow().core.alloc_threshold.store(8, Ordering::Relaxed));
    let node = || {
        GcCell::new(CellNode {
            children: Vec::new(),
        })
    };
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rand = move |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let ops = if cfg!(miri) { 300 } else { 5000 };
    let mut roots: Vec<GcCell<CellNode>> = (0..4).map(|_| node()).collect();
    let mut checked = 0;
    for _ in 0..ops {
        if roots.is_empty() {
            roots.push(node());
        }
        let i = rand(roots.len());
        match rand(6) {
            0 => roots.push(node()),
            // Add an edge (possibly a cycle).
            1 => {
                let j = rand(roots.len());
                let child = roots[j].clone();
                roots[i].borrow_mut().children.push(child);
            }
            // Drop a root handle.
            2 => drop(roots.swap_remove(i)),
            // Drop a node's edges.
            3 => roots[i].borrow_mut().children.clear(),
            // Move a root handle into another node.
            4 => {
                let moved = roots.swap_remove(i);
                if !roots.is_empty() {
                    let j = rand(roots.len());
                    roots[j].borrow_mut().children.push(moved);
                }
            }
            // Take an edge out of a node as a new root.
            _ => {
                let taken = roots[i].borrow_mut().children.pop();
                roots.extend(taken);
            }
        }
        // Check the widest scope whose bit is clear.
        let scope = [Generation::Gen2, Generation::Gen1, Generation::Gen0]
            .into_iter()
            .find(|&g| !garbage_possible(g));
        if let Some(max_gen) = scope {
            checked += 1;
            let stats = LOCAL_GC.with(|gc| unsafe { gc.borrow().core.collect_generation(max_gen) });
            assert_eq!(
                stats.objects_collected, 0,
                "{max_gen:?} collection freed garbage while its bit was clear"
            );
        } else if rand(4) == 0 {
            let max_gen = [Generation::Gen0, Generation::Gen1, Generation::Gen2][rand(3)];
            LOCAL_GC.with(|gc| unsafe { gc.borrow().core.collect_generation(max_gen) });
        }
    }
    assert!(
        checked > ops / 10,
        "a bit was clear in only {checked} states"
    );
    for r in &roots {
        r.borrow_mut().children.clear();
    }
}

#[test]
fn partial_collection_keeps_young_object_reached_only_from_promoted_one() {
    // Regression: X gets an edge to Y while both are in Gen0 (the write
    // barrier marks no card), then X is promoted and Y is not. A Gen0
    // collection whose root-discovery cascade reached X through a dead Gen0
    // cycle Z subtracted X's handle to Y; X itself is out of scope and its
    // card clean, so nothing traced it, and Y was freed while X still
    // pointed at it.
    use crate::generation::Generation;
    clean_gc_state();
    let gen0 =
        || LOCAL_GC.with(|gc| unsafe { gc.borrow().core.collect_generation(Generation::Gen0) });
    let node = || {
        GcCell::new(CellNode {
            children: Vec::new(),
        })
    };

    let x = node();
    gen0();
    gen0();
    let y = node();
    x.borrow_mut().children.push(y.clone());
    drop(y);
    // X's third survival promotes it; Y has survived once.
    assert_eq!(gen0().objects_promoted, 1);

    let z = node();
    let (z_self, z_to_x) = (z.clone(), x.clone());
    z.borrow_mut().children.extend([z_self, z_to_x]);
    drop(z);

    let stats = gen0();
    assert_eq!(stats.objects_collected, 1, "only the dead cycle is freed");
    assert_eq!(x.borrow().children.len(), 1);
    assert!(x.borrow().children[0].borrow().children.is_empty());
}

// --- The allocation trigger reaches every generation, on the owning thread ---

/// This thread's collector, for calling a local strategy's closure directly
/// (as `LocalStrategy::start` does).
fn this_thread_gc() -> &'static super::LocalGarbageCollector {
    // SAFETY: the thread-local outlives the test body, which only uses it on
    // this thread.
    LOCAL_GC.with(|gc| unsafe { &*gc.as_ptr() })
}

/// Promote a live self-cycle by `gen0` Gen0 then `gen1` Gen1 collections.
fn promoted_cycle(gen0: usize, gen1: usize) -> Gc<CyclicNode> {
    use crate::generation::Generation;
    let a = self_cycle();
    for (scope, n) in [(Generation::Gen0, gen0), (Generation::Gen1, gen1)] {
        for _ in 0..n {
            LOCAL_GC.with(|gc| unsafe { gc.borrow().core.collect_generation(scope) });
        }
    }
    a
}

#[test]
fn allocation_trigger_frees_promoted_dead_cycles() {
    // Regression: the trigger only ever ran Gen0 collections, so a dead
    // cycle that had been promoted was never freed. Now every
    // gen0_per_gen1-th triggered collection covers Gen1 and every
    // gen1_per_gen2-th of those Gen2 (a fixed small threshold keeps the
    // test short; the default schedule is 5/5).
    use super::{ThresholdRule, TriggerPolicy};
    clean_gc_state();
    let live_before = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    // Gen0 promotes after 3 survivals and Gen1 after 5 more.
    let in_gen2 = promoted_cycle(3, 5);
    let in_gen1 = promoted_cycle(3, 0);
    let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
    assert_eq!((stats.gen1_objects, stats.gen2_objects), (1, 1));
    this_thread_gc().configure_trigger(
        16,
        0,
        TriggerPolicy {
            gen0_per_gen1: 2,
            gen1_per_gen2: 2,
            threshold: ThresholdRule::Fixed,
        },
    );
    drop(in_gen1);
    drop(in_gen2);
    // Gen1 is covered on the 2nd trigger and Gen2 on the 4th.
    for i in 0..16 * 5 {
        drop(Gc::new(i));
    }
    let live = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    assert_eq!(live, live_before, "both promoted dead cycles were freed");
}

#[test]
fn gogc_threshold_follows_the_live_set() {
    // Regression: after a collection that freed next to nothing the
    // threshold doubled without bound, so the trigger's schedule never
    // reached older generations again. It now waits for as many
    // allocations as there are live objects (at least the floor).
    clean_gc_state();
    let kept: Vec<_> = (0..100).map(Gc::new).collect();
    let floor = super::LOCAL_GC_ALLOC_THRESHOLD;
    for _ in 0..3 {
        // A handle dropped while its object keeps others: garbage is
        // possible, so the next trigger collects (and frees nothing).
        drop(kept[0].clone());
        let before = LOCAL_GC.with(|gc| gc.borrow().stats().total_collections);
        for i in 0..floor + 1 {
            drop(Gc::new(i));
        }
        LOCAL_GC.with(|gc| {
            let g = gc.borrow();
            assert!(
                g.stats().total_collections > before,
                "the trigger collected"
            );
            let live = g.stats().live_objects;
            assert_eq!(
                g.core.alloc_threshold.load(Ordering::Relaxed),
                live.max(floor)
            );
        });
    }
    drop(kept);
}

#[test]
fn threshold_and_adaptive_local_strategies_collect_on_the_owning_thread() {
    // They used to spawn a thread that collected this thread's collector
    // while this thread mutated it - a data race. Now they only configure
    // the allocation trigger. (The collector reference a strategy gets is
    // used right away, as `LocalStrategy::start` does: an allocation's
    // `borrow_mut` invalidates it.)
    static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
    clean_gc_state();
    let threshold = || LOCAL_GC.with(|gc| gc.borrow().core.alloc_threshold.load(Ordering::Relaxed));
    let live_before = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);

    let mut strategy = crate::threshold_local_strategy(crate::ThresholdConfig {
        gen0_threshold: 32,
        ..Default::default()
    });
    assert!(
        strategy(this_thread_gc(), &ACTIVE).is_none(),
        "no collector thread"
    );
    assert_eq!(threshold(), 32);
    drop(self_cycle());
    for i in 0..32 {
        drop(Gc::new(i));
    }
    let live = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
    assert_eq!(live, live_before, "the dead cycle was freed on this thread");
    assert_eq!(
        threshold(),
        32,
        "the threshold strategy keeps its threshold"
    );

    let config = crate::AdaptiveConfig {
        initial_gen0_threshold: 64,
        min_threshold: 50,
        max_threshold: 80,
        ..Default::default()
    };
    let mut strategy = crate::adaptive_local_strategy(config);
    assert!(
        strategy(this_thread_gc(), &ACTIVE).is_none(),
        "no collector thread"
    );
    assert_eq!(threshold(), 64);
    // A collection that frees next to nothing raises the threshold, up to
    // the configured maximum.
    let kept = Gc::new(0);
    drop(kept.clone());
    for i in 0..64 {
        drop(Gc::new(i));
    }
    assert_eq!(threshold(), 80);
    drop(kept);
}

#[test]
fn skipped_trigger_trims_gen0_ids_after_a_large_free() {
    // After a large free the object map keeps more slots than the next
    // triggers' ids are worth rebuilding from: the trim filters the list
    // instead. Either way only the live Gen0 objects stay listed.
    clean_gc_state();
    let keep = Gc::new(0);
    drop((0..5000).map(Gc::new).collect::<Vec<_>>());
    let run_to_next_trigger = || {
        let threshold =
            LOCAL_GC.with(|gc| gc.borrow().core.alloc_threshold.load(Ordering::Relaxed));
        for i in 0..threshold {
            drop(Gc::new(i));
        }
    };
    // The first trigger rebuilds the list (fewer slots than ids); the
    // second, soon after, filters it (more slots than ids).
    run_to_next_trigger();
    LOCAL_GC.with(|gc| {
        gc.borrow()
            .core
            .alloc_threshold
            .store(100, Ordering::Relaxed)
    });
    run_to_next_trigger();
    LOCAL_GC.with(|gc| {
        let g = gc.borrow();
        let since_trigger = g.core.allocation_count.load(Ordering::Relaxed);
        // SAFETY: thread-local collector, no collection in progress.
        let maps = unsafe { g.core.gc_maps_unsync() };
        assert!(maps.objects.slot_count() >= 5000);
        assert!(
            maps.gen0_ids.len() <= 2 * maps.objects.len() + since_trigger,
            "gen0_ids kept {} ids for {} objects",
            maps.gen0_ids.len(),
            maps.objects.len()
        );
    });
    drop(keep);
}

/// Run an incremental mark of `max_gen` with `mutate` between its begin and
/// its steps, then finish it.
fn incremental_cycle_with(
    max_gen: crate::generation::Generation,
    mutate: impl FnOnce(),
) -> crate::generation::CollectionStats {
    LOCAL_GC.with(|gc| unsafe { gc.borrow().core.begin_collection(max_gen) });
    mutate();
    LOCAL_GC.with(|gc| unsafe {
        let gc = gc.borrow();
        while !gc.core.mark_step(1) {}
        gc.core.finish_collection()
    })
}

#[test]
fn handle_cloned_during_incremental_mark_survives() {
    // Regression: the mark followed the live graph, so an object whose only
    // in-graph handle went away mid-mark stayed White although a handle
    // cloned after the mark began still pointed at it - and was freed.
    clean_gc_state();
    let holder: GcCell<Option<Gc<i32>>> = GcCell::new(Some(Gc::new(7)));
    let mut taken = None;
    let stats = incremental_cycle_with(crate::generation::Generation::Gen2, || {
        taken = (**holder.borrow()).clone();
        **holder.borrow_mut() = None;
    });
    assert_eq!(stats.objects_collected, 0, "a live object was freed");
    assert_eq!(**taken.unwrap(), 7);
}

#[test]
fn handle_moved_out_during_incremental_mark_survives() {
    // As above, with the handle moved out of the cell (no new handle).
    clean_gc_state();
    let holder: GcCell<Option<Gc<i32>>> = GcCell::new(Some(Gc::new(7)));
    let mut taken = None;
    let stats = incremental_cycle_with(crate::generation::Generation::Gen2, || {
        taken = (**holder.borrow_mut()).take();
    });
    assert_eq!(stats.objects_collected, 0, "a live object was freed");
    assert_eq!(**taken.unwrap(), 7);
}

#[test]
fn partial_incremental_mark_keeps_young_objects_held_by_old_ones() {
    // Regression: a partial incremental mark discovered roots over the whole
    // heap, so the handle inside an old object did not make the young object
    // it points at a root; and the old object is out of scope, so nothing
    // reached the young one - it was freed while the old one held it.
    use crate::generation::Generation;
    clean_gc_state();
    let holder: GcCell<Option<Gc<i32>>> = GcCell::new(None);
    // Gen0 promotes after 3 survivals.
    for _ in 0..3 {
        LOCAL_GC.with(|gc| unsafe { gc.borrow().core.collect_generation(Generation::Gen0) });
    }
    **holder.borrow_mut() = Some(Gc::new(7));
    let stats = incremental_cycle_with(Generation::Gen0, || {});
    assert_eq!(
        stats.objects_collected, 0,
        "a young object an old one holds was freed"
    );
    assert_eq!(***(**holder.borrow()).as_ref().unwrap(), 7);
}

#[test]
fn weak_upgrade_during_incremental_mark_keeps_what_it_reaches() {
    // A weak upgrade can reach an object that was unreachable when the mark
    // began (a dead cycle not yet swept): the mark must keep it and what it
    // reaches.
    clean_gc_state();
    let a = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    let b = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(Some(a.clone())),
    });
    *a.next.borrow_mut() = Some(b.clone());
    let weak = Gc::downgrade(&a);
    drop(a);
    drop(b);
    let mut upgraded = None;
    let stats = incremental_cycle_with(crate::generation::Generation::Gen2, || {
        upgraded = weak.upgrade();
    });
    assert_eq!(stats.objects_collected, 0, "a resurrected object was freed");
    let a = upgraded.expect("the cycle is not swept yet");
    let b = a.next.borrow().clone().unwrap();
    assert!(b.next.borrow().is_some());
    drop(b);
    drop(a);
    clean_gc_state();
    assert!(!weak.is_alive(), "the dead cycle is freed later");
}

#[test]
fn allocation_inside_borrow_mut_scope_does_not_panic() {
    // Regression: the allocation trigger collected while the cell was
    // mutably borrowed, and tracing the cell panicked with "already mutably
    // borrowed". A promoted cell is traced by the trigger's Gen1/Gen2
    // collections, a young one by every collection.
    use crate::generation::Generation;
    clean_gc_state();
    LOCAL_GC.with(|gc| gc.borrow().core.alloc_threshold.store(8, Ordering::Relaxed));
    let old: GcCell<Vec<Gc<i32>>> = GcCell::new(Vec::new());
    // Gen0 promotes after 3 survivals.
    for _ in 0..3 {
        LOCAL_GC.with(|gc| unsafe { gc.borrow().core.collect_generation(Generation::Gen0) });
    }
    for i in 0..500 {
        let young: GcCell<Vec<Gc<i32>>> = GcCell::new(Vec::new());
        // A dropped handle whose object keeps another: garbage is possible,
        // so the trigger does not skip its collections.
        let x = Gc::new(i);
        drop(x.clone());
        old.borrow_mut().push(Gc::new(i));
        young.borrow_mut().push(Gc::new(i));
    }
    assert_eq!(old.borrow().len(), 500);
    assert_eq!(**old.borrow()[499], 499);
    LOCAL_GC.with(|gc| {
        gc.borrow()
            .core
            .alloc_threshold
            .store(super::LOCAL_GC_ALLOC_THRESHOLD, Ordering::Relaxed)
    });
}

/// A dead two-node cycle, and a weak reference to it.
fn dead_cycle() -> crate::GcWeak<CyclicNode> {
    let a = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(None),
    });
    let b = Gc::new(CyclicNode {
        next: std::cell::RefCell::new(Some(a.clone())),
    });
    *a.next.borrow_mut() = Some(b);
    Gc::downgrade(&a)
}

/// Holds a `GcCell`, so the cell is also reached through a handle in an
/// object, not only as a root.
struct CellHolder {
    cell: GcCell<Option<Gc<i32>>>,
}

impl Trace for CellHolder {
    fn is_root(&self) -> bool {
        false
    }
    fn reset_root(&self) {
        self.cell.reset_root();
    }
    fn trace(&self) {
        self.cell.trace();
    }
    fn reset(&self) {
        self.cell.reset();
    }
    fn is_traceable(&self) -> bool {
        false
    }
    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.cell.trace_children(children);
    }
}

impl Finalize for CellHolder {
    fn finalize(&self) {}
}

#[test]
fn collection_under_borrow_mut_frees_nothing_until_the_borrow_ends() {
    // While a cell is mutably borrowed the collector cannot see what it
    // holds, so that collection frees nothing (as sync::GcCell does); the
    // next one frees the garbage.
    use crate::generation::Generation;
    clean_gc_state();
    let cell: GcCell<Option<Gc<i32>>> = GcCell::new(Some(Gc::new(7)));
    let holder = Gc::new(CellHolder { cell: cell.clone() });
    let weak = dead_cycle();
    let mut guard = cell.borrow_mut();
    let stats =
        LOCAL_GC.with(|gc| unsafe { gc.borrow().core.collect_generation(Generation::Gen2) });
    assert_eq!(
        stats.objects_collected, 0,
        "a collection with a blocked scan freed objects"
    );
    assert!(weak.is_alive());
    // The borrow stays usable across the collection.
    **guard = Some(Gc::new(8));
    drop(guard);
    let stats =
        LOCAL_GC.with(|gc| unsafe { gc.borrow().core.collect_generation(Generation::Gen2) });
    assert_eq!(stats.objects_collected, 2);
    assert!(!weak.is_alive());
    assert_eq!(***(**holder.cell.borrow()).as_ref().unwrap(), 8);
}

#[test]
fn incremental_mark_begun_under_borrow_mut_frees_nothing() {
    // The snapshot cannot read a mutably borrowed cell, so the cycle that
    // began then frees nothing.
    clean_gc_state();
    let cell: GcCell<Option<Gc<i32>>> = GcCell::new(Some(Gc::new(7)));
    let weak = dead_cycle();
    let guard = cell.borrow_mut();
    LOCAL_GC.with(|gc| unsafe {
        gc.borrow()
            .core
            .begin_collection(crate::generation::Generation::Gen2)
    });
    drop(guard);
    let stats = LOCAL_GC.with(|gc| unsafe {
        let gc = gc.borrow();
        while !gc.core.mark_step(1) {}
        gc.core.finish_collection()
    });
    assert_eq!(stats.objects_collected, 0);
    assert!(weak.is_alive());
    clean_gc_state();
    assert!(!weak.is_alive());
    assert_eq!(***(**cell.borrow()).as_ref().unwrap(), 7);
}
