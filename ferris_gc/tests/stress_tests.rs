use ferris_gc::Gc;
use ferris_gc::sync;

// ---------------------------------------------------------------------------
// Thread-local stress tests
// ---------------------------------------------------------------------------

#[test]
fn stress_100k_objects() {
    let mut objects: Vec<Gc<i32>> = Vec::with_capacity(100_000);
    for i in 0..100_000i32 {
        objects.push(Gc::new(i));
    }
    // Verify all objects are accessible and hold the expected value
    for (i, gc) in objects.iter().enumerate() {
        assert_eq!(***gc, i as i32);
    }
    drop(objects);
}

#[test]
fn stress_alloc_drop_cycles() {
    for cycle in 0..1_000 {
        let mut batch: Vec<Gc<i32>> = Vec::with_capacity(100);
        for i in 0..100i32 {
            batch.push(Gc::new(i));
        }
        // Spot-check a few values each cycle
        assert_eq!(***batch.first().unwrap(), 0);
        assert_eq!(***batch.last().unwrap(), 99);
        let _ = cycle;
        drop(batch);
    }
}

#[test]
fn stress_large_objects() {
    // Allocate GC-managed vectors (larger heap objects)
    let mut objects: Vec<Gc<Vec<u8>>> = Vec::with_capacity(1_000);
    for i in 0..1_000 {
        objects.push(Gc::new(vec![i as u8; 1024]));
    }
    for (i, gc) in objects.iter().enumerate() {
        assert_eq!((***gc).len(), 1024);
        assert_eq!((***gc)[0], i as u8);
    }
    drop(objects);
}

#[test]
fn stress_rapid_create_and_forget() {
    // Create objects and immediately let them go out of scope
    for i in 0..50_000i32 {
        let _gc = Gc::new(i);
    }
}

// ---------------------------------------------------------------------------
// Thread-safe (sync) stress tests
// ---------------------------------------------------------------------------

#[test]
fn stress_concurrent_sync_alloc() {
    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            std::thread::spawn(move || {
                let mut objects: Vec<sync::Gc<i32>> = Vec::with_capacity(10_000);
                for i in 0..10_000i32 {
                    objects.push(sync::Gc::new(thread_id * 10_000 + i));
                }
                // Verify every object holds the correct value
                for (i, gc) in objects.iter().enumerate() {
                    assert_eq!(***gc, thread_id * 10_000 + i as i32);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked during concurrent alloc stress test");
    }
}

#[test]
fn stress_sync_alloc_drop_cycles() {
    for _ in 0..500 {
        let mut batch: Vec<sync::Gc<i32>> = Vec::with_capacity(100);
        for i in 0..100i32 {
            batch.push(sync::Gc::new(i));
        }
        assert_eq!(***batch.first().unwrap(), 0);
        assert_eq!(***batch.last().unwrap(), 99);
        drop(batch);
    }
}

#[test]
fn stress_concurrent_create_drop() {
    // Multiple threads rapidly creating and dropping objects
    let handles: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..1_000 {
                    let mut batch: Vec<sync::Gc<i32>> = Vec::with_capacity(50);
                    for i in 0..50i32 {
                        batch.push(sync::Gc::new(i));
                    }
                    drop(batch);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked during concurrent create/drop stress test");
    }
}

#[test]
fn stress_shared_across_threads() {
    // Create objects on one thread, share across many reader threads
    let objects: Vec<sync::Gc<i32>> = (0..1_000i32)
        .map(|i| sync::Gc::new(i))
        .collect();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let cloned: Vec<sync::Gc<i32>> = objects.clone();
            std::thread::spawn(move || {
                for (i, gc) in cloned.iter().enumerate() {
                    assert_eq!(***gc, i as i32);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked during shared read stress test");
    }
}
