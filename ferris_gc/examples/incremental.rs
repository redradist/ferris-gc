//! Incremental and time-budgeted garbage collection.

use ferris_gc::{Finalize, Gc, Generation, Trace};
use std::time::{Duration, Instant};

struct Payload {
    _data: Vec<u8>,
}

impl Finalize for Payload {
    fn finalize(&self) {}
}

impl Trace for Payload {
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

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;

    // Allocate many objects
    let mut objects: Vec<Gc<Payload>> = Vec::new();
    for _ in 0..10_000 {
        objects.push(Gc::new(Payload {
            _data: vec![0u8; 64],
        }));
    }
    println!("Allocated 10,000 objects");

    // Keep half alive
    let _alive: Vec<_> = objects.drain(..5_000).collect();
    drop(objects);

    ferris_gc::LOCAL_GC.with(|gc| {
        let gc_ref = gc.borrow();

        // Incremental collection with object-count budget
        let start = Instant::now();
        unsafe {
            let stats = gc_ref.collect_incremental(Generation::Gen2, 500);
            println!(
                "Incremental (budget=500 objects): collected {} in {:?}",
                stats.objects_collected,
                start.elapsed()
            );
        }

        // Time-budgeted collection: max 1ms per mark step
        let start = Instant::now();
        unsafe {
            let stats =
                gc_ref.collect_incremental_timed(Generation::Gen2, Duration::from_millis(1));
            println!(
                "Time-budgeted (max 1ms/step): scanned {} in {:?}",
                stats.objects_scanned,
                start.elapsed()
            );
        }

        // Concurrent collection: mark without STW
        let start = Instant::now();
        unsafe {
            let stats = gc_ref.collect_concurrent(Generation::Gen2, 1000);
            println!(
                "Concurrent (budget=1000): scanned {} in {:?}",
                stats.objects_scanned,
                start.elapsed()
            );
        }

        // Check stats
        let stats = gc_ref.stats();
        println!(
            "\nGC Stats: {} live objects, {} bytes heap, {} collections",
            stats.live_objects, stats.heap_size, stats.total_collections
        );
    });
}
