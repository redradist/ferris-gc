//! Demonstrates strategy configuration and GC diagnostics.
//!
//! Shows how to switch collection strategies and monitor GC behavior
//! without manual collection calls.

use ferris_gc::{Finalize, Gc, Trace};

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

    // Disable the default basic strategy and use threshold-based collection
    ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);
    ferris_gc::LOCAL_GC_STRATEGY.with(|s| {
        let strategy = ferris_gc::threshold_local_strategy(ferris_gc::ThresholdConfig::default());
        s.borrow().set_strategy(strategy);
    });

    // Monitor collection events
    ferris_gc::LOCAL_GC.with(|gc| {
        gc.borrow().set_on_collection(|stats| {
            println!(
                "  [GC] collected {} objects, freed {} bytes in {:?}",
                stats.objects_collected, stats.bytes_freed, stats.duration,
            );
        });
    });

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
    println!("Dropped 5,000 objects — strategy will collect them automatically");

    // Allocate more to trigger threshold-based collection
    for _ in 0..2_000 {
        let _ = Gc::new(Payload {
            _data: vec![0u8; 64],
        });
    }

    // Check stats
    ferris_gc::LOCAL_GC.with(|gc| {
        let stats = gc.borrow().stats();
        println!(
            "\nGC Stats: {} live objects, {} bytes heap, {} collections",
            stats.live_objects, stats.heap_size, stats.total_collections
        );
    });
}
