//! Thread-safe garbage collection with sync::Gc<T>.
//!
//! Demonstrates global GC configuration: custom strategy, promotion config,
//! monitoring callbacks, and multi-threaded usage.

use ferris_gc::sync;
use ferris_gc::{Finalize, PromotionConfig, Trace};
use std::thread;

struct SharedData {
    value: i32,
}

impl Finalize for SharedData {
    fn finalize(&self) {}
}

impl Trace for SharedData {
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

    // --- Strategy configuration ---
    // Disable the default basic strategy and use threshold-based collection.
    ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);
    {
        let strategy = ferris_gc::threshold_global_strategy(ferris_gc::ThresholdConfig::default());
        sync::GLOBAL_GC_STRATEGY.set_strategy(strategy);
    }

    // --- Promotion config ---
    // Tune how many collections an object must survive before promotion.
    sync::GLOBAL_GC.set_promotion_config(PromotionConfig {
        gen0_threshold: 3, // survive 3 Gen0 collections → promote to Gen1
        gen1_threshold: 5, // survive 5 Gen1 collections → promote to Gen2
    });
    println!("Promotion config: {:?}", sync::GLOBAL_GC.promotion_config());

    // --- Monitoring callback ---
    sync::GLOBAL_GC.set_on_collection(|stats| {
        println!(
            "  [global GC] collected {} objects, freed {} bytes in {:?}",
            stats.objects_collected, stats.bytes_freed, stats.duration,
        );
    });

    // --- Multi-threaded allocation ---
    let data = sync::Gc::new(SharedData { value: 42 });

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let data = data.clone();
            thread::spawn(move || {
                // Each thread can also allocate into the global GC
                let local = sync::Gc::new(SharedData { value: i * 10 });
                println!("Thread {i}: shared={}, local={}", data.value, local.value);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Allocate more objects to trigger collection via the strategy
    let mut objects: Vec<sync::Gc<SharedData>> = Vec::new();
    for i in 0..1_000 {
        objects.push(sync::Gc::new(SharedData { value: i }));
    }
    // Keep half alive
    let _alive: Vec<_> = objects.drain(..500).collect();
    drop(objects);

    // --- Stats ---
    let stats = sync::GLOBAL_GC.stats();
    println!(
        "\nGC Stats: {} live objects, {} bytes heap, {} collections",
        stats.live_objects, stats.heap_size, stats.total_collections
    );
}
