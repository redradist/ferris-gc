//! Demonstrates `#[ferris_gc_main]` with strategy configuration, promotion
//! tuning, monitoring callbacks, and region-based allocation.
//!
//! Run with:  cargo run --example macro_strategies --features proc-macro

use ferris_gc::sync;
use ferris_gc::{Finalize, Gc, GcCell, LOCAL_GC, LocalRegionId, PromotionConfig, Trace};

struct Node {
    value: i32,
    next: GcCell<Option<Gc<Node>>>,
}

impl Finalize for Node {
    fn finalize(&self) {}
}

impl Trace for Node {
    fn is_root(&self) -> bool {
        self.next.is_root()
    }
    fn reset_root(&self) {
        self.next.reset_root()
    }
    fn trace(&self) {
        self.next.trace()
    }
    fn reset(&self) {
        self.next.reset()
    }
    fn is_traceable(&self) -> bool {
        self.next.is_traceable()
    }
    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.next.trace_children(children);
    }
}

// The macro injects ApplicationCleanup and sets both local + global GC
// to the adaptive auto-tuning strategy. No manual setup needed.
#[ferris_gc::ferris_gc_main(strategy = "adaptive")]
fn main() {
    println!("=== Strategy: adaptive (set by #[ferris_gc_main]) ===\n");

    // --- Promotion config ---
    // Works with any strategy — tune how quickly objects get promoted.
    LOCAL_GC.with(|gc| {
        gc.borrow().set_promotion_config(PromotionConfig {
            gen0_threshold: 3,
            gen1_threshold: 5,
        });
    });
    sync::GLOBAL_GC.set_promotion_config(PromotionConfig {
        gen0_threshold: 3,
        gen1_threshold: 5,
    });
    println!("Promotion config set for both local and global GC");

    // --- Monitoring callback ---
    // Fires after every collection — useful for logging / metrics.
    LOCAL_GC.with(|gc| {
        gc.borrow().set_on_collection(|stats| {
            println!(
                "  [local GC] collected {} objects, freed {} bytes in {:?}",
                stats.objects_collected, stats.bytes_freed, stats.duration,
            );
        });
    });
    sync::GLOBAL_GC.set_on_collection(|stats| {
        println!(
            "  [global GC] collected {} objects, freed {} bytes in {:?}",
            stats.objects_collected, stats.bytes_freed, stats.duration,
        );
    });
    println!("Monitoring callbacks installed\n");

    // --- Thread-local GC: region-based allocation ---
    println!("--- Thread-local: region-based allocation ---");
    let region: LocalRegionId = LOCAL_GC.with(|gc| gc.borrow().new_region());
    println!("Created local region (id={})", region.id());

    // Allocate in specific region using the helper
    let a = region.gc(Node {
        value: 1,
        next: GcCell::new(None),
    });
    let b = region.gc(Node {
        value: 2,
        next: GcCell::new(None),
    });
    // Create a cycle
    **a.next.borrow_mut() = Some(b.clone());
    **b.next.borrow_mut() = Some(a.clone());
    println!("Created cycle in region {}", region.id());

    drop(a);
    drop(b);
    println!("Dropped cycle — the strategy will collect it\n");

    // --- Global GC: sync region-based allocation ---
    println!("--- Global: sync region-based allocation ---");
    let sync_region = sync::GLOBAL_GC.new_region();
    println!("Created sync region (id={})", sync_region.id());

    let _obj1 = sync_region.gc(42);
    let _obj2 = sync_region.gc(99);
    println!("Allocated 2 objects in sync region {}", sync_region.id());

    // --- Thread-local: allocate objects to trigger strategy-based collection ---
    println!("\n--- Triggering collection via allocation ---");
    let mut objects: Vec<Gc<Node>> = Vec::new();
    for i in 0..1_000 {
        objects.push(Gc::new(Node {
            value: i,
            next: GcCell::new(None),
        }));
    }
    let _alive: Vec<_> = objects.drain(..500).collect();
    drop(objects);

    // Allocate more to trigger the strategy's threshold
    for i in 0..2_000 {
        let _ = Gc::new(Node {
            value: i,
            next: GcCell::new(None),
        });
    }

    // --- Stats ---
    println!("\n--- Diagnostics ---");
    LOCAL_GC.with(|gc| {
        let stats = gc.borrow().stats();
        println!(
            "Local  GC: {} live objects, {} bytes heap, peak {} bytes, {} collections",
            stats.live_objects, stats.heap_size, stats.peak_heap_size, stats.total_collections,
        );
    });
    let stats = sync::GLOBAL_GC.stats();
    println!(
        "Global GC: {} live objects, {} bytes heap, peak {} bytes, {} collections",
        stats.live_objects, stats.heap_size, stats.peak_heap_size, stats.total_collections,
    );
}
