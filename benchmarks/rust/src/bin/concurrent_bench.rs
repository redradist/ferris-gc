// Benchmark: concurrent allocation from multiple threads using sync::Gc.
//
// Usage: cargo run --release --bin concurrent_bench -- [N] [--strategy=...]

use ferris_gc::sync;
use ferris_gc::{Finalize, Trace};
use ferris_gc_benchmarks::*;
use std::thread;

struct ConcNode {
    value: i32,
}

impl Trace for ConcNode {
    fn is_root(&self) -> bool { false }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool { false }
}

impl Finalize for ConcNode {
    fn finalize(&self) {}
}

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;
    let n = get_n();
    let strategy = configure_sync_strategy(&get_strategy());
    install_sync_gc_monitor();

    let num_workers = thread::available_parallelism()
        .map(|p| p.get().min(8))
        .unwrap_or(4);
    let per_worker = n / num_workers;

    unsafe { sync::GLOBAL_GC._collect() };

    let (_, elapsed) = timed(|| {
        let handles: Vec<_> = (0..num_workers)
            .map(|w| {
                thread::spawn(move || {
                    let objects: Vec<sync::Gc<ConcNode>> = (0..per_worker)
                        .map(|i| sync::Gc::new(ConcNode { value: (w * per_worker + i) as i32 }))
                        .collect();

                    // Read to prevent optimization
                    let sum: i64 = objects.iter().map(|gc| (**gc).value as i64).sum();
                    std::hint::black_box(sum);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    });

    let total_allocs = num_workers * per_worker;
    let stats = sync_gc_stats();
    make_result("concurrent", &strategy, total_allocs, elapsed, &stats).print();
}
