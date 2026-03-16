// Benchmark: generational stress — many short-lived + few long-lived objects.
//
// Usage: cargo run --release --bin generational_bench -- [N] [--strategy=...]

use ferris_gc::{Finalize, Gc, Trace, LOCAL_GC};
use ferris_gc_benchmarks::*;

struct GenNode {
    value: i32,
}

impl Trace for GenNode {
    fn is_root(&self) -> bool { false }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool { false }
}

impl Finalize for GenNode {
    fn finalize(&self) {}
}

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;
    let n = get_n();
    let strategy = configure_local_strategy(&get_strategy());
    install_local_gc_monitor();

    LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect() });

    // Long-lived objects (survive the whole benchmark)
    let long_lived: Vec<Gc<GenNode>> = (0..1000)
        .map(|i| Gc::new(GenNode { value: i }))
        .collect();

    let (total_allocs, elapsed) = timed(|| {
        let mut total_allocs = 0usize;

        // Create waves of short-lived objects
        for wave in 0..(n / 1000) {
            let batch: Vec<Gc<GenNode>> = (0..1000)
                .map(|i| {
                    total_allocs += 1;
                    Gc::new(GenNode { value: (wave * 1000 + i) as i32 })
                })
                .collect();

            // Read to prevent optimization
            let sum: i64 = batch.iter().map(|gc| (**gc).value as i64).sum();
            std::hint::black_box(sum);
            // batch drops here — short-lived garbage
        }

        total_allocs
    });

    // Verify long-lived objects are still alive
    let sum: i64 = long_lived.iter().map(|gc| (**gc).value as i64).sum();
    std::hint::black_box(sum);

    let stats = local_gc_stats();
    make_result("generational", &strategy, total_allocs, elapsed, &stats).print();
}
