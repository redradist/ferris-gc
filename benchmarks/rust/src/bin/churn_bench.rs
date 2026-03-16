// Benchmark: alloc/drop churn — constantly create and discard objects.
//
// Usage: cargo run --release --bin churn_bench -- [N] [--strategy=...]

use ferris_gc::{Finalize, Gc, Trace, LOCAL_GC};
use ferris_gc_benchmarks::*;

struct ChurnNode {
    value: i32,
}

impl Trace for ChurnNode {
    fn is_root(&self) -> bool { false }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool { false }
}

impl Finalize for ChurnNode {
    fn finalize(&self) {}
}

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;
    let n = get_n();
    let strategy = configure_local_strategy(&get_strategy());
    install_local_gc_monitor();

    LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect() });

    let window_size = 1000;

    let (_, elapsed) = timed(|| {
        let mut window: Vec<Option<Gc<ChurnNode>>> = (0..window_size).map(|_| None).collect();

        for i in 0..n {
            window[i % window_size] = Some(Gc::new(ChurnNode { value: i as i32 }));
        }

        // Prevent optimization
        let sum: i64 = window
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|gc| (**gc).value as i64)
            .sum();
        std::hint::black_box(sum);
    });

    let stats = local_gc_stats();
    make_result("churn", &strategy, n, elapsed, &stats).print();
}
