// Benchmark: mass allocation of N objects, measure time, memory, GC pauses.
//
// Usage: cargo run --release --bin alloc_bench -- [N] [--strategy=none|basic|threshold|adaptive|background|g1]

use ferris_gc::{Finalize, Gc, Trace, LOCAL_GC};
use ferris_gc_benchmarks::*;

struct Node {
    value: i32,
    _next: Option<Gc<Node>>,
}

impl Trace for Node {
    fn is_root(&self) -> bool { false }
    fn reset_root(&self) {}
    fn trace(&self) { if let Some(ref gc) = self._next { gc.trace(); } }
    fn reset(&self) { if let Some(ref gc) = self._next { gc.reset(); } }
    fn is_traceable(&self) -> bool { false }
}

impl Finalize for Node {
    fn finalize(&self) {}
}

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;
    let n = get_n();
    let strategy = configure_local_strategy(&get_strategy());
    install_local_gc_monitor();

    // Force initial GC
    LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect() });

    let (objects, elapsed) = timed(|| {
        let mut objects = Vec::with_capacity(n);
        for i in 0..n as i32 {
            objects.push(Gc::new(Node { value: i, _next: None }));
        }
        objects
    });

    // Prevent dead-code elimination
    let sum: i64 = objects.iter().map(|obj| (**obj).value as i64).sum();
    std::hint::black_box(sum);

    let stats = local_gc_stats();
    make_result("alloc", &strategy, n, elapsed, &stats).print();
}
