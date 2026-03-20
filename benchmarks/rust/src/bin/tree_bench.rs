// Benchmark: build a binary tree of depth D, then drop and collect.
//
// Usage: cargo run --release --bin tree_bench -- [depth] [--strategy=...]

use ferris_gc::{Finalize, Gc, Trace, LOCAL_GC};
use ferris_gc_benchmarks::*;
use std::time::Instant;

struct TreeNode {
    value: i32,
    left: Option<Gc<TreeNode>>,
    right: Option<Gc<TreeNode>>,
}

impl Trace for TreeNode {
    fn is_root(&self) -> bool { false }
    fn reset_root(&self) {
        if let Some(ref gc) = self.left { gc.reset_root(); }
        if let Some(ref gc) = self.right { gc.reset_root(); }
    }
    fn trace(&self) {
        if let Some(ref gc) = self.left { gc.trace(); }
        if let Some(ref gc) = self.right { gc.trace(); }
    }
    fn reset(&self) {
        if let Some(ref gc) = self.left { gc.reset(); }
        if let Some(ref gc) = self.right { gc.reset(); }
    }
    fn is_traceable(&self) -> bool { false }
}

impl Finalize for TreeNode {
    fn finalize(&self) {}
}

fn build_tree(depth: u32, value: i32) -> Gc<TreeNode> {
    if depth == 0 {
        return Gc::new(TreeNode { value, left: None, right: None });
    }
    let left = build_tree(depth - 1, 2 * value);
    let right = build_tree(depth - 1, 2 * value + 1);
    Gc::new(TreeNode {
        value,
        left: Some(left),
        right: Some(right),
    })
}

fn count_nodes(node: &Gc<TreeNode>) -> usize {
    let n = &**node;
    1 + n.left.as_ref().map_or(0, count_nodes)
      + n.right.as_ref().map_or(0, count_nodes)
}

fn sum_tree(node: &Gc<TreeNode>) -> i64 {
    let n = &**node;
    n.value as i64
        + n.left.as_ref().map_or(0, sum_tree)
        + n.right.as_ref().map_or(0, sum_tree)
}

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;
    // Default depth=20 (1M nodes), matching Go benchmark. Overridable via CLI arg.
    let depth: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20);
    let strategy = configure_local_strategy(&get_strategy());
    install_local_gc_monitor();

    LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect() });

    // Phase 1: Build tree
    let (root, build_time) = timed(|| build_tree(depth, 1));

    let node_count = count_nodes(&root);
    let tree_sum = sum_tree(&root);
    std::hint::black_box(tree_sum);

    // Phase 2: Drop tree and force GC
    drop(root);
    let gc_start = Instant::now();
    LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect() });
    let gc_time = gc_start.elapsed();

    let total_time = build_time + gc_time;

    eprintln!("Tree depth={depth}, nodes={node_count}");
    eprintln!("Build time: {:.2} ms", build_time.as_secs_f64() * 1000.0);
    eprintln!("GC time:    {:.2} ms", gc_time.as_secs_f64() * 1000.0);

    let stats = local_gc_stats();
    make_result("tree", &strategy, node_count, total_time, &stats).print();
}
