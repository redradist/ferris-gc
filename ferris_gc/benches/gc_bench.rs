use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ferris_gc::Gc;
use ferris_gc::sync;
use ferris_gc::{Generation, LOCAL_GC};

// ---------------------------------------------------------------------------
// Thread-local Gc<T> benchmarks (small)
// ---------------------------------------------------------------------------

fn bench_alloc(c: &mut Criterion) {
    c.bench_function("local/alloc_1000", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(1000);
            for i in 0..1000i32 {
                v.push(Gc::new(black_box(i)));
            }
            black_box(&v);
        });
    });
}

fn bench_alloc_drop(c: &mut Criterion) {
    c.bench_function("local/alloc_drop_1000", |b| {
        b.iter(|| {
            for i in 0..1000i32 {
                let _ = Gc::new(black_box(i));
            }
        });
    });
}

fn bench_alloc_deref(c: &mut Criterion) {
    c.bench_function("local/alloc_deref_1000", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(1000);
            for i in 0..1000i32 {
                v.push(Gc::new(black_box(i)));
            }
            let mut sum = 0i64;
            for gc in &v {
                sum += ***gc as i64;
            }
            black_box(sum);
        });
    });
}

fn bench_alloc_varying_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("local/alloc_varying");
    for size in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let mut v = Vec::with_capacity(n);
                for i in 0..n as i32 {
                    v.push(Gc::new(black_box(i)));
                }
                black_box(&v);
            });
        });
    }
    group.finish();
}

fn bench_alloc_drop_cycles(c: &mut Criterion) {
    c.bench_function("local/alloc_drop_cycles_100x100", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let mut batch = Vec::with_capacity(100);
                for i in 0..100i32 {
                    batch.push(Gc::new(black_box(i)));
                }
                drop(batch);
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Thread-safe sync::Gc<T> benchmarks (small)
// ---------------------------------------------------------------------------

fn bench_sync_alloc(c: &mut Criterion) {
    c.bench_function("sync/alloc_1000", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(1000);
            for i in 0..1000i32 {
                v.push(sync::Gc::new(black_box(i)));
            }
            black_box(&v);
        });
    });
}

fn bench_sync_alloc_drop(c: &mut Criterion) {
    c.bench_function("sync/alloc_drop_1000", |b| {
        b.iter(|| {
            for i in 0..1000i32 {
                let _ = sync::Gc::new(black_box(i));
            }
        });
    });
}

fn bench_sync_alloc_deref(c: &mut Criterion) {
    c.bench_function("sync/alloc_deref_1000", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(1000);
            for i in 0..1000i32 {
                v.push(sync::Gc::new(black_box(i)));
            }
            let mut sum = 0i64;
            for gc in &v {
                sum += ***gc as i64;
            }
            black_box(sum);
        });
    });
}

// ---------------------------------------------------------------------------
// Large-scale thread-local Gc<T> benchmarks
// ---------------------------------------------------------------------------

/// Allocate 1,000,000 Gc<i32> objects into a Vec, then drop.
/// Measures raw allocation throughput at scale.
fn bench_alloc_1m(c: &mut Criterion) {
    c.bench_function("local/alloc_1m", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(1_000_000);
            for i in 0..1_000_000i32 {
                v.push(Gc::new(black_box(i)));
            }
            black_box(&v);
        });
    });
}

/// Allocate and immediately drop 1,000,000 objects one at a time.
fn bench_alloc_drop_1m(c: &mut Criterion) {
    c.bench_function("local/alloc_drop_1m", |b| {
        b.iter(|| {
            for i in 0..1_000_000i32 {
                let _ = Gc::new(black_box(i));
            }
        });
    });
}

/// Allocate 100,000 objects, drop half (even indices), then run collect and
/// measure the collection time only.
fn bench_collection_100k(c: &mut Criterion) {
    let n = 100_000usize;
    c.bench_function("local/collection_100k", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                // Setup: allocate objects
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(Gc::new(i as i32));
                }
                // Drop half to create garbage
                for i in (0..n).step_by(2) {
                    v[i] = Gc::new(0);
                }
                // Measure only collection
                let start = std::time::Instant::now();
                LOCAL_GC.with(|gc| unsafe { gc.borrow_mut()._collect() });
                total += start.elapsed();
                drop(v);
                // Clean up for next iteration
                LOCAL_GC.with(|gc| unsafe { gc.borrow_mut()._collect() });
            }
            total
        });
    });
}

/// Allocate 1,000,000 objects, drop half (even indices), then run collect and
/// measure the collection time only.
fn bench_collection_1m(c: &mut Criterion) {
    let n = 1_000_000usize;
    c.bench_function("local/collection_1m", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                // Setup: allocate objects
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(Gc::new(i as i32));
                }
                // Drop half to create garbage
                for i in (0..n).step_by(2) {
                    v[i] = Gc::new(0);
                }
                // Measure only collection
                let start = std::time::Instant::now();
                LOCAL_GC.with(|gc| unsafe { gc.borrow_mut()._collect() });
                total += start.elapsed();
                drop(v);
                // Clean up for next iteration
                LOCAL_GC.with(|gc| unsafe { gc.borrow_mut()._collect() });
            }
            total
        });
    });
}

/// Allocate 100,000 objects, drop half, run collect_incremental with
/// step_budget=1000, measure total time.
fn bench_incremental_collection_100k(c: &mut Criterion) {
    let n = 100_000usize;
    c.bench_function("local/incremental_collection_100k", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                // Setup: allocate objects
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(Gc::new(i as i32));
                }
                // Drop half to create garbage
                for i in (0..n).step_by(2) {
                    v[i] = Gc::new(0);
                }
                // Measure only incremental collection
                let start = std::time::Instant::now();
                LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut()._collect_incremental(Generation::Gen2, 1000)
                });
                total += start.elapsed();
                drop(v);
                // Clean up for next iteration
                LOCAL_GC.with(|gc| unsafe { gc.borrow_mut()._collect() });
            }
            total
        });
    });
}

// ---------------------------------------------------------------------------
// Large-scale sync::Gc<T> benchmarks
// ---------------------------------------------------------------------------

/// Allocate 1,000,000 sync::Gc<i32> objects into a Vec, then drop.
/// Measures raw allocation throughput at scale for the thread-safe collector.
fn bench_sync_alloc_1m(c: &mut Criterion) {
    c.bench_function("sync/alloc_1m", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(1_000_000);
            for i in 0..1_000_000i32 {
                v.push(sync::Gc::new(black_box(i)));
            }
            black_box(&v);
        });
    });
}

/// Allocate 100,000 sync objects, drop half (even indices), then run collect
/// on the global GC and measure the collection time only.
fn bench_sync_collection_100k(c: &mut Criterion) {
    let n = 100_000usize;
    c.bench_function("sync/collection_100k", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                // Setup: allocate objects
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(sync::Gc::new(i as i32));
                }
                // Drop half to create garbage
                for i in (0..n).step_by(2) {
                    v[i] = sync::Gc::new(0);
                }
                // Measure only collection
                let start = std::time::Instant::now();
                unsafe { sync::GLOBAL_GC._collect() };
                total += start.elapsed();
                drop(v);
                // Clean up for next iteration
                unsafe { sync::GLOBAL_GC._collect() };
            }
            total
        });
    });
}

// ---------------------------------------------------------------------------
// Parameterized large-scale allocation benchmark
// ---------------------------------------------------------------------------

/// Parameterized benchmark with sizes [10_000, 100_000, 500_000, 1_000_000].
fn bench_alloc_varying_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("local/alloc_varying_large");
    // Use reduced sample size and longer measurement time for large sizes.
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(30));
    for size in [10_000, 100_000, 500_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let mut v = Vec::with_capacity(n);
                for i in 0..n as i32 {
                    v.push(Gc::new(black_box(i)));
                }
                black_box(&v);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Compaction & generational benchmarks
// ---------------------------------------------------------------------------

/// Allocate 10,000 objects, compact, and measure compaction time.
fn bench_compact_10k(c: &mut Criterion) {
    let _cleanup = ferris_gc::ApplicationCleanup;
    let n = 10_000usize;
    c.bench_function("local/compact_10k", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(Gc::new(i as i32));
                }
                // Drop half to create fragmentation
                for i in (0..n).step_by(2) {
                    v[i] = Gc::new(0);
                }
                let start = std::time::Instant::now();
                LOCAL_GC.with(|gc| unsafe { gc.borrow()._compact() });
                total += start.elapsed();
                drop(v);
                LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect() });
            }
            total
        });
    });
}

/// Compare Gen0-only vs full collection on 50,000 objects.
fn bench_gen0_vs_full_50k(c: &mut Criterion) {
    let n = 50_000usize;
    let mut group = c.benchmark_group("local/gen_comparison_50k");
    group.sample_size(10);

    group.bench_function("gen0_only", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(Gc::new(i as i32));
                }
                for i in (0..n).step_by(2) {
                    v[i] = Gc::new(0);
                }
                let start = std::time::Instant::now();
                LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect_generation(Generation::Gen0) });
                total += start.elapsed();
                drop(v);
                LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect() });
            }
            total
        });
    });

    group.bench_function("full_gen2", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(Gc::new(i as i32));
                }
                for i in (0..n).step_by(2) {
                    v[i] = Gc::new(0);
                }
                let start = std::time::Instant::now();
                LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect_generation(Generation::Gen2) });
                total += start.elapsed();
                drop(v);
                LOCAL_GC.with(|gc| unsafe { gc.borrow()._collect() });
            }
            total
        });
    });

    group.finish();
}

/// Benchmark deref throughput after compaction vs before.
fn bench_deref_after_compact(c: &mut Criterion) {
    let n = 10_000usize;
    let mut group = c.benchmark_group("local/deref_compact");

    group.bench_function("before_compact", |b| {
        let v: Vec<Gc<i32>> = (0..n as i32).map(|i| Gc::new(i)).collect();
        b.iter(|| {
            let mut sum = 0i64;
            for gc in &v {
                sum += ***gc as i64;
            }
            black_box(sum);
        });
    });

    group.bench_function("after_compact", |b| {
        let v: Vec<Gc<i32>> = (0..n as i32).map(|i| Gc::new(i)).collect();
        LOCAL_GC.with(|gc| unsafe { gc.borrow()._compact() });
        b.iter(|| {
            let mut sum = 0i64;
            for gc in &v {
                sum += ***gc as i64;
            }
            black_box(sum);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    small_benches,
    bench_alloc,
    bench_alloc_drop,
    bench_alloc_deref,
    bench_alloc_varying_sizes,
    bench_alloc_drop_cycles,
    bench_sync_alloc,
    bench_sync_alloc_drop,
    bench_sync_alloc_deref,
);

criterion_group!(
    name = large_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(30));
    targets =
        bench_alloc_1m,
        bench_alloc_drop_1m,
        bench_collection_100k,
        bench_collection_1m,
        bench_incremental_collection_100k,
        bench_sync_alloc_1m,
        bench_sync_collection_100k,
        bench_alloc_varying_large,
        bench_compact_10k,
        bench_gen0_vs_full_50k,
        bench_deref_after_compact,
);

criterion_main!(small_benches, large_benches);
