use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ferris_gc::Gc;
use ferris_gc::sync;

// ---------------------------------------------------------------------------
// Thread-local Gc<T> benchmarks
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
// Thread-safe sync::Gc<T> benchmarks
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

criterion_group!(
    benches,
    bench_alloc,
    bench_alloc_drop,
    bench_alloc_deref,
    bench_alloc_varying_sizes,
    bench_alloc_drop_cycles,
    bench_sync_alloc,
    bench_sync_alloc_drop,
    bench_sync_alloc_deref,
);
criterion_main!(benches);
