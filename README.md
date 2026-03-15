<a href="https://www.buymeacoffee.com/redradist" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

# FerrisGC

A production-grade garbage collection library for Rust with both thread-local and thread-safe collectors.

## Features

- **Generational collection** (Gen0/Gen1/Gen2) with configurable promotion thresholds
- **Incremental marking** with object-count or time-budgeted steps for predictable pause times
- **Concurrent marking** (snapshot-based, no STW during mark phase)
- **Region-based collection** for scoped memory management
- **Cycle detection** via tri-color mark-sweep
- **RC hybrid** — immediate deallocation for non-cyclic objects
- **Weak references** (`GcWeak<T>`)
- **Thread-safe GC** (`sync::Gc<T>`) for cross-thread sharing
- **OOM resilience** — emergency GC collection on allocation failure with automatic retry
- **Panic-safe** deallocation (finalizers wrapped in `catch_unwind`)
- **Derive macros** (`#[derive(Trace, Finalize)]`) with `#[unsafe_ignore_trace]`
- **`no_std` support** — core traits available without the allocator

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ferris-gc = { version = "0.2.0", features = ["proc-macro"] }
```

### Basic Usage

```rust
use ferris_gc::{Gc, GcCell, Trace, Finalize, ApplicationCleanup};

#[derive(Trace, Finalize)]
struct Node {
    value: i32,
    next: GcCell<Option<Gc<Node>>>,
}

#[ferris_gc::ferris_gc_main]
fn main() {
    // Thread-local GC
    let a = Gc::new(Node { value: 1, next: GcCell::new(None) });
    let b = Gc::new(Node { value: 2, next: GcCell::new(Some(a.clone())) });

    // Create a cycle — the GC will detect and collect it
    *a.next.borrow_mut() = Some(b.clone());
}
```

### Thread-Safe GC

```rust
use ferris_gc::sync;
use ferris_gc::{Trace, Finalize};
use std::thread;

#[derive(Trace, Finalize)]
struct Data { value: i32 }

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;
    let data = sync::Gc::new(Data { value: 42 });

    let handles: Vec<_> = (0..4).map(|_| {
        let data = data.clone();
        thread::spawn(move || println!("{}", data.value))
    }).collect();

    for h in handles { h.join().unwrap(); }
}
```

### Incremental Collection with Time Budget

**Thread-local:**

```rust
use ferris_gc::{Gc, Generation, LOCAL_GC};
use std::time::Duration;

// Allocate many objects...
let objects: Vec<Gc<i32>> = (0..100_000).map(|i| Gc::new(i)).collect();

// Collect with max 1ms pause per mark step
LOCAL_GC.with(|gc| unsafe {
    gc.borrow().collect_incremental_timed(
        Generation::Gen2,
        Duration::from_millis(1),
    );
});
```

**Global (thread-safe):**

```rust
use ferris_gc::{sync, Generation};
use std::time::Duration;

// Collect with max 1ms pause per mark step
unsafe {
    sync::GLOBAL_GC.collect_incremental_timed(
        Generation::Gen2,
        Duration::from_millis(1),
    );
}
```

### Configurable Promotion

**Thread-local:**

```rust
use ferris_gc::{PromotionConfig, LOCAL_GC};

LOCAL_GC.with(|gc| {
    gc.borrow().set_promotion_config(PromotionConfig {
        gen0_threshold: 5,  // survive 5 Gen0 collections before promotion
        gen1_threshold: 10, // survive 10 Gen1 collections before promotion
    });
});
```

**Global (thread-safe):**

```rust
use ferris_gc::{sync, PromotionConfig};

sync::GLOBAL_GC.set_promotion_config(PromotionConfig {
    gen0_threshold: 5,
    gen1_threshold: 10,
});
```

### Strategy Configuration via `#[ferris_gc_main]`

The `#[ferris_gc_main]` attribute macro sets up `ApplicationCleanup` and configures
the collection strategy for **both** thread-local and global GCs automatically:

```rust
use ferris_gc::{Gc, Trace, Finalize};

#[derive(Trace, Finalize)]
struct Data { value: i32 }

// Default: basic periodic strategy (background thread every 50ms)
#[ferris_gc::ferris_gc_main]
fn main() {
    let _obj = Gc::new(Data { value: 1 });
}
```

```rust
// Threshold strategy: collect when allocation count exceeds a threshold
#[ferris_gc::ferris_gc_main(strategy = "threshold")]
fn main() {
    let _obj = Gc::new(Data { value: 1 });
}
```

```rust
// Adaptive strategy: auto-tunes collection frequency based on allocation rate
#[ferris_gc::ferris_gc_main(strategy = "adaptive")]
fn main() {
    let _obj = Gc::new(Data { value: 1 });
}
```

```rust
// Background GC (.NET-style): foreground Gen0/Gen1, concurrent background Gen2
#[ferris_gc::ferris_gc_main(strategy = "background")]
fn main() {
    let _obj = Gc::new(Data { value: 1 });
}
```

```rust
// G1 (Garbage-First): pause-target collection with region prioritization
#[ferris_gc::ferris_gc_main(strategy = "g1")]
fn main() {
    let _obj = Gc::new(Data { value: 1 });
}
```

You can combine the macro with runtime configuration — promotion thresholds,
monitoring callbacks, and region-based allocation all work alongside any strategy:

```rust
use ferris_gc::{Gc, sync, PromotionConfig, Generation, LOCAL_GC};

#[ferris_gc::ferris_gc_main(strategy = "adaptive")]
fn main() {
    // Tune promotion thresholds (thread-local)
    LOCAL_GC.with(|gc| {
        gc.borrow().set_promotion_config(PromotionConfig {
            gen0_threshold: 5,
            gen1_threshold: 10,
        });
    });

    // Tune promotion thresholds (global)
    sync::GLOBAL_GC.set_promotion_config(PromotionConfig {
        gen0_threshold: 5,
        gen1_threshold: 10,
    });

    // Monitor collections
    LOCAL_GC.with(|gc| {
        gc.borrow().set_on_collection(|stats| {
            println!("GC: collected {} objects in {:?}", stats.objects_collected, stats.duration);
        });
    });

    // Region-based allocation
    let region = LOCAL_GC.with(|gc| gc.borrow().new_region());
    let _obj = region.gc(42);  // allocate in specific region
}
```

Each strategy accepts per-strategy parameters directly in the macro.
Unspecified parameters use their default values:

```rust
// Threshold strategy with custom gen0 threshold
#[ferris_gc::ferris_gc_main(strategy = "threshold", gen0_threshold = 200, poll_interval_ms = 100)]
fn main() { /* ... */ }

// Adaptive strategy with custom tuning
#[ferris_gc::ferris_gc_main(
    strategy = "adaptive",
    initial_gen0_threshold = 200,
    min_threshold = 30,
    max_threshold = 5000,
)]
fn main() { /* ... */ }

// Background GC with custom Gen2 trigger
#[ferris_gc::ferris_gc_main(
    strategy = "background",
    gen2_occupancy_trigger = 0.6,
    gen2_mark_step_budget = 128,
)]
fn main() { /* ... */ }

// G1 with custom pause target and heap occupancy threshold
#[ferris_gc::ferris_gc_main(
    strategy = "g1",
    pause_target_ms = 20,
    young_gen_threshold = 200,
    initiating_heap_occupancy_percent = 0.6,
)]
fn main() { /* ... */ }

// Basic strategy with custom poll interval
#[ferris_gc::ferris_gc_main(poll_interval_ms = 100)]
fn main() { /* ... */ }
```

### Custom Collection Strategy (Manual)

For fine-grained control without the macro, configure strategies manually:

```rust
use ferris_gc::sync;

// Disable basic strategy first
ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);

// Threshold strategy
let (start, stop) = ferris_gc::threshold_global_start(ferris_gc::ThresholdConfig::default());
sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);

// Adaptive strategy
let (start, stop) = ferris_gc::adaptive_global_start(ferris_gc::AdaptiveConfig::default());
sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);

// Background GC
let (start, stop) = ferris_gc::background_global_start(ferris_gc::BackgroundConfig::default());
sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);

// G1 strategy
let (start, stop) = ferris_gc::g1_global_start(ferris_gc::G1Config::default());
sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);
```

## Architecture

| | Thread-local | Thread-safe |
|---|---|---|
| Pointer | `Gc<T>` | `sync::Gc<T>` |
| Mutable cell | `GcCell<T>` | `sync::GcCell<T>` |
| Collector | `LocalGarbageCollector` | `GlobalGarbageCollector` |
| Weak ref | `GcWeak<T>` | `sync::GcWeak<T>` |

Internally uses slot-map arenas (`SlotMap<ObjectId, ObjectEntry>`) for O(1) insert/remove and cache-friendly iteration. Per-object memory overhead is ~97 bytes (vs ~880 bytes with the previous HashMap-based design).

## Collection Strategies

FerrisGC provides five macro-level strategies that control **when** and **how** garbage collection happens automatically. Each strategy is configured via `#[ferris_gc_main(strategy = "...")]`.

### Basic (default)

A simple periodic strategy. A background thread wakes every `poll_interval_ms` (default: 50ms) and calls `collect()` on all registered collectors. Minimal configuration, good for prototypes and small applications.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poll_interval_ms` | 50 | Background thread wake interval (ms) |

### Threshold

Triggers collection when the number of allocations since the last Gen0 collection exceeds a threshold. Generational: Gen1 runs after N Gen0 collections, Gen2 after M Gen1 collections.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gen0_threshold` | 100 | Allocations before triggering Gen0 collection |
| `gen0_collections_per_gen1` | 10 | Gen0 cycles before triggering Gen1 |
| `gen1_collections_per_gen2` | 5 | Gen1 cycles before triggering Gen2 |
| `poll_interval_ms` | 50 | Strategy thread wake interval (ms) |

### Adaptive

Self-tuning strategy that adjusts the Gen0 threshold based on the ratio of collected objects to scanned objects. If collections reclaim a lot (high garbage ratio), the threshold decreases to collect sooner. If most objects survive (low ratio), the threshold increases to avoid wasted work.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_gen0_threshold` | 100 | Starting Gen0 allocation threshold |
| `min_threshold` | 50 | Minimum Gen0 threshold (lower bound) |
| `max_threshold` | 10000 | Maximum Gen0 threshold (upper bound) |
| `high_ratio` | 0.7 | Garbage ratio above which threshold decreases |
| `low_ratio` | 0.3 | Garbage ratio below which threshold increases |
| `adjust_factor` | 0.5 | Multiplicative factor for threshold adjustment |
| `poll_interval_ms` | 50 | Strategy thread wake interval (ms) |

### Background (.NET-style)

Inspired by .NET's Background GC. Two threads work in parallel:
- **Foreground thread** — handles Gen0/Gen1 collections with short stop-the-world pauses
- **Background thread** — performs concurrent Gen2 marking using the existing incremental marking infrastructure, triggered when heap occupancy exceeds a configurable threshold

Gen2 marking happens concurrently without stopping the application, only requiring a brief STW pause for the final sweep.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gen0_threshold` | 100 | Allocations before triggering Gen0 collection |
| `gen2_occupancy_trigger` | 0.75 | Heap occupancy ratio to start background Gen2 mark |
| `gen2_mark_step_budget` | 64 | Objects to mark per concurrent step |
| `poll_interval_ms` | 50 | Strategy thread wake interval (ms) |

### G1 (Garbage-First)

Inspired by Java's G1 collector. Combines region-based garbage-first selection with concurrent Gen2 marking and a configurable pause target:
- Tracks per-region liveness statistics (bytes allocated, object count, estimated garbage ratio)
- Collects the dirtiest regions first within the pause-time budget
- Background thread concurrently marks Gen2 when heap occupancy crosses a threshold

Best for large heaps where predictable pause times are critical.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pause_target_ms` | 10 | Target max pause time per collection (ms) |
| `young_gen_threshold` | 100 | Allocations before triggering young gen collection |
| `initiating_heap_occupancy_percent` | 0.45 | Heap occupancy to start concurrent Gen2 mark |
| `concurrent_mark_budget` | 64 | Objects to mark per concurrent step |
| `poll_interval_ms` | 50 | Strategy thread wake interval (ms) |

### Which Strategy to Choose?

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| Prototype / small app | **basic** | Zero configuration, just works |
| Predictable allocation patterns | **threshold** | Direct control over collection frequency |
| Variable workloads | **adaptive** | Auto-tunes to changing allocation rates |
| Long-lived large heaps | **background** | Concurrent Gen2 avoids long pauses |
| Latency-sensitive with large heap | **g1** | Pause-target + garbage-first region selection |
| Maximum throughput | **threshold** or **adaptive** | Minimal overhead, no concurrent threads |

### Low-Level Collection Methods

In addition to automatic strategies, you can invoke collection manually:

| Method | Pause Profile |
|--------|---------------|
| `collect()` | Full STW — single long pause |
| `collect_generation(gen)` | Generational — shorter pauses (young gen only) |
| `collect_incremental(gen, budget)` | Incremental — multiple short pauses |
| `collect_incremental_timed(gen, duration)` | Time-budgeted — bounded pause time |
| `collect_concurrent(gen, budget)` | Concurrent — minimal STW (snapshot + sweep only) |
| `collect_region(region)` | Region-based — scoped collection |
| `collect_garbage_first(pause_target)` | G1-style — dirtiest regions first within time budget |
| `collect_parallel()` | Parallel sweep — uses rayon for deallocation (requires `parallel` feature) |

## Performance Features

### Card Table (Write Barrier)

FerrisGC uses a sparse card table for the write barrier instead of a mutex-protected remembered set. The card table divides the address space into 512-byte cards and tracks dirty cards with a `HashMap<usize, u8>`. This provides O(1) write barrier operations without locking, making pointer updates in inner loops fast.

### Thread-Local Allocation Buffers (TLAB)

The thread-local collector uses bump-pointer allocation within 64KB TLAB blocks. New allocations simply increment a pointer — no system allocator call or lock needed. When a block fills up, a new one is allocated. TLAB blocks are freed automatically when all objects within them are collected (via `Arc<TlabBlock>` reference counting).

### Parallel Sweep (requires `parallel` feature)

With the `parallel` feature enabled, `collect_parallel()` uses [rayon](https://crates.io/crates/rayon) to parallelize the deallocation phase. The mark phase remains serial (single-threaded graph traversal is typically faster due to pointer-chasing), but dead objects are deallocated in parallel across CPU cores.

```toml
[dependencies]
ferris-gc = { version = "0.2.0", features = ["parallel"] }
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Full GC runtime (collectors, strategies, threading) |
| `proc-macro` | no | `#[derive(Trace, Finalize)]` and `#[ferris_gc_main]` |
| `parallel` | no | Parallel sweep via rayon (`collect_parallel()`) |

With `--no-default-features`, only core traits (`Trace`, `Finalize`) and generation types are exported (`no_std` compatible).

## MSRV

The minimum supported Rust version is **1.85.0** (edition 2024).

## Examples

```bash
cargo run --example basic
cargo run --example cyclic
cargo run --example sync_gc
cargo run --example incremental
cargo run --example macro_strategies --features proc-macro   # #[ferris_gc_main] with strategies
```

## Benchmarks

```bash
cargo bench                    # Run all benchmarks (including million-object tests)
cargo bench -- "local/alloc"   # Run only local allocation benchmarks
```

## Fuzzing

```bash
rustup toolchain install nightly
cd ferris_gc && cargo +nightly fuzz run fuzz_alloc_drop -- -max_total_time=60
```

## License

Apache-2.0 / MIT
