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
ferris-gc = { version = "0.1.5", features = ["proc-macro"] }
```

### Basic Usage

```rust
use ferris_gc::{Gc, GcRefCell, Trace, Finalize, ApplicationCleanup};

#[derive(Trace, Finalize)]
struct Node {
    value: i32,
    next: GcRefCell<Option<Gc<Node>>>,
}

#[ferris_gc::ferris_gc_main]
fn main() {
    // Thread-local GC
    let a = Gc::new(Node { value: 1, next: GcRefCell::new(None) });
    let b = Gc::new(Node { value: 2, next: GcRefCell::new(Some(a.clone())) });

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

### Configurable Promotion

```rust
use ferris_gc::{PromotionConfig, LOCAL_GC};

LOCAL_GC.with(|gc| {
    gc.borrow().set_promotion_config(PromotionConfig {
        gen0_threshold: 5,  // survive 5 Gen0 collections before promotion
        gen1_threshold: 10, // survive 10 Gen1 collections before promotion
    });
});
```

## Architecture

| | Thread-local | Thread-safe |
|---|---|---|
| Pointer | `Gc<T>` | `sync::Gc<T>` |
| Mutable cell | `GcRefCell<T>` | `sync::GcRefCell<T>` |
| Collector | `LocalGarbageCollector` | `GlobalGarbageCollector` |
| Weak ref | `GcWeak<T>` | `sync::GcWeak<T>` |

Internally uses slot-map arenas (`SlotMap<ObjectId, ObjectEntry>`) for O(1) insert/remove and cache-friendly iteration. Per-object memory overhead is ~97 bytes (vs ~880 bytes with the previous HashMap-based design).

## Collection Strategies

| Strategy | Method | Pause Profile |
|----------|--------|---------------|
| Full STW | `collect()` | Single long pause |
| Generational | `collect_generation(gen)` | Shorter pauses (young gen only) |
| Incremental | `collect_incremental(gen, budget)` | Multiple short pauses |
| Time-budgeted | `collect_incremental_timed(gen, duration)` | Bounded pause time |
| Concurrent | `collect_concurrent(gen, budget)` | Minimal STW (snapshot + sweep only) |
| Region-based | `collect_region(region)` | Scoped collection |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Full GC runtime (collectors, strategies, threading) |
| `proc-macro` | no | `#[derive(Trace, Finalize)]` and `#[ferris_gc_main]` |

With `--no-default-features`, only core traits (`Trace`, `Finalize`) and generation types are exported (`no_std` compatible).

## MSRV

The minimum supported Rust version is **1.85.0** (edition 2024).

## Examples

```bash
cargo run --example basic
cargo run --example cyclic
cargo run --example sync_gc
cargo run --example incremental
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
