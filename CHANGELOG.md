# Changelog

## [0.2.0] - 2026-03-14

### Added
- **Slot-based arena** — replaced 11 HashMaps with 2 SlotMaps + 1 HashMap, reducing per-object memory overhead from ~880B to ~97B with O(1) insert/remove and cache-friendly iteration.
- **Time-budgeted collection** — `mark_step_timed()` and `concurrent_mark_step_timed()` for predictable pause times bounded by wall-clock duration.
- **Emergency GC on OOM** — `try_alloc_mem_with_gc()` runs a full GC cycle and retries allocation before returning an error.
- **Configurable promotion thresholds** — `PromotionConfig` struct allows tuning Gen0→Gen1 and Gen1→Gen2 promotion thresholds at runtime.
- **Fuzzing targets** — 3 cargo-fuzz targets for alloc/drop patterns, cyclic references, and incremental collection.
- **Million-object benchmarks** — 8 new benchmarks testing allocation and collection at 100K-1M object scale.
- **Examples** — 4 runnable examples: basic, cyclic, sync_gc, incremental.
- **API documentation** — doc comments for all public types, methods, and type aliases.
- **SAFETY audit** — SAFETY comments on all `unsafe impl` blocks and `unsafe {}` blocks throughout gc.rs and sync.rs.

### Fixed
- **Cyclic mark-sweep correctness** — `root_ref_count` no longer leaks in cyclic object graphs; replaced cascade-based reset with unconditional `clear_trace()`.
- **Re-entrant drop panic** — `Gc`/`GcCell` drop uses `try_borrow_mut()` to handle nested RC hybrid deallocation gracefully.
- **Use-after-free** in deallocation ordering — objects now freed before tracers in all collection methods.
- **Data races** on `Cell`/`RefCell` internals — switched to allocation-triggered collection for thread-local GC (no background thread access).
- **Stacked Borrows violations** (Miri) — fixed raw pointer provenance, finalizer fat pointer invalidation, and uninitialized memory races.
- All 102 tests pass under Miri with zero undefined behavior.

### Changed
- Minimum supported Rust version: **1.85.0** (edition 2024).
- License field uses SPDX expression: `Apache-2.0 OR MIT`.
- `ferris-gc-proc-macro` bumped to 0.2.0.
- **Breaking:** `ObjectMaps` and `TracerMaps` replaced with unified `GcMaps` (internal API).
- **Breaking:** `GcInternal<T>` now stores `TracerId` and `ObjectId` instead of raw pointer lookups.
- **Breaking:** `GarbageCollector` uses single `Mutex<GcMaps>` instead of separate `Mutex<ObjectMaps>` + `RwLock<TracerMaps>`.

## [0.1.5] - 2026-03-13

### Added
- Concurrent marking with snapshot-based edge traversal (no STW during mark).
- Region-based collection for scoped memory management.
- RC hybrid — immediate deallocation for non-cyclic objects on last handle drop.
- Write barriers with remembered set for generational correctness.
- Incremental tri-color marking with configurable step budgets.
- Weak references (`GcWeak<T>`) for both local and sync GC.
- Panic-safe deallocation (finalizers/drop wrapped in `catch_unwind`).
- `try_new()` / `Gc::try_new()` for fallible allocation.
- `GcAllocError` error type.
- Adaptive and threshold collection strategies.
- `no_std` support (core traits only, no allocator).
- Enum, generics, and unit struct support in derive macros.
- `#[ferris_gc_main(strategy = "...")]` macro parameter.

## [0.1.0] - Initial release

- Thread-local `Gc<T>` and `GcCell<T>`.
- Thread-safe `sync::Gc<T>` and `sync::GcCell<T>`.
- Mark-and-sweep collection with background strategy.
- `#[derive(Trace, Finalize)]` proc macros.
- `#[ferris_gc_main]` for graceful shutdown.
