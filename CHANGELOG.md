# Changelog

## [0.2.0] - 2026-03-21

### Added
- **TLAB bump allocator** — thread-local arena blocks for fast allocation. `Gc::new()` allocates `[GcPtr<T> | GcInternal<T> | ObjectEntry]` in a single TLAB bump (triple alloc), avoiding per-object system allocator calls.
- **Custom SlotMap** — generation-counted slot map (`slot_map.rs`) with O(1) insert/remove, cache-friendly iteration, and unsafe unchecked accessors for hot paths. ObjectIds are u64 (upper 32 = generation, lower 32 = index).
- **Generational collection** — Gen0/Gen1/Gen2 with configurable promotion thresholds (`PromotionConfig`). Adaptive threshold scaling proportional to live set size (GOGC-style).
- **gen0_ids** — dedicated Vec of Gen0 ObjectIds for O(Gen0) partial collections with heuristic fallback to full iteration when stale entries exceed live objects.
- **Card table write barriers** — track old→young references for generational correctness without scanning the full heap on partial collections.
- **Ephemeron tables** — weak key-value associations that are automatically cleaned up when keys are collected.
- **Compacting GC** — `compact()` relocates live objects to eliminate fragmentation.
- **Region-based collection** — G1-style region partitioning with `Gc::new_in(value, region)` and `collect_region()`.
- **Concurrent marking** — snapshot-based edge traversal without STW during mark phase.
- **Incremental marking** — tri-color marking with configurable step budgets and time-bounded collection (`mark_step_timed()`).
- **RC-hybrid deallocation** — objects freed immediately when last `Gc` handle is dropped (no waiting for GC cycle). GC only handles cyclic references.
- **Weak references** — `GcWeak<T>` / `sync::GcWeak<T>` with `upgrade()` / `downgrade()`.
- **Configurable strategies** — `adaptive`, `background`, `threshold`, `g1` strategies via `#[ferris_gc_main(strategy = "...")]`.
- **Emergency GC on OOM** — `try_alloc_mem_with_gc()` runs a full GC cycle and retries before returning error.
- **Fallible allocation** — `Gc::try_new()` / `GcCell::try_new()` with `GcAllocError`.
- **Production monitoring** — `GcStats`, collection duration, bytes_freed, peak_heap, `set_on_collection()` callback.
- **`no_std` support** — core traits (`Trace`, `Finalize`) and `Generation` types available without allocator.
- **Derive macro improvements** — enum, generics, and unit struct support in `#[derive(Trace)]`; span-based compile errors.
- **Fuzzing targets** — 3 cargo-fuzz targets for alloc/drop patterns, cyclic references, and incremental collection.
- **Go comparison benchmarks** — `alloc_bench`, `churn_bench`, `generational_bench`, `tree_bench`, `concurrent_bench` with Go baselines.
- **Miri integration tests** — 20 tests verifying absence of undefined behavior under Miri.
- **API documentation** — doc comments for all public types and methods.
- **SAFETY audit** — SAFETY comments on all `unsafe` blocks.

### Performance (9 rounds of optimization)
- **ObjectEntry**: 152B → 72B per object (53% reduction).
- **SlotMap slot**: 80B → 24B (ObjectEntryRef pointer).
- **Allocation**: triple TLAB bump (GcPtr + GcInternal + ObjectEntry in single allocation).
- **Drop path**: skip RefCell borrow, direct vtable finalize (no catch_unwind), inline hot-path functions.
- **Collection**: merged 4 passes into 1, gen0_ids for O(Gen0) iteration, adaptive threshold without fixed cap.
- **Thread-local GC**: UnsafeCell<GcMaps> instead of Mutex, non-atomic TLAB ref_count, Cell-based counters.
- **Benchmark results** (100K objects, best of 5 runs vs Go GC):

  | Benchmark | FerrisGC | Go GC | Ratio |
  |-----------|----------|-------|-------|
  | churn | 2.5 ms | 0.86 ms | 2.9x |
  | alloc | 4.8 ms | 1.7 ms | 2.8x |
  | generational | 2.4 ms | 0.99 ms | 2.4x |
  | tree (2M nodes) | 255 ms | 56 ms | 4.6x |

### Fixed
- **Cyclic mark-sweep correctness** — `root_ref_count` no longer leaks in cyclic object graphs.
- **Use-after-free** in concurrent collection (background/g1 strategies) — fixed deallocation ordering.
- **Data races** — `Cell<usize>` → `AtomicUsize` for fields shared across threads in sync GC.
- **Stacked Borrows violations** (Miri) — fixed raw pointer provenance, fat pointer invalidation, uninitialized memory access.
- **Race condition** in `sync::GcWeak::upgrade` vs concurrent drop.
- **Deadlock** in `GlobalStrategy::change_strategy`.
- **Infinite recursion** on cyclic references in trace/reset.
- **Stack overflow** in tree benchmark (was using N=100000 as depth instead of depth=20).
- All 148 tests pass, including Miri (zero undefined behavior).

### Changed
- Minimum supported Rust version: **1.85.0** (Rust edition 2024).
- License field uses SPDX expression: `Apache-2.0 OR MIT`.
- `ferris-gc-proc-macro` bumped to 0.2.0.
- Renamed `basic_gc_strategy.rs` → `basic_strategy.rs` and `basic_gc_strategy_start()` → `basic_strategy_start()` for naming consistency.
- Collection methods (`collect`, `collect_generation`, etc.) are now `pub(crate)` — users configure collection via strategies.
- **Breaking:** internal `GcMaps` replaces separate `ObjectMaps` + `TracerMaps`.
- **Breaking:** `GcInternal<T>` stores `ObjectId` instead of raw pointer lookups.

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
