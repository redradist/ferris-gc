# Changelog

## [0.2.0] - 2026-07-22

### Added
- **8-byte GC handles** — `Gc<T>` / `GcCell<T>` compacted to a single tagged pointer (bit 0 = root flag, bits 1.. = object pointer); `GcInternal<T>` eliminated for the thread-local GC (the sync GC still uses tracer-based `GcInternal`).
- **TLAB bump allocator** — thread-local arena blocks for fast allocation. `Gc::new()` allocates the object (`GcPtr<T>`) and its `ObjectEntry` in a single TLAB bump, avoiding per-object system allocator calls.
- **Custom SlotMap** — generation-counted slot map (`slot_map.rs`) with O(1) insert/remove, cache-friendly iteration, and unsafe unchecked accessors for hot paths. ObjectIds are u64 (upper 32 = generation, lower 32 = index).
- **Generational collection** — Gen0/Gen1/Gen2 with configurable promotion thresholds (`PromotionConfig`). Adaptive threshold scaling proportional to live set size (GOGC-style).
- **gen0_ids** — dedicated Vec of Gen0 ObjectIds for O(Gen0) partial collections with heuristic fallback to full iteration when stale entries exceed live objects.
- **Card table write barriers** — track old→young references for generational correctness without scanning the full heap on partial collections.
- **Ephemeron tables** — weak key-value associations that are automatically cleaned up when keys are collected.
- **Compacting GC (experimental, currently disabled)** — `compact()` infrastructure for relocating live objects into a contiguous buffer; disabled (returns 0, covered by test) until handle relocation for tagged-pointer handles lands.
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
  | churn | 2.37 ms | 0.86 ms | 2.8x |
  | alloc | 5.91 ms | 1.76 ms | 3.4x |
  | generational | 2.35 ms | 1.01 ms | 2.3x |
  | tree (2^18 nodes) | 60.0 ms | 13.5 ms | 4.4x |

  (best of 3 runs, Apple Silicon, Go 1.25 — see README for details and caveats)

### Fixed
- **Sweep use-after-free (Miri)** — two-phase sweep dealloc + dying-address registry: destructors of ALL dead objects run while every dead allocation is still valid, and interior handle drops become no-ops instead of dereferencing dead objects (fixes UAF / Stacked Borrows violations when collecting cycles).
- **Root-discovery double-decrement** — the phase-2 cascade now visits each object's fields exactly once; rooted self-referencing objects and rooted cycles are no longer swept while still referenced from the stack.
- **Gen0 cascade-guard leak** — partial collections no longer leave visit guards set on out-of-scope old-generation objects (the guard doubles as the mark bit, so the leak could suppress marking and sweep live young objects).
- **`Gc<T>` / `GcCell<T>` accidentally became `Send`** — restored `!Send` / `!Sync` (thread-local collector) with `compile_fail` doctests.
- **Sync GC tracer leak** — the sweep now frees dead cycle members' remaining `GcInternal` tracer allocations and reports them via `CollectionStats::tracers_collected`.
- **Cyclic mark-sweep correctness** — `root_ref_count` no longer leaks in cyclic object graphs.
- **Use-after-free** in concurrent collection (background/g1 strategies) — fixed deallocation ordering.
- **Data races** — `Cell<usize>` → `AtomicUsize` for fields shared across threads in sync GC.
- **Stacked Borrows violations** (Miri) — fixed raw pointer provenance, fat pointer invalidation, uninitialized memory access.
- **Race condition** in `sync::GcWeak::upgrade` vs concurrent drop.
- **Deadlock** in `GlobalStrategy::change_strategy`.
- **Infinite recursion** on cyclic references in trace/reset.
- **Stack overflow** in tree benchmark (was using N=100000 as depth instead of depth=20).
- All 154 tests pass, including the full Miri suite (zero undefined behavior).

### Changed
- **Breaking:** `Gc<T>` / `GcCell<T>` are `!Send` / `!Sync` — they are bound to the thread-local collector and were never sound to move across threads.
- Parallel sweep (`parallel` feature) now runs destructors and memory frees sequentially on the collecting thread (the parallel win is the mark phase); this also fixes an `ObjectEntry` leak in the old parallel path.
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
