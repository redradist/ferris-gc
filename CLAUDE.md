# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FerrisGC is a Rust garbage collection library providing both thread-local (`Gc<T>`) and thread-safe/global (`sync::Gc<T>`) garbage collectors with mark-and-sweep collection. It's a Cargo workspace with two crates: `ferris_gc` (main library) and `ferris_gc_proc_macro` (derive macros).

## Build Commands

```bash
cargo build                        # Build all workspace crates
cargo test                         # Run all tests (unit + integration)
cargo test -p ferris-gc --lib      # Run only library tests
cargo test -p ferris-gc --lib -- gc::tests::one_object  # Run a single test
cargo test --features _internal    # Run all tests including miri_tests integration test
cargo bench --features _internal   # Run benchmarks (requires _internal feature)
cargo check                        # Type-check without full compilation
```

## Architecture

### Core Traits

- **`Trace`** — Object graph traversal for mark-and-sweep. Methods: `trace()` (mark reachable), `is_traceable()` (check if marked), `reset()` (clear marks), `is_root()`/`reset_root()`.
- **`Finalize`** — Destructor callback (`finalize()`) invoked before deallocation.

### Two GC Implementations

Both share the same API surface but differ in synchronization:

| | Thread-local (`gc.rs`) | Thread-safe (`gc/sync.rs`) |
|---|---|---|
| Pointer type | `Gc<T>` | `sync::Gc<T>` |
| Mutable cell | `GcCell<T>` | `sync::GcCell<T>` |
| Collector | `LocalGarbageCollector` (thread_local!) | `GlobalGarbageCollector` (lazy_static + Mutex/RwLock) |
| Ref counting | `Cell<usize>` | `AtomicUsize` |
| Strategy | `LocalStrategy` | `GlobalStrategy` |

Internal type hierarchy differs per GC:

- **Thread-local:** `Gc<T>`/`GcCell<T>` is a single tagged pointer (8B `Cell<NonZeroUsize>`: bit 0 = root flag, bits 1.. = `*const GcPtr<T>`) straight to `GcPtr<T>` (`GcInfo` + `object_id` + data). Handle metadata (`handle_count`/`root_count`) lives in `ObjectEntry`. Both types are `!Send`/`!Sync`.
- **Sync (global):** `sync::Gc<T>` → `GcInternal<T>` (tracer: root flag + object_id) → `GcPtr<T>`. Tracers are registered in `ObjectEntry.tracers` (`TracerList`).

Raw allocations use TLAB bump allocation (thread-local arena blocks) with system allocator fallback. During a sweep, destructors of all dead objects run before any memory is freed (two-phase dealloc) and a dying-address registry makes interior handle drops no-ops — required for Miri/Stacked Borrows soundness of the tagged-pointer handles.

### Memory Management

- **TLAB (Thread-Local Allocation Buffer):** Bump allocator for fast allocation. `Gc::new()` allocates the `GcPtr<T>` and its `ObjectEntry` in a single TLAB bump.
- **SlotMap:** Custom generation-counted slot map (`slot_map.rs`) stores `ObjectEntryRef` pointers. O(1) insert/remove with cache-friendly iteration. Keys are `ObjectId` (u64: upper 32 = generation, lower 32 = index).
- **RC-hybrid deallocation:** Objects freed immediately when last tracer is removed (no waiting for GC cycle). GC only handles cyclic references.
- **Generational collection:** Gen0/Gen1/Gen2 with adaptive threshold scaling (GOGC-style: threshold proportional to live set size when <5% garbage collected).
- **gen0_ids:** Fast Vec of Gen0 ObjectIds for O(Gen0) partial collections, with heuristic fallback to full iteration when stale entries exceed live objects.

### Procedural Macros (`ferris_gc_proc_macro`)

- `#[derive(Trace)]` — Auto-implements Trace for structs and enums; supports `#[unsafe_ignore_trace]` to skip fields. Uses span-based compile errors (not panics).
- `#[derive(Finalize)]` — Generates empty finalize impl.
- `#[ferris_gc_main]` — Wraps `main()` to inject `ApplicationCleanup` for graceful background-thread shutdown.

### Collection Strategy (`basic_strategy.rs`)

Background thread wakes every 500ms to call `collect()` on registered GCs. `ApplicationCleanup` (RAII) signals shutdown via `APPLICATION_ACTIVE` atomic flag and joins background threads.

### Default Trace Implementations (`default_trace.rs`)

Pre-built `Trace`/`Finalize` impls for primitives, String, Box, Vec, HashMap, BTreeMap, Option, and other std collections.

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Full GC runtime (collectors, strategies, threading) |
| `proc-macro` | no | `#[derive(Trace, Finalize)]` and `#[ferris_gc_main]` |
| `_internal` | no | Exposes `_collect()`, `_compact()` etc. for benchmarks/integration tests |

With `--no-default-features`, only core traits (`Trace`, `Finalize`) and `generation` types are exported (`no_std` compatible).

Collection methods (`collect`, `collect_generation`, `collect_incremental`, `compact`, etc.) are `pub(crate)` — only accessible to strategies internally. Users configure collection behavior through strategies, not by calling these methods directly.

## Benchmarks

Benchmarks live in `benchmarks/rust/` (Rust) and `benchmarks/go/` (Go baseline). Run with:

```bash
cd benchmarks/rust && cargo run --release --bin <bench_name>
```

Available: `alloc_bench`, `churn_bench`, `generational_bench`, `tree_bench`, `concurrent_bench`.

Release profile uses `lto = "thin"` for cross-crate devirtualization.

## Known Issues

- `compact()` / `_compact()` is currently **disabled** — always returns 0 without moving objects (asserted by `compact_is_disabled_returns_zero`). Re-enabling requires a `relocate()` cascade through all Trace impls (derive macro + default_trace) plus pinning of rooted objects, since stack handles cannot be discovered after the GcInternal elimination.
- Some sync tests are flaky due to shared global GC state (test ordering sensitive). Run individual tests with `cargo test -- <test_name>` if a sync test fails in batch.
- `tree_bench` with `--strategy=adaptive` may collect live objects during build phase (node count mismatch). Use default `basic` strategy.
- Benchmark variance is high (2-5x between runs) due to system load, cache effects, and TLAB block reuse patterns.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
