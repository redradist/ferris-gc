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

## Code Guidelines

Every change, performance work included, should follow these:

- **Consistent naming.** One word per concept across the codebase, and the same word in both collectors.
  - A *handle* is a `Gc`/`GcCell` value.
  - An *object* is what a handle points to.
  - An *entry* is its `ObjectEntry`.
  - `drop_handle` takes a handle away; `release_object` frees an object that has no handles left.
  - Thread-local and sync counterparts get the same name, or an explicit `Local*`/`Sync*` prefix (`LocalGcInfo`/`SyncGcInfo`).
  - A rename updates every use, doc and test. A public name that changes keeps a `#[deprecated]` alias for a release.
- **Consistent logic.** The same problem is solved the same way in both collectors (handle counting, root discovery, object release). When they must differ, a comment says why at the place they differ.
- **SOLID where it applies.**
  - One responsibility per module and type (see the `gc/` module split).
  - Extend through traits (`ObjectMap`, `Counter`, `SyncObject`) rather than `if`-chains over kinds.
  - Small interfaces; dependencies on the trait, not the concrete map or counter.
  - Do not add an abstraction that costs the hot path or has a single implementation and no second one in sight. Static dispatch (generics, `const` flags like `O::SHARDED`) is preferred over `dyn` on hot paths.
- **DRY (Don't Repeat Yourself).** Each piece of knowledge has one home: shared steps live in one function used by every path (e.g. `unregister_object`, `CollectionStats::new`/`finish`). A specialized copy is allowed only when a measured hot-path cost justifies it, and then a comment says so.
- **KISS (Keep It Simple).** Prefer the simplest design that is correct and fast enough: plain data and functions over layers of indirection, one clear protocol over clever special cases. Unsafe and concurrent code especially should be easy to reason about. A simpler invariant beats a few saved instructions unless the benchmark says otherwise.
- **YAGNI (You Aren't Gonna Need It).** Build what the current task needs, not what might be needed later: no speculative options, hooks, generic parameters or feature flags without a present use. Remove dead code (unused functions, variants, fields) instead of keeping it "just in case".
- **Refactors are behaviour- and performance-neutral.** Verify with callgrind (same instruction counts on all benchmarks, local GC included), all tests, clippy (stable and nightly), and Miri. Do not mix refactors with functional or performance changes in one PR.
- **A cleanup pass every five working branches.** After about five feature/fix/perf branches, do a small refactor PR before the next phase:
  - look for what those changes made stale: comments and docs (this file included), dead code, workarounds whose reason is gone, Known Issues that are fixed;
  - keep it behaviour-neutral, as any refactor;
  - list what you found but did not fix (too big, or a functional change) for the next phases.

## Architecture

### Core Traits

- **`Trace`** — Object graph traversal for mark-and-sweep. Methods: `trace()` (mark reachable), `is_traceable()` (check if marked), `reset()` (clear marks), `is_root()`/`reset_root()`.
- **`Finalize`** — Destructor callback (`finalize()`) invoked before deallocation.

### Two GC Implementations

Both share the same API surface but differ in synchronization:

| | Thread-local (`gc/local.rs`) | Thread-safe (`gc/sync.rs`) |
|---|---|---|
| Pointer type | `Gc<T>` | `sync::Gc<T>` |
| Mutable cell | `GcCell<T>` | `sync::GcCell<T>` |
| Collector | `LocalGarbageCollector` (thread_local!) | `GlobalGarbageCollector` (lazy_static + Mutex/RwLock) |
| Handle count | `ObjectEntry::handle_count` (`u32`) | `SyncGcInfo::handle_count` (`AtomicU32`, object header) |
| Strategy | `LocalStrategy` | `GlobalStrategy` |

Internal type hierarchy differs per GC:

- **Thread-local:** `Gc<T>`/`GcCell<T>` is a single tagged pointer (8B `Cell<NonZeroUsize>`: bit 0 = root flag, bits 1.. = `*const GcPtr<T>`) straight to `GcPtr<T>` (`LocalGcInfo` + `object_id` + data). Handle metadata (`handle_count`/`root_count`) lives in `ObjectEntry`. Both types are `!Send`/`!Sync`.
- **Sync (global):** `sync::Gc<T>` is one pointer to `GcPtr<T>`, whose header (`SyncGcInfo`) holds an atomic `handle_count`, the collector's `root_count`, and the `object_id`. As with `Arc`:
  - clone is a relaxed `fetch_add`;
  - drop is a `fetch_sub`, or for the only handle just a load;
  - the last drop frees the object through `release_object`, which takes the object's shard lock.

  Root discovery starts each root count at the handle count, and the cascade subtracts the handles found inside scanned objects. It does this only while `SYNC_CASCADE` is set, i.e. during the global collector's own discovery. `sync::Gc`/`GcCell`/`GcWeak` need `T: Send + Sync` (like `Arc`), because the collector thread reaches every value.
- **`sync::GcCell<T>`** → `GcCellBox<T>` (`SyncGcInfo` header + `RwLock<()>` + `UnsafeCell<T>`), with reader-writer semantics. `borrow`/`borrow_mut` block, and the `try_` variants return `None`. The collector never blocks on a cell: it only uses `try_read`. If a scan is blocked by a held `borrow_mut`, `blocked_scans` is bumped, and every sweep path calls `keep_all_if_blocked_since`, so that collection frees nothing.
- **Thread-local `GcCell<T>`** → `RefCell<GcPtr<T>>`. An allocation can collect while the user holds a `borrow_mut`, so the collector never borrows the cell either: it reads the header through the cell's raw pointer (`cell_info`) and the value under `try_borrow` (`scan_cell_value`). A blocked scan bumps `blocked_scans`, as in sync.

Raw allocations use TLAB bump allocation (thread-local arena blocks) with system allocator fallback. During a sweep, destructors of all dead objects run before any memory is freed (two-phase dealloc) and a dying-address registry makes interior handle drops no-ops — required for Miri/Stacked Borrows soundness of the tagged-pointer handles.

### Memory Management

- **TLAB (Thread-Local Allocation Buffer):** Bump allocator for fast allocation. `Gc::new()` allocates the `GcPtr<T>` and its `ObjectEntry` in a single TLAB bump.
- **Sync pool (`pool.rs`):** `sync::Gc::new` allocates `[object | ObjectEntry]` as one block from a per-thread size-class pool. That is 96 bytes for a small `T`, with the 64-byte `ObjectEntry`. It serves blocks of at most 1 KiB with alignment of at most 16; larger or more-aligned layouts use the system allocator. All of these allocations are recorded as `GcObjMem::Mem`, and `free_sync_mem` tells pool blocks from system ones by layout alone. 64 KiB pages are carved from 2 MiB segments. Frees on the owning thread are plain stores; frees from other threads go through a per-page CAS stack. Heaps outlive their threads and are adopted by new ones.
- **SlotMap:** Custom generation-counted slot map (`slot_map.rs`) stores `ObjectEntryRef` pointers. O(1) insert/remove with cache-friendly iteration. Keys are `ObjectId` (u64: upper 32 = generation, lower 32 = index).
- **RC-hybrid deallocation:** Objects freed immediately when their last handle is dropped (no waiting for GC cycle). GC only handles cyclic references.
- **Thread-local allocation trigger:** a thread-local collector is only collected on its own thread (`LocalGarbageCollector` is neither `Send` nor `Sync`). Every local strategy acts through `configure_trigger`: a threshold, an incremental Gen2 budget (background/g1), and a `TriggerPolicy`:
  - the schedule: every `gen0_per_gen1`-th triggered collection covers Gen1 and every `gen1_per_gen2`-th of those Gen2 (5/5 by default), so promoted dead cycles are freed;
  - the threshold rule: `Gogc` (default: after a low-yield collection, as many allocations as live objects, at least 1000), `Fixed` (threshold strategy) or `Adaptive` (adaptive strategy).
- **Skipping futile collections (thread-local):** Unreachable objects that still have handles can only appear when a handle is dropped while its object keeps others. `GcMaps::garbage_possible` has one bit per collection scope. Such a drop sets every bit, and a collection clears its scope's bits when it starts. The allocation triggers skip a collection whose bit is clear (skips still advance the schedule) and trim `gen0_ids`. When no collection could free anything, the threshold grows (`Gogc`: it doubles, up to 8× the live set); while only older generations may hold garbage, it stays.
- **Generational collection:** Gen0/Gen1/Gen2. The thread-local trigger widens collections on its schedule; the sync collector's strategies schedule them on their threads.
- **Root discovery in partial collections:** phase 1 marks the objects in scope with `CASCADE_PENDING`, and the cascade walks only those. Handles inside older objects therefore stay counted, and every young object an older one points at is a root. So there is no card table and no write barrier: `GcCell::borrow_mut` does not touch the collector.
- **Incremental/concurrent marking:** both are snapshot-at-the-beginning. `begin_collection` records each in-scope object's children (`IncrementalState::children`), and the mark steps follow only that snapshot, so mutations cannot hide an object and the steps need no STW. Objects born mid-mark are greyed (they have no snapshot, so their step only blackens them). A weak upgrade mid-mark greys its object, which may be a dead cycle the snapshot does not reach. The concurrent entry points are the same cycle under other names.
- **gen0_ids:** Fast Vec of Gen0 ObjectIds for O(Gen0) partial collections, with heuristic fallback to full iteration when stale entries exceed live objects. Thread-local only: a sync allocation holds just its shard's lock and cannot append to it, so a sync Gen0 collection iterates every object, as Gen1 does.

### Procedural Macros (`ferris_gc_proc_macro`)

- `#[derive(Trace)]` — Auto-implements Trace for structs and enums; supports `#[unsafe_ignore_trace]` to skip fields. Uses span-based compile errors (not panics).
- `#[derive(Finalize)]` — Generates empty finalize impl.
- `#[ferris_gc_main]` — Wraps `main()` to inject `ApplicationCleanup` for graceful background-thread shutdown.

### Collection Strategy (`basic_strategy.rs`)

Background thread wakes every `BASIC_POLL_INTERVAL_MS` (50ms) to `collect()` the global GC; thread-local collectors are never collected from it. `ApplicationCleanup` (RAII) signals shutdown via `APPLICATION_ACTIVE` atomic flag and joins background threads.

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

The benchmark crate's release profile uses `lto = "fat"` for cross-crate inlining.

## Known Issues

- `compact()` / `_compact()` is currently **disabled** — always returns 0 without moving objects (asserted by `compact_is_disabled_returns_zero`). Re-enabling requires a `relocate()` cascade through all Trace impls (derive macro + default_trace) plus pinning of rooted objects, since stack handles cannot be discovered after the GcInternal elimination.
- Some sync tests are flaky due to shared global GC state (test ordering sensitive). Run individual tests with `cargo test -- <test_name>` if a sync test fails in batch.
- Benchmark variance is high (2-5x between runs) due to system load, cache effects, and TLAB block reuse patterns.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
