# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FerrisGC is a Rust garbage collection library providing both thread-local (`Gc<T>`) and thread-safe/global (`sync::Gc<T>`) garbage collectors with mark-and-sweep collection. It's a Cargo workspace with two crates: `ferris_gc` (main library) and `ferris_gc_proc_macro` (derive macros).

## Build Commands

```bash
cargo build                        # Build all workspace crates
cargo test                         # Run all tests
cargo test -p ferris-gc --lib      # Run only library tests
cargo test -p ferris-gc --lib -- gc::tests::one_object  # Run a single test
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
| Mutable cell | `GcRefCell<T>` | `sync::GcRefCell<T>` |
| Collector | `LocalGarbageCollector` (thread_local!) | `GlobalGarbageCollector` (lazy_static + Mutex/RwLock) |
| Ref counting | `Cell<usize>` | `AtomicUsize` |
| Strategy | `LocalStrategy` | `GlobalStrategy` |

Internal type hierarchy per GC: `Gc<T>` → `GcInternal<T>` (ref counting + root flag) → `GcPtr<T>` (data + `GcInfo` metadata). Raw allocations use `std::alloc::alloc`/`dealloc` with `Layout`.

### Procedural Macros (`ferris_gc_proc_macro`)

- `#[derive(Trace)]` — Auto-implements Trace; supports `#[unsafe_ignore_trace]` to skip fields. Only works on structs (panics on enums/unions).
- `#[derive(Finalize)]` — Generates empty finalize impl.
- `#[ferris_gc_main]` — Wraps `main()` to inject `ApplicationCleanup` for graceful background-thread shutdown.

### Collection Strategy (`basic_gc_strategy.rs`)

Background thread wakes every 500ms to call `collect()` on registered GCs. `ApplicationCleanup` (RAII) signals shutdown via `APPLICATION_ACTIVE` atomic flag and joins background threads.

### Default Trace Implementations (`default_trace.rs`)

Pre-built `Trace`/`Finalize` impls for primitives, String, Box, Vec, HashMap, BTreeMap, Option, and other std collections.

## Known Issues

Several tests are currently failing (e.g., `gc_collect_one_from_two`, `two_objects` in both local and sync modules). There are also unused import warnings across the codebase.
