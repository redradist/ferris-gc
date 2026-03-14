//! FerrisGC — a garbage collection library for Rust.
//!
//! # Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | **yes** | Full GC with background collection, thread-local and global collectors. |
//! | `proc-macro` | no | `#[derive(Trace, Finalize)]` and `#[ferris_gc_main]` macros. |
//!
//! With `default-features = false` (no `std`), only the core traits
//! ([`Trace`], [`Finalize`]) and the [`generation`] types are available.
//! This allows libraries to implement `Trace`/`Finalize` without pulling in
//! the allocator, threading, or collection machinery.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
#[macro_use]
extern crate lazy_static;

#[cfg(feature = "std")]
mod adaptive_strategy;
#[cfg(feature = "std")]
mod basic_gc_strategy;
#[cfg(feature = "std")]
mod default_trace;
#[cfg(feature = "std")]
mod gc;
mod generation;
#[cfg(feature = "std")]
pub(crate) mod slot_map;
#[cfg(feature = "std")]
mod threshold_strategy;

#[cfg(feature = "std")]
pub use adaptive_strategy::{AdaptiveConfig, adaptive_global_start, adaptive_local_start};
#[cfg(feature = "std")]
pub use basic_gc_strategy::{
    ApplicationCleanup, BASIC_POLL_INTERVAL_MS, BASIC_STRATEGY_DISABLED, BASIC_STRATEGY_GLOBAL_GC,
    BASIC_STRATEGY_LOCAL_GCS,
};
#[cfg(feature = "std")]
#[allow(unused_imports)]
pub use default_trace::*;
#[cfg(feature = "std")]
pub use gc::sync;
#[cfg(feature = "std")]
pub use gc::*;
#[cfg(feature = "std")]
pub use threshold_strategy::{ThresholdConfig, threshold_global_start, threshold_local_start};

pub use generation::*;

// In no_std mode, re-export the core traits from a minimal module.
#[cfg(not(feature = "std"))]
mod no_std_traits;
#[cfg(not(feature = "std"))]
pub use no_std_traits::*;

#[cfg(feature = "proc-macro")]
pub use ferris_gc_proc_macro::{Finalize, Trace, ferris_gc_main};
