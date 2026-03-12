#[macro_use]
extern crate lazy_static;

mod gc;
mod default_trace;
mod basic_gc_strategy;
mod generation;
mod threshold_strategy;
mod adaptive_strategy;

pub use gc::*;
pub use gc::sync;
#[allow(unused_imports)]
pub use default_trace::*;
pub use basic_gc_strategy::{BASIC_STRATEGY_LOCAL_GCS, BASIC_STRATEGY_GLOBAL_GC, BASIC_STRATEGY_DISABLED, ApplicationCleanup};
pub use threshold_strategy::{ThresholdConfig, threshold_local_start, threshold_global_start};
pub use adaptive_strategy::{AdaptiveConfig, adaptive_local_start, adaptive_global_start};

#[cfg(feature = "proc-macro")]
pub use ferris_gc_proc_macro::{Trace, Finalize, ferris_gc_main};