#[macro_use]
extern crate lazy_static;

mod gc;
mod basic_gc_strategy;

pub use gc::*;
pub use basic_gc_strategy::{BASIC_STRATEGY_LOCAL_GCS, BASIC_STRATEGY_GLOBAL_GC, ApplicationCleanup};

#[cfg(feature = "proc-macro")]
pub use ferris_gc_proc_macro::{Trace, Finalize, ferris_gc_main};