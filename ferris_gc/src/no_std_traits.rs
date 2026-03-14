//! Core GC traits for `no_std` environments.
//!
//! When the `std` feature is disabled, only these traits and the `generation`
//! types are exported. Libraries can implement `Trace` and `Finalize` without
//! depending on the full GC runtime.

extern crate alloc;
use alloc::vec::Vec;

/// Object graph traversal trait for mark-and-sweep garbage collection.
///
/// See the `std`-enabled [`crate::Trace`] for full documentation.
pub trait Trace: Finalize {
    fn is_root(&self) -> bool;
    fn reset_root(&self);
    fn trace(&self);
    fn reset(&self);
    fn is_traceable(&self) -> bool;
    fn trace_children(&self, _children: &mut Vec<*const dyn Trace>) {}
    fn clear_trace(&self) {}
}

/// Destructor callback invoked by the collector before deallocation.
pub trait Finalize {
    fn finalize(&self);
    fn as_finalize(&self) -> &dyn Finalize
    where
        Self: Sized,
    {
        self
    }
}
