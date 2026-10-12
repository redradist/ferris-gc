//! The traits every GC-managed type implements (`Trace`, `Finalize`), and
//! thin-pointer extraction from `dyn Trace`.

pub(crate) trait ThinPtr {
    fn get_thin_ptr(&self) -> usize;
}

impl ThinPtr for &dyn Trace {
    fn get_thin_ptr(&self) -> usize {
        (*self) as *const dyn Trace as *const () as usize
    }
}

impl ThinPtr for *const dyn Trace {
    fn get_thin_ptr(&self) -> usize {
        *self as *const () as usize
    }
}

/// Object graph traversal trait for mark-and-sweep garbage collection.
///
/// # Safety
///
/// This trait is safe to implement, but incorrect implementations can cause
/// the collector to miss live objects (leading to use-after-free) or loop
/// infinitely. Follow these rules:
///
/// - **`trace()`** must call `trace()` on every `Gc<T>`/`GcCell<T>` field.
///   Missing a field means the collector may free a live object.
/// - **`reset()`** must mirror `trace()` exactly — every field traced must
///   also be reset.
/// - **`trace_children()`** must push every immediate GC-managed child pointer
///   onto `children`. This enables incremental (tri-color) collection.
/// - Fields that do not contain `Gc` handles (primitives, `String`, etc.)
///   need not be traced. Use `#[unsafe_ignore_trace]` with `#[derive(Trace)]`
///   to skip such fields.
///
/// The easiest way to get a correct implementation is `#[derive(Trace)]`.
pub trait Trace: Finalize {
    /// Returns `true` if this handle is a root (stack-owned, not stored inside a GC object).
    fn is_root(&self) -> bool;
    /// Mark this handle as non-root. Called by the collector during root discovery
    /// to distinguish stack-owned handles from GC-internal references.
    fn reset_root(&self);
    /// Mark this object and its children as reachable. Called during the mark phase.
    fn trace(&self);
    /// Undo the mark set by `trace()`. Called after sweep to prepare for the next cycle.
    fn reset(&self);
    /// Returns `true` if this object was marked reachable during the current mark phase.
    fn is_traceable(&self) -> bool;
    /// Non-recursive child discovery for incremental tri-color marking.
    /// Pushes immediate GC-managed children (object pointers) onto `children`.
    ///
    /// **Every type with GC-managed children MUST implement this.** The
    /// incremental and concurrent collectors (background/g1 strategies, on-thread
    /// incremental collection) traverse the object graph through `trace_children`
    /// — *not* through `trace()`. The default below reports "no children", so a
    /// hand-written `Trace` impl that owns `Gc`/`GcCell` fields but forgets
    /// `trace_children` will have those children treated as unreachable and
    /// **swept while still live** (use-after-free). `#[derive(Trace)]` generates
    /// it correctly; only hand-written impls are at risk.
    fn trace_children(&self, _children: &mut Vec<*const dyn Trace>) {}
    /// Unconditionally clear mark-phase state without cascading.
    /// Used after sweep to reset surviving objects for the next collection cycle.
    fn clear_trace(&self) {}
    /// Update internal pointers after object relocation during compaction.
    /// Called during STW on each handle; `old_ptr` is the original object address
    /// and `new_ptr` is the new address after copying.
    ///
    /// # Safety
    /// Must only be called during stop-the-world compaction. No concurrent access
    /// to the relocated fields is permitted.
    unsafe fn relocate(&self, _old_ptr: *const u8, _new_ptr: *const u8) {}
}

/// Destructor callback invoked by the collector before an object is deallocated.
///
/// Called during the sweep phase. The default `#[derive(Finalize)]` generates
/// an empty body — override only when you need explicit cleanup of non-GC
/// resources (file handles, GPU buffers, etc.).
pub trait Finalize {
    fn finalize(&self);
    fn as_finalize(&self) -> &dyn Finalize
    where
        Self: Sized,
    {
        self
    }
}
