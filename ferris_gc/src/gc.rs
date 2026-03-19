use crate::card_table::CardTable;
use crate::generation::{
    CollectionPhase, CollectionStats, GcStats, Generation, MarkColor, PromotionConfig, RegionId,
};
use crate::slot_map::{ObjectId, SlotMap};
use std::alloc::{Layout, alloc, dealloc};
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Type-safe region identifier for the **thread-local** garbage collector.
/// Returned by [`LocalGarbageCollector::new_region`]. Cannot be used with
/// the global (`sync`) GC — the compiler will reject the mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalRegionId(RegionId);

impl LocalRegionId {
    /// Return the raw numeric id (for logging / diagnostics).
    pub fn id(self) -> u32 {
        (self.0).0
    }
}

use crate::basic_gc_strategy::basic_gc_strategy_start;

pub mod sync;

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
    fn trace_children(&self, _children: &mut Vec<*const dyn Trace>) {}
    /// Unconditionally clear mark-phase state without cascading.
    /// Used after sweep to reset surviving objects for the next collection cycle.
    fn clear_trace(&self) {}
    /// Update internal pointers after object relocation during compaction.
    /// Called during STW on each tracer; `old_ptr` is the original object address
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

/// Convenience alias for an optional GC-managed pointer.
pub type OptGc<T> = Option<Gc<T>>;
/// Convenience alias for an optional GC-managed interior-mutable cell.
pub type OptGcCell<T> = Option<GcCell<T>>;

struct GcInfo {
    root_ref_count: Cell<usize>,
}

impl GcInfo {
    #[inline]
    fn new() -> GcInfo {
        GcInfo {
            root_ref_count: Cell::new(0),
        }
    }
}

/// Internal pointer wrapper used as the `Deref` target of `Gc<T>` and `GcCell<T>`.
/// This type is `pub` because it appears in the public `Deref` interface, but users
/// should not need to name it directly — access `T` via double-deref (`**gc`).
#[doc(hidden)]
pub struct GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    info: GcInfo,
    t: T,
}

impl<T> GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    #[inline]
    fn new(t: T) -> GcPtr<T> {
        GcPtr {
            info: GcInfo::new(),
            t,
        }
    }
}

impl<T> Deref for GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.t
    }
}

impl<T> DerefMut for GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.t
    }
}

impl<T> Trace for GcPtr<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        unreachable!("is_root on GcPtr is unreachable !!");
    }

    fn reset_root(&self) {
        self.t.reset_root();
    }

    fn trace(&self) {
        // Guard: only recurse into children on first trace (breaks cycles)
        let prev = self.info.root_ref_count.get();
        self.info.root_ref_count.set(prev + 1);
        if prev == 0 {
            self.t.trace();
        }
    }

    fn reset(&self) {
        // Guard: only recurse into children on last reset (breaks cycles)
        let prev = self.info.root_ref_count.get();
        self.info.root_ref_count.set(prev - 1);
        if prev == 1 {
            self.t.reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.info.root_ref_count.get() > 0
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.t.trace_children(children);
    }

    fn clear_trace(&self) {
        self.info.root_ref_count.set(0);
    }
}

impl<T> Trace for RefCell<GcPtr<T>>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        unreachable!("is_root on GcPtr is unreachable !!");
    }

    fn reset_root(&self) {
        self.borrow().t.reset_root();
    }

    fn trace(&self) {
        let prev = self.borrow().info.root_ref_count.get();
        self.borrow().info.root_ref_count.set(prev + 1);
        if prev == 0 {
            self.borrow().t.trace();
        }
    }

    fn reset(&self) {
        let prev = self.borrow().info.root_ref_count.get();
        self.borrow().info.root_ref_count.set(prev - 1);
        if prev == 1 {
            self.borrow().t.reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.borrow().info.root_ref_count.get() > 0
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.borrow().t.trace_children(children);
    }

    fn clear_trace(&self) {
        self.borrow().info.root_ref_count.set(0);
    }
}

impl<T> Finalize for RefCell<GcPtr<T>>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

impl<T> Finalize for GcPtr<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

#[allow(dead_code)]
pub(crate) struct GcInternal<T>
where
    T: 'static + Sized + Trace,
{
    is_root: Cell<bool>,
    ptr: Cell<*const GcPtr<T>>,
    pub(crate) object_id: ObjectId,
}

impl<T> GcInternal<T>
where
    T: 'static + Sized + Trace,
{
    #[inline]
    fn new(ptr: *const GcPtr<T>, object_id: ObjectId) -> GcInternal<T> {
        GcInternal {
            is_root: Cell::new(true),
            ptr: Cell::new(ptr),
            object_id,
        }
    }
}

impl<T> Trace for GcInternal<T>
where
    T: Sized + Trace,
{
    #[inline]
    fn is_root(&self) -> bool {
        self.is_root.get()
    }

    #[inline]
    fn reset_root(&self) {
        if self.is_root.get() {
            self.is_root.set(false);
            unsafe {
                // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
                (*self.ptr.get()).reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr.get()).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr.get()).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr.get()).is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr.get() as *const dyn Trace);
    }

    unsafe fn relocate(&self, old_ptr: *const u8, new_ptr: *const u8) {
        if self.ptr.get() as *const u8 == old_ptr {
            // SAFETY: Called during STW compaction. Cell provides interior
            // mutability, and the stw_lock write guard guarantees no concurrent
            // access.
            self.ptr.set(new_ptr as *const GcPtr<T>);
        }
    }
}

impl<T> Finalize for GcInternal<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A garbage-collected smart pointer for thread-local use.
///
/// `Gc<T>` is **not** `Send`/`Sync` — it is bound to the thread that
/// created it. For cross-thread GC pointers, use [`sync::Gc<T>`].
///
/// Dereferences to `T` (via `GcPtr<T>`), so `*gc` gives you `&T`.
///
/// # Allocation
///
/// Created via [`Gc::new`] (infallible) or [`Gc::try_new`] (fallible).
/// Objects are automatically registered with the thread-local collector.
///
/// # Collection
///
/// The background strategy thread calls `collect()` periodically. You can
/// also trigger collection manually via the `LocalGarbageCollector`.
/// Unreachable cycles are detected and collected.
pub struct Gc<T>
where
    T: 'static + Sized + Trace,
{
    internal_ptr: *mut GcInternal<T>,
    ptr: *const GcPtr<T>,
    object_id: ObjectId,
}

impl<T> Deref for Gc<T>
where
    T: 'static + Sized + Trace,
{
    type Target = GcPtr<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: internal_ptr is valid for the lifetime of this Gc handle.
        // We dereference through internal_ptr→ptr so that compaction (which
        // updates GcInternal.ptr) is transparent to callers.
        unsafe { &*(*self.internal_ptr).ptr.get() }
    }
}

impl<T> std::fmt::Debug for Gc<T>
where
    T: 'static + Sized + Trace + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Gc").field(&***self).finish()
    }
}

impl<T> Gc<T>
where
    T: Sized + Trace,
{
    /// Allocate a new GC-managed object on the thread-local collector.
    /// Starts the background collection strategy if not already active.
    pub fn new(t: T) -> Gc<T> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe {
            let region = gc.borrow().core.current_region();
            gc.borrow_mut().create_gc(t, region)
        })
    }

    /// Allocate a new GC-managed object in the specified region.
    pub fn new_in(t: T, region: LocalRegionId) -> Gc<T> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().create_gc(t, region.0) })
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<Gc<T>, GcAllocError> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe {
            let region = gc.borrow().core.current_region();
            gc.borrow_mut().try_create_gc(t, region)
        })
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(t: T, region: LocalRegionId) -> Result<Gc<T>, GcAllocError> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().try_create_gc(t, region.0) })
    }
}

impl<T> Clone for Gc<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().clone_from_gc(self) })
    }
}

impl<T> Drop for Gc<T>
where
    T: Sized + Trace,
{
    fn drop(&mut self) {
        let tracer_ptr = self.internal_ptr as *const u8;
        let object_id = self.object_id;
        // SAFETY: Thread-local access ensures single-threaded borrow; try_borrow_mut avoids re-entrant panic.
        let _ = LOCAL_GC.try_with(move |gc| unsafe {
            // Use try_borrow_mut to avoid panic from re-entrant drops:
            // RC hybrid dealloc may drop inner Gc fields whose drop also calls remove_tracer.
            if let Ok(gc) = gc.try_borrow_mut() {
                gc.remove_tracer(object_id, tracer_ptr);
            }
        });
    }
}

impl<T> Trace for Gc<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        if self.internal_ptr.is_null() {
            return false;
        }
        // SAFETY: Null check above ensures the pointer is valid.
        unsafe { (*self.internal_ptr).is_root() }
    }

    fn reset_root(&self) {
        if self.internal_ptr.is_null() {
            return;
        }
        // SAFETY: Null check above ensures the pointer is valid.
        unsafe { (*self.internal_ptr).reset_root() }
    }

    fn trace(&self) {
        if self.internal_ptr.is_null() {
            return;
        }
        // SAFETY: Null check above ensures internal_ptr is valid.
        // Go through internal_ptr→ptr so compaction is transparent.
        unsafe { (*(*self.internal_ptr).ptr.get()).trace() }
    }

    fn reset(&self) {
        if self.internal_ptr.is_null() {
            return;
        }
        // SAFETY: Null check above ensures internal_ptr is valid.
        unsafe { (*(*self.internal_ptr).ptr.get()).reset() }
    }

    fn is_traceable(&self) -> bool {
        if self.internal_ptr.is_null() {
            return false;
        }
        // SAFETY: Null check above ensures internal_ptr is valid.
        unsafe { (*(*self.internal_ptr).ptr.get()).is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        if self.internal_ptr.is_null() {
            return;
        }
        // SAFETY: Null check above ensures internal_ptr is valid.
        children.push(unsafe { (*self.internal_ptr).ptr.get() as *const dyn Trace });
    }
}

impl<T> Finalize for Gc<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

#[allow(dead_code)]
pub(crate) struct GcCellInternal<T>
where
    T: 'static + Sized + Trace,
{
    is_root: Cell<bool>,
    ptr: Cell<*const RefCell<GcPtr<T>>>,
    pub(crate) object_id: ObjectId,
}

impl<T> GcCellInternal<T>
where
    T: 'static + Sized + Trace,
{
    fn new(
        ptr: *const RefCell<GcPtr<T>>,
        object_id: ObjectId,
    ) -> GcCellInternal<T> {
        GcCellInternal {
            is_root: Cell::new(true),
            ptr: Cell::new(ptr),
            object_id,
        }
    }
}

impl<T> Trace for GcCellInternal<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        self.is_root.get()
    }

    fn reset_root(&self) {
        if self.is_root.get() {
            self.is_root.set(false);
            unsafe {
                // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
                (*self.ptr.get()).borrow().reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr.get()).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr.get()).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr.get()).borrow().is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr.get() as *const dyn Trace);
    }

    unsafe fn relocate(&self, old_ptr: *const u8, new_ptr: *const u8) {
        if self.ptr.get() as *const u8 == old_ptr {
            // SAFETY: Called during STW compaction. Cell provides interior
            // mutability, and the stw_lock write guard guarantees no concurrent
            // access.
            self.ptr.set(new_ptr as *const RefCell<GcPtr<T>>);
        }
    }
}

impl<T> Finalize for GcCellInternal<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A garbage-collected mutable cell for thread-local use.
///
/// Provides interior mutability (`borrow()` / `borrow_mut()`) for GC-managed
/// objects. `borrow_mut()` triggers the **write barrier**, which records
/// cross-generation references so that young-generation collections can
/// discover pointers from old objects to young objects.
///
/// Thread-local only — for a cross-thread variant, use [`sync::GcCell<T>`].
pub struct GcCell<T>
where
    T: 'static + Sized + Trace,
{
    internal_ptr: *mut GcCellInternal<T>,
    ptr: *const RefCell<GcPtr<T>>,
    object_id: ObjectId,
}

impl<T> Drop for GcCell<T>
where
    T: Sized + Trace,
{
    fn drop(&mut self) {
        let tracer_ptr = self.internal_ptr as *const u8;
        let object_id = self.object_id;
        // SAFETY: Thread-local access ensures single-threaded borrow; try_borrow_mut avoids re-entrant panic.
        let _ = LOCAL_GC.try_with(move |gc| unsafe {
            // Use try_borrow_mut to avoid panic from re-entrant drops:
            // RC hybrid dealloc may drop inner Gc fields whose drop also calls remove_tracer.
            if let Ok(gc) = gc.try_borrow_mut() {
                gc.remove_tracer(object_id, tracer_ptr);
            }
        });
    }
}

impl<T> Deref for GcCell<T>
where
    T: 'static + Sized + Trace,
{
    type Target = RefCell<GcPtr<T>>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: internal_ptr is valid for the lifetime of this GcCell handle.
        // We dereference through internal_ptr→ptr so that compaction (which
        // updates GcCellInternal.ptr) is transparent to callers.
        unsafe { &*(*self.internal_ptr).ptr.get() }
    }
}

impl<T> std::fmt::Debug for GcCell<T>
where
    T: 'static + Sized + Trace + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GcCell").field(&**self.borrow()).finish()
    }
}

impl<T> GcCell<T>
where
    T: 'static + Sized + Trace,
{
    /// Allocate a new GC-managed interior-mutable cell on the thread-local collector.
    /// The contained value can be borrowed mutably via `borrow_mut()`.
    pub fn new(t: T) -> GcCell<T> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe {
            let region = gc.borrow().core.current_region();
            gc.borrow_mut().create_gc_cell(t, region)
        })
    }

    /// Allocate a new GC-managed interior-mutable cell in the specified region.
    pub fn new_in(t: T, region: LocalRegionId) -> GcCell<T> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().create_gc_cell(t, region.0) })
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<GcCell<T>, GcAllocError> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe {
            let region = gc.borrow().core.current_region();
            gc.borrow_mut().try_create_gc_cell(t, region)
        })
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(t: T, region: LocalRegionId) -> Result<GcCell<T>, GcAllocError> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().try_create_gc_cell(t, region.0) })
    }

    /// Mutable borrow with write barrier.
    /// Triggers the write barrier so that if this object is in an older generation,
    /// its card is marked dirty in the card table for young-generation collections.
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GcPtr<T>> {
        // SAFETY: internal_ptr is valid for the lifetime of this GcCell handle.
        // Go through internal_ptr→ptr so compaction relocation is transparent.
        let ptr = unsafe { (*self.internal_ptr).ptr.get() };
        LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .write_barrier(self.object_id, ptr as *const dyn Trace);
        });
        // SAFETY: ptr is valid for the lifetime of this GcCell handle.
        unsafe { (*ptr).borrow_mut() }
    }
}

impl<T> Clone for GcCell<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().clone_from_gc_cell(self) })
    }
}

impl<T> Trace for GcCell<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        // SAFETY: internal_ptr is valid for the lifetime of this GcCell handle.
        unsafe { (*self.internal_ptr).is_root() }
    }

    fn reset_root(&self) {
        // SAFETY: internal_ptr is valid for the lifetime of this GcCell handle.
        unsafe { (*self.internal_ptr).reset_root() }
    }

    fn trace(&self) {
        // SAFETY: internal_ptr→ptr is valid; go through internal_ptr for compaction safety.
        unsafe { (*(*self.internal_ptr).ptr.get()).borrow().trace() }
    }

    fn reset(&self) {
        // SAFETY: internal_ptr→ptr is valid; go through internal_ptr for compaction safety.
        unsafe { (*(*self.internal_ptr).ptr.get()).borrow().reset() }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: internal_ptr→ptr is valid; go through internal_ptr for compaction safety.
        unsafe { (*(*self.internal_ptr).ptr.get()).borrow().is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        // SAFETY: internal_ptr→ptr is valid; go through internal_ptr for compaction safety.
        children.push(unsafe { (*self.internal_ptr).ptr.get() as *const dyn Trace });
    }
}

impl<T> Finalize for GcCell<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A weak reference to a GC-managed object.
///
/// Does not prevent collection. Use [`upgrade()`](GcWeak::upgrade) to obtain
/// a strong `Gc<T>`. Returns `None` if the object has already been collected.
///
/// `GcWeak<T>` is `Send + Sync` when `T` is, so it can be shared across
/// threads to observe liveness of a thread-local object.
pub struct GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    alive: Arc<AtomicBool>,
    ptr: *const GcPtr<T>,
    object_id: ObjectId,
}

// SAFETY: GcWeak contains only an Arc<AtomicBool> (thread-safe) and a raw pointer
// to the GcPtr (never written through). The pointer is only used for upgrade(),
// which goes through the GC's Mutex-protected maps. Send/Sync bounds on T
// ensure the underlying data is safe to share.
unsafe impl<T> Send for GcWeak<T> where T: 'static + Sized + Trace + Send {}
unsafe impl<T> Sync for GcWeak<T> where T: 'static + Sized + Trace + Sync {}

impl<T> Clone for GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        GcWeak {
            alive: self.alive.clone(),
            ptr: self.ptr,
            object_id: self.object_id,
        }
    }
}

impl<T> std::fmt::Debug for GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.alive.load(Ordering::Relaxed) {
            f.write_str("GcWeak(alive)")
        } else {
            f.write_str("GcWeak(dead)")
        }
    }
}

impl<T> GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    /// Returns `true` if the referenced object is still alive (not yet collected).
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

    /// Try to upgrade this weak reference to a strong `Gc<T>`.
    /// Returns `None` if the object has been collected.
    pub fn upgrade(&self) -> Option<Gc<T>> {
        if !self.alive.load(Ordering::Acquire) {
            return None;
        }
        LOCAL_GC.with(|gc| {
            // Acquire STW read lock via raw pointer to avoid RefCell borrow conflict
            let gc_ptr = gc.as_ptr();
            let _stw = unsafe {
                // SAFETY: gc_ptr is valid for the duration of this thread_local closure; raw pointer avoids RefCell borrow conflict with the subsequent borrow_mut.
                (*gc_ptr)
                    .core
                    .stw_lock
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
            };
            // Re-check alive under STW protection
            if self.alive.load(Ordering::Acquire) {
                // SAFETY: Thread-local access ensures single-threaded borrow; alive check above guarantees the object has not been collected.
                Some(unsafe { gc.borrow_mut().upgrade_weak(self) })
            } else {
                None
            }
        })
    }
}

impl<T> Gc<T>
where
    T: 'static + Sized + Trace,
{
    /// Create a weak reference to this GC-managed object.
    pub fn downgrade(this: &Gc<T>) -> GcWeak<T> {
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let alive = gc_ref
                .core
                .get_or_create_weak_alive(this.object_id);
            GcWeak {
                alive,
                ptr: this.ptr,
                object_id: this.object_id,
            }
        })
    }
}

impl<T> Trace for GcWeak<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        unreachable!("is_root should never be called on GcWeak !!");
    }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool {
        unreachable!("is_traceable should never be called on GcWeak !!");
    }
}

impl<T> Finalize for GcWeak<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// Error returned when a GC allocation fails due to memory exhaustion.
#[derive(Debug, Clone, Copy)]
pub struct GcAllocError;

impl std::fmt::Display for GcAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GC allocation failed")
    }
}

impl std::error::Error for GcAllocError {}

/// Tracks the origin of a GC allocation so it can be correctly freed.
///
/// - `Mem` — allocated via `std::alloc::alloc`; freed with `std::alloc::dealloc`.
/// - `Tlab` — sub-allocated from a TLAB block; freed by releasing the block's
///   non-atomic ref_count, which frees the entire block once all refs are gone.
pub(crate) enum GcObjMem {
    /// System-allocated memory. The raw pointer is derived from the owning
    /// ObjectEntry/TracerInfo's fat `*const dyn Trace` thin pointer.
    Mem,
    /// TLAB sub-allocation. The `*mut TlabBlock` holds a ref_count reference
    /// to the backing block. Call `TlabBlock::release` to decrement.
    Tlab(*mut crate::tlab::TlabBlock),
    /// Compacted memory sub-allocation. The `Arc<CompactBlock>` keeps the
    /// contiguous buffer alive until all compacted objects are collected.
    Compact(Arc<CompactBlock>),
    /// Part of a combined allocation (e.g. initial tracer co-allocated with
    /// the object). No separate deallocation needed — freed with the object.
    Inline,
}

impl GcObjMem {
    /// Deallocate this memory. For `Mem`, calls `std::alloc::dealloc`.
    /// For `Tlab`, releases the block's ref_count (block freed when count hits 0).
    ///
    /// # Safety
    /// For `Mem` variant, `ptr` must have been allocated with the given layout.
    /// The memory must not have been previously deallocated.
    pub(crate) unsafe fn dealloc_mem(self, ptr: *mut u8, layout: Layout) {
        match self {
            GcObjMem::Mem => {
                // SAFETY: Caller guarantees ptr was allocated with this layout.
                unsafe { dealloc(ptr, layout) };
            }
            GcObjMem::Tlab(block) => {
                // SAFETY: block pointer is valid; decrement ref_count.
                // When ref_count reaches 0, the block buffer is freed.
                unsafe { crate::tlab::TlabBlock::release(block) };
            }
            GcObjMem::Compact(_block) => {
                // Dropping the Arc<CompactBlock>. When all Arcs for this block
                // are dropped, the CompactBlock::drop frees the entire buffer.
            }
            GcObjMem::Inline => {
                // No-op: inline allocations are freed with their parent object.
            }
        }
    }
}

/// Combined finalize + drop function pointer. Called during deallocation.
/// Finalizes the object first (catching panics), then drops it.
pub(crate) type DeallocFn = unsafe fn(*mut u8);

/// Per-tracer metadata stored inline in ObjectEntry's tracers Vec.
pub(crate) struct TracerInfo {
    pub(crate) tracer_ptr: *const dyn Trace,
    pub(crate) mem: GcObjMem,
    pub(crate) layout: Layout,
}

/// Inline-optimized tracer list: stores the first tracer pointer inline (16B),
/// boxing full TracerInfo only for non-inline or clone tracers.
/// Sized to 24B instead of 56B, saving 32B per ObjectEntry.
pub(crate) enum TracerList {
    /// Common case: single tracer with inline memory (GcObjMem::Inline).
    /// Only the pointer is needed — no dealloc info required.
    Inline(*const dyn Trace),
    /// Single tracer with separate allocation (needs dealloc info).
    One(Box<TracerInfo>),
    /// Multiple tracers (e.g. after Gc::clone). Heap-allocated.
    Many(Box<Vec<TracerInfo>>),
    /// Sentinel for drain operations.
    Empty,
}

impl TracerList {
    /// Create a TracerList for a tracer with a separate allocation.
    #[inline]
    fn new(t: TracerInfo) -> Self {
        TracerList::One(Box::new(t))
    }

    /// Create a TracerList for an inline tracer (GcObjMem::Inline).
    #[inline]
    fn new_inline(ptr: *const dyn Trace) -> Self {
        TracerList::Inline(ptr)
    }

    #[inline]
    fn push(&mut self, t: TracerInfo) {
        *self = match std::mem::replace(self, TracerList::Empty) {
            TracerList::Inline(ptr) => {
                // Synthesize TracerInfo for the inline tracer (dealloc is no-op)
                let first = TracerInfo {
                    tracer_ptr: ptr,
                    mem: GcObjMem::Inline,
                    layout: Layout::new::<()>(),
                };
                TracerList::Many(Box::new(vec![first, t]))
            }
            TracerList::One(first) => TracerList::Many(Box::new(vec![*first, t])),
            TracerList::Many(mut v) => { v.push(t); TracerList::Many(v) }
            TracerList::Empty => TracerList::One(Box::new(t)),
        };
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            TracerList::Inline(_) | TracerList::One(_) => 1,
            TracerList::Many(v) => v.len(),
            TracerList::Empty => 0,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        matches!(self, TracerList::Empty)
    }

    /// Call a closure for each tracer pointer.
    #[inline]
    fn for_each_tracer<F: FnMut(*const dyn Trace)>(&self, mut f: F) {
        match self {
            TracerList::Inline(ptr) => f(*ptr),
            TracerList::One(t) => f(t.tracer_ptr),
            TracerList::Many(v) => { for t in v.iter() { f(t.tracer_ptr); } }
            TracerList::Empty => {}
        }
    }

    /// Remove the tracer whose pointer matches `ptr` (as *const u8).
    /// Returns the removed TracerInfo, or None if not found.
    fn remove_by_ptr(&mut self, ptr: *const u8) -> Option<TracerInfo> {
        match self {
            TracerList::Inline(t) => {
                if (*t as *const u8) == ptr {
                    let t = *t;
                    *self = TracerList::Empty;
                    Some(TracerInfo {
                        tracer_ptr: t,
                        mem: GcObjMem::Inline,
                        layout: Layout::new::<()>(),
                    })
                } else {
                    None
                }
            }
            TracerList::One(t) => {
                if (t.tracer_ptr as *const u8) == ptr {
                    if let TracerList::One(t) = std::mem::replace(self, TracerList::Empty) {
                        Some(*t)
                    } else {
                        unreachable!()
                    }
                } else {
                    None
                }
            }
            TracerList::Many(v) => {
                if let Some(pos) = v.iter().position(|t| (t.tracer_ptr as *const u8) == ptr) {
                    let removed = v.swap_remove(pos);
                    if v.len() == 1 {
                        if let TracerList::Many(mut v) = std::mem::replace(self, TracerList::Empty) {
                            let last = v.pop().unwrap();
                            *self = TracerList::One(Box::new(last));
                        }
                    }
                    Some(removed)
                } else {
                    None
                }
            }
            TracerList::Empty => None,
        }
    }

    /// Drain all tracers out, leaving Empty.
    fn drain(&mut self) -> TracerDrain {
        match std::mem::replace(self, TracerList::Empty) {
            TracerList::Inline(ptr) => TracerDrain::One(Some(TracerInfo {
                tracer_ptr: ptr,
                mem: GcObjMem::Inline,
                layout: Layout::new::<()>(),
            })),
            TracerList::One(t) => TracerDrain::One(Some(*t)),
            TracerList::Many(v) => TracerDrain::Many(v.into_iter()),
            TracerList::Empty => TracerDrain::One(None),
        }
    }
}

/// Iterator returned by TracerList::drain().
pub(crate) enum TracerDrain {
    One(Option<TracerInfo>),
    Many(std::vec::IntoIter<TracerInfo>),
}

impl Iterator for TracerDrain {
    type Item = TracerInfo;
    fn next(&mut self) -> Option<TracerInfo> {
        match self {
            TracerDrain::One(opt) => opt.take(),
            TracerDrain::Many(it) => it.next(),
        }
    }
}

/// Compact layout: size as u32, alignment as log2.
/// Saves 8B vs std::alloc::Layout (8B vs 16B).
#[derive(Clone, Copy)]
pub(crate) struct CompactLayout {
    size: u32,
    align_shift: u8,
}

impl CompactLayout {
    #[inline]
    pub(crate) fn from_layout(layout: Layout) -> Self {
        CompactLayout {
            size: layout.size() as u32,
            align_shift: layout.align().trailing_zeros() as u8,
        }
    }
    #[inline]
    pub(crate) fn size(self) -> usize {
        self.size as usize
    }
    #[inline]
    pub(crate) fn to_layout(self) -> Layout {
        // SAFETY: Original Layout was valid; size fits in u32 (< 4GB),
        // alignment is a power of two reconstructed from trailing_zeros.
        unsafe { Layout::from_size_align_unchecked(self.size as usize, 1usize << self.align_shift) }
    }
}

/// Per-object metadata stored in a contiguous SlotMap.
pub(crate) struct ObjectEntry {
    pub(crate) ptr: *const dyn Trace,
    pub(crate) mem: GcObjMem,
    pub(crate) layout: CompactLayout,
    /// Packed generation (bits 31:30) + survive_count (bits 29:0).
    pub(crate) gen_survive: u32,
    pub(crate) dealloc_fn: DeallocFn,
    /// All Gc<T>/GcCell<T> handles pointing to this object.
    /// RC hybrid: when tracers is empty, the object is eagerly deallocated.
    pub(crate) tracers: TracerList,
    /// Region assignment for region-based collection.
    pub(crate) region: RegionId,
    /// Raw pointer to the object's `root_ref_count` Cell inside GcInfo.
    /// Allows inline mark-bit checks (is_traceable / clear_trace) without
    /// virtual dispatch through `*const dyn Trace`.
    pub(crate) root_ref_count_ptr: *const Cell<usize>,
}

impl ObjectEntry {
    #[inline]
    pub(crate) fn generation(&self) -> Generation {
        match self.gen_survive >> 30 {
            0 => Generation::Gen0,
            1 => Generation::Gen1,
            _ => Generation::Gen2,
        }
    }

    #[inline]
    pub(crate) fn set_generation(&mut self, generation: Generation) {
        self.gen_survive = (self.gen_survive & 0x3FFF_FFFF) | ((generation as u32) << 30);
    }

    #[inline]
    pub(crate) fn survive_count(&self) -> u32 {
        self.gen_survive & 0x3FFF_FFFF
    }

    #[inline]
    pub(crate) fn set_survive_count(&mut self, count: u32) {
        self.gen_survive = (self.gen_survive & 0xC000_0000) | (count & 0x3FFF_FFFF);
    }

    #[inline]
    pub(crate) fn increment_survive_count(&mut self) {
        let count = self.survive_count();
        self.set_survive_count(count + 1);
    }
}

/// Unified GC maps. Single Mutex protects all state to simplify locking.
pub(crate) struct GcMaps {
    pub(crate) objects: SlotMap<ObjectId, ObjectEntry>,
    /// Maps thin object pointer → ObjectId for trace_children resolution.
    pub(crate) ptr_to_object: HashMap<usize, ObjectId>,
    /// Weak-reference alive flags. Only populated for objects that have
    /// weak references — most objects never need this, so keeping it
    /// out of ObjectEntry saves 8B per object on the hot path.
    pub(crate) weak_alive_map: HashMap<ObjectId, Arc<AtomicBool>>,
}

impl GcMaps {
    /// Total number of tracers across all objects.
    #[cfg(test)]
    fn total_tracers(&self) -> usize {
        self.objects.values().map(|e| e.tracers.len()).sum()
    }

    /// Compute per-region total bytes and object counts on demand from the objects SlotMap.
    pub(crate) fn compute_region_stats(
        &self,
    ) -> (HashMap<RegionId, usize>, HashMap<RegionId, usize>) {
        let mut total_bytes: HashMap<RegionId, usize> = HashMap::new();
        let mut object_count: HashMap<RegionId, usize> = HashMap::new();
        for (_id, entry) in self.objects.iter() {
            *total_bytes.entry(entry.region).or_insert(0) += entry.layout.size();
            *object_count.entry(entry.region).or_insert(0) += 1;
        }
        (total_bytes, object_count)
    }

    /// Rebuild ptr_to_object from the current objects SlotMap.
    /// Called lazily before incremental/concurrent marking that needs
    /// pointer-to-ObjectId resolution.
    pub(crate) fn rebuild_ptr_to_object(&mut self) {
        self.ptr_to_object.clear();
        for (id, entry) in self.objects.iter() {
            let thin = entry.ptr.get_thin_ptr();
            self.ptr_to_object.insert(thin, id);
        }
    }
}

/// State for incremental tri-color marking.
pub(crate) struct IncrementalState {
    pub(crate) phase: CollectionPhase,
    pub(crate) max_gen: Generation,
    pub(crate) colors: HashMap<ObjectId, MarkColor>,
    pub(crate) gray_stack: Vec<ObjectId>,
    /// Snapshot of object graph edges for concurrent marking.
    /// Populated during begin_concurrent_collection so that mark steps
    /// can traverse the graph without following live pointers.
    pub(crate) edges: HashMap<ObjectId, Vec<ObjectId>>,
}

impl IncrementalState {
    fn new() -> IncrementalState {
        IncrementalState {
            phase: CollectionPhase::Idle,
            max_gen: Generation::Gen0,
            colors: HashMap::new(),
            gray_stack: Vec::new(),
            edges: HashMap::new(),
        }
    }
}

/// RAII guard for accessing GcMaps. Optionally holds a MutexGuard when
/// thread-safe access is needed (GlobalGarbageCollector, background strategies).
/// For thread-local access, the lock field is None and access is zero-cost.
pub(crate) struct GcMapsGuard<'a> {
    maps: &'a mut GcMaps,
    _lock: Option<std::sync::MutexGuard<'a, ()>>,
}

impl<'a> Deref for GcMapsGuard<'a> {
    type Target = GcMaps;
    fn deref(&self) -> &GcMaps {
        self.maps
    }
}

impl<'a> DerefMut for GcMapsGuard<'a> {
    fn deref_mut(&mut self) -> &mut GcMaps {
        self.maps
    }
}

/// Shared GC bookkeeping used by both LocalGarbageCollector and GlobalGarbageCollector.
/// gc_maps is behind UnsafeCell for zero-cost thread-local access; gc_maps_lock
/// provides Mutex protection when cross-thread access is needed.
pub(crate) struct GarbageCollector {
    pub(crate) gc_maps: UnsafeCell<GcMaps>,
    /// Mutex for thread-safe gc_maps access (used by GlobalGC and background strategies).
    pub(crate) gc_maps_lock: Mutex<()>,
    /// Allocation counter for adaptive collection triggering.
    /// Cell instead of AtomicUsize: thread-local GC is single-threaded,
    /// sync GC always holds gc_maps_lock. Saves ~15ns per alloc (no `lock xadd`).
    pub(crate) allocation_count: Cell<usize>,
    pub(crate) card_table: CardTable,
    pub(crate) stw_lock: RwLock<()>,
    pub(crate) incremental: Mutex<IncrementalState>,
    pub(crate) total_collections: AtomicUsize,
    pub(crate) last_collection: Mutex<Option<CollectionStats>>,
    /// Current region for new allocations.
    pub(crate) current_region: AtomicU32,
    /// Next region ID to assign.
    pub(crate) next_region_id: AtomicU32,
    /// Configurable promotion thresholds per generation.
    pub(crate) promotion_config: Mutex<PromotionConfig>,
    /// Current total heap size in bytes.
    /// Cell: same safety argument as allocation_count.
    pub(crate) current_heap_size: Cell<usize>,
    /// High-water mark of heap usage in bytes.
    pub(crate) peak_heap_size: Cell<usize>,
    /// Optional callback invoked after each collection cycle.
    #[allow(clippy::type_complexity)]
    pub(crate) on_collection: Mutex<Option<Box<dyn Fn(&CollectionStats) + Send + Sync>>>,
    /// Adaptive allocation threshold for maybe_collect.
    /// Cell: same safety argument as allocation_count.
    pub(crate) alloc_threshold: Cell<usize>,
}

// SAFETY: GcMaps is behind UnsafeCell but protected by gc_maps_lock (Mutex<()>)
// for cross-thread access. Thread-local access bypasses the lock (single-threaded
// guarantee from thread_local!). All other mutable state is behind Mutex or atomics.
unsafe impl Sync for GarbageCollector {}
unsafe impl Send for GarbageCollector {}

#[allow(dead_code)]
impl GarbageCollector {
    pub(crate) fn new() -> GarbageCollector {
        GarbageCollector {
            gc_maps: UnsafeCell::new(GcMaps {
                objects: SlotMap::new(),
                ptr_to_object: HashMap::new(),
                weak_alive_map: HashMap::new(),
            }),
            gc_maps_lock: Mutex::new(()),
            allocation_count: Cell::new(0),
            card_table: CardTable::new(),
            stw_lock: RwLock::new(()),
            incremental: Mutex::new(IncrementalState::new()),
            total_collections: AtomicUsize::new(0),
            last_collection: Mutex::new(None),
            current_region: AtomicU32::new(0),
            next_region_id: AtomicU32::new(1),
            promotion_config: Mutex::new(PromotionConfig::default()),
            current_heap_size: Cell::new(0),
            peak_heap_size: Cell::new(0),
            alloc_threshold: Cell::new(LOCAL_GC_ALLOC_THRESHOLD),
            on_collection: Mutex::new(None),
        }
    }

    /// Thread-safe locked access to gc_maps. Used by collection methods
    /// and GlobalGarbageCollector. Acquires gc_maps_lock Mutex.
    pub(crate) fn lock_gc_maps(&self) -> GcMapsGuard<'_> {
        let lock = self.gc_maps_lock.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: MutexGuard ensures exclusive access across threads.
        let maps = unsafe { &mut *self.gc_maps.get() };
        GcMapsGuard {
            maps,
            _lock: Some(lock),
        }
    }

    /// Unsynchronized zero-cost access to gc_maps for single-threaded use.
    ///
    /// # Safety
    /// Caller must guarantee no concurrent access (e.g., from a thread_local!
    /// context with no background collection thread accessing gc_maps).
    #[inline]
    pub(crate) unsafe fn gc_maps_unsync(&self) -> &mut GcMaps {
        unsafe { &mut *self.gc_maps.get() }
    }

    /// Set custom promotion thresholds.
    pub fn set_promotion_config(&self, config: PromotionConfig) {
        *self
            .promotion_config
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = config;
    }

    /// Get the current promotion config.
    pub fn promotion_config(&self) -> PromotionConfig {
        *self
            .promotion_config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> GcStats {
        let gc_maps = self.lock_gc_maps();
        let mut gen0 = 0usize;
        let mut gen1 = 0usize;
        let mut gen2 = 0usize;
        let mut heap_size = 0usize;
        for entry in gc_maps.objects.values() {
            match entry.generation() {
                Generation::Gen0 => gen0 += 1,
                Generation::Gen1 => gen1 += 1,
                Generation::Gen2 => gen2 += 1,
            }
            heap_size += entry.layout.size();
        }
        GcStats {
            heap_size,
            live_objects: gc_maps.objects.len(),
            live_tracers: gc_maps.objects.values().map(|e| e.tracers.len()).sum::<usize>(),
            gen0_objects: gen0,
            gen1_objects: gen1,
            gen2_objects: gen2,
            total_collections: self.total_collections.load(Ordering::Relaxed),
            last_collection: *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
            allocation_count: self.allocation_count.get(),
            peak_heap_size: self.peak_heap_size.get(),
        }
    }

    /// Register a callback invoked after each collection cycle.
    pub fn set_on_collection(&self, callback: impl Fn(&CollectionStats) + Send + Sync + 'static) {
        *self.on_collection.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(callback));
    }

    /// Remove the collection callback.
    pub fn clear_on_collection(&self) {
        *self.on_collection.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Fire the on_collection callback (panic-safe).
    fn fire_on_collection(&self, stats: &CollectionStats) {
        let guard = self.on_collection.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cb) = guard.as_ref() {
            let cb: &(dyn Fn(&CollectionStats) + Send + Sync) = cb.as_ref();
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cb(stats)));
        }
    }

    /// Track heap growth after an allocation.
    #[inline]
    fn track_alloc(&self, size: usize) {
        let new_size = self.current_heap_size.get() + size;
        self.current_heap_size.set(new_size);
        if new_size > self.peak_heap_size.get() {
            self.peak_heap_size.set(new_size);
        }
    }

    /// Track heap shrinkage after a deallocation.
    #[inline]
    fn track_dealloc(&self, size: usize) {
        self.current_heap_size.set(self.current_heap_size.get().saturating_sub(size));
    }

    /// Finalize, drop, and deallocate a collected object.
    ///
    /// # Safety
    /// The `ObjectEntry` must refer to a valid, initialized object that has been
    /// removed from the GC maps and is no longer reachable.
    unsafe fn dealloc_object_entry(entry: ObjectEntry) {
        let mem_ptr = entry.ptr.get_thin_ptr() as *mut u8;
        let dealloc_fn = entry.dealloc_fn;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: dealloc_fn finalizes then drops; wrapped in catch_unwind for panic safety.
            unsafe { (dealloc_fn)(mem_ptr) };
        }));
        // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem or TLAB.
        unsafe { entry.mem.dealloc_mem(mem_ptr, entry.layout.to_layout()) };
    }

    /// Deallocate a tracer entry's memory.
    ///
    /// # Safety
    /// The `GcObjMem` and `Layout` must refer to a valid allocation that has been
    /// removed from the GC maps.
    unsafe fn dealloc_tracer_mem(ptr: *mut u8, mem: GcObjMem, layout: Layout) {
        // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
        unsafe { mem.dealloc_mem(ptr, layout) };
    }

    pub(crate) unsafe fn alloc_mem<T>(&self) -> (*mut T, (GcObjMem, Layout))
    where
        T: Sized,
    {
        let layout = Layout::new::<T>();
        // SAFETY: Layout is non-zero-sized and properly aligned for T; alloc returns a valid pointer or null.
        let mem = unsafe { alloc(layout) };
        if mem.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let type_ptr: *mut T = mem as *mut _;
        (type_ptr, (GcObjMem::Mem, layout))
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` instead of aborting on OOM.
    pub(crate) unsafe fn try_alloc_mem<T>(
        &self,
    ) -> Result<(*mut T, (GcObjMem, Layout)), GcAllocError>
    where
        T: Sized,
    {
        let layout = Layout::new::<T>();
        // SAFETY: Layout is non-zero-sized and properly aligned for T; alloc returns a valid pointer or null.
        let mem = unsafe { alloc(layout) };
        if mem.is_null() {
            return Err(GcAllocError);
        }
        let type_ptr: *mut T = mem as *mut _;
        Ok((type_ptr, (GcObjMem::Mem, layout)))
    }

    /// Fallible allocation with emergency GC collection on first failure.
    /// If the initial allocation fails, runs a full GC cycle to free dead objects,
    /// then retries once. Returns `Err(GcAllocError)` only if the retry also fails.
    pub(crate) unsafe fn try_alloc_mem_with_gc<T>(
        &self,
    ) -> Result<(*mut T, (GcObjMem, Layout)), GcAllocError>
    where
        T: Sized,
    {
        // SAFETY: Delegates to try_alloc_mem and collect, both of which uphold their own safety invariants.
        match unsafe { self.try_alloc_mem::<T>() } {
            Ok(result) => Ok(result),
            Err(_) => {
                // Emergency collection: free dead objects and retry
                unsafe { self.collect() };
                unsafe { self.try_alloc_mem::<T>() }
            }
        }
    }

    /// Get or create the alive flag for an object (used by weak references).
    pub(crate) fn get_or_create_weak_alive(&self, obj_id: ObjectId) -> Arc<AtomicBool> {
        let mut gc_maps = self.lock_gc_maps();
        if gc_maps.objects.get(obj_id).is_some() {
            return gc_maps
                .weak_alive_map
                .entry(obj_id)
                .or_insert_with(|| Arc::new(AtomicBool::new(true)))
                .clone();
        }
        Arc::new(AtomicBool::new(false))
    }

    /// Write barrier: if the object is in Gen1+, mark its card dirty in the
    /// card table so that young-generation collections trace through it.
    /// During incremental marking, also re-grays Black objects to maintain
    /// the tri-color invariant (no Black->White edges).
    pub(crate) fn write_barrier(&self, obj_id: ObjectId, obj_ptr: *const dyn Trace) {
        let thin = obj_ptr.get_thin_ptr();
        let gc_maps = self.lock_gc_maps();
        if let Some(entry) = gc_maps.objects.get(obj_id) {
            if entry.generation() > Generation::Gen0 {
                self.card_table.mark_dirty(thin);
            }
        }
        drop(gc_maps);

        // During incremental marking, re-gray Black objects so their
        // new children are discovered in subsequent mark steps.
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        if incr.phase == CollectionPhase::Marking {
            if let Some(color) = incr.colors.get_mut(&obj_id) {
                if *color == MarkColor::Black {
                    *color = MarkColor::Gray;
                    incr.gray_stack.push(obj_id);
                }
            }
        }
    }

    pub(crate) unsafe fn remove_tracer(&self, object_id: ObjectId, tracer_ptr: *const u8) {
        let (tracer_dealloc, object_dealloc) = {
            let mut gc_maps = self.lock_gc_maps();
            Self::remove_tracer_inner(&mut gc_maps, &self.card_table, object_id, tracer_ptr)
        };

        Self::finalize_remove_tracer(self, tracer_dealloc, object_dealloc);
    }

    /// Thread-local fast path: no mutex, uses UnsafeCell directly.
    ///
    /// # Safety
    /// Must only be called from the owning thread (thread-local GC).
    pub(crate) unsafe fn remove_tracer_unsync(&self, object_id: ObjectId, tracer_ptr: *const u8) {
        let gc_maps = unsafe { self.gc_maps_unsync() };
        let (tracer_dealloc, object_dealloc) =
            Self::remove_tracer_inner(gc_maps, &self.card_table, object_id, tracer_ptr);

        Self::finalize_remove_tracer(self, tracer_dealloc, object_dealloc);
    }

    fn remove_tracer_inner(
        gc_maps: &mut GcMaps,
        card_table: &CardTable,
        object_id: ObjectId,
        tracer_ptr: *const u8,
    ) -> (Option<(*mut u8, GcObjMem, Layout)>, Option<ObjectEntry>) {
        let mut tracer_dealloc = None;
        let mut object_dealloc = None;

        if let Some(obj_entry) = gc_maps.objects.get_mut(object_id) {
            // Find and remove the tracer matching tracer_ptr
            if let Some(removed) = obj_entry.tracers.remove_by_ptr(tracer_ptr) {
                tracer_dealloc = Some((removed.tracer_ptr.get_thin_ptr() as *mut u8, removed.mem, removed.layout));

                // RC hybrid: eagerly dealloc object if no tracers remain
                if obj_entry.tracers.is_empty() {
                    let obj_gen = obj_entry.generation();
                    let obj_entry = gc_maps.objects.remove(object_id).expect(
                        "remove_tracer: object entry not found after tracers became empty",
                    );
                    // Skip ptr_to_object.remove — it's rebuilt lazily before any
                    // code that reads it (incremental/concurrent marking).
                    // Only clean card table for promoted objects (Gen0 objects are never registered).
                    if obj_gen != Generation::Gen0 {
                        let thin = obj_entry.ptr.get_thin_ptr();
                        card_table.remove_object(thin, object_id);
                    }
                    // Region stats are computed on-demand; no counters to decrement.
                    if let Some(alive) = gc_maps.weak_alive_map.remove(&object_id) {
                        alive.store(false, Ordering::Release);
                    }
                    object_dealloc = Some(obj_entry);
                }
            }
        }

        (tracer_dealloc, object_dealloc)
    }

    fn finalize_remove_tracer(&self, tracer_dealloc: Option<(*mut u8, GcObjMem, Layout)>, object_dealloc: Option<ObjectEntry>) {
        // Dealloc tracer outside lock scope
        if let Some((ptr, mem, layout)) = tracer_dealloc {
            // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
            unsafe { Self::dealloc_tracer_mem(ptr, mem, layout) };
        }

        // RC hybrid: dealloc object outside lock scope (prevents re-entrant deadlock)
        if let Some(obj_entry) = object_dealloc {
            self.track_dealloc(obj_entry.layout.size());
            // SAFETY: Object has been removed from gc_maps and is unreachable.
            unsafe { Self::dealloc_object_entry(obj_entry) };
        }
    }

    /// Full collection (all generations). Backwards-compatible with existing callers.
    pub unsafe fn collect(&self) {
        // SAFETY: Caller upholds the safety contract; delegates to collect_generation with Gen2.
        unsafe {
            self.collect_generation(Generation::Gen2);
        }
    }

    /// Generational collection: collect objects in generations 0..=max_gen.
    /// For partial collections (< Gen2), traces from in-scope root tracers + dirty card table entries.
    /// For full collections (Gen2), traces from ALL roots and clears the card table.
    /// Sweep phase only collects objects/tracers in target generations.
    /// Surviving objects in gens < max_gen may be promoted.
    pub unsafe fn collect_generation(&self, max_gen: Generation) -> CollectionStats {
        unsafe {
            let start = std::time::Instant::now();
            let mut stats = CollectionStats {
                generation: max_gen,
                objects_scanned: 0,
                objects_collected: 0,
                objects_promoted: 0,
                tracers_collected: 0,
                bytes_freed: 0,
                duration: std::time::Duration::ZERO,
            };

            let (tracer_deallocs, object_deallocs) = {
                // STW: block all mutator operations during mark+sweep
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                // Root discovery: cascade reset_root() into object fields.
                // For partial collections (max_gen < Gen2), only walk in-scope objects.
                // Gen1+ objects' Gc handles already have is_root=false from previous
                // collections or create_gc's reset_root call.
                if max_gen >= Generation::Gen2 {
                    for entry in gc_maps.objects.values() {
                        // SAFETY: Object pointer is valid while gc_maps lock is held.
                        (&*entry.ptr).reset_root();
                    }
                } else {
                    for (_id, entry) in gc_maps.objects.iter() {
                        if entry.generation() <= max_gen {
                            // SAFETY: Object pointer is valid while gc_maps lock is held.
                            (&*entry.ptr).reset_root();
                        }
                    }
                }

                // Mark phase: trace from ALL roots (needed for correctness —
                // Gen1+ roots may reference Gen0 objects through pre-promotion edges).
                for obj_entry in gc_maps.objects.values() {
                    obj_entry.tracers.for_each_tracer(|tracer_ptr| {
                        // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                        let t = &(*tracer_ptr);
                        if t.is_root() {
                            t.trace();
                        }
                    });
                }

                // For partial collections, also trace from dirty card table entries.
                if max_gen < Generation::Gen2 {
                    let dirty_ids = self.card_table.dirty_objects();
                    for obj_id in &dirty_ids {
                        if let Some(entry) = gc_maps.objects.get(*obj_id) {
                            // SAFETY: Object pointer is valid while gc_maps lock is held.
                            (&*entry.ptr).trace();
                        }
                    }
                }

                // No separate tracer sweep needed — dead tracers are collected
                // with their dead objects below.
                let mut tracer_deallocs = Vec::new();

                // Combined pass: sweep + count + clear_trace + promotion in a single
                // iteration over objects. Avoids 3 extra full iterations and uses
                // inline root_ref_count access instead of vtable dispatch.
                let promo_cfg = self.promotion_config();
                let mut collected_objects: Vec<ObjectId> = Vec::new();
                for (id, entry) in gc_maps.objects.iter_mut() {
                    let in_scope = entry.generation() <= max_gen;
                    // SAFETY: root_ref_count_ptr points into a valid GcInfo allocation.
                    let marked = (*entry.root_ref_count_ptr).get() > 0;

                    if in_scope {
                        stats.objects_scanned += 1;
                        if !marked {
                            collected_objects.push(id);
                            // Dead object — skip clear_trace and promotion.
                            continue;
                        }
                        // Surviving in-scope object: promote.
                        let cur_gen = entry.generation();
                        entry.increment_survive_count();
                        if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                            entry.set_survive_count(0);
                            if let Some(next_gen) = cur_gen.next() {
                                if cur_gen == Generation::Gen0 {
                                    // Register with card table on first promotion
                                    // so write barriers can track old→young references.
                                    let thin = entry.ptr.get_thin_ptr();
                                    self.card_table.register_object(thin, id);
                                }
                                entry.set_generation(next_gen);
                                stats.objects_promoted += 1;
                            }
                        }
                    }
                    // Clear mark state for all surviving objects (in-scope survivors
                    // + out-of-scope objects that may have been marked transitively).
                    (*entry.root_ref_count_ptr).set(0);
                }
                stats.objects_collected = collected_objects.len();

                // Remove collected objects and extract tracer deallocs
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(mut entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        // Drain tracers before consuming the object entry
                        stats.tracers_collected += entry.tracers.len();
                        for tracer in entry.tracers.drain() {
                            tracer_deallocs.push((tracer.tracer_ptr.get_thin_ptr() as *mut u8, tracer.mem, tracer.layout));
                        }
                        object_deallocs.push(entry);
                    }
                }

                // On full collection, clear dirty flags in the card table
                if max_gen >= Generation::Gen2 {
                    self.card_table.clear_dirty();
                }

                (tracer_deallocs, object_deallocs)
            };

            // Reset allocation counter after gen0 collection
            if max_gen >= Generation::Gen0 {
                self.allocation_count.set(0);
            }

            // Dealloc phase: all locks released.
            // Object drops must happen BEFORE tracer deallocs, because dropping
            // an object may drop inner Gc<T> handles whose Drop impl calls
            // remove_tracer (which looks up object_id in gc_maps — the object
            // has been removed so the lookup is a no-op, which is correct).
            for entry in object_deallocs {
                // SAFETY: Object has been removed from gc_maps and is unreachable.
                Self::dealloc_object_entry(entry);
            }
            for (ptr, mem, layout) in tracer_deallocs {
                // SAFETY: Tracer has been removed from gc_maps and is unreachable.
                Self::dealloc_tracer_mem(ptr, mem, layout);
            }

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            // Record diagnostics
            stats.duration = start.elapsed();
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);

            stats
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn collect_all(&self) {
        unsafe {
            let (tracer_deallocs, object_deallocs) = {
                // STW: block all mutator operations during cleanup
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();
                self.card_table.clear_all();

                let mut tracer_deallocs = Vec::new();
                let obj_entries = gc_maps.objects.drain();
                gc_maps.ptr_to_object.clear();
                // Signal all weak refs as dead
                for (_, alive) in gc_maps.weak_alive_map.drain() {
                    alive.store(false, Ordering::Release);
                }
                let object_deallocs: Vec<_> = obj_entries
                    .into_iter()
                    .map(|(_, mut entry)| {
                        for tracer in entry.tracers.drain() {
                            tracer_deallocs.push((tracer.tracer_ptr.get_thin_ptr() as *mut u8, tracer.mem, tracer.layout));
                        }
                        entry
                    })
                    .collect();
                (tracer_deallocs, object_deallocs)
            };
            self.allocation_count.set(0);
            // Object drops must happen BEFORE tracer deallocs, because dropping
            // an object may drop inner Gc<T> handles whose Drop impl calls
            // remove_tracer (which looks up object_id in gc_maps — the object
            // has been removed so the lookup is a no-op, which is correct).
            for entry in object_deallocs {
                // SAFETY: Object has been removed from gc_maps and is unreachable.
                Self::dealloc_object_entry(entry);
            }
            for (ptr, mem, layout) in tracer_deallocs {
                // SAFETY: Tracer has been removed from gc_maps and is unreachable.
                Self::dealloc_tracer_mem(ptr, mem, layout);
            }
        }
    }

    /// Parallel collection: mark phase is serial, sweep/dealloc phase uses rayon for
    /// parallel finalization and deallocation. Collects objects in generations 0..=max_gen.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    #[cfg(feature = "parallel")]
    pub unsafe fn collect_parallel(&self, max_gen: Generation) -> CollectionStats {
        use rayon::prelude::*;

        unsafe {
            let start = std::time::Instant::now();
            let mut stats = CollectionStats {
                generation: max_gen,
                objects_scanned: 0,
                objects_collected: 0,
                objects_promoted: 0,
                tracers_collected: 0,
                bytes_freed: 0,
                duration: std::time::Duration::ZERO,
            };

            let (tracer_deallocs, object_deallocs) = {
                // STW: block all mutator operations during mark+sweep
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                // Root discovery: for partial collections, only walk in-scope objects.
                if max_gen >= Generation::Gen2 {
                    for entry in gc_maps.objects.values() {
                        // SAFETY: Object pointer is valid while gc_maps lock is held.
                        (&*entry.ptr).reset_root();
                    }
                } else {
                    for (_id, entry) in gc_maps.objects.iter() {
                        if entry.generation() <= max_gen {
                            // SAFETY: Object pointer is valid while gc_maps lock is held.
                            (&*entry.ptr).reset_root();
                        }
                    }
                }

                // Mark phase: trace from ALL roots.
                for obj_entry in gc_maps.objects.values() {
                    obj_entry.tracers.for_each_tracer(|tracer_ptr| {
                        // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                        let t = &(*tracer_ptr);
                        if t.is_root() {
                            t.trace();
                        }
                    });
                }

                // For partial collections, also trace from dirty card table entries.
                if max_gen < Generation::Gen2 {
                    let dirty_ids = self.card_table.dirty_objects();
                    for obj_id in &dirty_ids {
                        if let Some(entry) = gc_maps.objects.get(*obj_id) {
                            // SAFETY: Object pointer is valid while gc_maps lock is held.
                            (&*entry.ptr).trace();
                        }
                    }
                }

                // Count in-scope objects directly (no HashSet allocation)
                stats.objects_scanned = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.generation() <= max_gen)
                    .count();

                // No separate tracer sweep needed — dead tracers are collected
                // with their dead objects below.
                let mut tracer_deallocs = Vec::new();

                // Sweep objects: collect unreachable in-scope objects (direct generation check)
                let collected_objects: Vec<ObjectId> = gc_maps
                    .objects
                    .iter()
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    .filter(|(_, e)| e.generation() <= max_gen && !(&*e.ptr).is_traceable())
                    .map(|(id, _)| id)
                    .collect();
                stats.objects_collected = collected_objects.len();

                // Unconditionally clear mark state for ALL objects.
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).clear_trace();
                }

                // Remove collected objects and extract tracer deallocs
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(mut entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        // Region stats are computed on-demand; no counters to decrement.
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        stats.tracers_collected += entry.tracers.len();
                        for tracer in entry.tracers.drain() {
                            tracer_deallocs.push((tracer.tracer_ptr.get_thin_ptr() as *mut u8, tracer.mem, tracer.layout));
                        }
                        object_deallocs.push(entry);
                    }
                }

                // On full collection, clear dirty flags in the card table
                if max_gen >= Generation::Gen2 {
                    self.card_table.clear_dirty();
                }

                // Promotion: surviving in-scope objects (already filtered by removal above)
                let promo_cfg = self.promotion_config();
                for (obj_id, entry) in gc_maps.objects.iter_mut() {
                    if entry.generation() > max_gen {
                        continue;
                    }
                    let cur_gen = entry.generation();
                    entry.increment_survive_count();
                    if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                        entry.set_survive_count(0);
                        if let Some(next_gen) = cur_gen.next() {
                            if cur_gen == Generation::Gen0 {
                                let thin = entry.ptr.get_thin_ptr();
                                self.card_table.register_object(thin, obj_id);
                            }
                            entry.set_generation(next_gen);
                            stats.objects_promoted += 1;
                        }
                    }
                }

                (tracer_deallocs, object_deallocs)
            };

            // Reset allocation counter after gen0 collection
            if max_gen >= Generation::Gen0 {
                self.allocation_count.set(0);
            }

            // Parallel dealloc phase: all locks released.
            // Object drops must happen BEFORE tracer deallocs, because dropping
            // an object may drop inner Gc<T> handles whose Drop impl calls
            // remove_tracer (which looks up object_id in gc_maps — the object
            // has been removed so the lookup is a no-op, which is correct).

            // Wrap dealloc info in Send-safe wrappers.
            // SAFETY: The raw pointers within are only used for deallocation
            // (finalize, drop, dealloc) and each entry is disjoint — no two
            // entries alias the same memory.
            struct SendObjectDealloc {
                ptr: *mut u8,
                mem: GcObjMem,
                layout: Layout,
                dealloc_fn: DeallocFn,
            }
            // SAFETY: Each SendObjectDealloc owns a unique allocation.
            // The raw pointer is only used for dealloc and is never shared between threads.
            unsafe impl Send for SendObjectDealloc {}
            unsafe impl Sync for SendObjectDealloc {}

            struct SendTracerDealloc {
                ptr: *mut u8,
                mem: GcObjMem,
                layout: Layout,
            }
            // SAFETY: Each SendTracerDealloc owns a unique allocation.
            unsafe impl Send for SendTracerDealloc {}
            unsafe impl Sync for SendTracerDealloc {}

            let obj_dealloc_items: Vec<SendObjectDealloc> = object_deallocs
                .into_iter()
                .map(|entry| {
                    let ptr = entry.ptr.get_thin_ptr() as *mut u8;
                    SendObjectDealloc {
                        ptr,
                        mem: entry.mem,
                        layout: entry.layout.to_layout(),
                        dealloc_fn: entry.dealloc_fn,
                    }
                })
                .collect();

            let tracer_dealloc_items: Vec<SendTracerDealloc> = tracer_deallocs
                .into_iter()
                .map(|(ptr, mem, layout)| SendTracerDealloc { ptr, mem, layout })
                .collect();

            obj_dealloc_items.into_par_iter().for_each(|item| {
                let mem_ptr = item.ptr;
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: dealloc_fn finalizes+drops; wrapped in catch_unwind for panic safety.
                    (item.dealloc_fn)(mem_ptr);
                }));
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem or TLAB.
                unsafe { item.mem.dealloc_mem(item.ptr, item.layout) };
            });
            tracer_dealloc_items.into_par_iter().for_each(|item| {
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                unsafe { item.mem.dealloc_mem(item.ptr, item.layout) };
            });

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            // Record diagnostics
            stats.duration = start.elapsed();
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);

            stats
        }
    }

    /// Collection with parallel mark AND parallel sweep.
    ///
    /// Uses rayon to parallelize the mark phase (root tracing) and the sweep phase
    /// (deallocation). The mark phase partitions root tracers across worker threads;
    /// each worker traces its assigned roots independently. This works because
    /// `GcInfo::root_ref_count` uses `Cell<usize>` for the thread-local GC.
    /// Parallel mark requires reverting to AtomicUsize or using UnsafeCell.
    ///
    /// Best for large heaps with many root objects. For small heaps, the serial
    /// `collect_generation()` is faster due to lower overhead.
    ///
    /// # Safety
    /// Caller must ensure no references to GC-managed objects are being
    /// actively dereferenced. STW write lock is acquired internally.
    #[cfg(feature = "parallel")]
    pub unsafe fn collect_parallel_mark(&self, max_gen: Generation) -> CollectionStats {
        use rayon::prelude::*;

        unsafe {
            let start = std::time::Instant::now();
            let mut stats = CollectionStats {
                generation: max_gen,
                objects_scanned: 0,
                objects_collected: 0,
                objects_promoted: 0,
                tracers_collected: 0,
                bytes_freed: 0,
                duration: std::time::Duration::ZERO,
            };

            let (tracer_deallocs, object_deallocs) = {
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                // Root discovery (serial — must complete before marking)
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).reset_root();
                }

                // Parallel mark phase: collect root tracers, then trace in parallel.
                // NOTE: root_ref_count is Cell<usize> — parallel mark would need
                // AtomicUsize or UnsafeCell. This feature is currently disabled.
                struct SendTracePtr(*const dyn Trace);
                unsafe impl Send for SendTracePtr {}
                unsafe impl Sync for SendTracePtr {}

                // SAFETY: Tracer pointers are valid while gc_maps lock is held.
                let mut root_tracers: Vec<SendTracePtr> = Vec::new();
                for obj in gc_maps.objects.values() {
                    obj.tracers.for_each_tracer(|tracer_ptr| {
                        if (&*tracer_ptr).is_root() {
                            root_tracers.push(SendTracePtr(tracer_ptr));
                        }
                    });
                }

                // SAFETY: Tracer pointers are valid; STW lock prevents concurrent mutation.
                root_tracers.par_iter().for_each(|stp| {
                    (&*stp.0).trace();
                });

                // Card table entries (serial, typically few entries)
                if max_gen < Generation::Gen2 {
                    let dirty_ids = self.card_table.dirty_objects();
                    for obj_id in &dirty_ids {
                        if let Some(entry) = gc_maps.objects.get(*obj_id) {
                            // SAFETY: Object pointer is valid while gc_maps lock is held.
                            (&*entry.ptr).trace();
                        }
                    }
                }

                // Count in-scope objects directly (no HashSet allocation)
                stats.objects_scanned = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.generation() <= max_gen)
                    .count();

                // No separate tracer sweep needed — dead tracers are collected
                // with their dead objects below.
                let mut tracer_deallocs = Vec::new();

                // Sweep objects: collect unreachable in-scope objects (direct generation check)
                // SAFETY: Object pointers are valid while gc_maps lock is held.
                let collected_objects: Vec<ObjectId> = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.generation() <= max_gen && !(&*e.ptr).is_traceable())
                    .map(|(id, _)| id)
                    .collect();
                stats.objects_collected = collected_objects.len();

                // Clear marks
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).clear_trace();
                }

                // Remove collected objects and extract tracer deallocs
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(mut entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        stats.tracers_collected += entry.tracers.len();
                        for tracer in entry.tracers.drain() {
                            tracer_deallocs.push((tracer.tracer_ptr.get_thin_ptr() as *mut u8, tracer.mem, tracer.layout));
                        }
                        object_deallocs.push(entry);
                    }
                }

                if max_gen >= Generation::Gen2 {
                    self.card_table.clear_dirty();
                }

                // Promotion: surviving in-scope objects (already filtered by removal above)
                let promo_cfg = self.promotion_config();
                for (obj_id, entry) in gc_maps.objects.iter_mut() {
                    if entry.generation() > max_gen {
                        continue;
                    }
                    let cur_gen = entry.generation();
                    entry.increment_survive_count();
                    if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                        entry.set_survive_count(0);
                        if let Some(next_gen) = cur_gen.next() {
                            if cur_gen == Generation::Gen0 {
                                let thin = entry.ptr.get_thin_ptr();
                                self.card_table.register_object(thin, obj_id);
                            }
                            entry.set_generation(next_gen);
                            stats.objects_promoted += 1;
                        }
                    }
                }

                (tracer_deallocs, object_deallocs)
            };

            if max_gen >= Generation::Gen0 {
                self.allocation_count.set(0);
            }

            // Parallel dealloc (same as collect_parallel)
            struct SendObjectDealloc {
                ptr: *mut u8,
                mem: GcObjMem,
                layout: Layout,
                dealloc_fn: DeallocFn,
            }
            unsafe impl Send for SendObjectDealloc {}
            unsafe impl Sync for SendObjectDealloc {}

            struct SendTracerDealloc {
                ptr: *mut u8,
                mem: GcObjMem,
                layout: Layout,
            }
            unsafe impl Send for SendTracerDealloc {}
            unsafe impl Sync for SendTracerDealloc {}

            let obj_dealloc_items: Vec<SendObjectDealloc> = object_deallocs
                .into_iter()
                .map(|entry| {
                    let ptr = entry.ptr.get_thin_ptr() as *mut u8;
                    SendObjectDealloc {
                        ptr,
                        mem: entry.mem,
                        layout: entry.layout.to_layout(),
                        dealloc_fn: entry.dealloc_fn,
                    }
                })
                .collect();

            let tracer_dealloc_items: Vec<SendTracerDealloc> = tracer_deallocs
                .into_iter()
                .map(|(ptr, mem, layout)| SendTracerDealloc { ptr, mem, layout })
                .collect();

            obj_dealloc_items.into_par_iter().for_each(|item| {
                let mem_ptr = item.ptr;
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: dealloc_fn finalizes+drops; wrapped in catch_unwind for panic safety.
                    (item.dealloc_fn)(mem_ptr);
                }));
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem or TLAB.
                unsafe { item.mem.dealloc_mem(item.ptr, item.layout) };
            });
            tracer_dealloc_items.into_par_iter().for_each(|item| {
                // SAFETY: Tracer memory was allocated with the same layout via alloc/alloc_mem.
                unsafe { item.mem.dealloc_mem(item.ptr, item.layout) };
            });

            self.track_dealloc(stats.bytes_freed);
            stats.duration = start.elapsed();
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);

            stats
        }
    }

    /// Begin an incremental collection cycle.
    /// Short STW: snapshots roots, initializes tri-color marks (all in-scope objects White,
    /// root-reachable objects Gray), and sets phase to Marking.
    pub unsafe fn begin_collection(&self, max_gen: Generation) {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let mut gc_maps = self.lock_gc_maps();

        // Lazily rebuild ptr_to_object for mark_step's trace_children resolution.
        gc_maps.rebuild_ptr_to_object();

        incr.phase = CollectionPhase::Marking;
        incr.max_gen = max_gen;
        incr.colors.clear();
        incr.gray_stack.clear();

        // Root discovery: cascade reset_root to mark internal Gc handles as non-root.
        for entry in gc_maps.objects.values() {
            unsafe {
                // SAFETY: Object pointer is valid while gc_maps lock is held.
                (&*entry.ptr).reset_root();
            }
        }

        // Initialize all in-scope objects as White
        for (obj_id, entry) in gc_maps.objects.iter() {
            if entry.generation() <= max_gen {
                incr.colors.insert(obj_id, MarkColor::White);
            }
        }

        // Gray objects reachable from root tracers
        for (obj_id, obj_entry) in gc_maps.objects.iter() {
            obj_entry.tracers.for_each_tracer(|tracer_ptr| {
                unsafe {
                    // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                    let t = &(*tracer_ptr);
                    if t.is_root() {
                        if let Some(color) = incr.colors.get_mut(&obj_id) {
                            if *color == MarkColor::White {
                                *color = MarkColor::Gray;
                                incr.gray_stack.push(obj_id);
                            }
                        }
                    }
                }
            });
        }
    }

    /// Process a batch of gray objects from the worklist.
    /// Short STW per batch. Returns `true` when the gray stack is empty (marking complete).
    /// Each step discovers immediate children of `budget` objects via `trace_children`.
    pub unsafe fn mark_step(&self, budget: usize) -> bool {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let gc_maps = self.lock_gc_maps();

        let mut processed = 0;
        let mut children_buf = Vec::new();

        while processed < budget {
            let obj_id = match incr.gray_stack.pop() {
                Some(id) => id,
                None => break,
            };

            // Discover immediate children via trace_children
            children_buf.clear();
            if let Some(entry) = gc_maps.objects.get(obj_id) {
                unsafe {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).trace_children(&mut children_buf);
                }
            }

            // Convert child raw ptrs to ObjectIds and gray any White children
            for &child_ptr in &children_buf {
                let thin = child_ptr.get_thin_ptr();
                if let Some(&child_id) = gc_maps.ptr_to_object.get(&thin) {
                    if let Some(color) = incr.colors.get_mut(&child_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(child_id);
                        }
                    }
                }
            }

            // This object is now fully scanned
            incr.colors.insert(obj_id, MarkColor::Black);
            processed += 1;
        }

        incr.gray_stack.is_empty()
    }

    /// Finish an incremental collection: re-mark from card table dirty entries, sweep White objects,
    /// promote survivors, and reset state.
    /// Short STW.
    pub unsafe fn finish_collection(&self) -> CollectionStats {
        unsafe {
            let start = std::time::Instant::now();
            let max_gen;
            let (tracer_deallocs, object_deallocs, mut stats) = {
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                max_gen = incr.max_gen;
                incr.phase = CollectionPhase::Sweeping;

                // Full STW re-mark using trace() instead of trace_children().
                // This is necessary because:
                // 1. Types may not implement trace_children (default is empty),
                //    making the concurrent edge snapshot incomplete.
                // 2. Objects allocated during concurrent marking are not in the
                //    snapshot and would be incorrectly swept.
                // The concurrent marking phase serves as a performance hint —
                // this re-mark guarantees correctness.

                // Root discovery: cascade reset_root into objects.
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).reset_root();
                }

                // Mark from all root tracers using trace() (same as collect_generation).
                for obj_entry in gc_maps.objects.values() {
                    obj_entry.tracers.for_each_tracer(|tracer_ptr| {
                        // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                        let t = &(*tracer_ptr);
                        if t.is_root() {
                            t.trace();
                        }
                    });
                }

                // For partial collections, also trace from dirty card table entries.
                if max_gen < Generation::Gen2 {
                    let dirty_ids = self.card_table.dirty_objects();
                    for obj_id in &dirty_ids {
                        if let Some(entry) = gc_maps.objects.get(*obj_id) {
                            // SAFETY: Object pointer is valid while gc_maps lock is held.
                            (&*entry.ptr).trace();
                        }
                    }
                }

                let mut stats = CollectionStats {
                    generation: max_gen,
                    objects_scanned: 0,
                    objects_collected: 0,
                    objects_promoted: 0,
                    tracers_collected: 0,
                    bytes_freed: 0,
                    duration: std::time::Duration::ZERO,
                };

                // No separate tracer sweep needed — dead tracers are collected
                // with their dead objects below.
                let mut tracer_deallocs = Vec::new();

                // Combined pass: sweep + count + clear_trace + promotion.
                let promo_cfg = self.promotion_config();
                let mut collected_objects: Vec<ObjectId> = Vec::new();
                for (id, entry) in gc_maps.objects.iter_mut() {
                    let in_scope = entry.generation() <= max_gen;
                    let marked = (*entry.root_ref_count_ptr).get() > 0;

                    if in_scope {
                        stats.objects_scanned += 1;
                        if !marked {
                            collected_objects.push(id);
                            continue;
                        }
                        let cur_gen = entry.generation();
                        entry.increment_survive_count();
                        if entry.survive_count() >= promo_cfg.threshold_for(cur_gen) {
                            entry.set_survive_count(0);
                            if let Some(next_gen) = cur_gen.next() {
                                if cur_gen == Generation::Gen0 {
                                    let thin = entry.ptr.get_thin_ptr();
                                    self.card_table.register_object(thin, id);
                                }
                                entry.set_generation(next_gen);
                                stats.objects_promoted += 1;
                            }
                        }
                    }
                    (*entry.root_ref_count_ptr).set(0);
                }
                stats.objects_collected = collected_objects.len();

                // Remove collected objects and extract tracer deallocs
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(mut entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        // Region stats are computed on-demand; no counters to decrement.
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        stats.tracers_collected += entry.tracers.len();
                        for tracer in entry.tracers.drain() {
                            tracer_deallocs.push((tracer.tracer_ptr.get_thin_ptr() as *mut u8, tracer.mem, tracer.layout));
                        }
                        object_deallocs.push(entry);
                    }
                }

                // Clear card table dirty flags on full collection
                if max_gen >= Generation::Gen2 {
                    self.card_table.clear_dirty();
                }

                // Reset incremental state
                incr.phase = CollectionPhase::Idle;
                incr.colors.clear();
                incr.gray_stack.clear();

                (tracer_deallocs, object_deallocs, stats)
            };

            // Reset allocation counter
            if max_gen >= Generation::Gen0 {
                self.allocation_count.set(0);
            }

            // Dealloc phase: all locks released.
            // Object drops must happen BEFORE tracer deallocs, because dropping
            // an object may drop inner Gc<T> handles whose Drop impl calls
            // remove_tracer (which looks up object_id in gc_maps — the object
            // has been removed so the lookup is a no-op, which is correct).
            for entry in object_deallocs {
                // SAFETY: Object has been removed from gc_maps and is unreachable.
                Self::dealloc_object_entry(entry);
            }
            for (ptr, mem, layout) in tracer_deallocs {
                // SAFETY: Tracer has been removed from gc_maps and is unreachable.
                Self::dealloc_tracer_mem(ptr, mem, layout);
            }

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            // Record diagnostics
            stats.duration = start.elapsed();
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);

            stats
        }
    }

    /// Convenience: run a complete incremental collection.
    /// Breaks the mark phase into steps of `step_budget` objects each,
    /// releasing the STW lock between steps to reduce pause times.
    pub unsafe fn collect_incremental(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to begin_collection, mark_step, finish_collection.
        unsafe {
            self.begin_collection(max_gen);
            loop {
                let done = self.mark_step(step_budget);
                if done {
                    break;
                }
            }
            self.finish_collection()
        }
    }

    /// Time-budgeted incremental mark step.
    /// Processes gray objects until `max_duration` elapses or the gray stack is empty.
    /// Returns `true` when marking is complete.
    ///
    /// Unlike `mark_step` which limits by object count, this limits by wall-clock
    /// time, making pause times predictable regardless of object graph shape.
    pub unsafe fn mark_step_timed(&self, max_duration: Duration) -> bool {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let gc_maps = self.lock_gc_maps();

        let deadline = Instant::now() + max_duration;
        let mut children_buf = Vec::new();

        loop {
            if Instant::now() >= deadline {
                break;
            }

            let obj_id = match incr.gray_stack.pop() {
                Some(id) => id,
                None => break,
            };

            children_buf.clear();
            if let Some(entry) = gc_maps.objects.get(obj_id) {
                // SAFETY: entry.ptr is valid for the lifetime of the GcMaps lock.
                unsafe {
                    (&*entry.ptr).trace_children(&mut children_buf);
                }
            }

            for &child_ptr in &children_buf {
                let thin = child_ptr.get_thin_ptr();
                if let Some(&child_id) = gc_maps.ptr_to_object.get(&thin) {
                    if let Some(color) = incr.colors.get_mut(&child_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(child_id);
                        }
                    }
                }
            }

            incr.colors.insert(obj_id, MarkColor::Black);
        }

        incr.gray_stack.is_empty()
    }

    /// Run a complete incremental collection with time-budgeted mark steps.
    /// Each mark step runs for at most `max_step_duration`, then releases
    /// the STW lock to allow mutator progress.
    pub unsafe fn collect_incremental_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to begin_collection, mark_step_timed, finish_collection.
        unsafe {
            self.begin_collection(max_gen);
            loop {
                let done = self.mark_step_timed(max_step_duration);
                if done {
                    break;
                }
            }
            self.finish_collection()
        }
    }

    // ---- Concurrent marking (Strategy 21) ----

    /// Begin a concurrent collection cycle.
    /// Short STW: snapshots roots AND object graph edges so that subsequent
    /// `concurrent_mark_step` calls can traverse the graph without STW.
    pub unsafe fn begin_concurrent_collection(&self, max_gen: Generation) {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let mut gc_maps = self.lock_gc_maps();

        // Lazily rebuild ptr_to_object for edge snapshot resolution.
        gc_maps.rebuild_ptr_to_object();

        incr.phase = CollectionPhase::Marking;
        incr.max_gen = max_gen;
        incr.colors.clear();
        incr.gray_stack.clear();
        incr.edges.clear();

        // Root discovery: cascade reset_root into objects to mark internal
        // Gc handles as non-root (enables cycle collection).
        for entry in gc_maps.objects.values() {
            unsafe {
                // SAFETY: Object pointer is valid while gc_maps lock is held.
                (&*entry.ptr).reset_root();
            }
        }

        // Initialize all in-scope objects as White and snapshot their edges
        let mut children_buf = Vec::new();
        for (obj_id, entry) in gc_maps.objects.iter() {
            if entry.generation() <= max_gen {
                incr.colors.insert(obj_id, MarkColor::White);
                children_buf.clear();
                unsafe {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).trace_children(&mut children_buf);
                }
                let child_ids: Vec<ObjectId> = children_buf
                    .iter()
                    .filter_map(|&ptr| gc_maps.ptr_to_object.get(&ptr.get_thin_ptr()).copied())
                    .collect();
                incr.edges.insert(obj_id, child_ids);
            }
        }

        // Gray objects reachable from root tracers
        for (obj_id, obj_entry) in gc_maps.objects.iter() {
            obj_entry.tracers.for_each_tracer(|tracer_ptr| {
                unsafe {
                    // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                    let t = &(*tracer_ptr);
                    if t.is_root() {
                        if let Some(color) = incr.colors.get_mut(&obj_id) {
                            if *color == MarkColor::White {
                                *color = MarkColor::Gray;
                                incr.gray_stack.push(obj_id);
                            }
                        }
                    }
                }
            });
        }
    }

    /// Process gray objects using the edge snapshot. NO STW lock required.
    /// Safe to call concurrently with mutators because edges were snapshotted.
    /// Returns `true` when marking is complete (gray stack empty).
    pub fn concurrent_mark_step(&self, budget: usize) -> bool {
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let mut processed = 0;

        while processed < budget {
            let obj_id = match incr.gray_stack.pop() {
                Some(id) => id,
                None => break,
            };

            // Use snapshotted edges instead of following live pointers
            if let Some(children) = incr.edges.get(&obj_id).cloned() {
                for child_id in children {
                    if let Some(color) = incr.colors.get_mut(&child_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(child_id);
                        }
                    }
                }
            }

            incr.colors.insert(obj_id, MarkColor::Black);
            processed += 1;
        }

        incr.gray_stack.is_empty()
    }

    /// Convenience: run a complete concurrent collection.
    /// begin = short STW (snapshot), mark = no STW, finish = short STW (re-mark + sweep).
    pub unsafe fn collect_concurrent(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to begin_concurrent_collection, concurrent_mark_step, finish_collection.
        unsafe {
            self.begin_concurrent_collection(max_gen);
            loop {
                let done = self.concurrent_mark_step(step_budget);
                if done {
                    break;
                }
            }
            // finish_collection handles re-graying from card table + new roots
            self.finish_collection()
        }
    }

    /// Time-budgeted concurrent mark step. NO STW lock required.
    /// Processes gray objects from the edge snapshot until `max_duration` elapses
    /// or the gray stack is empty. Returns `true` when marking is complete.
    pub fn concurrent_mark_step_timed(&self, max_duration: Duration) -> bool {
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Instant::now() + max_duration;

        loop {
            if Instant::now() >= deadline {
                break;
            }

            let obj_id = match incr.gray_stack.pop() {
                Some(id) => id,
                None => break,
            };

            if let Some(children) = incr.edges.get(&obj_id).cloned() {
                for child_id in children {
                    if let Some(color) = incr.colors.get_mut(&child_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(child_id);
                        }
                    }
                }
            }

            incr.colors.insert(obj_id, MarkColor::Black);
        }

        incr.gray_stack.is_empty()
    }

    /// Run a complete concurrent collection with time-budgeted mark steps.
    /// Each mark step runs for at most `max_step_duration` with no STW overhead.
    pub unsafe fn collect_concurrent_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to begin_concurrent_collection, concurrent_mark_step_timed, finish_collection.
        unsafe {
            self.begin_concurrent_collection(max_gen);
            loop {
                let done = self.concurrent_mark_step_timed(max_step_duration);
                if done {
                    break;
                }
            }
            self.finish_collection()
        }
    }

    // ---- Region-based collection (Strategy 22) ----

    /// Create a new region. Future allocations go into this region.
    pub fn new_region(&self) -> RegionId {
        let id = self.next_region_id.fetch_add(1, Ordering::Relaxed);
        self.current_region.store(id, Ordering::Relaxed);
        RegionId(id)
    }

    /// Get the current allocation region.
    pub fn current_region(&self) -> RegionId {
        RegionId(self.current_region.load(Ordering::Relaxed))
    }

    /// Collect only objects in the specified region.
    /// Traces from all roots (needed for correctness), but sweeps only objects in the target region.
    pub unsafe fn collect_region(&self, region: RegionId) -> CollectionStats {
        unsafe {
            let start = std::time::Instant::now();
            let mut stats = CollectionStats {
                generation: Generation::Gen2,
                objects_scanned: 0,
                objects_collected: 0,
                objects_promoted: 0,
                tracers_collected: 0,
                bytes_freed: 0,
                duration: std::time::Duration::ZERO,
            };

            let (tracer_deallocs, object_deallocs) = {
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                // Root discovery
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).reset_root();
                }

                // Mark phase: trace from ALL roots
                for obj_entry in gc_maps.objects.values() {
                    obj_entry.tracers.for_each_tracer(|tracer_ptr| {
                        // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                        let t = &(*tracer_ptr);
                        if t.is_root() {
                            t.trace();
                        }
                    });
                }

                // Also trace from dirty card table entries
                let dirty_ids = self.card_table.dirty_objects();
                for obj_id in &dirty_ids {
                    if let Some(entry) = gc_maps.objects.get(*obj_id) {
                        // SAFETY: Object pointer is valid while gc_maps lock is held.
                        (&*entry.ptr).trace();
                    }
                }

                // No separate tracer sweep needed — dead tracers are collected
                // with their dead objects below.
                let mut tracer_deallocs = Vec::new();

                // Combined pass: sweep + count + clear_trace in a single iteration.
                let mut collected_objects: Vec<ObjectId> = Vec::new();
                for (id, entry) in gc_maps.objects.iter_mut() {
                    let in_region = entry.region == region;
                    let marked = (*entry.root_ref_count_ptr).get() > 0;

                    if in_region {
                        stats.objects_scanned += 1;
                        if !marked {
                            collected_objects.push(id);
                            continue;
                        }
                    }
                    (*entry.root_ref_count_ptr).set(0);
                }
                stats.objects_collected = collected_objects.len();

                // Remove collected objects and extract tracer deallocs
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(mut entry) = gc_maps.objects.remove(obj_id) {
                        let thin = entry.ptr.get_thin_ptr();
                        gc_maps.ptr_to_object.remove(&thin);
                        self.card_table.remove_object(thin, obj_id);
                        stats.bytes_freed += entry.layout.size();
                        // Region stats are computed on-demand; no counters to decrement.
                        if let Some(alive) = gc_maps.weak_alive_map.remove(&obj_id) {
                            alive.store(false, Ordering::Release);
                        }
                        stats.tracers_collected += entry.tracers.len();
                        for tracer in entry.tracers.drain() {
                            tracer_deallocs.push((tracer.tracer_ptr.get_thin_ptr() as *mut u8, tracer.mem, tracer.layout));
                        }
                        object_deallocs.push(entry);
                    }
                }

                (tracer_deallocs, object_deallocs)
            };

            // Dealloc phase: all locks released.
            // Object drops must happen BEFORE tracer deallocs, because dropping
            // an object may drop inner Gc<T> handles whose Drop impl calls
            // remove_tracer (which looks up object_id in gc_maps — the object
            // has been removed so the lookup is a no-op, which is correct).
            for entry in object_deallocs {
                // SAFETY: Object has been removed from gc_maps and is unreachable.
                Self::dealloc_object_entry(entry);
            }
            for (ptr, mem, layout) in tracer_deallocs {
                // SAFETY: Tracer has been removed from gc_maps and is unreachable.
                Self::dealloc_tracer_mem(ptr, mem, layout);
            }

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            stats.duration = start.elapsed();
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);
            stats
        }
    }

    /// Return per-region liveness statistics without running a collection.
    pub fn region_stats(&self) -> Vec<crate::generation::RegionStats> {
        let gc_maps = self.lock_gc_maps();
        let (total_bytes, object_count) = gc_maps.compute_region_stats();
        let mut region_ids: Vec<RegionId> = total_bytes.keys().copied().collect();
        for key in object_count.keys() {
            if !region_ids.contains(key) {
                region_ids.push(*key);
            }
        }
        region_ids
            .into_iter()
            .map(|rid| crate::generation::RegionStats {
                region_id: rid,
                total_bytes: total_bytes.get(&rid).copied().unwrap_or(0),
                object_count: object_count.get(&rid).copied().unwrap_or(0),
                estimated_garbage_ratio: 0.0,
            })
            .collect()
    }

    /// G1-style "Garbage First" collection. Runs a mark phase, computes per-region
    /// garbage ratios, then sweeps regions with the highest garbage ratio first,
    /// stopping when the elapsed time reaches `pause_target`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are held
    /// across this call (stop-the-world requirement).
    pub unsafe fn collect_garbage_first(&self, pause_target: Duration) -> CollectionStats {
        unsafe {
            let start = Instant::now();
            let mut stats = CollectionStats {
                generation: Generation::Gen2,
                objects_scanned: 0,
                objects_collected: 0,
                objects_promoted: 0,
                tracers_collected: 0,
                bytes_freed: 0,
                duration: Duration::ZERO,
            };

            let (tracer_deallocs, object_deallocs) = {
                // STW: block all mutator operations during mark+sweep
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.lock_gc_maps();

                // ---- Mark phase (same as collect_generation Gen2) ----

                // Root discovery
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).reset_root();
                }

                // Mark from all roots
                for obj_entry in gc_maps.objects.values() {
                    obj_entry.tracers.for_each_tracer(|tracer_ptr| {
                        // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                        let t = &(*tracer_ptr);
                        if t.is_root() {
                            t.trace();
                        }
                    });
                }

                stats.objects_scanned = gc_maps.objects.len();

                // ---- Compute per-region garbage ratio using inline mark bits ----
                let mut region_total: HashMap<RegionId, usize> = HashMap::new();
                let mut region_marked: HashMap<RegionId, usize> = HashMap::new();

                for (_id, entry) in gc_maps.objects.iter() {
                    *region_total.entry(entry.region).or_insert(0) += 1;
                    if (*entry.root_ref_count_ptr).get() > 0 {
                        *region_marked.entry(entry.region).or_insert(0) += 1;
                    }
                }

                // Build sorted list: highest garbage ratio first
                let mut regions_by_garbage: Vec<(RegionId, f64)> = region_total
                    .iter()
                    .map(|(&rid, &total)| {
                        let marked = region_marked.get(&rid).copied().unwrap_or(0);
                        let ratio = if total > 0 {
                            (total - marked) as f64 / total as f64
                        } else {
                            0.0
                        };
                        (rid, ratio)
                    })
                    .collect();
                regions_by_garbage
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // ---- Sweep regions one by one, highest garbage first ----
                let mut tracer_deallocs = Vec::new();
                let mut object_deallocs = Vec::new();

                for (region, _ratio) in &regions_by_garbage {
                    // Check time budget before sweeping another region
                    if start.elapsed() >= pause_target && stats.objects_collected > 0 {
                        break;
                    }

                    // Find unreachable objects in this region using inline mark bits
                    let dead_in_region: Vec<ObjectId> = gc_maps
                        .objects
                        .iter()
                        .filter(|(_id, e)| {
                            e.region == *region
                                && (*e.root_ref_count_ptr).get() == 0
                        })
                        .map(|(id, _)| id)
                        .collect();

                    if dead_in_region.is_empty() {
                        continue;
                    }

                    // Remove dead objects and extract tracer deallocs
                    for obj_id in &dead_in_region {
                        if let Some(mut entry) = gc_maps.objects.remove(*obj_id) {
                            let thin = entry.ptr.get_thin_ptr();
                            gc_maps.ptr_to_object.remove(&thin);
                            self.card_table.remove_object(thin, *obj_id);
                            stats.bytes_freed += entry.layout.size();
                            if let Some(alive) = gc_maps.weak_alive_map.remove(obj_id) {
                                alive.store(false, Ordering::Release);
                            }
                            stats.tracers_collected += entry.tracers.len();
                            for tracer in entry.tracers.drain() {
                                tracer_deallocs.push((tracer.tracer_ptr.get_thin_ptr() as *mut u8, tracer.mem, tracer.layout));
                            }
                            object_deallocs.push(entry);
                        }
                    }
                    stats.objects_collected += dead_in_region.len();
                }

                // Clear mark state for ALL surviving objects using inline access.
                for entry in gc_maps.objects.values() {
                    (*entry.root_ref_count_ptr).set(0);
                }

                (tracer_deallocs, object_deallocs)
            };

            // Dealloc phase: all locks released.
            // Object drops must happen BEFORE tracer deallocs.
            for entry in object_deallocs {
                // SAFETY: Object has been removed from gc_maps and is unreachable.
                Self::dealloc_object_entry(entry);
            }
            for (ptr, mem, layout) in tracer_deallocs {
                // SAFETY: Tracer has been removed from gc_maps and is unreachable.
                Self::dealloc_tracer_mem(ptr, mem, layout);
            }

            // Track heap shrinkage
            self.track_dealloc(stats.bytes_freed);

            stats.duration = start.elapsed();
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            self.fire_on_collection(&stats);
            stats
        }
    }

    /// Compact the heap by copying all live objects into a contiguous buffer.
    ///
    /// Performs a full collection first (to free dead objects), then:
    /// 1. Allocates a single contiguous buffer for all live objects.
    /// 2. Copies each live object (GcPtr) into the buffer at aligned offsets.
    /// 3. Updates tracer pointers via `Trace::relocate()`.
    /// 4. Updates ObjectEntry metadata.
    /// 5. Frees old individual allocations.
    ///
    /// Returns the number of objects compacted.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    /// For thread-local GC this is trivially safe (single-threaded).
    /// For global GC, STW write lock is acquired internally.
    pub unsafe fn compact(&self) -> usize {
        unsafe {
            // Full collection first to free dead objects
            self.collect();

            let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
            let mut gc_maps = self.lock_gc_maps();

            let num_objects = gc_maps.objects.len();
            if num_objects == 0 {
                return 0;
            }

            // Calculate total size needed with proper alignment
            let mut total_size: usize = 0;
            let mut obj_layouts: Vec<(ObjectId, Layout)> = Vec::with_capacity(num_objects);
            for (id, entry) in gc_maps.objects.iter() {
                let layout = entry.layout.to_layout();
                // Align up
                let align_offset = total_size % layout.align();
                if align_offset != 0 {
                    total_size += layout.align() - align_offset;
                }
                obj_layouts.push((id, layout));
                total_size += layout.size();
            }

            if total_size == 0 {
                return 0;
            }

            // Allocate contiguous buffer with max alignment
            let max_align = obj_layouts
                .iter()
                .map(|(_, l)| l.align())
                .max()
                .unwrap_or(1);
            let compact_layout = match Layout::from_size_align(total_size, max_align) {
                Ok(l) => l,
                Err(_) => return 0,
            };
            // SAFETY: compact_layout has non-zero size and valid alignment.
            let compact_buf = alloc(compact_layout);
            if compact_buf.is_null() {
                return 0; // allocation failed, skip compaction
            }

            // Copy each object into the contiguous buffer
            let mut offset: usize = 0;
            let mut relocations: Vec<(ObjectId, *const u8, *mut u8, Layout)> =
                Vec::with_capacity(num_objects);

            for (id, layout) in &obj_layouts {
                // Align offset
                let align_offset = offset % layout.align();
                if align_offset != 0 {
                    offset += layout.align() - align_offset;
                }

                let entry = gc_maps
                    .objects
                    .get(*id)
                    .expect("compact: object entry not found for id in obj_layouts");
                let old_ptr = entry.ptr.get_thin_ptr() as *mut u8;
                // SAFETY: offset is within compact_buf bounds (calculated from accumulated layout sizes).
                let new_ptr = compact_buf.add(offset);

                // SAFETY: old_ptr and new_ptr point to valid, non-overlapping memory of at least layout.size() bytes.
                std::ptr::copy_nonoverlapping(old_ptr, new_ptr, layout.size());

                relocations.push((*id, old_ptr as *const u8, new_ptr, *layout));

                offset += layout.size();
            }

            // Update ObjectEntry pointers and create new GcObjMem for each object.
            // All objects now live in the compact buffer. We use Mem variant pointing
            // into the buffer. Only the LAST object "owns" the buffer for dealloc;
            // others use a Tlab-like scheme. For simplicity, we track the buffer
            // separately and never dealloc individual objects from it.
            // Instead, we store a shared Arc that owns the compact buffer.
            let compact_block = Arc::new(CompactBlock {
                ptr: compact_buf,
                layout: compact_layout,
            });

            let mut old_mems: Vec<(*mut u8, GcObjMem, Layout)> = Vec::with_capacity(num_objects);

            for (id, _old_raw, new_raw, layout) in &relocations {
                // Two-phase update: first update ObjectEntry (needs &mut objects),
                // then update ptr_to_object (needs &mut ptr_to_object).
                // We collect the old/new thin ptrs to avoid overlapping borrows.
                let (old_thin, new_thin) = {
                    let entry = match gc_maps.objects.get_mut(*id) {
                        Some(e) => e,
                        None => continue,
                    };
                    let old_thin = entry.ptr.get_thin_ptr();

                    // Reconstruct fat pointer: preserve vtable, update data pointer.
                    // SAFETY: new_raw points to a valid copy of the original object
                    // within the compact buffer. The object was fully copied above.
                    let mut fat_ptr_bytes = [0u8; std::mem::size_of::<*const dyn Trace>()];
                    std::ptr::copy_nonoverlapping(
                        &entry.ptr as *const *const dyn Trace as *const u8,
                        fat_ptr_bytes.as_mut_ptr(),
                        fat_ptr_bytes.len(),
                    );
                    // Overwrite data pointer (first pointer-sized field)
                    std::ptr::copy_nonoverlapping(
                        &(*new_raw as *const u8) as *const *const u8 as *const u8,
                        fat_ptr_bytes.as_mut_ptr(),
                        std::mem::size_of::<*const u8>(),
                    );
                    // SAFETY: read_unaligned because fat_ptr_bytes is a [u8] (align 1)
                    // but *const dyn Trace requires pointer alignment.
                    let new_fat_ptr: *const dyn Trace =
                        std::ptr::read_unaligned(fat_ptr_bytes.as_ptr() as *const *const dyn Trace);
                    entry.ptr = new_fat_ptr;
                    let new_thin = entry.ptr.get_thin_ptr();

                    // Update root_ref_count_ptr: same offset, new base address.
                    let rrc_offset = entry.root_ref_count_ptr as usize - old_thin;
                    entry.root_ref_count_ptr = (new_thin + rrc_offset) as *const Cell<usize>;

                    // Replace memory tracking with compact block reference.
                    // Use std::mem::replace to properly move the old GcObjMem out,
                    // avoiding double-drop of Arc refs in Compact/Tlab variants.
                    let old_mem = std::mem::replace(
                        &mut entry.mem,
                        GcObjMem::Compact(compact_block.clone()),
                    );
                    old_mems.push((old_thin as *mut u8, old_mem, *layout));
                    entry.layout = CompactLayout::from_layout(*layout);

                    (old_thin, new_thin)
                };

                // Update ptr_to_object mapping (borrows gc_maps, not gc_maps.objects)
                gc_maps.ptr_to_object.remove(&old_thin);
                gc_maps.ptr_to_object.insert(new_thin, *id);

                // Update card table
                self.card_table.remove_object(old_thin, *id);
                self.card_table.register_object(new_thin, *id);
            }

            // Relocate tracer pointers: tell each GcInternal to update its gc_ptr field
            for (_obj_id, obj_entry) in gc_maps.objects.iter() {
                obj_entry.tracers.for_each_tracer(|tracer_ptr| {
                    for (_, old_raw, new_raw, _) in &relocations {
                        // SAFETY: tracer_ptr is valid while gc_maps lock is held; relocate is called during STW with exclusive access.
                        (&*tracer_ptr).relocate(*old_raw, *new_raw);
                    }
                });
            }

            drop(gc_maps);

            for (ptr, mem, layout) in old_mems {
                // SAFETY: Old allocation is no longer in use; objects have been relocated to the compact buffer.
                mem.dealloc_mem(ptr, layout);
            }

            num_objects
        }
    }
}

/// RAII guard for a compacted memory block.
/// Frees the contiguous buffer when all objects in it have been collected.
pub(crate) struct CompactBlock {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: The block is only freed when all Arc references are dropped (during sweep),
// and the GC ensures no references exist at that point.
unsafe impl Send for CompactBlock {}
unsafe impl Sync for CompactBlock {}

impl Drop for CompactBlock {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            // SAFETY: ptr was allocated with this layout via alloc().
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }
}

/// Thread-local garbage collector wrapper.
/// Each thread gets its own `LocalGarbageCollector` via the `LOCAL_GC` thread-local.
/// Provides allocation (`create_gc`), collection (`collect`), and incremental/concurrent
/// collection methods. All operations are single-threaded (no cross-thread sharing).
/// Allocation threshold for triggering automatic Gen0 collection on the owning thread.
/// Local GCs collect on allocation (not from a background thread) to avoid data races
/// on non-atomic Cell/RefCell internals.
const LOCAL_GC_ALLOC_THRESHOLD: usize = 1_000;

/// Thread-local garbage collector.
///
/// Each thread gets its own instance via the [`LOCAL_GC`] thread-local.
/// Provides allocation, collection, and diagnostic methods. All operations
/// are single-threaded — for cross-thread GC, use [`sync::GlobalGarbageCollector`].
///
/// Collection is triggered automatically after every `LOCAL_GC_ALLOC_THRESHOLD`
/// allocations, or manually via `collect()` / `collect_generation()`.
pub struct LocalGarbageCollector {
    pub(crate) core: GarbageCollector,
    /// Thread-local allocation buffer for fast bump-pointer allocation.
    /// Lazily initialized on first allocation.
    tlab: Option<crate::tlab::Tlab>,
}

// SAFETY: Delegates to GarbageCollector which protects all state with Mutex/atomics.
// The Tlab is only accessed from the owning thread (thread_local).
unsafe impl Sync for LocalGarbageCollector {}
unsafe impl Send for LocalGarbageCollector {}

#[allow(dead_code)]
impl LocalGarbageCollector {
    fn new() -> LocalGarbageCollector {
        LocalGarbageCollector {
            core: GarbageCollector::new(),
            tlab: None,
        }
    }

    /// Try to allocate A and B in a single TLAB bump (combined allocation).
    /// Returns `Some((ptr_a, ptr_b, combined_mem, combined_layout))` on success,
    /// or `None` if the TLAB cannot satisfy the combined layout.
    ///
    /// # Safety
    /// Both returned pointers must be initialized before use.
    #[inline]
    unsafe fn try_alloc_combined_tlab<A: Sized, B: Sized>(
        &mut self,
    ) -> Option<(*mut A, *mut B, GcObjMem, Layout)> {
        let layout_a = Layout::new::<A>();
        let layout_b = Layout::new::<B>();
        let (combined, offset_b) = layout_a.extend(layout_b).ok()?;
        let combined = combined.pad_to_align();

        let tlab_result = if let Some(ref mut tlab) = self.tlab {
            tlab.alloc_or_grow(combined)
        } else {
            let mut tlab = crate::tlab::Tlab::new()?;
            let r = tlab.alloc(combined);
            self.tlab = Some(tlab);
            r
        };
        let (ptr, block) = tlab_result?;
        let ptr_a = ptr as *mut A;
        // SAFETY: offset_b is within the combined allocation.
        let ptr_b = unsafe { ptr.add(offset_b) } as *mut B;
        Some((ptr_a, ptr_b, GcObjMem::Tlab(block), combined))
    }

    /// Try to allocate memory from the TLAB. Falls back to system allocator if
    /// the TLAB is exhausted or cannot be created.
    ///
    /// Returns `(typed_ptr, (GcObjMem, Layout))` like `GarbageCollector::alloc_mem`.
    ///
    /// # Safety
    /// The returned pointer must be initialized before use.
    #[inline]
    unsafe fn alloc_mem_tlab<T: Sized>(&mut self) -> (*mut T, (GcObjMem, Layout)) {
        let layout = Layout::new::<T>();

        // Try TLAB allocation first
        if let Some(ref mut tlab) = self.tlab {
            if let Some((ptr, block)) = tlab.alloc_or_grow(layout) {
                let type_ptr: *mut T = ptr as *mut T;
                return (type_ptr, (GcObjMem::Tlab(block), layout));
            }
        } else {
            // Lazily initialize TLAB
            if let Some(mut tlab) = crate::tlab::Tlab::new() {
                if let Some((ptr, block)) = tlab.alloc(layout) {
                    let type_ptr: *mut T = ptr as *mut T;
                    self.tlab = Some(tlab);
                    return (type_ptr, (GcObjMem::Tlab(block), layout));
                }
                self.tlab = Some(tlab);
            }
        }

        // Fall back to system allocator
        // SAFETY: Delegates to alloc_mem which upholds its own safety invariants.
        unsafe { self.core.alloc_mem::<T>() }
    }

    /// Fallible TLAB allocation with GC retry.
    ///
    /// # Safety
    /// The returned pointer must be initialized before use.
    unsafe fn try_alloc_mem_tlab<T: Sized>(
        &mut self,
    ) -> Result<(*mut T, (GcObjMem, Layout)), GcAllocError> {
        let layout = Layout::new::<T>();

        // Try TLAB allocation first
        if let Some(ref mut tlab) = self.tlab {
            if let Some((ptr, block)) = tlab.alloc_or_grow(layout) {
                let type_ptr: *mut T = ptr as *mut T;
                return Ok((type_ptr, (GcObjMem::Tlab(block), layout)));
            }
        } else {
            // Lazily initialize TLAB
            if let Some(mut tlab) = crate::tlab::Tlab::new() {
                if let Some((ptr, block)) = tlab.alloc(layout) {
                    let type_ptr: *mut T = ptr as *mut T;
                    self.tlab = Some(tlab);
                    return Ok((type_ptr, (GcObjMem::Tlab(block), layout)));
                }
                self.tlab = Some(tlab);
            }
        }

        // Fall back to system allocator with GC retry
        // SAFETY: Delegates to try_alloc_mem_with_gc which upholds its own safety invariants.
        unsafe { self.core.try_alloc_mem_with_gc::<T>() }
    }

    /// Check allocation count and trigger Gen0 collection if threshold exceeded.
    /// The threshold adapts based on collection efficiency: if a collection frees
    /// little garbage, the threshold doubles (up to 100K) to avoid wasteful scans
    /// of a mostly-live heap. If a collection frees a lot, the threshold shrinks
    /// (down to 1K) to reclaim memory promptly.
    #[inline]
    unsafe fn maybe_collect(&self) {
        let count = self.core.allocation_count.get();
        let threshold = self.core.alloc_threshold.get();
        if count >= threshold {
            let stats = unsafe { self.core.collect_generation(Generation::Gen0) };
            // Adapt threshold based on collection efficiency
            if stats.objects_scanned > 0 {
                let collected_ratio =
                    stats.objects_collected as f64 / stats.objects_scanned as f64;
                if collected_ratio < 0.05 {
                    // Less than 5% garbage — heap is mostly live, double threshold
                    let new = (threshold * 2).min(100_000);
                    self.core.alloc_threshold.set(new);
                } else if collected_ratio > 0.25 {
                    // More than 25% garbage — shrink threshold to collect sooner
                    let new = (threshold / 2).max(LOCAL_GC_ALLOC_THRESHOLD);
                    self.core.alloc_threshold.set(new);
                }
            }
        }
    }

    #[inline]
    unsafe fn create_gc<T>(&mut self, t: T, region: RegionId) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            // Try combined allocation: [GcPtr<T> | GcInternal<T>] in one TLAB bump.
            let (gc_ptr, gc_inter_ptr, obj_mem, obj_layout, tracer_list);
            if let Some((a, b, mem, layout)) =
                self.try_alloc_combined_tlab::<GcPtr<T>, GcInternal<T>>()
            {
                gc_ptr = a;
                gc_inter_ptr = b;
                obj_mem = mem;
                obj_layout = layout;
                tracer_list = TracerList::new_inline(b as *const dyn Trace);
            } else {
                // Fallback: separate allocations
                let (a, (mem_a, layout_a)) = self.alloc_mem_tlab::<GcPtr<T>>();
                let (b, (mem_b, layout_b)) = self.alloc_mem_tlab::<GcInternal<T>>();
                gc_ptr = a;
                gc_inter_ptr = b;
                obj_mem = mem_a;
                obj_layout = layout_a;
                tracer_list = TracerList::new(TracerInfo {
                    tracer_ptr: b as *const dyn Trace,
                    mem: mem_b,
                    layout: layout_b,
                });
            }
            std::ptr::write(gc_ptr, GcPtr::new(t));

            let gc_maps = self.core.gc_maps_unsync();
            unsafe fn dealloc_gc_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (*(ptr as *const GcPtr<T>)).t.finalize();
                    }));
                    std::ptr::drop_in_place(ptr as *mut GcPtr<T>);
                }
            }
            let root_ref_count_ptr = &(*gc_ptr).info.root_ref_count as *const Cell<usize>;
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: obj_mem,
                layout: CompactLayout::from_layout(obj_layout),
                gen_survive: 0,
                dealloc_fn: dealloc_gc_ptr::<T>,

                tracers: tracer_list,
                region,
                root_ref_count_ptr,
            });
            self.core.track_alloc(obj_layout.size());
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr, obj_id));

            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
                object_id: obj_id,
            };
            // SAFETY: gc_ptr was just initialized above; call reset_root directly
            // instead of going through gc.internal_ptr→ptr.get() (avoids triple dereference).
            (*gc_ptr).reset_root();
            self.core.allocation_count.set(self.core.allocation_count.get() + 1);
            self.maybe_collect();
            gc
        }
    }

    unsafe fn clone_from_gc<T>(&self, gc: &Gc<T>) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();
            // SAFETY: internal_ptr is valid for the lifetime of the source Gc handle.
            let object_id = (*gc.internal_ptr).object_id;

            let gc_maps = self.core.gc_maps_unsync();
            // RC hybrid: push tracer to the object's tracers Vec
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.tracers.push(TracerInfo {
                    tracer_ptr: gc_inter_ptr as *const dyn Trace,
                    mem: mem_info_internal_ptr.0,
                    layout: mem_info_internal_ptr.1,
                });
            }
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc.ptr, object_id));
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
                object_id,
            };
            // SAFETY: Both internal_ptr and ptr are valid; internal_ptr was just initialized, ptr comes from the source Gc.
            (*gc.ptr).reset_root();
            gc
        }
    }

    unsafe fn create_gc_cell<T>(&mut self, t: T, region: RegionId) -> GcCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_ptr, gc_cell_inter_ptr, obj_mem, obj_layout, tracer_list);
            if let Some((a, b, mem, layout)) =
                self.try_alloc_combined_tlab::<RefCell<GcPtr<T>>, GcCellInternal<T>>()
            {
                gc_ptr = a;
                gc_cell_inter_ptr = b;
                obj_mem = mem;
                obj_layout = layout;
                tracer_list = TracerList::new_inline(b as *const dyn Trace);
            } else {
                let (a, (mem_a, layout_a)) = self.alloc_mem_tlab::<RefCell<GcPtr<T>>>();
                let (b, (mem_b, layout_b)) = self.alloc_mem_tlab::<GcCellInternal<T>>();
                gc_ptr = a;
                gc_cell_inter_ptr = b;
                obj_mem = mem_a;
                obj_layout = layout_a;
                tracer_list = TracerList::new(TracerInfo {
                    tracer_ptr: b as *const dyn Trace,
                    mem: mem_b,
                    layout: layout_b,
                });
            }
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let gc_maps = self.core.gc_maps_unsync();
            unsafe fn dealloc_gc_cell_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (*(*(ptr as *const RefCell<GcPtr<T>>)).as_ptr())
                            .t
                            .finalize();
                    }));
                    std::ptr::drop_in_place(ptr as *mut RefCell<GcPtr<T>>);
                }
            }
            let root_ref_count_ptr =
                &(*(*gc_ptr).as_ptr()).info.root_ref_count as *const Cell<usize>;
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: obj_mem,
                layout: CompactLayout::from_layout(obj_layout),
                gen_survive: 0,
                dealloc_fn: dealloc_gc_cell_ptr::<T>,

                tracers: tracer_list,
                region,
                root_ref_count_ptr,
            });
            self.core.track_alloc(obj_layout.size());
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_cell_inter_ptr,
                GcCellInternal::new(gc_ptr, obj_id),
            );

            let gc = GcCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
                object_id: obj_id,
            };
            // SAFETY: gc_ptr/gc.ptr was initialized above; call reset_root directly.
            (*gc.ptr).reset_root();
            self.core.allocation_count.set(self.core.allocation_count.get() + 1);
            self.maybe_collect();
            gc
        }
    }

    unsafe fn clone_from_gc_cell<T>(&self, gc: &GcCell<T>) -> GcCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_inter_ptr, mem_info) = self.core.alloc_mem::<GcCellInternal<T>>();
            // SAFETY: internal_ptr is valid for the lifetime of the source GcCell handle.
            let object_id = (*gc.internal_ptr).object_id;

            let gc_maps = self.core.gc_maps_unsync();
            // RC hybrid: push tracer to the object's tracers Vec
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.tracers.push(TracerInfo {
                    tracer_ptr: gc_inter_ptr as *const dyn Trace,
                    mem: mem_info.0,
                    layout: mem_info.1,
                });
            }
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_inter_ptr,
                GcCellInternal::new(gc.ptr, object_id),
            );
            let gc = GcCell {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
                object_id,
            };
            // SAFETY: Both internal_ptr and ptr are valid; internal_ptr was just initialized, ptr comes from the source GcCell.
            (*gc.ptr).reset_root();
            gc
        }
    }

    /// Fallible version of `create_gc`. Returns `Err(GcAllocError)` on OOM.
    unsafe fn try_create_gc<T>(&mut self, t: T, region: RegionId) -> Result<Gc<T>, GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_ptr, gc_inter_ptr, obj_mem, obj_layout, tracer_list);
            if let Some((a, b, mem, layout)) =
                self.try_alloc_combined_tlab::<GcPtr<T>, GcInternal<T>>()
            {
                gc_ptr = a;
                gc_inter_ptr = b;
                obj_mem = mem;
                obj_layout = layout;
                tracer_list = TracerList::new_inline(b as *const dyn Trace);
            } else {
                let (a, (mem_a, layout_a)) = self.try_alloc_mem_tlab::<GcPtr<T>>()?;
                let (b, (mem_b, layout_b)) =
                    match self.try_alloc_mem_tlab::<GcInternal<T>>() {
                        Ok(v) => v,
                        Err(e) => {
                            mem_a.dealloc_mem(a as *mut u8, layout_a);
                            return Err(e);
                        }
                    };
                gc_ptr = a;
                gc_inter_ptr = b;
                obj_mem = mem_a;
                obj_layout = layout_a;
                tracer_list = TracerList::new(TracerInfo {
                    tracer_ptr: b as *const dyn Trace,
                    mem: mem_b,
                    layout: layout_b,
                });
            }
            std::ptr::write(gc_ptr, GcPtr::new(t));

            let gc_maps = self.core.gc_maps_unsync();
            unsafe fn dealloc_gc_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (*(ptr as *const GcPtr<T>)).t.finalize();
                    }));
                    std::ptr::drop_in_place(ptr as *mut GcPtr<T>);
                }
            }
            let root_ref_count_ptr = &(*gc_ptr).info.root_ref_count as *const Cell<usize>;
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: obj_mem,
                layout: CompactLayout::from_layout(obj_layout),
                gen_survive: 0,
                dealloc_fn: dealloc_gc_ptr::<T>,

                tracers: tracer_list,
                region,
                root_ref_count_ptr,
            });
            self.core.track_alloc(obj_layout.size());
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr, obj_id));

            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
                object_id: obj_id,
            };
            // SAFETY: gc_ptr/gc.ptr was initialized above; call reset_root directly.
            (*gc.ptr).reset_root();
            self.core.allocation_count.set(self.core.allocation_count.get() + 1);
            self.maybe_collect();
            Ok(gc)
        }
    }

    /// Fallible version of `create_gc_cell`. Returns `Err(GcAllocError)` on OOM.
    unsafe fn try_create_gc_cell<T>(
        &mut self,
        t: T,
        region: RegionId,
    ) -> Result<GcCell<T>, GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_ptr, gc_cell_inter_ptr, obj_mem, obj_layout, tracer_list);
            if let Some((a, b, mem, layout)) =
                self.try_alloc_combined_tlab::<RefCell<GcPtr<T>>, GcCellInternal<T>>()
            {
                gc_ptr = a;
                gc_cell_inter_ptr = b;
                obj_mem = mem;
                obj_layout = layout;
                tracer_list = TracerList::new_inline(b as *const dyn Trace);
            } else {
                let (a, (mem_a, layout_a)) = self.try_alloc_mem_tlab::<RefCell<GcPtr<T>>>()?;
                let (b, (mem_b, layout_b)) =
                    match self.try_alloc_mem_tlab::<GcCellInternal<T>>() {
                        Ok(v) => v,
                        Err(e) => {
                            mem_a.dealloc_mem(a as *mut u8, layout_a);
                            return Err(e);
                        }
                    };
                gc_ptr = a;
                gc_cell_inter_ptr = b;
                obj_mem = mem_a;
                obj_layout = layout_a;
                tracer_list = TracerList::new(TracerInfo {
                    tracer_ptr: b as *const dyn Trace,
                    mem: mem_b,
                    layout: layout_b,
                });
            }
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let gc_maps = self.core.gc_maps_unsync();
            unsafe fn dealloc_gc_cell_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (*(*(ptr as *const RefCell<GcPtr<T>>)).as_ptr())
                            .t
                            .finalize();
                    }));
                    std::ptr::drop_in_place(ptr as *mut RefCell<GcPtr<T>>);
                }
            }
            let root_ref_count_ptr =
                &(*(*gc_ptr).as_ptr()).info.root_ref_count as *const Cell<usize>;
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: obj_mem,
                layout: CompactLayout::from_layout(obj_layout),
                gen_survive: 0,
                dealloc_fn: dealloc_gc_cell_ptr::<T>,

                tracers: tracer_list,
                region,
                root_ref_count_ptr,
            });
            self.core.track_alloc(obj_layout.size());
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_cell_inter_ptr,
                GcCellInternal::new(gc_ptr, obj_id),
            );

            let gc = GcCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
                object_id: obj_id,
            };
            // SAFETY: gc_ptr/gc.ptr was initialized above; call reset_root directly.
            (*gc.ptr).reset_root();
            self.core.allocation_count.set(self.core.allocation_count.get() + 1);
            self.maybe_collect();
            Ok(gc)
        }
    }

    /// Create a new strong Gc<T> from a weak reference (for upgrade).
    unsafe fn upgrade_weak<T>(&self, weak: &GcWeak<T>) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();

            let gc_maps = self.core.gc_maps_unsync();
            let object_id = weak.object_id;
            // RC hybrid: push tracer to the object's tracers Vec
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.tracers.push(TracerInfo {
                    tracer_ptr: gc_inter_ptr as *const dyn Trace,
                    mem: mem_info_internal_ptr.0,
                    layout: mem_info_internal_ptr.1,
                });
            }

            // SAFETY: Pointer was just allocated via alloc_mem and is properly aligned for GcInternal<T>.
            std::ptr::write(
                gc_inter_ptr,
                GcInternal::new(weak.ptr, object_id),
            );
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: weak.ptr,
                object_id,
            };
            // SAFETY: Both internal_ptr (just initialized) and ptr (verified alive via STW lock) are valid.
            (*gc.ptr).reset_root();
            gc
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect(&self) {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect().
        unsafe {
            self.core.collect();
        }
    }

    /// Collect objects up to the specified generation.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_generation(&self, max_gen: Generation) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_generation().
        unsafe { self.core.collect_generation(max_gen) }
    }

    #[allow(dead_code)]
    unsafe fn collect_all(&self) {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_all().
        unsafe {
            self.core.collect_all();
        }
    }

    /// Parallel collection: serial mark phase, parallel sweep/dealloc using rayon.
    /// Collects objects in generations 0..=max_gen.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn collect_parallel(&self, max_gen: Generation) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_parallel().
        unsafe { self.core.collect_parallel(max_gen) }
    }

    /// Collection with parallel mark AND parallel sweep using rayon.
    /// Parallelizes root tracing across worker threads for faster marking
    /// on large heaps with many roots.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn collect_parallel_mark(&self, max_gen: Generation) -> CollectionStats {
        unsafe { self.core.collect_parallel_mark(max_gen) }
    }

    pub(crate) unsafe fn remove_tracer(&self, object_id: ObjectId, tracer_ptr: *const u8) {
        // SAFETY: Thread-local GC, no concurrent access; caller provides valid object_id and tracer_ptr.
        unsafe {
            self.core.remove_tracer_unsync(object_id, tracer_ptr);
        }
    }

    /// Begin an incremental collection cycle.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn begin_collection(&self, max_gen: Generation) {
        // SAFETY: Caller upholds the safety contract; delegates to core.begin_collection().
        unsafe {
            self.core.begin_collection(max_gen);
        }
    }

    /// Process a batch of gray objects. Returns true when marking is complete.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn mark_step(&self, budget: usize) -> bool {
        // SAFETY: Caller upholds the safety contract; delegates to core.mark_step().
        unsafe { self.core.mark_step(budget) }
    }

    /// Finish incremental collection: sweep white objects, promote survivors.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn finish_collection(&self) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.finish_collection().
        unsafe { self.core.finish_collection() }
    }

    /// Run a complete incremental collection with the given step budget.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_incremental(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_incremental().
        unsafe { self.core.collect_incremental(max_gen, step_budget) }
    }

    /// Begin a concurrent collection: short STW to snapshot roots + edges.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn begin_concurrent_collection(&self, max_gen: Generation) {
        // SAFETY: Caller upholds the safety contract; delegates to core.begin_concurrent_collection().
        unsafe {
            self.core.begin_concurrent_collection(max_gen);
        }
    }

    /// Process gray objects using edge snapshot. NO STW lock — safe for concurrent use.
    pub(crate) fn concurrent_mark_step(&self, budget: usize) -> bool {
        self.core.concurrent_mark_step(budget)
    }

    /// Run a complete concurrent collection (snapshot → concurrent mark → STW sweep).
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_concurrent(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_concurrent().
        unsafe { self.core.collect_concurrent(max_gen, step_budget) }
    }

    /// Time-budgeted incremental mark step. See `GarbageCollector::mark_step_timed`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn mark_step_timed(&self, max_duration: Duration) -> bool {
        // SAFETY: Caller upholds the safety contract; delegates to core.mark_step_timed().
        unsafe { self.core.mark_step_timed(max_duration) }
    }

    /// Run a complete incremental collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_incremental_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_incremental_timed().
        unsafe {
            self.core
                .collect_incremental_timed(max_gen, max_step_duration)
        }
    }

    /// Time-budgeted concurrent mark step. NO STW lock — safe for concurrent use.
    pub(crate) fn concurrent_mark_step_timed(&self, max_duration: Duration) -> bool {
        self.core.concurrent_mark_step_timed(max_duration)
    }

    /// Run a complete concurrent collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_concurrent_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_concurrent_timed().
        unsafe {
            self.core
                .collect_concurrent_timed(max_gen, max_step_duration)
        }
    }

    /// Create a new region. Future allocations go into this region.
    pub fn new_region(&self) -> LocalRegionId {
        LocalRegionId(self.core.new_region())
    }

    /// Get the current allocation region.
    pub fn current_region(&self) -> LocalRegionId {
        LocalRegionId(self.core.current_region())
    }

    /// Collect only objects in the specified region.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_region(&self, region: LocalRegionId) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_region().
        unsafe { self.core.collect_region(region.0) }
    }

    /// G1-style "Garbage First" collection. Marks all objects, then sweeps
    /// regions with the highest garbage ratio first, stopping when the elapsed
    /// time reaches `pause_target`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are held
    /// across this call (stop-the-world requirement).
    pub(crate) unsafe fn collect_garbage_first(&self, pause_target: Duration) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.
        unsafe { self.core.collect_garbage_first(pause_target) }
    }

    /// Return per-region liveness statistics without running a collection.
    pub fn region_stats(&self) -> Vec<crate::generation::RegionStats> {
        self.core.region_stats()
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> GcStats {
        self.core.stats()
    }

    /// Set custom promotion thresholds (how many collections an object must survive
    /// before being promoted to the next generation).
    pub fn set_promotion_config(&self, config: PromotionConfig) {
        self.core.set_promotion_config(config);
    }

    /// Get the current promotion config.
    pub fn promotion_config(&self) -> PromotionConfig {
        self.core.promotion_config()
    }

    /// Register a callback invoked after each collection cycle.
    pub fn set_on_collection(&self, callback: impl Fn(&CollectionStats) + Send + Sync + 'static) {
        self.core.set_on_collection(callback);
    }

    /// Remove the collection callback.
    pub fn clear_on_collection(&self) {
        self.core.clear_on_collection();
    }

    /// Compact the heap by copying all live objects into a contiguous buffer.
    /// Runs a full collection first, then relocates surviving objects for
    /// improved cache locality and reduced fragmentation.
    ///
    /// Returns the number of objects compacted.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    pub(crate) unsafe fn compact(&self) -> usize {
        unsafe { self.core.compact() }
    }
}

// ---- Feature-gated public access for benchmarks and integration tests ----

#[cfg(feature = "_internal")]
impl LocalGarbageCollector {
    pub unsafe fn _collect(&self) {
        unsafe { self.collect() }
    }
    pub unsafe fn _collect_generation(&self, max_gen: Generation) -> CollectionStats {
        unsafe { self.collect_generation(max_gen) }
    }
    pub unsafe fn _collect_incremental(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        unsafe { self.collect_incremental(max_gen, step_budget) }
    }
    pub unsafe fn _collect_incremental_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        unsafe { self.collect_incremental_timed(max_gen, max_step_duration) }
    }
    pub unsafe fn _collect_concurrent(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        unsafe { self.collect_concurrent(max_gen, step_budget) }
    }
    pub unsafe fn _collect_concurrent_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        unsafe { self.collect_concurrent_timed(max_gen, max_step_duration) }
    }
    pub unsafe fn _collect_region(&self, region: LocalRegionId) -> CollectionStats {
        unsafe { self.collect_region(region) }
    }
    pub unsafe fn _collect_garbage_first(&self, pause_target: Duration) -> CollectionStats {
        unsafe { self.collect_garbage_first(pause_target) }
    }
    pub unsafe fn _compact(&self) -> usize {
        unsafe { self.compact() }
    }
}

// ---- LocalRegionId helpers for thread-local GC ----

impl LocalRegionId {
    /// Allocate a new `Gc<T>` in this region (thread-local collector).
    pub fn gc<T: 'static + Sized + Trace>(self, t: T) -> Gc<T> {
        Gc::new_in(t, self)
    }

    /// Fallible `Gc<T>` allocation in this region (thread-local collector).
    pub fn try_gc<T: 'static + Sized + Trace>(self, t: T) -> Result<Gc<T>, GcAllocError> {
        Gc::try_new_in(t, self)
    }

    /// Allocate a new `GcCell<T>` in this region (thread-local collector).
    pub fn gc_cell<T: 'static + Sized + Trace>(self, t: T) -> GcCell<T> {
        GcCell::new_in(t, self)
    }

    /// Fallible `GcCell<T>` allocation in this region (thread-local collector).
    pub fn try_gc_cell<T: 'static + Sized + Trace>(self, t: T) -> Result<GcCell<T>, GcAllocError> {
        GcCell::try_new_in(t, self)
    }
}

impl PartialEq for &LocalGarbageCollector {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(
            *self as *const LocalGarbageCollector,
            *other as *const LocalGarbageCollector,
        )
    }
}

/// Callback type for a collection strategy. Receives the GC and an "active" flag;
/// may spawn a background thread and return its `JoinHandle`.
pub type LocalStrategyFn =
    Box<dyn FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;

/// Pluggable collection strategy for the thread-local GC.
/// Controls when and how garbage collection runs (e.g., periodic background thread,
/// allocation-triggered, or manual). Use `set_strategy` to swap at runtime.
///
/// Internally stores a raw pointer to the `LocalGarbageCollector` to avoid
/// Stacked Borrows violations during thread-local destruction ordering.
pub struct LocalStrategy {
    gc: Cell<*const LocalGarbageCollector>,
    is_active: AtomicBool,
    strategy_func: RefCell<LocalStrategyFn>,
    join_handle: RefCell<Option<JoinHandle<()>>>,
}

impl LocalStrategy {
    fn new<StrategyFn>(
        gc_ptr: *const LocalGarbageCollector,
        strategy_fn: StrategyFn,
    ) -> LocalStrategy
    where
        StrategyFn: 'static
            + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
    {
        LocalStrategy {
            gc: Cell::new(gc_ptr),
            is_active: AtomicBool::new(false),
            strategy_func: RefCell::new(Box::new(strategy_fn)),
            join_handle: RefCell::new(None),
        }
    }

    /// Replace the current collection strategy. Stops the current strategy first if active.
    pub fn set_strategy<StrategyFn>(&self, strategy_fn: StrategyFn)
    where
        StrategyFn: 'static
            + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
    {
        if self.is_active() {
            self.stop();
        }
        let _ = self.strategy_func.replace(Box::new(strategy_fn));
    }

    /// Returns `true` if the strategy's background collection is currently running.
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Start the collection strategy (e.g., spawn a background collection thread).
    pub fn start(&'static self) {
        self.is_active.store(true, Ordering::Release);
        // SAFETY: gc pointer is valid for the lifetime of the thread-local and is only
        // accessed from the owning thread.
        let gc_ref = unsafe { &*self.gc.get() };
        self.join_handle
            .replace((*(self.strategy_func.borrow_mut()))(
                gc_ref,
                &self.is_active,
            ));
    }

    /// Stop the collection strategy and join any background thread.
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        if let Some(join_handle) = self.join_handle.borrow_mut().take() {
            join_handle
                .join()
                .expect("LocalStrategy::stop: strategy thread panicked");
        }
    }
}

impl Drop for LocalStrategy {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }
}

thread_local! {
    pub static LOCAL_GC: RefCell<LocalGarbageCollector> = RefCell::new(LocalGarbageCollector::new());
    pub static LOCAL_GC_STRATEGY: RefCell<LocalStrategy> = {
        LOCAL_GC.with(move |gc| {
            // Use as_ptr() directly to get a raw pointer without creating an intermediate
            // reference, avoiding Stacked Borrows provenance issues during thread-local
            // destruction ordering.
            let gc_ptr = gc.as_ptr() as *const LocalGarbageCollector;
            // NOTE: Local GCs use allocation-triggered collection on the owning thread
            // instead of registering with the background thread. Collecting from a
            // background thread would be a data race because thread-local objects use
            // non-atomic Cell/RefCell internals. The background thread is only used
            // for the global/sync GC which uses atomic operations.
            RefCell::new(LocalStrategy::new(gc_ptr, move |_local_gc, _| {
                None
            }))
        })
    };
}

#[cfg(test)]
mod tests {
    use crate::gc::{Gc, GcCell, LOCAL_GC};
    use crate::{Finalize, Trace};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Clean residual state from previous tests that may have run on this thread.
    fn clean_gc_state() {
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
    }

    #[test]
    fn object_entry_size() {
        // Track ObjectEntry size to catch regressions.
        // After weak_alive move + GcObjMem ptr removal: should be ~96B.
        let size = std::mem::size_of::<super::ObjectEntry>();
        eprintln!("ObjectEntry size: {}B", size);
        eprintln!("GcObjMem size: {}B", std::mem::size_of::<super::GcObjMem>());
        eprintln!("TracerList size: {}B", std::mem::size_of::<super::TracerList>());
        assert!(size <= 88, "ObjectEntry grew unexpectedly to {size}B");
    }

    #[test]
    fn one_object() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
        });
        let _one = Gc::new(1);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .total_tracers()
                    - baseline,
                1
            );
        });
    }

    #[test]
    fn gc_collect_one_from_one() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
        });
        {
            let _one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .total_tracers()
                    - baseline,
                0
            );
        });
    }

    #[test]
    #[allow(unused_assignments)]
    fn two_objects_reassign() {
        // Reassigning drops the old Gc (remove_tracer removes it from trs),
        // so only 1 tracer remains for the surviving Gc.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
        });
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| {
            assert_eq!(
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .total_tracers()
                    - baseline,
                1
            );
        });
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_after_reassign() {
        // After reassign, one live Gc remains. collect() keeps live objects.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
        });
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .total_tracers()
                    - baseline,
                1
            );
        });
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_two_from_two() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
        });
        {
            let mut one = Gc::new(1);
            one = Gc::new(2);
            drop(one);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .total_tracers()
                    - baseline,
                0
            );
        });
    }

    #[test]
    fn ptr_to_object_cleaned_on_collect() {
        clean_gc_state();
        let baseline_trs = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
        });
        let baseline_p2o = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .ptr_to_object
                .len()
        });
        {
            let _one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .total_tracers()
                    - baseline_trs,
                0
            );
            assert_eq!(
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .ptr_to_object
                    .len()
                    - baseline_p2o,
                0
            );
        });
    }

    #[test]
    fn collect_from_another_thread() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
        });
        let _one = Gc::new(42);
        LOCAL_GC.with(|gc| {
            let gc_ptr = gc.as_ptr();
            let gc_ref = unsafe { &*gc_ptr };
            std::thread::scope(|s| {
                s.spawn(|| unsafe {
                    gc_ref.collect();
                });
            });
            assert_eq!(
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .total_tracers()
                    - baseline,
                1
            );
        });
    }

    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct DropCounter {
        _value: String,
    }

    impl Trace for DropCounter {
        fn is_root(&self) -> bool {
            false
        }
        fn reset_root(&self) {}
        fn trace(&self) {}
        fn reset(&self) {}
        fn is_traceable(&self) -> bool {
            false
        }
    }

    impl Finalize for DropCounter {
        fn finalize(&self) {}
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn collect_calls_drop_on_gc_objects() {
        clean_gc_state();
        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let _obj = Gc::new(DropCounter {
                _value: String::from("hello"),
            });
        }
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        assert_eq!(
            DROP_COUNT.load(Ordering::SeqCst),
            1,
            "Drop should be called during collect"
        );
    }

    #[test]
    fn clone_from_registers_with_gc() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
        });
        let source = Gc::new(42);
        let mut target = Gc::new(99);
        target.clone_from(&source);
        LOCAL_GC.with(|gc| {
            let delta = gc
                .borrow()
                .core
                .lock_gc_maps()
                .total_tracers()
                - baseline;
            // source + target + the new clone's tracer = at least 2 alive
            assert!(
                delta >= 2,
                "clone_from should register new tracer with GC, got {delta}"
            );
        });
    }

    struct CyclicNode {
        next: std::cell::RefCell<Option<Gc<CyclicNode>>>,
    }

    impl Trace for CyclicNode {
        fn is_root(&self) -> bool {
            false
        }
        fn reset_root(&self) {
            if let Some(ref gc) = *self.next.borrow() {
                gc.reset_root();
            }
        }
        fn trace(&self) {
            if let Some(ref gc) = *self.next.borrow() {
                gc.trace();
            }
        }
        fn reset(&self) {
            if let Some(ref gc) = *self.next.borrow() {
                gc.reset();
            }
        }
        fn is_traceable(&self) -> bool {
            false
        }
    }

    impl Finalize for CyclicNode {
        fn finalize(&self) {}
    }

    #[test]
    fn objects_start_in_gen0() {
        clean_gc_state();
        let _obj = Gc::new(42);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let gc_maps = gc_ref
                .core
                .lock_gc_maps();
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation() == crate::generation::Generation::Gen0),
                "newly allocated objects should be in Gen0"
            );
        });
    }

    #[test]
    fn gen0_collection_does_not_collect_promoted_objects() {
        clean_gc_state();
        let _obj = Gc::new(100);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            // Survive 3 Gen0 collections to promote to Gen1
            for _ in 0..3 {
                unsafe {
                    gc_ref
                        .core
                        .collect_generation(crate::generation::Generation::Gen0);
                }
            }
            // Verify object was promoted to Gen1
            {
                let gc_maps = gc_ref
                    .core
                    .lock_gc_maps();
                assert!(
                    gc_maps
                        .objects
                        .values()
                        .any(|e| e.generation() == crate::generation::Generation::Gen1),
                    "object should be promoted to Gen1 after surviving 3 Gen0 collections"
                );
            }

            let baseline_objs = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            // Gen0-only collection should not touch Gen1 objects
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
            let after_objs = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            assert_eq!(
                baseline_objs, after_objs,
                "Gen0 collection must not collect Gen1 objects"
            );
        });
    }

    #[test]
    fn promotion_gen0_to_gen1() {
        clean_gc_state();
        let _obj = Gc::new(77);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            // Gen0 promotion threshold is 3
            for _ in 0..2 {
                unsafe {
                    gc_ref
                        .core
                        .collect_generation(crate::generation::Generation::Gen0);
                }
            }
            // Still Gen0 after 2 survivals
            {
                let gc_maps = gc_ref
                    .core
                    .lock_gc_maps();
                assert!(
                    gc_maps
                        .objects
                        .values()
                        .all(|e| e.generation() == crate::generation::Generation::Gen0),
                    "object should still be in Gen0 after 2 survivals"
                );
            }

            // Third survival triggers promotion
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
            let gc_maps = gc_ref
                .core
                .lock_gc_maps();
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation() == crate::generation::Generation::Gen1),
                "object should be promoted to Gen1 after 3 survivals"
            );
        });
    }

    #[test]
    fn promotion_gen1_to_gen2() {
        clean_gc_state();
        let _obj = Gc::new(55);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            // Promote to Gen1 first (3 Gen0 collections)
            for _ in 0..3 {
                unsafe {
                    gc_ref
                        .core
                        .collect_generation(crate::generation::Generation::Gen0);
                }
            }
            // Now survive 5 Gen1 collections to promote to Gen2
            for _ in 0..5 {
                unsafe {
                    gc_ref
                        .core
                        .collect_generation(crate::generation::Generation::Gen1);
                }
            }
            let gc_maps = gc_ref
                .core
                .lock_gc_maps();
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation() == crate::generation::Generation::Gen2),
                "object should be promoted to Gen2 after surviving Gen1 threshold"
            );
        });
    }

    #[test]
    fn dead_object_collected_by_gen0() {
        // RC hybrid: object is freed immediately when last handle drops.
        // Verify the object is gone without needing an explicit Gen0 collection.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        {
            let _obj = Gc::new(99);
        }
        let after = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert_eq!(
            after, baseline,
            "RC hybrid should free object immediately when last handle drops"
        );
    }

    #[test]
    fn allocation_count_tracks_new_objects() {
        clean_gc_state();
        let before = LOCAL_GC.with(|gc| gc.borrow().core.allocation_count.get());
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let after = LOCAL_GC.with(|gc| gc.borrow().core.allocation_count.get());
        assert_eq!(
            after - before,
            2,
            "allocation_count should increment per new object"
        );
    }

    #[test]
    fn write_barrier_marks_old_gen_object_card_dirty() {
        clean_gc_state();
        use crate::gc::GcCell;
        let cell = GcCell::new(Option::<Gc<i32>>::None);

        // Promote cell's object to Gen1 (survive 3 Gen0 collections)
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            for _ in 0..3 {
                unsafe {
                    gc_ref
                        .core
                        .collect_generation(crate::generation::Generation::Gen0);
                }
            }
            // Verify it's in Gen1 (check by value to avoid fat pointer comparison issues)
            let gc_maps = gc_ref
                .core
                .lock_gc_maps();
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation() == crate::generation::Generation::Gen1),
                "cell should be promoted to Gen1 after 3 Gen0 collections"
            );
        });

        // Card table should be clean before mutation
        LOCAL_GC.with(|gc| {
            assert!(
                gc.borrow().core.card_table.is_clean(),
                "card table should be clean before write barrier"
            );
        });

        // Mutate via borrow_mut() — triggers write barrier
        // (must be outside LOCAL_GC.with borrow to avoid RefCell conflict)
        {
            let young = Gc::new(42);
            **cell.borrow_mut() = Some(young);
        }

        // Verify cell's card is now dirty in the card table
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            assert!(
                gc_ref.core.card_table.has_dirty(),
                "write barrier should mark card dirty in card table"
            );
        });
    }

    #[test]
    fn card_table_cleared_on_full_collection() {
        clean_gc_state();
        use crate::gc::GcCell;
        let cell = GcCell::new(Option::<Gc<i32>>::None);

        // Promote to Gen1
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            for _ in 0..3 {
                unsafe {
                    gc_ref
                        .core
                        .collect_generation(crate::generation::Generation::Gen0);
                }
            }
        });

        // Trigger write barrier (outside LOCAL_GC borrow)
        **cell.borrow_mut() = Some(Gc::new(99));

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            assert!(
                gc_ref.core.card_table.has_dirty(),
                "card table should have dirty cards after write barrier"
            );

            // Full collection (Gen2) should clear card table dirty flags
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen2);
            }
            assert!(
                gc_ref.core.card_table.is_clean(),
                "card table should be clean after full collection"
            );
        });
    }

    #[test]
    fn write_barrier_preserves_young_object_during_gen0_collection() {
        // Key correctness test: an old-gen object holds a reference to a young
        // object. Without write barrier, the young object would be collected.
        clean_gc_state();
        use crate::gc::GcCell;
        let cell = GcCell::new(Option::<Gc<i32>>::None);

        // Promote cell to Gen1
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            for _ in 0..3 {
                unsafe {
                    gc_ref
                        .core
                        .collect_generation(crate::generation::Generation::Gen0);
                }
            }
        });

        // Create young object and store reference in old-gen cell
        let young = Gc::new(777);
        **cell.borrow_mut() = Some(young.clone());
        drop(young); // Drop user handle — only reference is from old-gen cell

        // Gen0 collection — the young object must survive because
        // it's referenced by an old-gen object tracked via the card table
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen0);
            }
            let objs_after = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            assert_eq!(
                objs_before, objs_after,
                "young object referenced from old-gen (via card table) must survive Gen0 collection"
            );
        });
    }

    #[test]
    fn compact_preserves_live_objects() {
        clean_gc_state();
        let n = 100;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(Gc::new(i as i32));
        }
        // Drop half to create fragmentation
        for i in (0..n).step_by(2) {
            v[i] = Gc::new(0);
        }
        LOCAL_GC.with(|gc| unsafe { gc.borrow().compact() });
        // Verify values are still accessible after compaction
        assert_eq!(***v.get(1).unwrap(), 1);
        assert_eq!(***v.get(3).unwrap(), 3);
        drop(v);
        LOCAL_GC.with(|gc| unsafe { gc.borrow().collect() });
    }

    #[test]
    fn compact_repeated_iterations() {
        clean_gc_state();
        for _ in 0..20 {
            let n = 1000;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(Gc::new(i as i32));
            }
            for i in (0..n).step_by(2) {
                v[i] = Gc::new(0);
            }
            LOCAL_GC.with(|gc| unsafe { gc.borrow().compact() });
            assert_eq!(***v.get(1).unwrap(), 1);
            drop(v);
            LOCAL_GC.with(|gc| unsafe { gc.borrow().collect() });
        }
    }

    #[test]
    fn no_write_barrier_for_gen0_objects() {
        clean_gc_state();
        use crate::gc::GcCell;
        let cell = GcCell::new(Option::<Gc<i32>>::None);

        // Cell is still in Gen0 — borrow_mut should NOT mark card dirty
        **cell.borrow_mut() = Some(Gc::new(1));

        LOCAL_GC.with(|gc| {
            assert!(
                gc.borrow().core.card_table.is_clean(),
                "Gen0 objects should not mark cards dirty in card table"
            );
        });
    }

    #[test]
    fn cyclic_gc_collect_does_not_overflow() {
        // Create a cycle: a → b → a. Without cycle protection in trace()/reset(),
        // collect() will recurse infinitely and overflow the stack.
        let result = std::thread::Builder::new()
            .stack_size(256 * 1024) // small stack to detect infinite recursion quickly
            .spawn(|| {
                let a = Gc::new(CyclicNode {
                    next: std::cell::RefCell::new(None),
                });
                let b = Gc::new(CyclicNode {
                    next: std::cell::RefCell::new(Some(a.clone())),
                });
                *a.next.borrow_mut() = Some(b.clone());
                // Drop user handles — cycle keeps objects alive internally
                drop(a);
                drop(b);
                LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut().collect();
                });
            })
            .unwrap()
            .join();
        assert!(
            result.is_ok(),
            "collect() with cyclic references must not stack overflow"
        );
    }

    #[test]
    fn weak_upgrade_returns_some_while_alive() {
        clean_gc_state();
        let strong = Gc::new(42);
        let weak = Gc::downgrade(&strong);
        let upgraded = weak.upgrade();
        assert!(
            upgraded.is_some(),
            "upgrade should succeed while object is alive"
        );
        assert_eq!(**upgraded.unwrap(), 42);
    }

    #[test]
    fn weak_upgrade_returns_none_after_collection() {
        clean_gc_state();
        let weak = {
            let strong = Gc::new(99);
            Gc::downgrade(&strong)
        }; // strong dropped here
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        assert!(
            weak.upgrade().is_none(),
            "upgrade should return None after object is collected"
        );
    }

    #[test]
    fn weak_clone_shares_alive_flag() {
        clean_gc_state();
        let strong = Gc::new(7);
        let weak1 = Gc::downgrade(&strong);
        let weak2 = weak1.clone();
        drop(strong);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        assert!(weak1.upgrade().is_none());
        assert!(weak2.upgrade().is_none());
    }

    #[test]
    fn weak_does_not_prevent_collection() {
        // RC hybrid: object is freed immediately when last strong handle drops.
        // Weak reference does not count as a tracer, so it doesn't affect RC.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        let weak = {
            let strong = Gc::new(123);
            Gc::downgrade(&strong)
        };
        // RC should have freed the object immediately
        let after = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert_eq!(
            after, baseline,
            "RC hybrid should free object immediately, weak doesn't prevent it"
        );
        assert!(
            weak.upgrade().is_none(),
            "weak upgrade must return None after strong dropped"
        );
    }

    // --- Incremental tri-color collection tests ---

    #[test]
    fn incremental_collects_dead_objects() {
        // RC hybrid frees non-cyclic objects immediately. Verify object is gone.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        {
            let _obj = Gc::new(42);
        }
        // Object already freed by RC
        let after = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert_eq!(
            after, baseline,
            "RC hybrid should free dead object immediately"
        );
        // Incremental collection still works correctly (just nothing to collect)
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let _stats = unsafe {
                gc_ref
                    .core
                    .collect_incremental(crate::generation::Generation::Gen2, 10)
            };
        });
    }

    #[test]
    fn incremental_preserves_live_objects() {
        clean_gc_state();
        let _live = Gc::new(99);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            let stats = unsafe {
                gc_ref
                    .core
                    .collect_incremental(crate::generation::Generation::Gen2, 10)
            };
            assert_eq!(
                stats.objects_collected, 0,
                "incremental must not collect live objects"
            );
            assert_eq!(
                gc_ref
                    .core
                    .lock_gc_maps()
                    .objects
                    .len(),
                objs_before
            );
        });
    }

    #[test]
    fn incremental_step_by_step() {
        // Verify that begin_collection + mark_step + finish_collection works.
        // With RC hybrid, dead non-cyclic objects are already freed, so test with live objects.
        clean_gc_state();
        let _live = Gc::new(2);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            unsafe {
                gc_ref
                    .core
                    .begin_collection(crate::generation::Generation::Gen2);
                while !gc_ref.core.mark_step(1) {}
                let stats = gc_ref.core.finish_collection();
                assert_eq!(
                    stats.objects_collected, 0,
                    "live object must not be collected"
                );
            }
            let objs_after = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            assert_eq!(objs_before, objs_after, "live object count must not change");
        });
    }

    #[test]
    fn incremental_with_child_references() {
        // Test that trace_children discovers object graph correctly
        clean_gc_state();
        use crate::gc::GcCell;
        let parent = GcCell::new(Option::<Gc<i32>>::None);
        let child = Gc::new(42);
        **parent.borrow_mut() = Some(child.clone());
        drop(child); // only reference is from parent

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            unsafe {
                gc_ref
                    .core
                    .collect_incremental(crate::generation::Generation::Gen2, 1);
            }
            let objs_after = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            assert_eq!(
                objs_before, objs_after,
                "child referenced from parent must survive incremental collection"
            );
        });
    }

    #[test]
    fn incremental_promotes_survivors() {
        clean_gc_state();
        let _obj = Gc::new(55);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            // 3 Gen0 incremental collections should promote to Gen1
            for _ in 0..3 {
                unsafe {
                    gc_ref
                        .core
                        .collect_incremental(crate::generation::Generation::Gen0, 10);
                }
            }
            let gc_maps = gc_ref
                .core
                .lock_gc_maps();
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation() == crate::generation::Generation::Gen1),
                "incremental collection should promote survivors"
            );
        });
    }

    #[test]
    fn incremental_phase_tracking() {
        clean_gc_state();
        let _obj = Gc::new(1);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            // Idle before collection
            assert_eq!(
                gc_ref
                    .core
                    .incremental
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .phase,
                crate::generation::CollectionPhase::Idle
            );

            unsafe {
                gc_ref
                    .core
                    .begin_collection(crate::generation::Generation::Gen2);
            }
            assert_eq!(
                gc_ref
                    .core
                    .incremental
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .phase,
                crate::generation::CollectionPhase::Marking
            );

            while unsafe { !gc_ref.core.mark_step(1) } {}
            // Still marking until finish_collection
            assert_eq!(
                gc_ref
                    .core
                    .incremental
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .phase,
                crate::generation::CollectionPhase::Marking
            );

            unsafe {
                gc_ref.core.finish_collection();
            }
            assert_eq!(
                gc_ref
                    .core
                    .incremental
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .phase,
                crate::generation::CollectionPhase::Idle
            );
        });
    }

    // --- OOM handling tests ---

    #[test]
    fn try_new_succeeds_for_normal_allocation() {
        clean_gc_state();
        let result = Gc::try_new(42);
        assert!(
            result.is_ok(),
            "try_new should succeed for normal allocation"
        );
        assert_eq!(**result.unwrap(), 42);
    }

    #[test]
    fn try_new_gc_ref_cell_succeeds() {
        clean_gc_state();
        use crate::gc::GcCell;
        let result = GcCell::try_new(100);
        assert!(result.is_ok(), "try_new should succeed for GcCell");
        let cell = result.unwrap();
        assert_eq!(**cell.borrow(), 100);
    }

    #[test]
    fn try_new_object_participates_in_gc() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        {
            let _obj = Gc::try_new(77).unwrap();
        }
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        let after = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert_eq!(
            after, baseline,
            "try_new objects should be collected when dead"
        );
    }

    #[test]
    fn gc_alloc_error_is_display() {
        use crate::gc::GcAllocError;
        let err = GcAllocError;
        assert_eq!(format!("{}", err), "GC allocation failed");
    }

    // --- Diagnostics API tests ---

    #[test]
    fn stats_reports_live_objects() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().stats().live_objects);
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert_eq!(
            stats.live_objects - baseline,
            2,
            "stats should report 2 new live objects"
        );
    }

    #[test]
    fn stats_reports_live_tracers() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().stats().live_tracers);
        let _a = Gc::new(1);
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert_eq!(
            stats.live_tracers - baseline,
            1,
            "stats should report 1 new tracer"
        );
    }

    #[test]
    fn stats_reports_heap_size() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().stats().heap_size);
        let _a = Gc::new(42i32);
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(
            stats.heap_size > baseline,
            "heap_size should increase after allocation"
        );
    }

    #[test]
    fn stats_reports_generation_counts() {
        clean_gc_state();
        let _obj = Gc::new(1);
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(stats.gen0_objects >= 1, "new object should be in Gen0");

        // Promote to Gen1
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            for _ in 0..3 {
                unsafe {
                    gc_ref
                        .core
                        .collect_generation(crate::generation::Generation::Gen0);
                }
            }
        });
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(stats.gen1_objects >= 1, "promoted object should be in Gen1");
    }

    #[test]
    fn stats_tracks_total_collections() {
        clean_gc_state();
        let before = LOCAL_GC.with(|gc| gc.borrow().stats().total_collections);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        let after = LOCAL_GC.with(|gc| gc.borrow().stats().total_collections);
        assert_eq!(
            after - before,
            2,
            "total_collections should increment per collect call"
        );
    }

    #[test]
    fn stats_tracks_last_collection() {
        clean_gc_state();
        // With RC hybrid, dead objects are already freed. Just verify stats recording works.
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(
            stats.last_collection.is_some(),
            "last_collection should be Some after collect"
        );
    }

    #[test]
    fn stats_reports_allocation_count() {
        clean_gc_state();
        // After clean_gc_state (which calls collect), allocation_count is reset
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(
            stats.allocation_count >= 2,
            "allocation_count should track new objects"
        );
    }

    // --- Debug impl tests ---

    #[test]
    fn debug_gc_prints_value() {
        clean_gc_state();
        let gc = Gc::new(42);
        let s = format!("{:?}", gc);
        assert_eq!(s, "Gc(42)");
    }

    #[test]
    fn debug_gc_ref_cell_prints_value() {
        clean_gc_state();
        let cell = GcCell::new(7);
        let s = format!("{:?}", cell);
        assert_eq!(s, "GcCell(7)");
    }

    #[test]
    fn debug_gc_weak_alive() {
        clean_gc_state();
        let strong = Gc::new(1);
        let weak = Gc::downgrade(&strong);
        assert_eq!(format!("{:?}", weak), "GcWeak(alive)");
    }

    #[test]
    fn debug_gc_weak_dead() {
        clean_gc_state();
        let weak = {
            let strong = Gc::new(1);
            Gc::downgrade(&strong)
        };
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        assert_eq!(format!("{:?}", weak), "GcWeak(dead)");
    }

    // --- Cycle tests ---

    #[test]
    fn self_cycle_collected() {
        // A single node pointing to itself should be collected without overflow.
        let result = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let a = Gc::new(CyclicNode {
                    next: std::cell::RefCell::new(None),
                });
                *a.next.borrow_mut() = Some(a.clone());
                drop(a);
                LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut().collect();
                });
            })
            .unwrap()
            .join();
        assert!(result.is_ok(), "self-cycle must not stack overflow");
    }

    #[test]
    fn three_node_cycle_no_overflow() {
        // a → b → c → a — verify no stack overflow on collect
        let result = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let a = Gc::new(CyclicNode {
                    next: std::cell::RefCell::new(None),
                });
                let b = Gc::new(CyclicNode {
                    next: std::cell::RefCell::new(None),
                });
                let c = Gc::new(CyclicNode {
                    next: std::cell::RefCell::new(None),
                });
                *a.next.borrow_mut() = Some(b.clone());
                *b.next.borrow_mut() = Some(c.clone());
                *c.next.borrow_mut() = Some(a.clone());
                drop(a);
                drop(b);
                drop(c);
                LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut().collect();
                });
            })
            .unwrap()
            .join();
        assert!(result.is_ok(), "3-node cycle must not stack overflow");
    }

    struct VecNode {
        children: std::cell::RefCell<Vec<Gc<VecNode>>>,
    }
    impl Trace for VecNode {
        fn is_root(&self) -> bool {
            false
        }
        fn reset_root(&self) {
            for gc in self.children.borrow().iter() {
                gc.reset_root();
            }
        }
        fn trace(&self) {
            for gc in self.children.borrow().iter() {
                gc.trace();
            }
        }
        fn reset(&self) {
            for gc in self.children.borrow().iter() {
                gc.reset();
            }
        }
        fn is_traceable(&self) -> bool {
            false
        }
        fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
            for gc in self.children.borrow().iter() {
                gc.trace_children(children);
            }
        }
    }
    impl Finalize for VecNode {
        fn finalize(&self) {}
    }

    #[test]
    fn cycle_in_vec_no_overflow() {
        // Two nodes each holding the other in their Vec of children.
        let result = std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(|| {
                let a = Gc::new(VecNode {
                    children: std::cell::RefCell::new(vec![]),
                });
                let b = Gc::new(VecNode {
                    children: std::cell::RefCell::new(vec![]),
                });
                let b_clone = b.clone();
                a.children.borrow_mut().push(b_clone);
                let a_clone = a.clone();
                b.children.borrow_mut().push(a_clone);
                drop(a);
                drop(b);
                LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut().collect();
                });
            })
            .unwrap()
            .join();
        assert!(result.is_ok(), "cycle in Vec must not stack overflow");
    }

    // --- Concurrent marking tests (Strategy 21) ---

    #[test]
    fn concurrent_collects_dead_objects_via_cycle() {
        // RC hybrid handles simple drops, but cycles need tracing.
        // Test concurrent marking on a cycle.
        clean_gc_state();
        {
            let a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
        }
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            let _stats = unsafe {
                gc_ref
                    .core
                    .collect_concurrent(crate::generation::Generation::Gen2, 10)
            };
            let objs_after = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            assert!(
                objs_after < objs_before,
                "concurrent collection should collect cyclic garbage"
            );
        });
    }

    #[test]
    fn concurrent_preserves_live_objects() {
        clean_gc_state();
        let _live = Gc::new(42);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            let stats = unsafe {
                gc_ref
                    .core
                    .collect_concurrent(crate::generation::Generation::Gen2, 10)
            };
            assert_eq!(
                stats.objects_collected, 0,
                "concurrent must not collect live objects"
            );
            let objs_after = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            assert_eq!(objs_before, objs_after);
        });
    }

    #[test]
    fn concurrent_step_by_step() {
        clean_gc_state();
        let _live = Gc::new(99);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            unsafe {
                gc_ref
                    .core
                    .begin_concurrent_collection(crate::generation::Generation::Gen2);
            }
            // concurrent_mark_step does NOT require STW
            while !gc_ref.core.concurrent_mark_step(1) {}
            let stats = unsafe { gc_ref.core.finish_collection() };
            assert_eq!(
                stats.objects_collected, 0,
                "live objects must survive concurrent collection"
            );
        });
    }

    // --- Region-based collection tests (Strategy 22) ---

    #[test]
    fn objects_assigned_to_current_region() {
        clean_gc_state();
        let _obj = Gc::new(42);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let gc_maps = gc_ref
                .core
                .lock_gc_maps();
            assert!(
                gc_maps.objects.values().any(|e| e.region.0 == 0),
                "new objects should be in region 0 (default)"
            );
        });
    }

    #[test]
    fn new_region_changes_allocation_target() {
        clean_gc_state();
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let region1 = gc_ref.core.new_region();
            assert!(region1.0 > 0, "new region should have non-zero ID");
            assert_eq!(gc_ref.core.current_region(), region1);
        });
        let _obj = Gc::new(77);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let region_id = gc_ref.core.current_region();
            let gc_maps = gc_ref
                .core
                .lock_gc_maps();
            assert!(
                gc_maps.objects.values().any(|e| e.region == region_id),
                "object should be in the new region"
            );
        });
    }

    #[test]
    fn collect_region_sweeps_only_target_region() {
        clean_gc_state();
        // Create objects in region 0 (default)
        let _keep = Gc::new(1);

        // Switch to a new region and create + drop objects there
        LOCAL_GC.with(|gc| {
            gc.borrow().core.new_region();
        });
        // Objects in new region will be RC-freed when dropped, so create a cycle
        let region_id = LOCAL_GC.with(|gc| gc.borrow().core.current_region());
        {
            let a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
        }

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let stats = unsafe { gc_ref.core.collect_region(region_id) };
            assert!(
                stats.objects_collected >= 2,
                "should collect cycle in target region"
            );
        });

        // Object in region 0 should still be alive
        assert_eq!(**_keep, 1, "object in other region must survive");
    }

    // --- RC hybrid tests (Strategy 23) ---

    #[test]
    fn rc_immediate_dealloc_on_last_drop() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        {
            let _obj = Gc::new(42);
            let during = LOCAL_GC.with(|gc| {
                gc.borrow()
                    .core
                    .lock_gc_maps()
                    .objects
                    .len()
            });
            assert_eq!(during - baseline, 1, "object should be alive during scope");
        }
        let after = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert_eq!(
            after, baseline,
            "RC should free object immediately when last handle drops"
        );
    }

    #[test]
    fn rc_clone_keeps_object_alive() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        let a = Gc::new(99);
        let b = a.clone();
        drop(a);
        // Object should still be alive (b holds a reference)
        let after_drop_a = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert_eq!(
            after_drop_a - baseline,
            1,
            "object should survive when clone exists"
        );
        assert_eq!(**b, 99);
        drop(b);
        // Now object should be freed
        let after_drop_b = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert_eq!(
            after_drop_b, baseline,
            "RC should free object when last clone drops"
        );
    }

    #[test]
    fn rc_does_not_free_cyclic_objects() {
        // Cycles keep ref_count > 0, so RC doesn't free them.
        // Tracing GC is needed.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        {
            let a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
        }
        // Cycle objects should still be alive (ref_count > 0 due to internal refs)
        let after = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert!(after > baseline, "cyclic objects should not be freed by RC");

        // Tracing GC should collect them
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        let final_count = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        assert_eq!(
            final_count, baseline,
            "tracing GC should collect cycles that RC cannot"
        );
    }

    #[test]
    fn rc_ref_count_tracks_correctly() {
        // Verify RC behavior: object stays alive while any handle exists,
        // freed when last handle drops.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()
        });
        let a = Gc::new(42);
        assert_eq!(
            LOCAL_GC.with(|gc| gc
                .borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()),
            baseline + 1,
            "one object after creation"
        );

        let b = a.clone();
        // Clone shares the same object — count unchanged
        assert_eq!(
            LOCAL_GC.with(|gc| gc
                .borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()),
            baseline + 1,
            "still one object after clone"
        );

        drop(b);
        // Object survives (a still holds it)
        assert_eq!(
            LOCAL_GC.with(|gc| gc
                .borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()),
            baseline + 1,
            "object alive while handle exists"
        );
        assert_eq!(**a, 42);

        drop(a);
        // Last handle dropped — RC frees object
        assert_eq!(
            LOCAL_GC.with(|gc| gc
                .borrow()
                .core
                .lock_gc_maps()
                .objects
                .len()),
            baseline,
            "object freed after last handle dropped"
        );
    }

    #[test]
    fn timed_incremental_collection_works() {
        use std::time::Duration;
        clean_gc_state();
        // Keep objects alive during collection to verify timed marking scans them
        let v: Vec<Gc<i32>> = (0..100).map(|i| Gc::new(i)).collect();

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            unsafe {
                let stats = gc_ref.core.collect_incremental_timed(
                    crate::generation::Generation::Gen2,
                    Duration::from_secs(1),
                );
                // All live objects should be scanned, none collected
                assert_eq!(stats.objects_collected, 0);
                assert!(stats.objects_scanned > 0);
            }
            let objs_after = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            assert_eq!(objs_before, objs_after, "no objects should be collected");
        });
        drop(v);
    }

    #[test]
    fn timed_concurrent_collection_works() {
        use std::time::Duration;
        clean_gc_state();
        let v: Vec<Gc<i32>> = (0..100).map(|i| Gc::new(i)).collect();

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            unsafe {
                let stats = gc_ref.core.collect_concurrent_timed(
                    crate::generation::Generation::Gen2,
                    Duration::from_secs(1),
                );
                assert_eq!(stats.objects_collected, 0);
                assert!(stats.objects_scanned > 0);
            }
            let objs_after = gc_ref
                .core
                .lock_gc_maps()
                .objects
                .len();
            assert_eq!(objs_before, objs_after, "no objects should be collected");
        });
        drop(v);
    }

    #[test]
    fn stats_tracks_collection_duration() {
        clean_gc_state();
        let _obj = Gc::new(42);
        drop(_obj);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            unsafe {
                gc_ref.core.collect();
            }
            let stats = gc_ref.stats();
            let last = stats
                .last_collection
                .expect("should have a last collection");
            assert!(
                last.duration > std::time::Duration::ZERO,
                "duration should be > 0"
            );
        });
    }

    #[test]
    fn stats_tracks_bytes_freed() {
        clean_gc_state();
        // Create two objects forming a cycle so RC hybrid does NOT eagerly free them.
        // Only mark-sweep collection can reclaim cycles.
        let a = Gc::new(GcCell::new(
            Option::<Gc<GcCell<Option<Gc<GcCell<Option<i32>>>>>>>::None,
        ));
        let b = Gc::new(GcCell::new(Some(a.clone())));
        // We just need objects that survive past drop for the collector to find them.
        // Instead, use a simpler approach: just allocate, drop, and collect in separate scopes.
        drop(a);
        drop(b);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow().core.collect();
        });
        LOCAL_GC.with(|gc| {
            let stats = gc.borrow().stats();
            let last = stats
                .last_collection
                .expect("should have a last collection");
            assert!(last.bytes_freed > 0, "bytes_freed should be > 0");
        });
    }

    #[test]
    fn stats_tracks_peak_heap_size() {
        clean_gc_state();
        let _obj1 = Gc::new(1_i64);
        let _obj2 = Gc::new(2_i64);
        let peak_with_objects = LOCAL_GC.with(|gc| gc.borrow().stats().peak_heap_size);
        assert!(peak_with_objects > 0, "peak should be > 0 after alloc");
        drop(_obj1);
        drop(_obj2);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow().core.collect();
        });
        let peak_after = LOCAL_GC.with(|gc| gc.borrow().stats().peak_heap_size);
        assert!(
            peak_after >= peak_with_objects,
            "peak should not decrease after collection"
        );
    }

    #[test]
    fn on_collection_callback_fires() {
        clean_gc_state();
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();
        LOCAL_GC.with(|gc| {
            gc.borrow().set_on_collection(move |_stats| {
                called_clone.store(true, std::sync::atomic::Ordering::Release);
            });
        });
        let _obj = Gc::new(42);
        drop(_obj);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow().core.collect();
        });
        assert!(
            called.load(std::sync::atomic::Ordering::Acquire),
            "callback should have been called"
        );
        LOCAL_GC.with(|gc| {
            gc.borrow().clear_on_collection();
        });
    }

    #[test]
    fn on_collection_callback_panic_does_not_break_gc() {
        clean_gc_state();
        LOCAL_GC.with(|gc| {
            gc.borrow().set_on_collection(|_stats| {
                panic!("intentional panic in callback");
            });
        });
        let _obj = Gc::new(42);
        drop(_obj);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow().core.collect();
        });
        // GC should still work after panicking callback
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(stats.total_collections > 0);
        LOCAL_GC.with(|gc| {
            gc.borrow().clear_on_collection();
        });
    }

    // ---- G1 Garbage-First tests ----

    #[test]
    fn region_stats_tracks_allocations() {
        clean_gc_state();
        // Allocate objects in default region (0)
        let _a = Gc::new(1u64);
        let _b = Gc::new(2u64);

        // Switch to a new region and allocate there
        let region1 = LOCAL_GC.with(|gc| gc.borrow().new_region());

        let _c = Gc::new_in(3u64, region1);

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let stats = gc_ref.region_stats();
            // Should have at least 2 regions (default 0 + region1)
            assert!(
                stats.len() >= 2,
                "expected at least 2 regions, got {}",
                stats.len()
            );

            let r0_stats = stats
                .iter()
                .find(|s| s.region_id == crate::generation::RegionId(0));
            assert!(r0_stats.is_some(), "default region stats should exist");
            let r0 = r0_stats.unwrap();
            assert!(
                r0.object_count >= 2,
                "default region should have at least 2 objects, got {}",
                r0.object_count
            );
            assert!(r0.total_bytes > 0, "default region should have > 0 bytes");

            let r1_stats = stats
                .iter()
                .find(|s| s.region_id == crate::generation::RegionId(region1.id()));
            assert!(r1_stats.is_some(), "region1 stats should exist");
            let r1 = r1_stats.unwrap();
            assert_eq!(r1.object_count, 1, "region1 should have 1 object");
            assert!(r1.total_bytes > 0, "region1 should have > 0 bytes");
        });
    }

    #[test]
    fn collect_garbage_first_collects_dirtiest_region() {
        clean_gc_state();

        // Region 0: keep objects alive
        let _keep1 = Gc::new(100u64);
        let _keep2 = Gc::new(200u64);

        // Create a new region with garbage (cyclic references that become unreachable)
        let dirty_region = LOCAL_GC.with(|gc| gc.borrow().new_region());
        {
            let a = Gc::new_in(
                CyclicNode {
                    next: std::cell::RefCell::new(None),
                },
                dirty_region,
            );
            let b = Gc::new_in(
                CyclicNode {
                    next: std::cell::RefCell::new(None),
                },
                dirty_region,
            );
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            // Drop all handles => cycles become garbage
            drop(a);
            drop(b);
        }

        // Run G1 collection with a generous pause target
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let stats = unsafe { gc_ref.collect_garbage_first(std::time::Duration::from_secs(10)) };
            assert!(
                stats.objects_collected >= 2,
                "G1 should collect at least the 2 cyclic garbage objects, got {}",
                stats.objects_collected
            );
        });

        // Objects in region 0 should still be alive
        assert_eq!(**_keep1, 100u64);
        assert_eq!(**_keep2, 200u64);
    }

    #[test]
    fn collect_garbage_first_respects_pause_target() {
        clean_gc_state();

        // Create some objects
        let _keep = Gc::new(42u64);

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let start = std::time::Instant::now();
            let stats =
                unsafe { gc_ref.collect_garbage_first(std::time::Duration::from_millis(100)) };
            let elapsed = start.elapsed();
            // The method should return in reasonable time (within 2x the target + mark overhead)
            assert!(
                elapsed < std::time::Duration::from_secs(5),
                "G1 collection took too long: {:?}",
                elapsed
            );
            // Stats should be valid
            assert_eq!(stats.generation, crate::generation::Generation::Gen2);
            // Duration should be recorded
            assert!(stats.duration > std::time::Duration::ZERO);
        });
    }

    // --- Parallel collection tests ---

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_collect_basic() {
        clean_gc_state();
        // RC hybrid immediately frees non-cyclic objects on drop, so we need
        // cycles to ensure objects survive until the collector runs.
        {
            let a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let c = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            // a → b → c → a
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(c.clone());
            *c.next.borrow_mut() = Some(a.clone());
        }
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let stats = unsafe {
                gc_ref
                    .core
                    .collect_parallel(crate::generation::Generation::Gen2)
            };
            // All 3 cyclic objects should have been collected
            assert!(
                stats.objects_collected >= 3,
                "expected at least 3 objects collected, got {}",
                stats.objects_collected,
            );
            // bytes_freed should be positive
            assert!(
                stats.bytes_freed > 0,
                "expected bytes_freed > 0, got {}",
                stats.bytes_freed,
            );
            // No objects should remain from this batch
            let maps = gc_ref
                .core
                .lock_gc_maps();
            assert_eq!(
                maps.objects.len(),
                0,
                "all objects should be collected after parallel collect"
            );
        });
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_collect_preserves_live() {
        clean_gc_state();
        // Keep one object alive on the stack
        let live = Gc::new(42);
        // Create a dead cycle that the collector should reclaim
        {
            let dead_a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let dead_b = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *dead_a.next.borrow_mut() = Some(dead_b.clone());
            *dead_b.next.borrow_mut() = Some(dead_a.clone());
        }
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let stats = unsafe {
                gc_ref
                    .core
                    .collect_parallel(crate::generation::Generation::Gen2)
            };
            // The 2 dead cyclic objects should have been collected
            assert!(
                stats.objects_collected >= 2,
                "expected at least 2 objects collected, got {}",
                stats.objects_collected,
            );
            // The live object should still be accessible
            assert_eq!(**live, 42);
            // There should be exactly 1 object remaining (the live one)
            let maps = gc_ref
                .core
                .lock_gc_maps();
            assert_eq!(
                maps.objects.len(),
                1,
                "live object should survive parallel collect"
            );
        });
        drop(live);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_collect_cycles() {
        clean_gc_state();
        // Create a cycle: a → b → a
        {
            let a = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(CyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            // Drop user handles — cycle keeps objects alive internally
        }
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let stats = unsafe {
                gc_ref
                    .core
                    .collect_parallel(crate::generation::Generation::Gen2)
            };
            // Both cyclic objects should have been collected
            assert!(
                stats.objects_collected >= 2,
                "expected at least 2 cyclic objects collected, got {}",
                stats.objects_collected,
            );
            let maps = gc_ref
                .core
                .lock_gc_maps();
            assert_eq!(
                maps.objects.len(),
                0,
                "cyclic objects should be collected by parallel collect"
            );
        });
    }

    // ---- TLAB integration tests ----

    #[test]
    fn tlab_basic_allocation() {
        // Objects allocated via the TLAB should work identically to system-allocated objects.
        clean_gc_state();
        let a = Gc::new(42i32);
        let b = Gc::new(100i32);
        assert_eq!(**a, 42);
        assert_eq!(**b, 100);
        // Verify the objects are tracked by the GC
        LOCAL_GC.with(|gc| {
            let gc = gc.borrow();
            let maps = gc.core.lock_gc_maps();
            assert!(maps.objects.len() >= 2);
        });
        drop(a);
        drop(b);
        // After collection, all should be cleaned up
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
    }

    #[test]
    fn tlab_many_allocations() {
        // Allocate enough objects to exhaust one TLAB block and require a new one.
        clean_gc_state();
        // Use String which is ~24 bytes on stack + heap allocation.
        // With GcPtr overhead, we need many allocations to exhaust a 64KB TLAB block.
        let mut gcs: Vec<Gc<String>> = Vec::new();
        for i in 0..2000 {
            gcs.push(Gc::new(format!("item_{}", i)));
        }
        // All objects should be valid
        for (i, gc) in gcs.iter().enumerate() {
            assert_eq!(&***gc, &format!("item_{}", i));
        }
        // TLAB should have been initialized
        LOCAL_GC.with(|gc| {
            let gc = gc.borrow();
            assert!(
                gc.tlab.is_some(),
                "TLAB should be initialized after allocations"
            );
        });
        drop(gcs);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
    }

    #[test]
    fn tlab_fallback_to_system() {
        // When TLAB cannot service a request, it should fall back to the system allocator.
        // We test by allocating a very large String to stress the allocator path.
        // Even if the TLAB grows to accommodate, the fallback is always available.
        clean_gc_state();
        let big = Gc::new("x".repeat(256 * 1024));
        assert!(big.len() == 256 * 1024);
        drop(big);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
    }

    #[test]
    fn tlab_dealloc_frees_block() {
        // When all objects in a TLAB block are collected, the block should be freed.
        // We verify this indirectly: allocate objects, force a new block, drop all
        // objects from the first block, collect, and check that no crash occurs.
        clean_gc_state();
        // Allocate some objects into the initial TLAB block
        let a = Gc::new(1u64);
        let b = Gc::new(2u64);
        let c = Gc::new(3u64);
        assert_eq!(**a, 1);
        assert_eq!(**b, 2);
        assert_eq!(**c, 3);
        // Drop all and collect — the TLAB block's ref_count should reach 0
        drop(a);
        drop(b);
        drop(c);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
            // After collection, all objects should be gone
            let gc_ref = gc.borrow();
            let maps = gc_ref
                .core
                .lock_gc_maps();
            assert_eq!(
                maps.objects.len(),
                0,
                "all TLAB-allocated objects should be collected"
            );
        });
    }

    #[test]
    fn tlab_gc_ref_cell_allocation() {
        // GcCell should also use the TLAB for allocation.
        clean_gc_state();
        let cell = GcCell::new(String::from("hello"));
        {
            let borrowed = cell.borrow();
            assert_eq!(&***borrowed, "hello");
        }
        {
            let mut borrowed = cell.borrow_mut();
            **borrowed = String::from("world");
        }
        {
            let borrowed = cell.borrow();
            assert_eq!(&***borrowed, "world");
        }
        drop(cell);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
    }

    #[test]
    fn tlab_mixed_alloc_and_collect() {
        // Interleave TLAB allocations with collections to ensure correctness.
        clean_gc_state();
        for _ in 0..10 {
            let _a = Gc::new(42i32);
            let _b = Gc::new(String::from("test"));
            let _c = GcCell::new(vec![1, 2, 3]);
            // Drop _a, _b, _c at end of iteration
        }
        // Force collection
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
            let gc_ref = gc.borrow();
            let maps = gc_ref
                .core
                .lock_gc_maps();
            assert_eq!(maps.objects.len(), 0, "all objects should be collected");
        });
    }

    #[test]
    fn tlab_try_new_allocation() {
        // Fallible allocation should also use TLAB.
        clean_gc_state();
        let result = Gc::try_new(99u32);
        assert!(result.is_ok());
        let gc = result.unwrap();
        assert_eq!(**gc, 99);
        drop(gc);
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
    }
}
