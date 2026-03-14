use crate::generation::{
    CollectionPhase, CollectionStats, GcStats, Generation, MarkColor, PromotionConfig, RegionId,
};
use crate::slot_map::{ObjectId, SlotMap, TracerId};
use std::alloc::{Layout, alloc, dealloc};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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
/// - **`trace()`** must call `trace()` on every `Gc<T>`/`GcRefCell<T>` field.
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
pub type OptGcCell<T> = Option<GcRefCell<T>>;

struct GcInfo {
    root_ref_count: AtomicUsize,
}

impl GcInfo {
    fn new() -> GcInfo {
        GcInfo {
            root_ref_count: AtomicUsize::new(0),
        }
    }
}

/// Internal pointer wrapper used as the `Deref` target of `Gc<T>` and `GcRefCell<T>`.
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
        let prev = self.info.root_ref_count.fetch_add(1, Ordering::AcqRel);
        if prev == 0 {
            self.t.trace();
        }
    }

    fn reset(&self) {
        // Guard: only recurse into children on last reset (breaks cycles)
        let prev = self.info.root_ref_count.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.t.reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.info.root_ref_count.load(Ordering::Acquire) > 0
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.t.trace_children(children);
    }

    fn clear_trace(&self) {
        self.info.root_ref_count.store(0, Ordering::Release);
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
        let prev = self
            .borrow()
            .info
            .root_ref_count
            .fetch_add(1, Ordering::AcqRel);
        if prev == 0 {
            self.borrow().t.trace();
        }
    }

    fn reset(&self) {
        let prev = self
            .borrow()
            .info
            .root_ref_count
            .fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.borrow().t.reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.borrow().info.root_ref_count.load(Ordering::Acquire) > 0
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.borrow().t.trace_children(children);
    }

    fn clear_trace(&self) {
        self.borrow()
            .info
            .root_ref_count
            .store(0, Ordering::Release);
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

pub(crate) struct GcInternal<T>
where
    T: 'static + Sized + Trace,
{
    is_root: AtomicBool,
    ptr: *const GcPtr<T>,
    pub(crate) tracer_id: TracerId,
    pub(crate) object_id: ObjectId,
}

impl<T> GcInternal<T>
where
    T: 'static + Sized + Trace,
{
    fn new(ptr: *const GcPtr<T>, tracer_id: TracerId, object_id: ObjectId) -> GcInternal<T> {
        GcInternal {
            is_root: AtomicBool::new(true),
            ptr,
            tracer_id,
            object_id,
        }
    }
}

impl<T> Trace for GcInternal<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        self.is_root.load(Ordering::Acquire)
    }

    fn reset_root(&self) {
        if self.is_root.load(Ordering::Acquire) {
            self.is_root.store(false, Ordering::Release);
            unsafe {
                // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
                (*self.ptr).reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr).is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
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
    tracer_id: TracerId,
    object_id: ObjectId,
}

impl<T> Deref for Gc<T>
where
    T: 'static + Sized + Trace,
{
    type Target = GcPtr<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { &(*self.ptr) }
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
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().create_gc(t) })
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
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().try_create_gc(t) })
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
        let tracer_id = self.tracer_id;
        let object_id = self.object_id;
        let _ = LOCAL_GC.try_with(move |gc| unsafe {
            // Use try_borrow_mut to avoid panic from re-entrant drops:
            // RC hybrid dealloc may drop inner Gc fields whose drop also calls remove_tracer.
            if let Ok(gc) = gc.try_borrow_mut() {
                gc.remove_tracer(tracer_id, object_id);
            }
        });
    }
}

impl<T> Trace for Gc<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        unsafe {
            if self.internal_ptr.is_null() {
                return false;
            }
            // SAFETY: Null check above ensures the pointer is valid.
            (*self.internal_ptr).is_root()
        }
    }

    fn reset_root(&self) {
        unsafe {
            if self.internal_ptr.is_null() {
                return;
            }
            // SAFETY: Null check above ensures the pointer is valid.
            (*self.internal_ptr).reset_root();
        }
    }

    fn trace(&self) {
        unsafe {
            if self.ptr.is_null() {
                return;
            }
            // SAFETY: Null check above ensures the pointer is valid.
            (*self.ptr).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            if self.ptr.is_null() {
                return;
            }
            // SAFETY: Null check above ensures the pointer is valid.
            (*self.ptr).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        unsafe {
            if self.ptr.is_null() {
                return false;
            }
            // SAFETY: Null check above ensures the pointer is valid.
            (*self.ptr).is_traceable()
        }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for Gc<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

pub(crate) struct GcRefCellInternal<T>
where
    T: 'static + Sized + Trace,
{
    is_root: AtomicBool,
    ptr: *const RefCell<GcPtr<T>>,
    pub(crate) tracer_id: TracerId,
    pub(crate) object_id: ObjectId,
}

impl<T> GcRefCellInternal<T>
where
    T: 'static + Sized + Trace,
{
    fn new(
        ptr: *const RefCell<GcPtr<T>>,
        tracer_id: TracerId,
        object_id: ObjectId,
    ) -> GcRefCellInternal<T> {
        GcRefCellInternal {
            is_root: AtomicBool::new(true),
            ptr,
            tracer_id,
            object_id,
        }
    }
}

impl<T> Trace for GcRefCellInternal<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        self.is_root.load(Ordering::Acquire)
    }

    fn reset_root(&self) {
        if self.is_root.load(Ordering::Acquire) {
            self.is_root.store(false, Ordering::Release);
            unsafe {
                // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
                (*self.ptr).borrow().reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr).borrow().is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for GcRefCellInternal<T>
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
/// Thread-local only — for a cross-thread variant, use [`sync::GcRefCell<T>`].
pub struct GcRefCell<T>
where
    T: 'static + Sized + Trace,
{
    internal_ptr: *mut GcRefCellInternal<T>,
    ptr: *const RefCell<GcPtr<T>>,
    tracer_id: TracerId,
    object_id: ObjectId,
}

impl<T> Drop for GcRefCell<T>
where
    T: Sized + Trace,
{
    fn drop(&mut self) {
        let tracer_id = self.tracer_id;
        let object_id = self.object_id;
        let _ = LOCAL_GC.try_with(move |gc| unsafe {
            // Use try_borrow_mut to avoid panic from re-entrant drops:
            // RC hybrid dealloc may drop inner Gc fields whose drop also calls remove_tracer.
            if let Ok(gc) = gc.try_borrow_mut() {
                gc.remove_tracer(tracer_id, object_id);
            }
        });
    }
}

impl<T> Deref for GcRefCell<T>
where
    T: 'static + Sized + Trace,
{
    type Target = RefCell<GcPtr<T>>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { &(*self.ptr) }
    }
}

impl<T> std::fmt::Debug for GcRefCell<T>
where
    T: 'static + Sized + Trace + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GcRefCell").field(&**self.borrow()).finish()
    }
}

impl<T> GcRefCell<T>
where
    T: 'static + Sized + Trace,
{
    /// Allocate a new GC-managed interior-mutable cell on the thread-local collector.
    /// The contained value can be borrowed mutably via `borrow_mut()`.
    pub fn new(t: T) -> GcRefCell<T> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().create_gc_cell(t) })
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<GcRefCell<T>, GcAllocError> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().try_create_gc_cell(t) })
    }

    /// Mutable borrow with write barrier.
    /// Triggers the write barrier so that if this object is in an older generation,
    /// it gets added to the remembered set for young-generation collections.
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GcPtr<T>> {
        LOCAL_GC.with(|gc| {
            gc.borrow().core.write_barrier(self.ptr as *const dyn Trace);
        });
        // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr).borrow_mut() }
    }
}

impl<T> Clone for GcRefCell<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().clone_from_gc_cell(self) })
    }
}

impl<T> Trace for GcRefCell<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        // SAFETY: internal_ptr is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.internal_ptr).is_root() }
    }

    fn reset_root(&self) {
        unsafe {
            // SAFETY: internal_ptr is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.internal_ptr).reset_root();
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this GcRefCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr).borrow().is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for GcRefCell<T>
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
                .get_or_create_weak_alive(this.ptr as *const dyn Trace);
            GcWeak {
                alive,
                ptr: this.ptr,
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

pub(crate) type GcObjMem = *mut u8;

pub(crate) type DropFn = unsafe fn(*mut u8);
/// Function pointer that calls `finalize()` on the object at the given memory address.
/// Derives the `&dyn Finalize` reference at call time to avoid Stacked Borrows
/// provenance issues from storing a pre-computed `*const dyn Finalize`.
pub(crate) type FinalizeFn = unsafe fn(*mut u8);

/// Per-object metadata stored in a contiguous SlotMap.
pub(crate) struct ObjectEntry {
    pub(crate) ptr: *const dyn Trace,
    pub(crate) mem: GcObjMem,
    pub(crate) layout: Layout,
    pub(crate) generation: Generation,
    pub(crate) survive_count: u32,
    pub(crate) finalize_fn: FinalizeFn,
    pub(crate) drop_fn: DropFn,
    pub(crate) weak_alive: Option<Arc<AtomicBool>>,
    /// Number of tracers pointing to this object (RC hybrid).
    pub(crate) ref_count: usize,
    /// Region assignment for region-based collection.
    pub(crate) region: RegionId,
}

/// Per-tracer metadata stored in a contiguous SlotMap.
pub(crate) struct TracerEntry {
    pub(crate) tracer_ptr: *const dyn Trace,
    pub(crate) mem: GcObjMem,
    pub(crate) layout: Layout,
    pub(crate) object_id: ObjectId,
}

/// Unified GC maps replacing separate TracerMaps + ObjectMaps.
/// Single Mutex protects all state to simplify locking.
pub(crate) struct GcMaps {
    pub(crate) objects: SlotMap<ObjectId, ObjectEntry>,
    pub(crate) tracers: SlotMap<TracerId, TracerEntry>,
    /// Maps thin object pointer → ObjectId for trace_children resolution.
    pub(crate) ptr_to_object: HashMap<usize, ObjectId>,
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

/// Shared GC bookkeeping used by both LocalGarbageCollector and GlobalGarbageCollector.
/// All object/tracer state is consolidated under a single Mutex<GcMaps> for
/// simpler locking and slot-map-based O(1) operations.
pub(crate) struct GarbageCollector {
    pub(crate) gc_maps: Mutex<GcMaps>,
    pub(crate) allocation_count: AtomicUsize,
    pub(crate) remembered_set: Mutex<HashSet<ObjectId>>,
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
}

// SAFETY: All mutable state in GarbageCollector is behind Mutex or atomic types,
// ensuring exclusive access across threads. Raw pointers stored in ObjectEntry/TracerEntry
// are only dereferenced while holding the gc_maps Mutex lock.
unsafe impl Sync for GarbageCollector {}
unsafe impl Send for GarbageCollector {}

impl GarbageCollector {
    pub(crate) fn new() -> GarbageCollector {
        GarbageCollector {
            gc_maps: Mutex::new(GcMaps {
                objects: SlotMap::new(),
                tracers: SlotMap::new(),
                ptr_to_object: HashMap::new(),
            }),
            allocation_count: AtomicUsize::new(0),
            remembered_set: Mutex::new(HashSet::new()),
            stw_lock: RwLock::new(()),
            incremental: Mutex::new(IncrementalState::new()),
            total_collections: AtomicUsize::new(0),
            last_collection: Mutex::new(None),
            current_region: AtomicU32::new(0),
            next_region_id: AtomicU32::new(1),
            promotion_config: Mutex::new(PromotionConfig::default()),
        }
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
        let gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
        let mut gen0 = 0usize;
        let mut gen1 = 0usize;
        let mut gen2 = 0usize;
        let mut heap_size = 0usize;
        for entry in gc_maps.objects.values() {
            match entry.generation {
                Generation::Gen0 => gen0 += 1,
                Generation::Gen1 => gen1 += 1,
                Generation::Gen2 => gen2 += 1,
            }
            heap_size += entry.layout.size();
        }
        GcStats {
            heap_size,
            live_objects: gc_maps.objects.len(),
            live_tracers: gc_maps.tracers.len(),
            gen0_objects: gen0,
            gen1_objects: gen1,
            gen2_objects: gen2,
            total_collections: self.total_collections.load(Ordering::Relaxed),
            last_collection: *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
        }
    }

    pub(crate) unsafe fn alloc_mem<T>(&self) -> (*mut T, (GcObjMem, Layout))
    where
        T: Sized,
    {
        unsafe {
            let layout = Layout::new::<T>();
            // SAFETY: Layout is non-zero-sized and properly aligned for T; alloc returns a valid pointer or null.
            let mem = alloc(layout);
            if mem.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            // SAFETY: Null check above ensures the pointer is valid; layout guarantees proper alignment for T.
            let type_ptr: *mut T = mem as *mut _;
            (type_ptr, (mem, layout))
        }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` instead of aborting on OOM.
    pub(crate) unsafe fn try_alloc_mem<T>(
        &self,
    ) -> Result<(*mut T, (GcObjMem, Layout)), GcAllocError>
    where
        T: Sized,
    {
        unsafe {
            let layout = Layout::new::<T>();
            // SAFETY: Layout is non-zero-sized and properly aligned for T; alloc returns a valid pointer or null.
            let mem = alloc(layout);
            if mem.is_null() {
                return Err(GcAllocError);
            }
            // SAFETY: Null check above ensures the pointer is valid; layout guarantees proper alignment for T.
            let type_ptr: *mut T = mem as *mut _;
            Ok((type_ptr, (mem, layout)))
        }
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
        unsafe {
            // SAFETY: Delegates to try_alloc_mem and collect, both of which uphold their own safety invariants.
            match self.try_alloc_mem::<T>() {
                Ok(result) => Ok(result),
                Err(_) => {
                    // Emergency collection: free dead objects and retry
                    self.collect();
                    self.try_alloc_mem::<T>()
                }
            }
        }
    }

    /// Get or create the alive flag for an object (used by weak references).
    pub(crate) fn get_or_create_weak_alive(&self, obj_ptr: *const dyn Trace) -> Arc<AtomicBool> {
        let thin = obj_ptr.get_thin_ptr();
        let mut gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&obj_id) = gc_maps.ptr_to_object.get(&thin) {
            if let Some(entry) = gc_maps.objects.get_mut(obj_id) {
                return entry
                    .weak_alive
                    .get_or_insert_with(|| Arc::new(AtomicBool::new(true)))
                    .clone();
            }
        }
        Arc::new(AtomicBool::new(false))
    }

    /// Write barrier: if the object is in Gen1+, add it to the remembered set
    /// so that young-generation collections trace through it.
    /// During incremental marking, also re-grays Black objects to maintain
    /// the tri-color invariant (no Black→White edges).
    pub(crate) fn write_barrier(&self, obj_ptr: *const dyn Trace) {
        let thin = obj_ptr.get_thin_ptr();
        let gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&obj_id) = gc_maps.ptr_to_object.get(&thin) {
            if let Some(entry) = gc_maps.objects.get(obj_id) {
                if entry.generation > Generation::Gen0 {
                    self.remembered_set
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .insert(obj_id);
                }
            }
        }
        let obj_id_opt = gc_maps.ptr_to_object.get(&thin).copied();
        drop(gc_maps);

        // During incremental marking, re-gray Black objects so their
        // new children are discovered in subsequent mark steps.
        if let Some(obj_id) = obj_id_opt {
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
    }

    pub(crate) unsafe fn remove_tracer(&self, tracer_id: TracerId, object_id: ObjectId) {
        unsafe {
            let (tracer_dealloc, object_dealloc) = {
                let mut gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
                let mut tracer_dealloc = None;
                let mut object_dealloc = None;

                if let Some(tracer_entry) = gc_maps.tracers.remove(tracer_id) {
                    tracer_dealloc = Some((tracer_entry.mem, tracer_entry.layout));

                    // RC hybrid: decrement ref count and eagerly dealloc if zero
                    if let Some(obj_entry) = gc_maps.objects.get_mut(object_id) {
                        obj_entry.ref_count -= 1;
                        if obj_entry.ref_count == 0 {
                            let obj_entry = gc_maps.objects.remove(object_id).unwrap();
                            gc_maps.ptr_to_object.remove(&obj_entry.ptr.get_thin_ptr());
                            self.remembered_set
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .remove(&object_id);
                            if let Some(alive) = &obj_entry.weak_alive {
                                alive.store(false, Ordering::Release);
                            }
                            object_dealloc = Some(obj_entry);
                        }
                    }
                }

                (tracer_dealloc, object_dealloc)
            };

            // Dealloc tracer outside lock scope
            if let Some((mem, layout)) = tracer_dealloc {
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(mem, layout);
            }

            // RC hybrid: dealloc object outside lock scope (prevents re-entrant deadlock)
            if let Some(obj_entry) = object_dealloc {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: finalize_fn and mem are valid; wrapped in catch_unwind for panic safety.
                    (obj_entry.finalize_fn)(obj_entry.mem);
                }));
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: Drop function and memory pointer are valid; wrapped in catch_unwind for panic safety.
                    (obj_entry.drop_fn)(obj_entry.mem);
                }));
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(obj_entry.mem, obj_entry.layout);
            }
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
    /// For partial collections (< Gen2), traces from in-scope root tracers + remembered set.
    /// For full collections (Gen2), traces from ALL roots and clears the remembered set.
    /// Sweep phase only collects objects/tracers in target generations.
    /// Surviving objects in gens < max_gen may be promoted.
    pub unsafe fn collect_generation(&self, max_gen: Generation) -> CollectionStats {
        unsafe {
            let mut stats = CollectionStats {
                generation: max_gen,
                objects_scanned: 0,
                objects_collected: 0,
                objects_promoted: 0,
                tracers_collected: 0,
            };

            let (tracer_deallocs, object_deallocs) = {
                // STW: block all mutator operations during mark+sweep
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
                let mut remembered_set = self
                    .remembered_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());

                // Root discovery: walk all objects and cascade reset_root() into their
                // fields. This marks Gc handles stored *inside* GC-managed objects as
                // non-root (is_root = false), while stack-owned handles remain root.
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).reset_root();
                }

                // Mark phase: trace from ALL roots (needed for correctness)
                for entry in gc_maps.tracers.values() {
                    // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                    let tracer = &(*entry.tracer_ptr);
                    if tracer.is_root() {
                        tracer.trace();
                    }
                }

                // For partial collections, also trace from remembered set entries.
                if max_gen < Generation::Gen2 {
                    for &obj_id in remembered_set.iter() {
                        if let Some(entry) = gc_maps.objects.get(obj_id) {
                            // SAFETY: Object pointer is valid while gc_maps lock is held.
                            (&*entry.ptr).trace();
                        }
                    }
                }

                // Build set of in-scope objects (in target generations)
                let in_scope_objs: HashSet<ObjectId> = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.generation <= max_gen)
                    .map(|(id, _)| id)
                    .collect();
                stats.objects_scanned = in_scope_objs.len();

                // Identify in-scope tracers (their object is in target generations)
                let in_scope_tracers: HashSet<TracerId> = gc_maps
                    .tracers
                    .iter()
                    .filter(|(_, e)| in_scope_objs.contains(&e.object_id))
                    .map(|(id, _)| id)
                    .collect();

                // Sweep tracers: only collect unreachable in-scope tracers
                let collected_tracers: Vec<TracerId> = gc_maps
                    .tracers
                    .iter()
                    .filter(|(id, e)| {
                        // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                        in_scope_tracers.contains(id) && !(&*e.tracer_ptr).is_traceable()
                    })
                    .map(|(id, _)| id)
                    .collect();

                let mut tracer_deallocs = Vec::new();
                for tracer_id in &collected_tracers {
                    if let Some(entry) = gc_maps.tracers.remove(*tracer_id) {
                        tracer_deallocs.push((entry.mem, entry.layout));
                    }
                }
                stats.tracers_collected = collected_tracers.len();

                // Sweep objects: only collect unreachable in-scope objects
                let collected_objects: Vec<ObjectId> = gc_maps
                    .objects
                    .iter()
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    .filter(|(id, e)| in_scope_objs.contains(id) && !(&*e.ptr).is_traceable())
                    .map(|(id, _)| id)
                    .collect();
                stats.objects_collected = collected_objects.len();

                // Unconditionally clear mark state for ALL objects.
                // This replaces the old cascade-based reset which failed on cycles
                // (root_ref_count leaked due to trace/reset guard asymmetry).
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).clear_trace();
                }

                // Remove collected objects
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(entry) = gc_maps.objects.remove(obj_id) {
                        gc_maps.ptr_to_object.remove(&entry.ptr.get_thin_ptr());
                        remembered_set.remove(&obj_id);
                        if let Some(alive) = &entry.weak_alive {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push(entry);
                    }
                }

                // On full collection, clear the entire remembered set
                if max_gen >= Generation::Gen2 {
                    remembered_set.clear();
                }

                // Promotion: surviving in-scope objects may be promoted
                let surviving_objs: Vec<ObjectId> = in_scope_objs
                    .iter()
                    .filter(|id| !collected_objects.contains(id))
                    .copied()
                    .collect();

                for obj_id in surviving_objs {
                    if let Some(entry) = gc_maps.objects.get_mut(obj_id) {
                        if entry.generation <= max_gen {
                            let cur_gen = entry.generation;
                            entry.survive_count += 1;
                            let promo_cfg = self.promotion_config();
                            let should_promote =
                                entry.survive_count >= promo_cfg.threshold_for(cur_gen);
                            if should_promote {
                                entry.survive_count = 0;
                                if let Some(next_gen) = cur_gen.next() {
                                    entry.generation = next_gen;
                                    stats.objects_promoted += 1;
                                }
                            }
                        }
                    }
                }

                (tracer_deallocs, object_deallocs)
            };

            // Reset allocation counter after gen0 collection
            if max_gen >= Generation::Gen0 {
                self.allocation_count.store(0, Ordering::Relaxed);
            }

            // Dealloc phase: all locks released
            // IMPORTANT: Object drops must happen BEFORE tracer deallocs, because
            // dropping an object may drop inner Gc<T> handles whose Drop impl
            // dereferences tracer memory to read tracer_id/object_id.
            for entry in object_deallocs {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: finalize_fn and mem are valid; wrapped in catch_unwind for panic safety.
                    (entry.finalize_fn)(entry.mem);
                }));
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: Drop function and memory pointer are valid; wrapped in catch_unwind for panic safety.
                    (entry.drop_fn)(entry.mem);
                }));
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(entry.mem, entry.layout);
            }
            for (mem, layout) in tracer_deallocs {
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(mem, layout);
            }

            // Record diagnostics
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);

            stats
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn collect_all(&self) {
        unsafe {
            let (tracer_deallocs, object_deallocs) = {
                // STW: block all mutator operations during cleanup
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
                self.remembered_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();

                let tracer_entries = gc_maps.tracers.drain();
                let tracer_deallocs: Vec<_> = tracer_entries
                    .into_iter()
                    .map(|(_, e)| (e.mem, e.layout))
                    .collect();

                let obj_entries = gc_maps.objects.drain();
                gc_maps.ptr_to_object.clear();
                let object_deallocs: Vec<_> = obj_entries
                    .into_iter()
                    .map(|(_, entry)| {
                        if let Some(alive) = &entry.weak_alive {
                            alive.store(false, Ordering::Release);
                        }
                        entry
                    })
                    .collect();
                (tracer_deallocs, object_deallocs)
            };
            self.allocation_count.store(0, Ordering::Relaxed);
            // IMPORTANT: Object drops must happen BEFORE tracer deallocs, because
            // dropping an object may drop inner Gc<T> handles whose Drop impl
            // dereferences tracer memory to read tracer_id/object_id.
            for entry in object_deallocs {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: finalize_fn and mem are valid; wrapped in catch_unwind for panic safety.
                    (entry.finalize_fn)(entry.mem);
                }));
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: Drop function and memory pointer are valid; wrapped in catch_unwind for panic safety.
                    (entry.drop_fn)(entry.mem);
                }));
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(entry.mem, entry.layout);
            }
            for (mem, layout) in tracer_deallocs {
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(mem, layout);
            }
        }
    }

    /// Begin an incremental collection cycle.
    /// Short STW: snapshots roots, initializes tri-color marks (all in-scope objects White,
    /// root-reachable objects Gray), and sets phase to Marking.
    pub unsafe fn begin_collection(&self, max_gen: Generation) {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());

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
            if entry.generation <= max_gen {
                incr.colors.insert(obj_id, MarkColor::White);
            }
        }

        // Gray objects reachable from root tracers
        for (_tracer_id, tracer_entry) in gc_maps.tracers.iter() {
            unsafe {
                // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                let tracer = &(*tracer_entry.tracer_ptr);
                if tracer.is_root() {
                    if let Some(color) = incr.colors.get_mut(&tracer_entry.object_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(tracer_entry.object_id);
                        }
                    }
                }
            }
        }
    }

    /// Process a batch of gray objects from the worklist.
    /// Short STW per batch. Returns `true` when the gray stack is empty (marking complete).
    /// Each step discovers immediate children of `budget` objects via `trace_children`.
    pub unsafe fn mark_step(&self, budget: usize) -> bool {
        let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
        let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
        let gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());

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

    /// Finish an incremental collection: re-mark from remembered set, sweep White objects,
    /// promote survivors, and reset state.
    /// Short STW.
    pub unsafe fn finish_collection(&self) -> CollectionStats {
        unsafe {
            let max_gen;
            let (tracer_deallocs, object_deallocs, stats) = {
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut incr = self.incremental.lock().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
                let mut remembered_set = self
                    .remembered_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());

                max_gen = incr.max_gen;
                incr.phase = CollectionPhase::Sweeping;

                // Root discovery for objects allocated during concurrent marking.
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).reset_root();
                }

                // Re-gray any remembered-set entries that were already Black.
                for &obj_id in remembered_set.iter() {
                    if let Some(color) = incr.colors.get_mut(&obj_id) {
                        if *color == MarkColor::Black {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(obj_id);
                        }
                    }
                }

                // Also gray objects reachable from NEW root tracers (allocated during marking).
                for (_tracer_id, tracer_entry) in gc_maps.tracers.iter() {
                    // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                    let tracer = &(*tracer_entry.tracer_ptr);
                    if tracer.is_root() {
                        if let Some(color) = incr.colors.get_mut(&tracer_entry.object_id) {
                            if *color == MarkColor::White {
                                *color = MarkColor::Gray;
                                incr.gray_stack.push(tracer_entry.object_id);
                            }
                        }
                    }
                }

                // Drain remaining gray (from re-graying + new roots)
                let mut children_buf = Vec::new();
                while let Some(obj_id) = incr.gray_stack.pop() {
                    children_buf.clear();
                    if let Some(entry) = gc_maps.objects.get(obj_id) {
                        // SAFETY: Object pointer is valid while gc_maps lock is held.
                        (&*entry.ptr).trace_children(&mut children_buf);
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

                // Sweep: collect White objects
                let mut stats = CollectionStats {
                    generation: max_gen,
                    objects_scanned: incr.colors.len(),
                    objects_collected: 0,
                    objects_promoted: 0,
                    tracers_collected: 0,
                };

                let white_objects: HashSet<ObjectId> = incr
                    .colors
                    .iter()
                    .filter(|(_, color)| **color == MarkColor::White)
                    .map(|(id, _)| *id)
                    .collect();
                stats.objects_collected = white_objects.len();

                // Sweep dead tracers (those pointing to white objects)
                let dead_tracers: Vec<TracerId> = gc_maps
                    .tracers
                    .iter()
                    .filter(|(_, e)| white_objects.contains(&e.object_id))
                    .map(|(id, _)| id)
                    .collect();
                stats.tracers_collected = dead_tracers.len();

                let mut tracer_deallocs = Vec::new();
                for tracer_id in &dead_tracers {
                    if let Some(entry) = gc_maps.tracers.remove(*tracer_id) {
                        tracer_deallocs.push((entry.mem, entry.layout));
                    }
                }

                // Sweep dead objects
                let mut object_deallocs = Vec::new();
                for &obj_id in &white_objects {
                    if let Some(entry) = gc_maps.objects.remove(obj_id) {
                        gc_maps.ptr_to_object.remove(&entry.ptr.get_thin_ptr());
                        remembered_set.remove(&obj_id);
                        if let Some(alive) = &entry.weak_alive {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push(entry);
                    }
                }

                // Clear remembered set on full collection
                if max_gen >= Generation::Gen2 {
                    remembered_set.clear();
                }

                // Promote surviving in-scope objects
                let surviving: Vec<ObjectId> = incr
                    .colors
                    .iter()
                    .filter(|(_, color)| **color != MarkColor::White)
                    .map(|(id, _)| *id)
                    .collect();

                for obj_id in surviving {
                    if let Some(entry) = gc_maps.objects.get_mut(obj_id) {
                        if entry.generation <= max_gen {
                            let cur_gen = entry.generation;
                            entry.survive_count += 1;
                            let promo_cfg = self.promotion_config();
                            let should_promote =
                                entry.survive_count >= promo_cfg.threshold_for(cur_gen);
                            if should_promote {
                                entry.survive_count = 0;
                                if let Some(next_gen) = cur_gen.next() {
                                    entry.generation = next_gen;
                                    stats.objects_promoted += 1;
                                }
                            }
                        }
                    }
                }

                // Reset incremental state
                incr.phase = CollectionPhase::Idle;
                incr.colors.clear();
                incr.gray_stack.clear();

                (tracer_deallocs, object_deallocs, stats)
            };

            // Reset allocation counter
            if max_gen >= Generation::Gen0 {
                self.allocation_count.store(0, Ordering::Relaxed);
            }

            // Dealloc phase: all locks released
            // IMPORTANT: Object drops must happen BEFORE tracer deallocs, because
            // dropping an object may drop inner Gc<T> handles whose Drop impl
            // dereferences tracer memory to read tracer_id/object_id.
            for entry in object_deallocs {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: finalize_fn and mem are valid; wrapped in catch_unwind for panic safety.
                    (entry.finalize_fn)(entry.mem);
                }));
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: Drop function and memory pointer are valid; wrapped in catch_unwind for panic safety.
                    (entry.drop_fn)(entry.mem);
                }));
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(entry.mem, entry.layout);
            }
            for (mem, layout) in tracer_deallocs {
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(mem, layout);
            }

            // Record diagnostics
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);

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
        let gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());

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
        let gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());

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
            if entry.generation <= max_gen {
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
        for (_tracer_id, tracer_entry) in gc_maps.tracers.iter() {
            unsafe {
                // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                let tracer = &(*tracer_entry.tracer_ptr);
                if tracer.is_root() {
                    if let Some(color) = incr.colors.get_mut(&tracer_entry.object_id) {
                        if *color == MarkColor::White {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(tracer_entry.object_id);
                        }
                    }
                }
            }
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
            // finish_collection handles re-graying from remembered set + new roots
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
            let mut stats = CollectionStats {
                generation: Generation::Gen2,
                objects_scanned: 0,
                objects_collected: 0,
                objects_promoted: 0,
                tracers_collected: 0,
            };

            let (tracer_deallocs, object_deallocs) = {
                let _stw = self.stw_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut gc_maps = self.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
                let mut remembered_set = self
                    .remembered_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());

                // Root discovery
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).reset_root();
                }

                // Mark phase: trace from ALL roots
                for entry in gc_maps.tracers.values() {
                    // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                    let tracer = &(*entry.tracer_ptr);
                    if tracer.is_root() {
                        tracer.trace();
                    }
                }

                // Also trace from remembered set
                for &obj_id in remembered_set.iter() {
                    if let Some(entry) = gc_maps.objects.get(obj_id) {
                        // SAFETY: Object pointer is valid while gc_maps lock is held.
                        (&*entry.ptr).trace();
                    }
                }

                // Objects in target region
                let in_region: HashSet<ObjectId> = gc_maps
                    .objects
                    .iter()
                    .filter(|(_, e)| e.region == region)
                    .map(|(id, _)| id)
                    .collect();
                stats.objects_scanned = in_region.len();

                // Tracers pointing to objects in this region
                let region_tracers: HashSet<TracerId> = gc_maps
                    .tracers
                    .iter()
                    .filter(|(_, e)| in_region.contains(&e.object_id))
                    .map(|(id, _)| id)
                    .collect();

                // Sweep unreachable tracers in this region
                let collected_tracers: Vec<TracerId> = gc_maps
                    .tracers
                    .iter()
                    .filter(|(id, e)| {
                        // SAFETY: Tracer pointer is valid while gc_maps lock is held.
                        region_tracers.contains(id) && !(&*e.tracer_ptr).is_traceable()
                    })
                    .map(|(id, _)| id)
                    .collect();
                let mut tracer_deallocs = Vec::new();
                for tracer_id in &collected_tracers {
                    if let Some(entry) = gc_maps.tracers.remove(*tracer_id) {
                        tracer_deallocs.push((entry.mem, entry.layout));
                    }
                }
                stats.tracers_collected = collected_tracers.len();

                // Sweep unreachable objects in this region
                let collected_objects: Vec<ObjectId> = gc_maps
                    .objects
                    .iter()
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    .filter(|(id, e)| in_region.contains(id) && !(&*e.ptr).is_traceable())
                    .map(|(id, _)| id)
                    .collect();
                stats.objects_collected = collected_objects.len();

                // Unconditionally clear mark state for ALL objects.
                for entry in gc_maps.objects.values() {
                    // SAFETY: Object pointer is valid while gc_maps lock is held.
                    (&*entry.ptr).clear_trace();
                }

                // Remove collected objects
                let mut object_deallocs = Vec::new();
                for &obj_id in &collected_objects {
                    if let Some(entry) = gc_maps.objects.remove(obj_id) {
                        gc_maps.ptr_to_object.remove(&entry.ptr.get_thin_ptr());
                        remembered_set.remove(&obj_id);
                        if let Some(alive) = &entry.weak_alive {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push(entry);
                    }
                }

                (tracer_deallocs, object_deallocs)
            };

            // Dealloc phase: all locks released
            // IMPORTANT: Object drops must happen BEFORE tracer deallocs, because
            // dropping an object may drop inner Gc<T> handles whose Drop impl
            // dereferences tracer memory to read tracer_id/object_id.
            for entry in object_deallocs {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: finalize_fn and mem are valid; wrapped in catch_unwind for panic safety.
                    (entry.finalize_fn)(entry.mem);
                }));
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: Drop function and memory pointer are valid; wrapped in catch_unwind for panic safety.
                    (entry.drop_fn)(entry.mem);
                }));
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(entry.mem, entry.layout);
            }
            for (mem, layout) in tracer_deallocs {
                // SAFETY: Memory was allocated with the same layout via alloc/alloc_mem.
                dealloc(mem, layout);
            }

            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self
                .last_collection
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stats);
            stats
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
const LOCAL_GC_ALLOC_THRESHOLD: usize = 1000;

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
}

// SAFETY: Delegates to GarbageCollector which protects all state with Mutex/atomics.
unsafe impl Sync for LocalGarbageCollector {}
unsafe impl Send for LocalGarbageCollector {}

impl LocalGarbageCollector {
    fn new() -> LocalGarbageCollector {
        LocalGarbageCollector {
            core: GarbageCollector::new(),
        }
    }

    /// Check allocation count and trigger Gen0 collection if threshold exceeded.
    unsafe fn maybe_collect(&self) {
        let count = self.core.allocation_count.load(Ordering::Relaxed);
        if count >= LOCAL_GC_ALLOC_THRESHOLD {
            unsafe {
                self.core.collect_generation(Generation::Gen0);
            }
        }
    }

    unsafe fn create_gc<T>(&self, t: T) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_ptr, mem_info_gc_ptr) = self.core.alloc_mem::<GcPtr<T>>();
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();
            // SAFETY: Pointer was just allocated via alloc_mem and is properly aligned for GcPtr<T>.
            std::ptr::write(gc_ptr, GcPtr::new(t));

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            unsafe fn drop_gc_ptr<T: 'static + Trace>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: Pointer points to a valid, initialized GcPtr<T> that is being deallocated.
                    std::ptr::drop_in_place(ptr as *mut GcPtr<T>);
                }
            }
            unsafe fn finalize_gc_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: Pointer points to a valid GcPtr<T>; derives reference at call
                    // time to avoid Stacked Borrows provenance issues.
                    (*(ptr as *const GcPtr<T>)).t.finalize();
                }
            }
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: mem_info_gc_ptr.0,
                layout: mem_info_gc_ptr.1,
                generation: Generation::Gen0,
                survive_count: 0,
                finalize_fn: finalize_gc_ptr::<T>,
                drop_fn: drop_gc_ptr::<T>,
                weak_alive: None,
                ref_count: 1,
                region: self.core.current_region(),
            });
            gc_maps
                .ptr_to_object
                .insert((gc_ptr as *const dyn Trace).get_thin_ptr(), obj_id);

            let tracer_id = gc_maps.tracers.insert(TracerEntry {
                tracer_ptr: gc_inter_ptr as *const dyn Trace,
                mem: mem_info_internal_ptr.0,
                layout: mem_info_internal_ptr.1,
                object_id: obj_id,
            });
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr, tracer_id, obj_id));
            drop(gc_maps);

            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
                tracer_id,
                object_id: obj_id,
            };
            // SAFETY: Both internal_ptr and ptr were just initialized above; derefs are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
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

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            let tracer_id = gc_maps.tracers.insert(TracerEntry {
                tracer_ptr: gc_inter_ptr as *const dyn Trace,
                mem: mem_info_internal_ptr.0,
                layout: mem_info_internal_ptr.1,
                object_id,
            });
            // RC hybrid: increment ref count for the cloned object
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.ref_count += 1;
            }
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc.ptr, tracer_id, object_id));
            drop(gc_maps);
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
                tracer_id,
                object_id,
            };
            // SAFETY: Both internal_ptr and ptr are valid; internal_ptr was just initialized, ptr comes from the source Gc.
            (*(*gc.internal_ptr).ptr).reset_root();
            gc
        }
    }

    unsafe fn create_gc_cell<T>(&self, t: T) -> GcRefCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_ptr, mem_info_gc_ptr) = self.core.alloc_mem::<RefCell<GcPtr<T>>>();
            let (gc_cell_inter_ptr, mem_info_internal_ptr) =
                self.core.alloc_mem::<GcRefCellInternal<T>>();
            // SAFETY: Pointer was just allocated via alloc_mem and is properly aligned for RefCell<GcPtr<T>>.
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            unsafe fn drop_gc_cell_ptr<T: 'static + Trace>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: Pointer points to a valid, initialized RefCell<GcPtr<T>> that is being deallocated.
                    std::ptr::drop_in_place(ptr as *mut RefCell<GcPtr<T>>);
                }
            }
            unsafe fn finalize_gc_cell_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: Pointer points to a valid RefCell<GcPtr<T>>; derives reference
                    // at call time to avoid Stacked Borrows provenance issues.
                    (*(*(ptr as *const RefCell<GcPtr<T>>)).as_ptr())
                        .t
                        .finalize();
                }
            }
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: mem_info_gc_ptr.0,
                layout: mem_info_gc_ptr.1,
                generation: Generation::Gen0,
                survive_count: 0,
                finalize_fn: finalize_gc_cell_ptr::<T>,
                drop_fn: drop_gc_cell_ptr::<T>,
                weak_alive: None,
                ref_count: 1,
                region: self.core.current_region(),
            });
            gc_maps
                .ptr_to_object
                .insert((gc_ptr as *const dyn Trace).get_thin_ptr(), obj_id);

            let tracer_id = gc_maps.tracers.insert(TracerEntry {
                tracer_ptr: gc_cell_inter_ptr as *const dyn Trace,
                mem: mem_info_internal_ptr.0,
                layout: mem_info_internal_ptr.1,
                object_id: obj_id,
            });
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_cell_inter_ptr,
                GcRefCellInternal::new(gc_ptr, tracer_id, obj_id),
            );
            drop(gc_maps);

            let gc = GcRefCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
                tracer_id,
                object_id: obj_id,
            };
            // SAFETY: Both internal_ptr and ptr were just initialized above; derefs are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            self.maybe_collect();
            gc
        }
    }

    unsafe fn clone_from_gc_cell<T>(&self, gc: &GcRefCell<T>) -> GcRefCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_inter_ptr, mem_info) = self.core.alloc_mem::<GcRefCellInternal<T>>();
            // SAFETY: internal_ptr is valid for the lifetime of the source GcRefCell handle.
            let object_id = (*gc.internal_ptr).object_id;

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            let tracer_id = gc_maps.tracers.insert(TracerEntry {
                tracer_ptr: gc_inter_ptr as *const dyn Trace,
                mem: mem_info.0,
                layout: mem_info.1,
                object_id,
            });
            // RC hybrid: increment ref count for the cloned object
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.ref_count += 1;
            }
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_inter_ptr,
                GcRefCellInternal::new(gc.ptr, tracer_id, object_id),
            );
            drop(gc_maps);
            let gc = GcRefCell {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
                tracer_id,
                object_id,
            };
            // SAFETY: Both internal_ptr and ptr are valid; internal_ptr was just initialized, ptr comes from the source GcRefCell.
            (*(*gc.internal_ptr).ptr).reset_root();
            gc
        }
    }

    /// Fallible version of `create_gc`. Returns `Err(GcAllocError)` on OOM.
    unsafe fn try_create_gc<T>(&self, t: T) -> Result<Gc<T>, GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_ptr, mem_info_gc_ptr) = self.core.try_alloc_mem_with_gc::<GcPtr<T>>()?;
            let (gc_inter_ptr, mem_info_internal_ptr) =
                match self.core.try_alloc_mem_with_gc::<GcInternal<T>>() {
                    Ok(v) => v,
                    Err(e) => {
                        // SAFETY: Memory was allocated with the same layout via try_alloc_mem_with_gc.
                        dealloc(mem_info_gc_ptr.0, mem_info_gc_ptr.1);
                        return Err(e);
                    }
                };
            // SAFETY: Pointer was just allocated via try_alloc_mem_with_gc and is properly aligned for GcPtr<T>.
            std::ptr::write(gc_ptr, GcPtr::new(t));

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            unsafe fn drop_gc_ptr<T: 'static + Trace>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: Pointer points to a valid, initialized GcPtr<T> that is being deallocated.
                    std::ptr::drop_in_place(ptr as *mut GcPtr<T>);
                }
            }
            unsafe fn finalize_gc_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: Pointer points to a valid GcPtr<T>; derives reference at call
                    // time to avoid Stacked Borrows provenance issues.
                    (*(ptr as *const GcPtr<T>)).t.finalize();
                }
            }
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: mem_info_gc_ptr.0,
                layout: mem_info_gc_ptr.1,
                generation: Generation::Gen0,
                survive_count: 0,
                finalize_fn: finalize_gc_ptr::<T>,
                drop_fn: drop_gc_ptr::<T>,
                weak_alive: None,
                ref_count: 1,
                region: self.core.current_region(),
            });
            gc_maps
                .ptr_to_object
                .insert((gc_ptr as *const dyn Trace).get_thin_ptr(), obj_id);

            let tracer_id = gc_maps.tracers.insert(TracerEntry {
                tracer_ptr: gc_inter_ptr as *const dyn Trace,
                mem: mem_info_internal_ptr.0,
                layout: mem_info_internal_ptr.1,
                object_id: obj_id,
            });
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr, tracer_id, obj_id));
            drop(gc_maps);

            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
                tracer_id,
                object_id: obj_id,
            };
            // SAFETY: Both internal_ptr and ptr were just initialized above; derefs are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            self.maybe_collect();
            Ok(gc)
        }
    }

    /// Fallible version of `create_gc_cell`. Returns `Err(GcAllocError)` on OOM.
    unsafe fn try_create_gc_cell<T>(&self, t: T) -> Result<GcRefCell<T>, GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            let (gc_ptr, mem_info_gc_ptr) =
                self.core.try_alloc_mem_with_gc::<RefCell<GcPtr<T>>>()?;
            let (gc_cell_inter_ptr, mem_info_internal_ptr) =
                match self.core.try_alloc_mem_with_gc::<GcRefCellInternal<T>>() {
                    Ok(v) => v,
                    Err(e) => {
                        // SAFETY: Memory was allocated with the same layout via try_alloc_mem_with_gc.
                        dealloc(mem_info_gc_ptr.0, mem_info_gc_ptr.1);
                        return Err(e);
                    }
                };
            // SAFETY: Pointer was just allocated via try_alloc_mem_with_gc and is properly aligned for RefCell<GcPtr<T>>.
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            unsafe fn drop_gc_cell_ptr<T: 'static + Trace>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: Pointer points to a valid, initialized RefCell<GcPtr<T>> that is being deallocated.
                    std::ptr::drop_in_place(ptr as *mut RefCell<GcPtr<T>>);
                }
            }
            unsafe fn finalize_gc_cell_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: Pointer points to a valid RefCell<GcPtr<T>>; derives reference
                    // at call time to avoid Stacked Borrows provenance issues.
                    (*(*(ptr as *const RefCell<GcPtr<T>>)).as_ptr())
                        .t
                        .finalize();
                }
            }
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: mem_info_gc_ptr.0,
                layout: mem_info_gc_ptr.1,
                generation: Generation::Gen0,
                survive_count: 0,
                finalize_fn: finalize_gc_cell_ptr::<T>,
                drop_fn: drop_gc_cell_ptr::<T>,
                weak_alive: None,
                ref_count: 1,
                region: self.core.current_region(),
            });
            gc_maps
                .ptr_to_object
                .insert((gc_ptr as *const dyn Trace).get_thin_ptr(), obj_id);

            let tracer_id = gc_maps.tracers.insert(TracerEntry {
                tracer_ptr: gc_cell_inter_ptr as *const dyn Trace,
                mem: mem_info_internal_ptr.0,
                layout: mem_info_internal_ptr.1,
                object_id: obj_id,
            });
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_cell_inter_ptr,
                GcRefCellInternal::new(gc_ptr, tracer_id, obj_id),
            );
            drop(gc_maps);

            let gc = GcRefCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
                tracer_id,
                object_id: obj_id,
            };
            // SAFETY: Both internal_ptr and ptr were just initialized above; derefs are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
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
            let thin = (weak.ptr as *const dyn Trace).get_thin_ptr();

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            let object_id = *gc_maps
                .ptr_to_object
                .get(&thin)
                .expect("upgrade_weak: object not found");
            let tracer_id = gc_maps.tracers.insert(TracerEntry {
                tracer_ptr: gc_inter_ptr as *const dyn Trace,
                mem: mem_info_internal_ptr.0,
                layout: mem_info_internal_ptr.1,
                object_id,
            });
            // RC hybrid: increment ref count for upgraded object
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.ref_count += 1;
            }
            drop(gc_maps);

            // SAFETY: Pointer was just allocated via alloc_mem and is properly aligned for GcInternal<T>.
            std::ptr::write(
                gc_inter_ptr,
                GcInternal::new(weak.ptr, tracer_id, object_id),
            );
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: weak.ptr,
                tracer_id,
                object_id,
            };
            // SAFETY: Both internal_ptr (just initialized) and ptr (verified alive via STW lock) are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            gc
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect(&self) {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect().
        unsafe {
            self.core.collect();
        }
    }

    #[allow(dead_code)]
    unsafe fn collect_all(&self) {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_all().
        unsafe {
            self.core.collect_all();
        }
    }

    pub(crate) unsafe fn remove_tracer(&self, tracer_id: TracerId, object_id: ObjectId) {
        // SAFETY: Caller provides valid tracer_id and object_id obtained from a live Gc handle.
        unsafe {
            self.core.remove_tracer(tracer_id, object_id);
        }
    }

    /// Begin an incremental collection cycle.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn begin_collection(&self, max_gen: Generation) {
        // SAFETY: Caller upholds the safety contract; delegates to core.begin_collection().
        unsafe {
            self.core.begin_collection(max_gen);
        }
    }

    /// Process a batch of gray objects. Returns true when marking is complete.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn mark_step(&self, budget: usize) -> bool {
        // SAFETY: Caller upholds the safety contract; delegates to core.mark_step().
        unsafe { self.core.mark_step(budget) }
    }

    /// Finish incremental collection: sweep white objects, promote survivors.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn finish_collection(&self) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.finish_collection().
        unsafe { self.core.finish_collection() }
    }

    /// Run a complete incremental collection with the given step budget.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_incremental(
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
    pub unsafe fn begin_concurrent_collection(&self, max_gen: Generation) {
        // SAFETY: Caller upholds the safety contract; delegates to core.begin_concurrent_collection().
        unsafe {
            self.core.begin_concurrent_collection(max_gen);
        }
    }

    /// Process gray objects using edge snapshot. NO STW lock — safe for concurrent use.
    pub fn concurrent_mark_step(&self, budget: usize) -> bool {
        self.core.concurrent_mark_step(budget)
    }

    /// Run a complete concurrent collection (snapshot → concurrent mark → STW sweep).
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_concurrent(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_concurrent().
        unsafe { self.core.collect_concurrent(max_gen, step_budget) }
    }

    /// Time-budgeted incremental mark step. See [`GarbageCollector::mark_step_timed`].
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn mark_step_timed(&self, max_duration: Duration) -> bool {
        // SAFETY: Caller upholds the safety contract; delegates to core.mark_step_timed().
        unsafe { self.core.mark_step_timed(max_duration) }
    }

    /// Run a complete incremental collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_incremental_timed(
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
    pub fn concurrent_mark_step_timed(&self, max_duration: Duration) -> bool {
        self.core.concurrent_mark_step_timed(max_duration)
    }

    /// Run a complete concurrent collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_concurrent_timed(
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
    pub fn new_region(&self) -> RegionId {
        self.core.new_region()
    }

    /// Get the current allocation region.
    pub fn current_region(&self) -> RegionId {
        self.core.current_region()
    }

    /// Collect only objects in the specified region.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_region(&self, region: RegionId) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_region().
        unsafe { self.core.collect_region(region) }
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
}

impl PartialEq for &LocalGarbageCollector {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(
            *self as *const LocalGarbageCollector,
            *other as *const LocalGarbageCollector,
        )
    }
}

/// Callback to start a collection strategy. Receives the GC and an "active" flag;
/// may spawn a background thread and return its `JoinHandle`.
pub type StartLocalStrategyFn =
    Box<dyn FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;
/// Callback to stop a collection strategy.
pub type StopLocalStrategyFn = Box<dyn FnMut(&'static LocalGarbageCollector)>;

/// Pluggable collection strategy for the thread-local GC.
/// Controls when and how garbage collection runs (e.g., periodic background thread,
/// allocation-triggered, or manual). Use `change_strategy` to swap at runtime.
///
/// Internally stores a raw pointer to the `LocalGarbageCollector` to avoid
/// Stacked Borrows violations during thread-local destruction ordering.
pub struct LocalStrategy {
    gc: Cell<*const LocalGarbageCollector>,
    is_active: AtomicBool,
    start_func: RefCell<StartLocalStrategyFn>,
    stop_func: RefCell<StopLocalStrategyFn>,
    join_handle: RefCell<Option<JoinHandle<()>>>,
}

impl LocalStrategy {
    fn new<StartFn, StopFn>(
        gc_ptr: *const LocalGarbageCollector,
        start_fn: StartFn,
        stop_fn: StopFn,
    ) -> LocalStrategy
    where
        StartFn: 'static
            + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
        StopFn: 'static + FnMut(&'static LocalGarbageCollector),
    {
        LocalStrategy {
            gc: Cell::new(gc_ptr),
            is_active: AtomicBool::new(false),
            start_func: RefCell::new(Box::new(start_fn)),
            stop_func: RefCell::new(Box::new(stop_fn)),
            join_handle: RefCell::new(None),
        }
    }

    /// Replace the current collection strategy. Stops the current strategy first if active.
    pub fn change_strategy<StartFn, StopFn>(&self, start_fn: StartFn, stop_fn: StopFn)
    where
        StartFn: 'static
            + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
        StopFn: 'static + FnMut(&'static LocalGarbageCollector),
    {
        if self.is_active() {
            self.stop();
        }
        let _ = self.start_func.replace(Box::new(start_fn));
        let _ = self.stop_func.replace(Box::new(stop_fn));
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
            .replace((*(self.start_func.borrow_mut()))(gc_ref, &self.is_active));
    }

    /// Stop the collection strategy and join any background thread.
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        if let Some(join_handle) = self.join_handle.borrow_mut().take() {
            join_handle
                .join()
                .expect("LocalStrategy::stop, LocalStrategy Thread being joined has panicked !!");
        }
        // SAFETY: gc pointer is valid for the lifetime of the thread-local and is only
        // accessed from the owning thread.
        let gc_ref = unsafe { &*self.gc.get() };
        (*(self.stop_func.borrow_mut()))(gc_ref);
    }
}

impl Drop for LocalStrategy {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
        // SAFETY: gc pointer is valid for the lifetime of the thread-local and is only
        // accessed from the owning thread. Using a raw pointer avoids Stacked Borrows
        // violations during thread-local destruction ordering.
        let gc_ref = unsafe { &*self.gc.get() };
        (*(self.stop_func.borrow_mut()))(gc_ref);
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
            RefCell::new(LocalStrategy::new(gc_ptr,
            move |_local_gc, _| {
                None
            },
            move |_local_gc| {
            }))
        })
    };
}

#[cfg(test)]
mod tests {
    use crate::gc::{Gc, GcRefCell, LOCAL_GC};
    use crate::{Finalize, Trace};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Clean residual state from previous tests that may have run on this thread.
    fn clean_gc_state() {
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
    }

    #[test]
    fn one_object() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
        });
        let _one = Gc::new(1);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(
                gc.borrow()
                    .core
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .tracers
                    .len()
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
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
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .tracers
                    .len()
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
        });
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| {
            assert_eq!(
                gc.borrow()
                    .core
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .tracers
                    .len()
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
        });
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(
                gc.borrow()
                    .core
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .tracers
                    .len()
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
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
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .tracers
                    .len()
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
        });
        let baseline_p2o = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .tracers
                    .len()
                    - baseline_trs,
                0
            );
            assert_eq!(
                gc.borrow()
                    .core
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
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
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .tracers
                    .len()
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
        });
        let source = Gc::new(42);
        let mut target = Gc::new(99);
        target.clone_from(&source);
        LOCAL_GC.with(|gc| {
            let delta = gc
                .borrow()
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation == crate::generation::Generation::Gen0),
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
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                assert!(
                    gc_maps
                        .objects
                        .values()
                        .any(|e| e.generation == crate::generation::Generation::Gen1),
                    "object should be promoted to Gen1 after surviving 3 Gen0 collections"
                );
            }

            let baseline_objs = gc_ref
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                assert!(
                    gc_maps
                        .objects
                        .values()
                        .all(|e| e.generation == crate::generation::Generation::Gen0),
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation == crate::generation::Generation::Gen1),
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation == crate::generation::Generation::Gen2),
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .objects
                .len()
        });
        {
            let _obj = Gc::new(99);
        }
        let after = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
        let before = LOCAL_GC.with(|gc| gc.borrow().core.allocation_count.load(Ordering::Relaxed));
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let after = LOCAL_GC.with(|gc| gc.borrow().core.allocation_count.load(Ordering::Relaxed));
        assert_eq!(
            after - before,
            2,
            "allocation_count should increment per new object"
        );
    }

    #[test]
    fn write_barrier_adds_old_gen_object_to_remembered_set() {
        clean_gc_state();
        use crate::gc::GcRefCell;
        let cell = GcRefCell::new(Option::<Gc<i32>>::None);

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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation == crate::generation::Generation::Gen1),
                "cell should be promoted to Gen1 after 3 Gen0 collections"
            );
        });

        // Remembered set should be empty before mutation
        LOCAL_GC.with(|gc| {
            assert!(
                gc.borrow()
                    .core
                    .remembered_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty(),
                "remembered set should be empty before write barrier"
            );
        });

        // Mutate via borrow_mut() — triggers write barrier
        // (must be outside LOCAL_GC.with borrow to avoid RefCell conflict)
        {
            let young = Gc::new(42);
            **cell.borrow_mut() = Some(young);
        }

        // Verify cell is now in the remembered set
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let rs = gc_ref
                .core
                .remembered_set
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            assert!(
                !rs.is_empty(),
                "write barrier should add old-gen object to remembered set"
            );
        });
    }

    #[test]
    fn remembered_set_cleared_on_full_collection() {
        clean_gc_state();
        use crate::gc::GcRefCell;
        let cell = GcRefCell::new(Option::<Gc<i32>>::None);

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
                !gc_ref
                    .core
                    .remembered_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty(),
                "remembered set should not be empty after write barrier"
            );

            // Full collection (Gen2) should clear remembered set
            unsafe {
                gc_ref
                    .core
                    .collect_generation(crate::generation::Generation::Gen2);
            }
            assert!(
                gc_ref
                    .core
                    .remembered_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty(),
                "remembered set should be cleared after full collection"
            );
        });
    }

    #[test]
    fn write_barrier_preserves_young_object_during_gen0_collection() {
        // Key correctness test: an old-gen object holds a reference to a young
        // object. Without write barrier, the young object would be collected.
        clean_gc_state();
        use crate::gc::GcRefCell;
        let cell = GcRefCell::new(Option::<Gc<i32>>::None);

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
        // it's referenced by an old-gen object in the remembered set
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner()).objects.len();
            unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            let objs_after = gc_ref.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner()).objects.len();
            assert_eq!(objs_before, objs_after,
                "young object referenced from old-gen (via remembered set) must survive Gen0 collection");
        });
    }

    #[test]
    fn no_write_barrier_for_gen0_objects() {
        clean_gc_state();
        use crate::gc::GcRefCell;
        let cell = GcRefCell::new(Option::<Gc<i32>>::None);

        // Cell is still in Gen0 — borrow_mut should NOT add to remembered set
        **cell.borrow_mut() = Some(Gc::new(1));

        LOCAL_GC.with(|gc| {
            assert!(
                gc.borrow()
                    .core
                    .remembered_set
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty(),
                "Gen0 objects should not be added to remembered set"
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .objects
                .len();
            assert_eq!(objs_before, objs_after, "live object count must not change");
        });
    }

    #[test]
    fn incremental_with_child_references() {
        // Test that trace_children discovers object graph correctly
        clean_gc_state();
        use crate::gc::GcRefCell;
        let parent = GcRefCell::new(Option::<Gc<i32>>::None);
        let child = Gc::new(42);
        **parent.borrow_mut() = Some(child.clone());
        drop(child); // only reference is from parent

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .objects
                .len();
            unsafe {
                gc_ref
                    .core
                    .collect_incremental(crate::generation::Generation::Gen2, 1);
            }
            let objs_after = gc_ref
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            assert!(
                gc_maps
                    .objects
                    .values()
                    .any(|e| e.generation == crate::generation::Generation::Gen1),
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
        use crate::gc::GcRefCell;
        let result = GcRefCell::try_new(100);
        assert!(result.is_ok(), "try_new should succeed for GcRefCell");
        let cell = result.unwrap();
        assert_eq!(**cell.borrow(), 100);
    }

    #[test]
    fn try_new_object_participates_in_gc() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
        let cell = GcRefCell::new(7);
        let s = format!("{:?}", cell);
        assert_eq!(s, "GcRefCell(7)");
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .objects
                .len();
            let _stats = unsafe {
                gc_ref
                    .core
                    .collect_concurrent(crate::generation::Generation::Gen2, 10)
            };
            let objs_after = gc_ref
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner());
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner());
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .objects
                .len()
        });
        {
            let _obj = Gc::new(42);
            let during = LOCAL_GC.with(|gc| {
                gc.borrow()
                    .core
                    .gc_maps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .objects
                    .len()
            });
            assert_eq!(during - baseline, 1, "object should be alive during scope");
        }
        let after = LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .objects
                .len()
        });
        let a = Gc::new(42);
        assert_eq!(
            LOCAL_GC.with(|gc| gc
                .borrow()
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
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
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .objects
                .len();
            assert_eq!(objs_before, objs_after, "no objects should be collected");
        });
        drop(v);
    }
}
