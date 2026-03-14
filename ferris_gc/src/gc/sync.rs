use std::cell::{Cell, RefCell};
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::basic_gc_strategy::{BASIC_STRATEGY_GLOBAL_GC, basic_gc_strategy_start};
use crate::gc::{Finalize, GarbageCollector, ObjectEntry, ThinPtr, Trace, TracerEntry};
use crate::generation::Generation;
use crate::slot_map::{ObjectId, TracerId};

/// Convenience alias for an optional thread-safe GC pointer.
pub type OptGc<T> = Option<Gc<T>>;
/// Convenience alias for an optional thread-safe GC interior-mutable cell.
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

/// Internal pointer wrapper used as the `Deref` target of `sync::Gc<T>` and `sync::GcRefCell<T>`.
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
                // SAFETY: Pointer is valid for the lifetime of this GcInternal handle; the GC guarantees the allocation is not freed while any handle exists.
                (*self.ptr).reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcInternal handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcInternal handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this GcInternal handle; the GC guarantees the allocation is not freed while any handle exists.
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

/// A garbage-collected smart pointer that can be shared across threads.
///
/// `sync::Gc<T>` is `Send + Sync` when `T` is. All allocations and
/// collections go through the global `GlobalGarbageCollector`, which uses
/// an `RwLock`-based stop-the-world protocol: mutators hold a read lock,
/// the collector acquires a write lock for mark-and-sweep.
///
/// Dereferences to `T` (via `GcPtr<T>`).
pub struct Gc<T>
where
    T: 'static + Sized + Trace,
{
    internal_ptr: *mut GcInternal<T>,
    ptr: *const GcPtr<T>,
    tracer_id: TracerId,
    object_id: ObjectId,
}

/// # Safety
/// The underlying GcInternal uses atomic ref-counting, and the global GC
/// protects all structural mutations with Mutex/RwLock. Gc<T> is safe to
/// send across threads when T itself is Send.
unsafe impl<T> Sync for Gc<T> where T: 'static + Sized + Trace + Sync {}
unsafe impl<T> Send for Gc<T> where T: 'static + Sized + Trace + Send {}

impl<T> Deref for Gc<T>
where
    T: 'static + Sized + Trace,
{
    type Target = GcPtr<T>;

    /// Safe without holding the STW lock because as long as this `Gc<T>` handle
    /// exists, the object is reachable (root) and cannot be collected.
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
    /// Allocate a new thread-safe GC-managed object on the global collector.
    /// Starts the background collection strategy if not already active.
    pub fn new(t: T) -> Gc<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).create_gc(t, region)
        }
    }

    /// Allocate a new thread-safe GC-managed object in the specified region.
    pub fn new_in(t: T, region: crate::generation::RegionId) -> Gc<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).create_gc(t, region) }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<Gc<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).try_create_gc(t, region)
        }
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(
        t: T,
        region: crate::generation::RegionId,
    ) -> Result<Gc<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).try_create_gc(t, region) }
    }
}

impl<T> Clone for Gc<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).clone_from_gc(self) }
    }
}

impl<T> Drop for Gc<T>
where
    T: Sized + Trace,
{
    fn drop(&mut self) {
        let tracer_id = self.tracer_id;
        let object_id = self.object_id;
        unsafe {
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            (*GLOBAL_GC).remove_tracer(tracer_id, object_id);
        }
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
                // SAFETY: Pointer is valid for the lifetime of this GcRefCellInternal handle; the GC guarantees the allocation is not freed while any handle exists.
                (*self.ptr).borrow().reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcRefCellInternal handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcRefCellInternal handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this GcRefCellInternal handle; the GC guarantees the allocation is not freed while any handle exists.
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

/// A garbage-collected mutable cell that can be shared across threads.
///
/// The thread-safe counterpart of the local `GcRefCell<T>`. `borrow_mut()`
/// triggers the write barrier for generational collection.
pub struct GcRefCell<T>
where
    T: 'static + Sized + Trace,
{
    internal_ptr: *mut GcRefCellInternal<T>,
    ptr: *const RefCell<GcPtr<T>>,
    tracer_id: TracerId,
    object_id: ObjectId,
}

// SAFETY: GcRefCell uses atomic ref-counting internally. All structural mutations
// go through the global GC's Mutex. The RefCell itself is only accessed while
// holding the STW read lock, preventing data races.
unsafe impl<T> Sync for GcRefCell<T> where T: 'static + Sized + Trace + Sync {}
unsafe impl<T> Send for GcRefCell<T> where T: 'static + Sized + Trace + Send {}

impl<T> Drop for GcRefCell<T>
where
    T: Sized + Trace,
{
    fn drop(&mut self) {
        let tracer_id = self.tracer_id;
        let object_id = self.object_id;
        unsafe {
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            (*GLOBAL_GC).remove_tracer(tracer_id, object_id);
        }
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
    /// Allocate a new thread-safe GC-managed interior-mutable cell on the global collector.
    pub fn new(t: T) -> GcRefCell<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).create_gc_cell(t, region)
        }
    }

    /// Allocate a new thread-safe GC-managed interior-mutable cell in the specified region.
    pub fn new_in(t: T, region: crate::generation::RegionId) -> GcRefCell<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).create_gc_cell(t, region) }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<GcRefCell<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).try_create_gc_cell(t, region)
        }
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(
        t: T,
        region: crate::generation::RegionId,
    ) -> Result<GcRefCell<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).try_create_gc_cell(t, region) }
    }

    /// Mutable borrow with write barrier.
    /// Triggers the write barrier so that if this object is in an older generation,
    /// it gets added to the remembered set for young-generation collections.
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GcPtr<T>> {
        unsafe {
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            // self.ptr is valid for the lifetime of this GcRefCell handle.
            let _stw = GLOBAL_GC
                .core
                .stw_lock
                .read()
                .unwrap_or_else(|e| e.into_inner());
            GLOBAL_GC.core.write_barrier(self.ptr as *const dyn Trace);
            (*self.ptr).borrow_mut()
        }
    }
}

impl<T> Clone for GcRefCell<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).clone_from_gc_cell(self) }
    }
}

impl<T> Trace for GcRefCell<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        // SAFETY: internal_ptr is valid for the lifetime of this GcRefCell handle.
        unsafe { (*self.internal_ptr).is_root() }
    }

    fn reset_root(&self) {
        unsafe {
            // SAFETY: internal_ptr is valid for the lifetime of this GcRefCell handle.
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

/// A weak reference to a global GC-managed object.
///
/// Does not prevent collection. `upgrade()` acquires the STW read lock
/// to safely check liveness before returning a strong `Gc<T>`.
pub struct GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    alive: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ptr: *const GcPtr<T>,
}

// SAFETY: GcWeak holds only an Arc<AtomicBool> (thread-safe) and a raw pointer
// that is only dereferenced through the Mutex-protected global GC during upgrade().
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
        if self.alive.load(std::sync::atomic::Ordering::Relaxed) {
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
    /// Attempt to upgrade the weak reference to a strong `Gc<T>`.
    /// Returns `None` if the object has been collected.
    pub fn upgrade(&self) -> Option<Gc<T>> {
        if !self.alive.load(Ordering::Acquire) {
            return None;
        }
        unsafe {
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            // Acquire STW read lock to prevent collection during upgrade
            let _stw = GLOBAL_GC
                .core
                .stw_lock
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if self.alive.load(Ordering::Acquire) {
                Some((*GLOBAL_GC).upgrade_weak(self))
            } else {
                None
            }
        }
    }
}

impl<T> Gc<T>
where
    T: 'static + Sized + Trace,
{
    /// Create a weak reference that does not prevent collection.
    pub fn downgrade(this: &Gc<T>) -> GcWeak<T> {
        let alive = GLOBAL_GC
            .core
            .get_or_create_weak_alive(this.ptr as *const dyn Trace);
        GcWeak {
            alive,
            ptr: this.ptr,
        }
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

/// Thread-safe global garbage collector.
/// Shared across all threads via the `GLOBAL_GC` lazy_static.
/// Internally delegates to `GarbageCollector` which protects all state
/// with Mutex, RwLock, and atomic operations.
pub struct GlobalGarbageCollector {
    pub(crate) core: GarbageCollector,
}

// SAFETY: Delegates to GarbageCollector which protects all state with Mutex/atomics.
unsafe impl Sync for GlobalGarbageCollector {}
unsafe impl Send for GlobalGarbageCollector {}

impl GlobalGarbageCollector {
    fn new() -> GlobalGarbageCollector {
        GlobalGarbageCollector {
            core: GarbageCollector::new(),
        }
    }

    unsafe fn create_gc<T>(&self, t: T, region: crate::generation::RegionId) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (gc_ptr, mem_info_gc_ptr) = self.core.alloc_mem::<GcPtr<T>>();
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();
            // SAFETY: Pointer was just allocated via alloc_mem and is properly aligned for GcPtr<T>.
            std::ptr::write(gc_ptr, GcPtr::new(t));

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            unsafe fn drop_gc_ptr<T: 'static + Trace>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: ptr was originally allocated as a GcPtr<T> and has not been dropped yet.
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
                region,
            });
            gc_maps
                .ptr_to_object
                .insert((gc_ptr as *const dyn Trace).get_thin_ptr(), obj_id);
            self.core.track_alloc(mem_info_gc_ptr.1.size());

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
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            gc
        }
    }

    unsafe fn clone_from_gc<T>(&self, gc: &Gc<T>) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();
            // SAFETY: internal_ptr is valid; the source Gc handle is alive during clone.
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
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            gc
        }
    }

    unsafe fn create_gc_cell<T>(&self, t: T, region: crate::generation::RegionId) -> GcRefCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (gc_ptr, mem_info_gc_ptr) = self.core.alloc_mem::<RefCell<GcPtr<T>>>();
            let (gc_cell_inter_ptr, mem_info_internal_ptr) =
                self.core.alloc_mem::<GcRefCellInternal<T>>();
            // SAFETY: Pointer was just allocated via alloc_mem and is properly aligned for RefCell<GcPtr<T>>.
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let mut gc_maps = self.core.gc_maps.lock().unwrap_or_else(|e| e.into_inner());
            unsafe fn drop_gc_cell_ptr<T: 'static + Trace>(ptr: *mut u8) {
                unsafe {
                    // SAFETY: ptr was originally allocated as a RefCell<GcPtr<T>> and has not been dropped yet.
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
                region,
            });
            gc_maps
                .ptr_to_object
                .insert((gc_ptr as *const dyn Trace).get_thin_ptr(), obj_id);
            self.core.track_alloc(mem_info_gc_ptr.1.size());

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
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            gc
        }
    }

    unsafe fn clone_from_gc_cell<T>(&self, gc: &GcRefCell<T>) -> GcRefCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (gc_inter_ptr, mem_info) = self.core.alloc_mem::<GcRefCellInternal<T>>();
            // SAFETY: internal_ptr is valid; the source GcRefCell handle is alive during clone.
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
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            gc
        }
    }

    unsafe fn try_create_gc<T>(
        &self,
        t: T,
        region: crate::generation::RegionId,
    ) -> Result<Gc<T>, crate::gc::GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            use std::alloc::dealloc;
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
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
                    // SAFETY: ptr was originally allocated as a GcPtr<T> and has not been dropped yet.
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
                region,
            });
            gc_maps
                .ptr_to_object
                .insert((gc_ptr as *const dyn Trace).get_thin_ptr(), obj_id);
            self.core.track_alloc(mem_info_gc_ptr.1.size());

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
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            Ok(gc)
        }
    }

    unsafe fn try_create_gc_cell<T>(
        &self,
        t: T,
        region: crate::generation::RegionId,
    ) -> Result<GcRefCell<T>, crate::gc::GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            use std::alloc::dealloc;
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
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
                    // SAFETY: ptr was originally allocated as a RefCell<GcPtr<T>> and has not been dropped yet.
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
                region,
            });
            gc_maps
                .ptr_to_object
                .insert((gc_ptr as *const dyn Trace).get_thin_ptr(), obj_id);
            self.core.track_alloc(mem_info_gc_ptr.1.size());

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
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            Ok(gc)
        }
    }

    /// Create a new strong Gc<T> from a weak reference (for upgrade).
    /// Caller must hold STW read lock.
    unsafe fn upgrade_weak<T>(&self, weak: &GcWeak<T>) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
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
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_inter_ptr,
                GcInternal::new(weak.ptr, tracer_id, object_id),
            );
            drop(gc_maps);
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: weak.ptr,
                tracer_id,
                object_id,
            };
            // SAFETY: internal_ptr was just written above; weak.ptr is valid because alive check passed.
            (*(*gc.internal_ptr).ptr).reset_root();
            gc
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect(&self) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.collect();
        }
    }

    #[allow(dead_code)]
    unsafe fn collect_all(&self) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.collect_all();
        }
    }

    pub(crate) unsafe fn remove_tracer(&self, tracer_id: TracerId, object_id: ObjectId) {
        unsafe {
            // SAFETY: Caller provides valid tracer_id and object_id from a live Gc/GcRefCell handle.
            self.core.remove_tracer(tracer_id, object_id);
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn begin_collection(&self, max_gen: crate::generation::Generation) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.begin_collection(max_gen);
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn mark_step(&self, budget: usize) -> bool {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.mark_step(budget) }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn finish_collection(&self) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.finish_collection() }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_incremental(
        &self,
        max_gen: crate::generation::Generation,
        step_budget: usize,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_incremental(max_gen, step_budget) }
    }

    /// Begin concurrent collection: short STW to snapshot roots + edges.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn begin_concurrent_collection(&self, max_gen: crate::generation::Generation) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.begin_concurrent_collection(max_gen);
        }
    }

    /// Process gray objects using edge snapshot. NO STW lock — safe for concurrent use.
    pub fn concurrent_mark_step(&self, budget: usize) -> bool {
        self.core.concurrent_mark_step(budget)
    }

    /// Run a complete concurrent collection.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_concurrent(
        &self,
        max_gen: crate::generation::Generation,
        step_budget: usize,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_concurrent(max_gen, step_budget) }
    }

    /// Time-budgeted incremental mark step. See [`GarbageCollector::mark_step_timed`].
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn mark_step_timed(&self, max_duration: Duration) -> bool {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.mark_step_timed(max_duration) }
    }

    /// Run a complete incremental collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_incremental_timed(
        &self,
        max_gen: crate::generation::Generation,
        max_step_duration: Duration,
    ) -> crate::generation::CollectionStats {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
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
        max_gen: crate::generation::Generation,
        max_step_duration: Duration,
    ) -> crate::generation::CollectionStats {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core
                .collect_concurrent_timed(max_gen, max_step_duration)
        }
    }

    /// Create a new region. Future allocations go into this region.
    pub fn new_region(&self) -> crate::generation::RegionId {
        self.core.new_region()
    }

    /// Get the current allocation region.
    pub fn current_region(&self) -> crate::generation::RegionId {
        self.core.current_region()
    }

    /// Collect only objects in the specified region.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub unsafe fn collect_region(
        &self,
        region: crate::generation::RegionId,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_region(region) }
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> crate::generation::GcStats {
        self.core.stats()
    }

    /// Set custom promotion thresholds.
    pub fn set_promotion_config(&self, config: crate::generation::PromotionConfig) {
        self.core.set_promotion_config(config);
    }

    /// Get the current promotion config.
    pub fn promotion_config(&self) -> crate::generation::PromotionConfig {
        self.core.promotion_config()
    }

    /// Register a callback invoked after each collection cycle.
    pub fn set_on_collection(
        &self,
        callback: impl Fn(&crate::generation::CollectionStats) + Send + Sync + 'static,
    ) {
        self.core.set_on_collection(callback);
    }

    /// Remove the collection callback.
    pub fn clear_on_collection(&self) {
        self.core.clear_on_collection();
    }
}

/// Callback to start a global collection strategy.
pub type StartGlobalStrategyFn =
    Box<dyn FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;
/// Callback to stop a global collection strategy.
pub type StopGlobalStrategyFn = Box<dyn FnMut(&'static GlobalGarbageCollector)>;

/// Pluggable collection strategy for the global (thread-safe) GC.
/// Controls when and how garbage collection runs. Use `change_strategy` to swap at runtime.
pub struct GlobalStrategy {
    gc: Cell<&'static GlobalGarbageCollector>,
    is_active: AtomicBool,
    start_func: Mutex<StartGlobalStrategyFn>,
    stop_func: Mutex<StopGlobalStrategyFn>,
    join_handle: Mutex<Option<JoinHandle<()>>>,
}

// SAFETY: The Cell<&'static GlobalGarbageCollector> is only written during initialization
// (before any thread can access it). All other fields use Mutex or atomics.
unsafe impl Sync for GlobalStrategy {}
unsafe impl Send for GlobalStrategy {}

impl GlobalStrategy {
    fn new<StartFn, StopFn>(
        gc: &'static GlobalGarbageCollector,
        start_fn: StartFn,
        stop_fn: StopFn,
    ) -> GlobalStrategy
    where
        StartFn: 'static
            + FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
        StopFn: 'static + FnMut(&'static GlobalGarbageCollector),
    {
        GlobalStrategy {
            gc: Cell::new(gc),
            is_active: AtomicBool::new(false),
            start_func: Mutex::new(Box::new(start_fn)),
            stop_func: Mutex::new(Box::new(stop_fn)),
            join_handle: Mutex::new(None),
        }
    }

    /// Replace the current collection strategy. Stops the current strategy first if active.
    pub fn change_strategy<StartFn, StopFn>(&self, start_fn: StartFn, stop_fn: StopFn)
    where
        StartFn: 'static
            + FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
        StopFn: 'static + FnMut(&'static GlobalGarbageCollector),
    {
        if self.is_active() {
            self.stop();
        }
        let mut start_func = self.start_func.lock().unwrap_or_else(|e| e.into_inner());
        let mut stop_func = self.stop_func.lock().unwrap_or_else(|e| e.into_inner());
        *start_func = Box::new(start_fn);
        *stop_func = Box::new(stop_fn);
    }

    /// Returns `true` if the strategy's background collection is currently running.
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Atomically check if inactive and start. Safe to call from multiple threads.
    pub fn ensure_started(&'static self) {
        if self
            .is_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let mut start_func = self.start_func.lock().unwrap_or_else(|e| e.into_inner());
            let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
            *join_handle = (*(start_func))(self.gc.get(), &self.is_active);
        }
    }

    /// Start the collection strategy (e.g., spawn a background collection thread).
    pub fn start(&'static self) {
        self.is_active.store(true, Ordering::Release);
        let mut start_func = self.start_func.lock().unwrap_or_else(|e| e.into_inner());
        let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
        *join_handle = (*(start_func))(self.gc.get(), &self.is_active);
    }

    /// Stop the collection strategy and join any background thread.
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(join_handle) = join_handle.take() {
            join_handle
                .join()
                .expect("GlobalStrategy::stop, GlobalStrategy Thread being joined has panicked !!");
        }
        let mut stop_func = self.stop_func.lock().unwrap_or_else(|e| e.into_inner());
        (*(stop_func))(self.gc.get());
    }
}

impl Drop for GlobalStrategy {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
        let mut stop_func = self.stop_func.lock().unwrap_or_else(|e| e.into_inner());
        (*(stop_func))(self.gc.get());
    }
}

lazy_static! {
    pub static ref GLOBAL_GC: GlobalGarbageCollector = GlobalGarbageCollector::new();
    pub static ref GLOBAL_GC_STRATEGY: GlobalStrategy = {
        let gc = &(*GLOBAL_GC);
        GlobalStrategy::new(
            gc,
            move |global_gc, _| {
                let mut basic_strategy_global_gc = BASIC_STRATEGY_GLOBAL_GC
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                *basic_strategy_global_gc = Some(global_gc);
                None
            },
            move |_global_gc| {
                let mut basic_strategy_global_gc = BASIC_STRATEGY_GLOBAL_GC
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                *basic_strategy_global_gc = None;
            },
        )
    };
}

#[cfg(test)]
mod tests {
    use crate::gc::sync::{GLOBAL_GC, Gc};
    use std::sync::Mutex;

    // Serialize sync GC tests since they share GLOBAL_GC.
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    /// Clean residual state and return baseline trs count.
    fn setup() -> (std::sync::MutexGuard<'static, ()>, usize) {
        let guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        let baseline = (*GLOBAL_GC)
            .core
            .gc_maps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .tracers
            .len();
        (guard, baseline)
    }

    #[test]
    fn one_object() {
        let (_guard, baseline) = setup();
        let _one = Gc::new(1);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            1
        );
    }

    #[test]
    fn gc_collect_one_from_one() {
        let (_guard, baseline) = setup();
        {
            let _one = Gc::new(1);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            0
        );
    }

    #[test]
    #[allow(unused_assignments)]
    fn two_objects_reassign() {
        let (_guard, baseline) = setup();
        let mut one = Gc::new(1);
        one = Gc::new(2);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        // Reassignment drops old Gc (remove_tracer), so only 1 tracer remains
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            1
        );
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_after_reassign() {
        let (_guard, baseline) = setup();
        let mut one = Gc::new(1);
        one = Gc::new(2);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        // one is still live, so 1 tracer remains
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            1
        );
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_two_from_two() {
        let (_guard, baseline) = setup();
        {
            let mut one = Gc::new(1);
            one = Gc::new(2);
            drop(one);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            0
        );
    }

    #[test]
    fn stw_blocks_allocation_during_collection() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let (_guard, _) = setup();

        let allocated_during_stw = Arc::new(AtomicBool::new(false));
        let collection_started = Arc::new(AtomicBool::new(false));
        let collection_done = Arc::new(AtomicBool::new(false));

        // Hold the STW write lock to simulate an ongoing collection
        let stw_guard = (*GLOBAL_GC)
            .core
            .stw_lock
            .write()
            .unwrap_or_else(|e| e.into_inner());
        collection_started.store(true, Ordering::Release);

        let alloc_flag = allocated_during_stw.clone();
        let started = collection_started.clone();
        let _done = collection_done.clone();

        let handle = std::thread::spawn(move || {
            // Wait for collection to be in progress
            while !started.load(Ordering::Acquire) {}
            // Try to allocate — should block on STW read lock
            let _obj = Gc::new(42);
            alloc_flag.store(true, Ordering::Release);
        });

        // Give the allocator thread time to block
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Allocation should NOT have completed while we hold the write lock
        assert!(
            !allocated_during_stw.load(Ordering::Acquire),
            "allocation must block while STW write lock is held"
        );

        // Release the STW lock — simulates end of collection
        drop(stw_guard);
        collection_done.store(true, Ordering::Release);

        handle.join().expect("allocator thread panicked");
        assert!(
            allocated_during_stw.load(Ordering::Acquire),
            "allocation should complete after STW lock is released"
        );
    }

    #[test]
    fn stw_allows_concurrent_allocations() {
        let (_guard, _) = setup();
        // Multiple threads allocating concurrently should all succeed (read locks are shared)
        let handles: Vec<_> = (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    let _obj = Gc::new(i);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("concurrent allocation thread panicked");
        }
    }

    #[test]
    fn change_strategy_while_active_does_not_deadlock() {
        let (_guard, _) = setup();
        let _gc = Gc::new(1); // ensures strategy is started
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use crate::gc::sync::GLOBAL_GC_STRATEGY;
            GLOBAL_GC_STRATEGY.change_strategy(|_gc, _| None, |_gc| {});
            tx.send(()).unwrap();
        });
        rx.recv_timeout(std::time::Duration::from_secs(3))
            .expect("change_strategy deadlocked when called while active");
    }

    #[test]
    fn incremental_collects_dead_objects() {
        // RC hybrid frees non-cyclic objects immediately. Verify object is gone.
        let (_guard, baseline) = setup();
        {
            let _obj = Gc::new(42);
        }
        // Object already freed by RC
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            0
        );
        // Incremental still works correctly
        let _stats =
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            unsafe { (*GLOBAL_GC).collect_incremental(crate::generation::Generation::Gen2, 10) };
    }

    #[test]
    fn incremental_preserves_live_objects() {
        let (_guard, baseline) = setup();
        let _live = Gc::new(99);
        let stats =
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            unsafe { (*GLOBAL_GC).collect_incremental(crate::generation::Generation::Gen2, 10) };
        assert_eq!(
            stats.objects_collected, 0,
            "incremental must not collect live objects"
        );
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            1
        );
    }

    #[test]
    fn incremental_step_by_step_sync() {
        // With RC hybrid, dead objects are already freed. Test with live object.
        let (_guard, _) = setup();
        let _live = Gc::new(2);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe {
            (*GLOBAL_GC).begin_collection(crate::generation::Generation::Gen2);
            while !(*GLOBAL_GC).mark_step(1) {}
            let stats = (*GLOBAL_GC).finish_collection();
            assert_eq!(
                stats.objects_collected, 0,
                "live object must not be collected"
            );
        }
    }

    #[test]
    fn try_new_succeeds() {
        let (_guard, _) = setup();
        let result = Gc::try_new(42);
        assert!(
            result.is_ok(),
            "try_new should succeed for normal allocation"
        );
        assert_eq!(**result.unwrap(), 42);
    }

    #[test]
    fn try_new_object_collected_when_dead() {
        let (_guard, baseline) = setup();
        {
            let _obj = Gc::try_new(77).unwrap();
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            0
        );
    }

    // --- Diagnostics API tests ---

    #[test]
    fn sync_stats_reports_live_objects() {
        let (_guard, _) = setup();
        let baseline = (*GLOBAL_GC).stats().live_objects;
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let stats = (*GLOBAL_GC).stats();
        assert_eq!(stats.live_objects - baseline, 2);
    }

    #[test]
    fn sync_stats_reports_heap_size() {
        let (_guard, _) = setup();
        let baseline = (*GLOBAL_GC).stats().heap_size;
        let _a = Gc::new(42i32);
        let stats = (*GLOBAL_GC).stats();
        assert!(stats.heap_size > baseline);
    }

    #[test]
    fn sync_stats_tracks_total_collections() {
        let (_guard, _) = setup();
        let before = (*GLOBAL_GC).stats().total_collections;
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        let after = (*GLOBAL_GC).stats().total_collections;
        assert_eq!(after - before, 2);
    }

    #[test]
    fn sync_stats_tracks_last_collection() {
        let (_guard, _) = setup();
        // With RC hybrid, dead objects are already freed. Just verify stats recording.
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        let stats = (*GLOBAL_GC).stats();
        assert!(
            stats.last_collection.is_some(),
            "last_collection should be Some after collect"
        );
    }

    #[test]
    fn sync_stats_reports_allocation_count() {
        let (_guard, _) = setup();
        // collect resets allocation_count
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let stats = (*GLOBAL_GC).stats();
        assert!(stats.allocation_count >= 2);
    }

    // --- Debug impl tests ---

    #[test]
    fn sync_debug_gc_prints_value() {
        let (_guard, _) = setup();
        let gc = Gc::new(42);
        assert_eq!(format!("{:?}", gc), "Gc(42)");
    }

    #[test]
    fn sync_debug_gc_weak() {
        let (_guard, _) = setup();
        let strong = Gc::new(1);
        let weak = Gc::downgrade(&strong);
        assert_eq!(format!("{:?}", weak), "GcWeak(alive)");
    }

    // --- Cycle tests ---

    struct SyncCyclicNode {
        next: std::cell::RefCell<Option<Gc<SyncCyclicNode>>>,
    }
    unsafe impl Send for SyncCyclicNode {}
    unsafe impl Sync for SyncCyclicNode {}
    impl crate::gc::Trace for SyncCyclicNode {
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
        fn trace_children(&self, children: &mut Vec<*const dyn crate::gc::Trace>) {
            if let Some(ref gc) = *self.next.borrow() {
                gc.trace_children(children);
            }
        }
    }
    impl crate::gc::Finalize for SyncCyclicNode {
        fn finalize(&self) {}
    }

    #[test]
    fn sync_self_cycle_collected() {
        let (_guard, _) = setup();
        {
            let a = Gc::new(SyncCyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(a.clone());
            drop(a);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
    }

    #[test]
    fn sync_three_node_cycle_collected() {
        let (_guard, _) = setup();
        {
            let a = Gc::new(SyncCyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(SyncCyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let c = Gc::new(SyncCyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(c.clone());
            *c.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
            drop(c);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
    }

    // --- Concurrent marking tests (Strategy 21) ---

    #[test]
    fn sync_concurrent_preserves_live_objects() {
        let (_guard, baseline) = setup();
        let _live = Gc::new(42);
        let stats =
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            unsafe { (*GLOBAL_GC).collect_concurrent(crate::generation::Generation::Gen2, 10) };
        assert_eq!(
            stats.objects_collected, 0,
            "concurrent must not collect live objects"
        );
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .gc_maps
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tracers
                .len()
                - baseline,
            1
        );
    }

    #[test]
    fn sync_concurrent_step_by_step() {
        let (_guard, _) = setup();
        let _live = Gc::new(99);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe {
            (*GLOBAL_GC).begin_concurrent_collection(crate::generation::Generation::Gen2);
            while !(*GLOBAL_GC).concurrent_mark_step(1) {}
            let stats = (*GLOBAL_GC).finish_collection();
            assert_eq!(stats.objects_collected, 0, "live objects must survive");
        }
    }

    // --- Region-based collection tests (Strategy 22) ---

    #[test]
    fn sync_region_assignment() {
        let (_guard, _) = setup();
        let _obj = Gc::new(1);
        let region = (*GLOBAL_GC).current_region();
        let gc_maps = (*GLOBAL_GC)
            .core
            .gc_maps
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert!(
            gc_maps.objects.values().any(|e| e.region == region),
            "object should be assigned to current region"
        );
    }

    #[test]
    fn sync_collect_region() {
        let (_guard, _) = setup();
        // Create cycle in a new region
        let region = (*GLOBAL_GC).new_region();
        {
            let a = Gc::new(SyncCyclicNode {
                next: std::cell::RefCell::new(None),
            });
            let b = Gc::new(SyncCyclicNode {
                next: std::cell::RefCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        let stats = unsafe { (*GLOBAL_GC).collect_region(region) };
        assert!(
            stats.objects_collected >= 2,
            "should collect cycle in target region"
        );
    }

    // --- RC hybrid tests (Strategy 23) ---

    #[test]
    fn sync_rc_immediate_dealloc() {
        let (_guard, baseline) = setup();
        {
            let _obj = Gc::new(42);
        }
        // RC should have freed it — check tracers are back to baseline
        let gc_maps = (*GLOBAL_GC)
            .core
            .gc_maps
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            gc_maps.tracers.len(),
            baseline,
            "RC should free object immediately"
        );
    }

    #[test]
    fn sync_rc_clone_keeps_alive() {
        let (_guard, _) = setup();
        let a = Gc::new(99);
        let b = a.clone();
        let baseline = (*GLOBAL_GC)
            .core
            .gc_maps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .objects
            .len();
        drop(a);
        let after = (*GLOBAL_GC)
            .core
            .gc_maps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .objects
            .len();
        assert_eq!(after, baseline, "object should survive when clone exists");
        assert_eq!(**b, 99);
    }
}
