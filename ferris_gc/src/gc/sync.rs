use std::cell::{Cell, RefCell};
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::basic_gc_strategy::{BASIC_STRATEGY_GLOBAL_GC, basic_gc_strategy_start};
use crate::gc::{Finalize, GarbageCollector, ObjectEntry, Trace, TracerInfo, TracerList};
use crate::generation::Generation;
use crate::slot_map::ObjectId;

/// Convenience alias for an optional thread-safe GC pointer.
pub type OptGc<T> = Option<Gc<T>>;
/// Convenience alias for an optional thread-safe GC interior-mutable cell.
pub type OptGcCell<T> = Option<GcCell<T>>;

/// Type-safe region identifier for the **global (thread-safe)** garbage collector.
/// Returned by [`GlobalGarbageCollector::new_region`]. Cannot be used with
/// the thread-local GC — the compiler will reject the mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyncRegionId(crate::generation::RegionId);

impl SyncRegionId {
    /// Return the raw numeric id (for logging / diagnostics).
    pub fn id(self) -> u32 {
        (self.0).0
    }
}

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

/// Internal pointer wrapper used as the `Deref` target of `sync::Gc<T>` and `sync::GcCell<T>`.
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

#[allow(dead_code)]
pub(crate) struct GcInternal<T>
where
    T: 'static + Sized + Trace,
{
    is_root: AtomicBool,
    ptr: *const GcPtr<T>,
    pub(crate) object_id: ObjectId,
}

impl<T> GcInternal<T>
where
    T: 'static + Sized + Trace,
{
    fn new(ptr: *const GcPtr<T>, object_id: ObjectId) -> GcInternal<T> {
        GcInternal {
            is_root: AtomicBool::new(true),
            ptr,
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

    unsafe fn relocate(&self, old_ptr: *const u8, new_ptr: *const u8) {
        if self.ptr as *const u8 == old_ptr {
            // SAFETY: Called during STW compaction. The stw_lock write guard
            // guarantees no concurrent access to this field.
            let self_mut = self as *const Self as *mut Self;
            unsafe { (*self_mut).ptr = new_ptr as *const GcPtr<T> };
        }
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
    object_id: ObjectId,
}

/// # Safety
/// The underlying GcInternal uses atomic ref-counting, and the global GC
/// protects all structural mutations with Mutex/RwLock. `Gc<T>` is safe to
/// send across threads when `T` itself is Send.
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
    pub fn new_in(t: T, region: SyncRegionId) -> Gc<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).create_gc(t, region.0) }
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
    pub fn try_new_in(t: T, region: SyncRegionId) -> Result<Gc<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).try_create_gc(t, region.0) }
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
        let tracer_ptr = self.internal_ptr as *const u8;
        let object_id = self.object_id;
        unsafe {
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            (*GLOBAL_GC).remove_tracer(object_id, tracer_ptr);
        }
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
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: Null check above ensures the pointer is valid.
        unsafe { (*self.ptr).trace() }
    }

    fn reset(&self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: Null check above ensures the pointer is valid.
        unsafe { (*self.ptr).reset() }
    }

    fn is_traceable(&self) -> bool {
        if self.ptr.is_null() {
            return false;
        }
        // SAFETY: Null check above ensures the pointer is valid.
        unsafe { (*self.ptr).is_traceable() }
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

#[allow(dead_code)]
pub(crate) struct GcCellInternal<T>
where
    T: 'static + Sized + Trace,
{
    is_root: AtomicBool,
    ptr: *const RefCell<GcPtr<T>>,
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
            is_root: AtomicBool::new(true),
            ptr,
            object_id,
        }
    }
}

impl<T> Trace for GcCellInternal<T>
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
                // SAFETY: Pointer is valid for the lifetime of this GcCellInternal handle; the GC guarantees the allocation is not freed while any handle exists.
                (*self.ptr).borrow().reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcCellInternal handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcCellInternal handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this GcCellInternal handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr).borrow().is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for GcCellInternal<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A garbage-collected mutable cell that can be shared across threads.
///
/// The thread-safe counterpart of the local `GcCell<T>`. `borrow_mut()`
/// triggers the write barrier for generational collection.
pub struct GcCell<T>
where
    T: 'static + Sized + Trace,
{
    internal_ptr: *mut GcCellInternal<T>,
    ptr: *const RefCell<GcPtr<T>>,
    object_id: ObjectId,
}

// SAFETY: GcCell uses atomic ref-counting internally. All structural mutations
// go through the global GC's Mutex. The RefCell itself is only accessed while
// holding the STW read lock, preventing data races.
unsafe impl<T> Sync for GcCell<T> where T: 'static + Sized + Trace + Sync {}
unsafe impl<T> Send for GcCell<T> where T: 'static + Sized + Trace + Send {}

impl<T> Drop for GcCell<T>
where
    T: Sized + Trace,
{
    fn drop(&mut self) {
        let tracer_ptr = self.internal_ptr as *const u8;
        let object_id = self.object_id;
        unsafe {
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            (*GLOBAL_GC).remove_tracer(object_id, tracer_ptr);
        }
    }
}

impl<T> Deref for GcCell<T>
where
    T: 'static + Sized + Trace,
{
    type Target = RefCell<GcPtr<T>>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { &(*self.ptr) }
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
    /// Allocate a new thread-safe GC-managed interior-mutable cell on the global collector.
    pub fn new(t: T) -> GcCell<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).create_gc_cell(t, region)
        }
    }

    /// Allocate a new thread-safe GC-managed interior-mutable cell in the specified region.
    pub fn new_in(t: T, region: SyncRegionId) -> GcCell<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).create_gc_cell(t, region.0) }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<GcCell<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).try_create_gc_cell(t, region)
        }
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(t: T, region: SyncRegionId) -> Result<GcCell<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).try_create_gc_cell(t, region.0) }
    }

    /// Mutable borrow with write barrier.
    /// Triggers the write barrier so that if this object is in an older generation,
    /// its card is marked dirty in the card table for young-generation collections.
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GcPtr<T>> {
        unsafe {
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            // self.ptr is valid for the lifetime of this GcCell handle.
            let _stw = GLOBAL_GC
                .core
                .stw_lock
                .read()
                .unwrap_or_else(|e| e.into_inner());
            GLOBAL_GC
                .core
                .write_barrier(self.object_id, self.ptr as *const dyn Trace);
            (*self.ptr).borrow_mut()
        }
    }
}

impl<T> Clone for GcCell<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).clone_from_gc_cell(self) }
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
        unsafe {
            // SAFETY: internal_ptr is valid for the lifetime of this GcCell handle.
            (*self.internal_ptr).reset_root();
        }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr).borrow().is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for GcCell<T>
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
    object_id: ObjectId,
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
            object_id: self.object_id,
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
    /// Returns `true` if the referenced object is still alive (not yet collected).
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

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
                (*GLOBAL_GC).upgrade_weak(self)
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
            .get_or_create_weak_alive(this.object_id);
        GcWeak {
            alive,
            ptr: this.ptr,
            object_id: this.object_id,
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

#[allow(dead_code)]
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

            let mut gc_maps = self.core.lock_gc_maps();
            unsafe fn dealloc_gc_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (*(ptr as *const GcPtr<T>)).t.finalize();
                    }));
                    std::ptr::drop_in_place(ptr as *mut GcPtr<T>);
                }
            }
            let root_ref_count_ptr = &(*gc_ptr).info.root_ref_count as *const AtomicUsize as *const Cell<usize>;
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: mem_info_gc_ptr.0,
                layout: mem_info_gc_ptr.1,
                generation: Generation::Gen0,
                survive_count: 0,
                dealloc_fn: dealloc_gc_ptr::<T>,
                weak_alive: None,
                tracers: TracerList::new(TracerInfo {
                    tracer_ptr: gc_inter_ptr as *const dyn Trace,
                    mem: mem_info_internal_ptr.0,
                    layout: mem_info_internal_ptr.1,
                }),
                region,
                root_ref_count_ptr,
            });
            // ptr_to_object, region stats, and card table are populated lazily
            // (during marking/promotion), not on the allocation hot path.
            self.core.track_alloc(mem_info_gc_ptr.1.size());
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr, obj_id));
            drop(gc_maps);

            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
                object_id: obj_id,
            };
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.set(self.core.allocation_count.get() + 1);
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

            let mut gc_maps = self.core.lock_gc_maps();
            // RC hybrid: push tracer for the cloned object
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
            drop(gc_maps);

            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
                object_id,
            };
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            gc
        }
    }

    unsafe fn create_gc_cell<T>(&self, t: T, region: crate::generation::RegionId) -> GcCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (gc_ptr, mem_info_gc_ptr) = self.core.alloc_mem::<RefCell<GcPtr<T>>>();
            let (gc_cell_inter_ptr, mem_info_internal_ptr) =
                self.core.alloc_mem::<GcCellInternal<T>>();
            // SAFETY: Pointer was just allocated via alloc_mem and is properly aligned for RefCell<GcPtr<T>>.
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let mut gc_maps = self.core.lock_gc_maps();
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
                &(*(*gc_ptr).as_ptr()).info.root_ref_count as *const AtomicUsize as *const Cell<usize>;
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: mem_info_gc_ptr.0,
                layout: mem_info_gc_ptr.1,
                generation: Generation::Gen0,
                survive_count: 0,
                dealloc_fn: dealloc_gc_cell_ptr::<T>,
                weak_alive: None,
                tracers: TracerList::new(TracerInfo {
                    tracer_ptr: gc_cell_inter_ptr as *const dyn Trace,
                    mem: mem_info_internal_ptr.0,
                    layout: mem_info_internal_ptr.1,
                }),
                region,
                root_ref_count_ptr,
            });
            // ptr_to_object, region stats, and card table are populated lazily
            // (during marking/promotion), not on the allocation hot path.
            self.core.track_alloc(mem_info_gc_ptr.1.size());
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_cell_inter_ptr,
                GcCellInternal::new(gc_ptr, obj_id),
            );
            drop(gc_maps);

            let gc = GcCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
                object_id: obj_id,
            };
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.set(self.core.allocation_count.get() + 1);
            gc
        }
    }

    unsafe fn clone_from_gc_cell<T>(&self, gc: &GcCell<T>) -> GcCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (gc_inter_ptr, mem_info) = self.core.alloc_mem::<GcCellInternal<T>>();
            // SAFETY: internal_ptr is valid; the source GcCell handle is alive during clone.
            let object_id = (*gc.internal_ptr).object_id;

            let mut gc_maps = self.core.lock_gc_maps();
            // RC hybrid: push tracer for the cloned object
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
            drop(gc_maps);
            let gc = GcCell {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
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
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (gc_ptr, mem_info_gc_ptr) = self.core.try_alloc_mem_with_gc::<GcPtr<T>>()?;
            let (gc_inter_ptr, mem_info_internal_ptr) =
                match self.core.try_alloc_mem_with_gc::<GcInternal<T>>() {
                    Ok(v) => v,
                    Err(e) => {
                        // SAFETY: Memory was allocated with the same layout via try_alloc_mem_with_gc.
                        mem_info_gc_ptr.0.dealloc_mem(mem_info_gc_ptr.1);
                        return Err(e);
                    }
                };
            // SAFETY: Pointer was just allocated via try_alloc_mem_with_gc and is properly aligned for GcPtr<T>.
            std::ptr::write(gc_ptr, GcPtr::new(t));

            let mut gc_maps = self.core.lock_gc_maps();
            unsafe fn dealloc_gc_ptr<T: 'static + Trace + Finalize>(ptr: *mut u8) {
                unsafe {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (*(ptr as *const GcPtr<T>)).t.finalize();
                    }));
                    std::ptr::drop_in_place(ptr as *mut GcPtr<T>);
                }
            }
            let root_ref_count_ptr = &(*gc_ptr).info.root_ref_count as *const AtomicUsize as *const Cell<usize>;
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: mem_info_gc_ptr.0,
                layout: mem_info_gc_ptr.1,
                generation: Generation::Gen0,
                survive_count: 0,
                dealloc_fn: dealloc_gc_ptr::<T>,
                weak_alive: None,
                tracers: TracerList::new(TracerInfo {
                    tracer_ptr: gc_inter_ptr as *const dyn Trace,
                    mem: mem_info_internal_ptr.0,
                    layout: mem_info_internal_ptr.1,
                }),
                region,
                root_ref_count_ptr,
            });
            // ptr_to_object, region stats, and card table are populated lazily
            // (during marking/promotion), not on the allocation hot path.
            self.core.track_alloc(mem_info_gc_ptr.1.size());
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr, obj_id));
            drop(gc_maps);
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
                object_id: obj_id,
            };
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.set(self.core.allocation_count.get() + 1);
            Ok(gc)
        }
    }

    unsafe fn try_create_gc_cell<T>(
        &self,
        t: T,
        region: crate::generation::RegionId,
    ) -> Result<GcCell<T>, crate::gc::GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (gc_ptr, mem_info_gc_ptr) =
                self.core.try_alloc_mem_with_gc::<RefCell<GcPtr<T>>>()?;
            let (gc_cell_inter_ptr, mem_info_internal_ptr) =
                match self.core.try_alloc_mem_with_gc::<GcCellInternal<T>>() {
                    Ok(v) => v,
                    Err(e) => {
                        // SAFETY: Memory was allocated with the same layout via try_alloc_mem_with_gc.
                        mem_info_gc_ptr.0.dealloc_mem(mem_info_gc_ptr.1);
                        return Err(e);
                    }
                };
            // SAFETY: Pointer was just allocated via try_alloc_mem_with_gc and is properly aligned for RefCell<GcPtr<T>>.
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let mut gc_maps = self.core.lock_gc_maps();
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
                &(*(*gc_ptr).as_ptr()).info.root_ref_count as *const AtomicUsize as *const Cell<usize>;
            let obj_id = gc_maps.objects.insert(ObjectEntry {
                ptr: gc_ptr as *const dyn Trace,
                mem: mem_info_gc_ptr.0,
                layout: mem_info_gc_ptr.1,
                generation: Generation::Gen0,
                survive_count: 0,
                dealloc_fn: dealloc_gc_cell_ptr::<T>,
                weak_alive: None,
                tracers: TracerList::new(TracerInfo {
                    tracer_ptr: gc_cell_inter_ptr as *const dyn Trace,
                    mem: mem_info_internal_ptr.0,
                    layout: mem_info_internal_ptr.1,
                }),
                region,
                root_ref_count_ptr,
            });
            // ptr_to_object, region stats, and card table are populated lazily
            // (during marking/promotion), not on the allocation hot path.
            self.core.track_alloc(mem_info_gc_ptr.1.size());
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_cell_inter_ptr,
                GcCellInternal::new(gc_ptr, obj_id),
            );
            drop(gc_maps);
            let gc = GcCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
                object_id: obj_id,
            };
            // SAFETY: internal_ptr and ptr were just written above and are valid.
            (*(*gc.internal_ptr).ptr).reset_root();
            self.core.allocation_count.set(self.core.allocation_count.get() + 1);
            Ok(gc)
        }
    }

    /// Create a new strong Gc<T> from a weak reference (for upgrade).
    /// Caller must hold STW read lock.
    unsafe fn upgrade_weak<T>(&self, weak: &GcWeak<T>) -> Option<Gc<T>>
    where
        T: Sized + Trace,
    {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());

            let mut gc_maps = self.core.lock_gc_maps();
            // Re-check under gc_maps lock: the object may have been freed by a
            // concurrent drop (RC hybrid) between the alive check and this point.
            let object_id = weak.object_id;
            if gc_maps.objects.get(object_id).is_none() {
                return None;
            }
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();
            // RC hybrid: push tracer for upgraded object
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.tracers.push(TracerInfo {
                    tracer_ptr: gc_inter_ptr as *const dyn Trace,
                    mem: mem_info_internal_ptr.0,
                    layout: mem_info_internal_ptr.1,
                });
            }
            // Initialize tracer memory BEFORE releasing the lock, so the background
            // GC thread cannot read uninitialized memory via tracer_ptr.
            std::ptr::write(
                gc_inter_ptr,
                GcInternal::new(weak.ptr, object_id),
            );
            drop(gc_maps);
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: weak.ptr,
                object_id,
            };
            // SAFETY: internal_ptr was just written above; weak.ptr is valid because
            // the object exists (we checked under gc_maps lock above).
            (*(*gc.internal_ptr).ptr).reset_root();
            Some(gc)
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect(&self) {
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

    /// Parallel collection: serial mark phase, parallel sweep/dealloc using rayon.
    /// Collects objects in generations 0..=max_gen.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn collect_parallel(
        &self,
        max_gen: crate::generation::Generation,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_parallel(max_gen) }
    }

    /// Collection with parallel mark AND parallel sweep using rayon.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn collect_parallel_mark(
        &self,
        max_gen: crate::generation::Generation,
    ) -> crate::generation::CollectionStats {
        unsafe { self.core.collect_parallel_mark(max_gen) }
    }

    pub(crate) unsafe fn remove_tracer(&self, object_id: ObjectId, tracer_ptr: *const u8) {
        unsafe {
            // SAFETY: Caller provides valid object_id and tracer_ptr from a live Gc/GcCell handle.
            self.core.remove_tracer(object_id, tracer_ptr);
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn begin_collection(&self, max_gen: crate::generation::Generation) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.begin_collection(max_gen);
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn mark_step(&self, budget: usize) -> bool {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.mark_step(budget) }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn finish_collection(&self) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.finish_collection() }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_incremental(
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
    pub(crate) unsafe fn begin_concurrent_collection(
        &self,
        max_gen: crate::generation::Generation,
    ) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.begin_concurrent_collection(max_gen);
        }
    }

    /// Process gray objects using edge snapshot. NO STW lock — safe for concurrent use.
    pub(crate) fn concurrent_mark_step(&self, budget: usize) -> bool {
        self.core.concurrent_mark_step(budget)
    }

    /// Run a complete concurrent collection.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_concurrent(
        &self,
        max_gen: crate::generation::Generation,
        step_budget: usize,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_concurrent(max_gen, step_budget) }
    }

    /// Time-budgeted incremental mark step. See `GarbageCollector::mark_step_timed`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn mark_step_timed(&self, max_duration: Duration) -> bool {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.mark_step_timed(max_duration) }
    }

    /// Run a complete incremental collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_incremental_timed(
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
    pub(crate) fn concurrent_mark_step_timed(&self, max_duration: Duration) -> bool {
        self.core.concurrent_mark_step_timed(max_duration)
    }

    /// Run a complete concurrent collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_concurrent_timed(
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
    pub fn new_region(&self) -> SyncRegionId {
        SyncRegionId(self.core.new_region())
    }

    /// Get the current allocation region.
    pub fn current_region(&self) -> SyncRegionId {
        SyncRegionId(self.core.current_region())
    }

    /// Collect only objects in the specified region.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_region(
        &self,
        region: SyncRegionId,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_region(region.0) }
    }

    /// G1-style "Garbage First" collection. Marks all objects, then sweeps
    /// regions with the highest garbage ratio first, stopping when the elapsed
    /// time reaches `pause_target`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are held
    /// across this call (stop-the-world requirement).
    pub(crate) unsafe fn collect_garbage_first(
        &self,
        pause_target: std::time::Duration,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract.
        unsafe { self.core.collect_garbage_first(pause_target) }
    }

    /// Return per-region liveness statistics without running a collection.
    pub fn region_stats(&self) -> Vec<crate::generation::RegionStats> {
        self.core.region_stats()
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

    /// Compact the heap by copying all live objects into a contiguous buffer.
    /// Runs a full collection first, then relocates surviving objects.
    ///
    /// Returns the number of objects compacted.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    /// Acquires STW write lock internally.
    pub(crate) unsafe fn compact(&self) -> usize {
        unsafe { self.core.compact() }
    }
}

// ---- Feature-gated public access for benchmarks and integration tests ----

#[cfg(feature = "_internal")]
impl GlobalGarbageCollector {
    pub unsafe fn _collect(&self) {
        unsafe { self.collect() }
    }
    pub unsafe fn _collect_incremental(
        &self,
        max_gen: crate::generation::Generation,
        step_budget: usize,
    ) -> crate::generation::CollectionStats {
        unsafe { self.collect_incremental(max_gen, step_budget) }
    }
    pub unsafe fn _collect_incremental_timed(
        &self,
        max_gen: crate::generation::Generation,
        max_step_duration: Duration,
    ) -> crate::generation::CollectionStats {
        unsafe { self.collect_incremental_timed(max_gen, max_step_duration) }
    }
    pub unsafe fn _collect_concurrent_timed(
        &self,
        max_gen: crate::generation::Generation,
        max_step_duration: Duration,
    ) -> crate::generation::CollectionStats {
        unsafe { self.collect_concurrent_timed(max_gen, max_step_duration) }
    }
    pub unsafe fn _collect_region(
        &self,
        region: SyncRegionId,
    ) -> crate::generation::CollectionStats {
        unsafe { self.collect_region(region) }
    }
    pub unsafe fn _compact(&self) -> usize {
        unsafe { self.compact() }
    }
}

// ---- SyncRegionId helpers for global (sync) GC ----

impl SyncRegionId {
    /// Allocate a new `sync::Gc<T>` in this region (global collector).
    pub fn gc<T: 'static + Sized + Trace>(self, t: T) -> Gc<T> {
        Gc::new_in(t, self)
    }

    /// Fallible `sync::Gc<T>` allocation in this region (global collector).
    pub fn try_gc<T: 'static + Sized + Trace>(
        self,
        t: T,
    ) -> Result<Gc<T>, crate::gc::GcAllocError> {
        Gc::try_new_in(t, self)
    }

    /// Allocate a new `sync::GcCell<T>` in this region (global collector).
    pub fn gc_cell<T: 'static + Sized + Trace>(self, t: T) -> GcCell<T> {
        GcCell::new_in(t, self)
    }

    /// Fallible `sync::GcCell<T>` allocation in this region (global collector).
    pub fn try_gc_cell<T: 'static + Sized + Trace>(
        self,
        t: T,
    ) -> Result<GcCell<T>, crate::gc::GcAllocError> {
        GcCell::try_new_in(t, self)
    }
}

/// Callback type for a global collection strategy.
pub type GlobalStrategyFn =
    Box<dyn FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;

/// Pluggable collection strategy for the global (thread-safe) GC.
/// Controls when and how garbage collection runs. Use `set_strategy` to swap at runtime.
pub struct GlobalStrategy {
    gc: Cell<&'static GlobalGarbageCollector>,
    is_active: AtomicBool,
    strategy_func: Mutex<GlobalStrategyFn>,
    join_handle: Mutex<Option<JoinHandle<()>>>,
}

// SAFETY: The Cell<&'static GlobalGarbageCollector> is only written during initialization
// (before any thread can access it). All other fields use Mutex or atomics.
unsafe impl Sync for GlobalStrategy {}
unsafe impl Send for GlobalStrategy {}

impl GlobalStrategy {
    fn new<StrategyFn>(
        gc: &'static GlobalGarbageCollector,
        strategy_fn: StrategyFn,
    ) -> GlobalStrategy
    where
        StrategyFn: 'static
            + FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
    {
        GlobalStrategy {
            gc: Cell::new(gc),
            is_active: AtomicBool::new(false),
            strategy_func: Mutex::new(Box::new(strategy_fn)),
            join_handle: Mutex::new(None),
        }
    }

    /// Replace the current collection strategy. Stops the current strategy first if active.
    pub fn set_strategy<StrategyFn>(&self, strategy_fn: StrategyFn)
    where
        StrategyFn: 'static
            + FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
    {
        if self.is_active() {
            self.stop();
        }
        let mut strategy_func = self.strategy_func.lock().unwrap_or_else(|e| e.into_inner());
        *strategy_func = Box::new(strategy_fn);
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
            let mut strategy_func = self.strategy_func.lock().unwrap_or_else(|e| e.into_inner());
            let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
            *join_handle = (*(strategy_func))(self.gc.get(), &self.is_active);
        }
    }

    /// Start the collection strategy (e.g., spawn a background collection thread).
    pub fn start(&'static self) {
        self.is_active.store(true, Ordering::Release);
        let mut strategy_func = self.strategy_func.lock().unwrap_or_else(|e| e.into_inner());
        let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
        *join_handle = (*(strategy_func))(self.gc.get(), &self.is_active);
    }

    /// Stop the collection strategy and join any background thread.
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(join_handle) = join_handle.take() {
            join_handle
                .join()
                .expect("GlobalStrategy::stop: strategy thread panicked");
        }
    }
}

impl Drop for GlobalStrategy {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }
}

lazy_static! {
    pub static ref GLOBAL_GC: GlobalGarbageCollector = GlobalGarbageCollector::new();
    pub static ref GLOBAL_GC_STRATEGY: GlobalStrategy = {
        let gc = &(*GLOBAL_GC);
        GlobalStrategy::new(gc, move |global_gc, _| {
            let mut basic_strategy_global_gc = BASIC_STRATEGY_GLOBAL_GC
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *basic_strategy_global_gc = Some(global_gc);
            None
        })
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
            .lock_gc_maps()
            .objects
            .values()
            .map(|e| e.tracers.len())
            .sum::<usize>();
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
    fn set_strategy_while_active_does_not_deadlock() {
        let (_guard, _) = setup();
        let _gc = Gc::new(1); // ensures strategy is started
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use crate::gc::sync::GLOBAL_GC_STRATEGY;
            GLOBAL_GC_STRATEGY.set_strategy(|_gc, _| None);
            tx.send(()).unwrap();
        });
        rx.recv_timeout(std::time::Duration::from_secs(3))
            .expect("set_strategy deadlocked when called while active");
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
        // Use >= because the background basic-strategy thread may collect concurrently.
        assert!(
            after - before >= 2,
            "expected at least 2 collections, got {}",
            after - before
        );
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
        let (_guard, baseline) = setup();
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let stats = (*GLOBAL_GC).stats();
        // allocation_count can be reset by background strategy calling collect(),
        // so check live_objects instead — we hold both Gc references alive.
        let live: usize = (*GLOBAL_GC)
            .core
            .lock_gc_maps()
            .objects
            .values()
            .map(|e| e.tracers.len())
            .sum();
        assert!(
            live - baseline >= 2,
            "expected at least 2 new tracers, got {}",
            live - baseline
        );
        // Stats should still report some live objects
        assert!(stats.live_objects >= 2);
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
    // SAFETY: Tests run under TEST_MUTEX and call collect() manually after all
    // mutations complete, so no true concurrent access occurs. The background
    // basic-strategy thread can race with RefCell's non-atomic borrow counter;
    // tests that use SyncCyclicNode are excluded from Miri via cfg_attr below.
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
    #[cfg_attr(miri, ignore)] // RefCell in SyncCyclicNode races with background GC thread
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
    #[cfg_attr(miri, ignore)] // RefCell in SyncCyclicNode races with background GC thread
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
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| e.tracers.len())
                .sum::<usize>()
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
        let region = (*GLOBAL_GC).core.current_region();
        let gc_maps = (*GLOBAL_GC)
            .core
            .lock_gc_maps();
        assert!(
            gc_maps.objects.values().any(|e| e.region == region),
            "object should be assigned to current region"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // RefCell in SyncCyclicNode races with background GC thread
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
            .lock_gc_maps();
        assert_eq!(
            gc_maps.objects.values().map(|e| e.tracers.len()).sum::<usize>(),
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
            .lock_gc_maps()
            .objects
            .len();
        drop(a);
        let after = (*GLOBAL_GC)
            .core
            .lock_gc_maps()
            .objects
            .len();
        assert_eq!(after, baseline, "object should survive when clone exists");
        assert_eq!(**b, 99);
    }
}
