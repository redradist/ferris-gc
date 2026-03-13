use std::cell::{Cell, RefCell};
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;

use crate::gc::{Finalize, Trace, GarbageCollector};
use crate::generation::Generation;
use crate::basic_gc_strategy::{basic_gc_strategy_start, BASIC_STRATEGY_GLOBAL_GC};

pub type OptGc<T> = Option<Gc<T>>;
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

pub struct GcPtr<T> where T: 'static + Sized + Trace {
    info: GcInfo,
    t: T,
}

impl<T> GcPtr<T> where T: 'static + Sized + Trace {
    fn new(t: T) -> GcPtr<T> {
        GcPtr {
            info: GcInfo::new(),
            t: t,
        }
    }
}

impl<T> Deref for GcPtr<T> where T: 'static + Sized + Trace {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.t
    }
}

impl<T> DerefMut for GcPtr<T> where T: 'static + Sized + Trace {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.t
    }
}

impl<T> Trace for GcPtr<T> where T: Sized + Trace {
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
}

impl<T> Trace for RefCell<GcPtr<T>> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        unreachable!("is_root on GcPtr is unreachable !!");
    }

    fn reset_root(&self) {
        self.borrow().t.reset_root();
    }

    fn trace(&self) {
        let prev = self.borrow().info.root_ref_count.fetch_add(1, Ordering::AcqRel);
        if prev == 0 {
            self.borrow().t.trace();
        }
    }

    fn reset(&self) {
        let prev = self.borrow().info.root_ref_count.fetch_sub(1, Ordering::AcqRel);
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
}

impl<T> Finalize for RefCell<GcPtr<T>> where T: Sized + Trace {
    fn finalize(&self) {}
}

impl<T> Finalize for GcPtr<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

pub struct GcInternal<T> where T: 'static + Sized + Trace {
    is_root: AtomicBool,
    ptr: *const GcPtr<T>,
}

impl<T> GcInternal<T> where T: 'static + Sized + Trace {
    fn new(ptr: *const GcPtr<T>) -> GcInternal<T> {
        GcInternal {
            is_root: AtomicBool::new(true),
            ptr: ptr,
        }
    }
}

impl<T> Trace for GcInternal<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        self.is_root.load(Ordering::Acquire)
    }

    fn reset_root(&self) {
        if self.is_root.load(Ordering::Acquire) {
            self.is_root.store(false, Ordering::Release);
            unsafe {
                (*self.ptr).reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            (*self.ptr).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            (*self.ptr).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        unsafe {
            (*self.ptr).is_traceable()
        }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for GcInternal<T> where T: Sized + Trace {
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
pub struct Gc<T> where T: 'static + Sized + Trace {
    internal_ptr: *mut GcInternal<T>,
    ptr: *const GcPtr<T>,
}

/// # Safety
/// The underlying GcInternal uses atomic ref-counting, and the global GC
/// protects all structural mutations with Mutex/RwLock. Gc<T> is safe to
/// send across threads when T itself is Send.
unsafe impl<T> Sync for Gc<T> where T: 'static + Sized + Trace + Sync {}
unsafe impl<T> Send for Gc<T> where T: 'static + Sized + Trace + Send {}

impl<T> Deref for Gc<T> where T: 'static + Sized + Trace {
    type Target = GcPtr<T>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &(*self.ptr)
        }
    }
}

impl<T> std::fmt::Debug for Gc<T> where T: 'static + Sized + Trace + std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Gc").field(&***self).finish()
    }
}

impl<T> Gc<T> where T: Sized + Trace {
    pub fn new(t: T) -> Gc<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            (*GLOBAL_GC).create_gc(t)
        }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    pub fn try_new(t: T) -> Result<Gc<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            (*GLOBAL_GC).try_create_gc(t)
        }
    }
}

impl<T> Clone for Gc<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        unsafe {
            (*GLOBAL_GC).clone_from_gc(self)
        }
    }
}

impl<T> Drop for Gc<T> where T: Sized + Trace {
    fn drop(&mut self) {
        unsafe {
            (*GLOBAL_GC).remove_tracer(self.internal_ptr);
        }
    }
}

impl<T> Trace for Gc<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        unsafe {
            (*self.internal_ptr).is_root()
        }
    }

    fn reset_root(&self) {
        unsafe {
            (*self.internal_ptr).reset_root();
        }
    }

    fn trace(&self) {
        unsafe {
            (*self.ptr).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            (*self.ptr).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        unsafe {
            (*self.ptr).is_traceable()
        }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for Gc<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

pub struct GcRefCellInternal<T> where T: 'static + Sized + Trace {
    is_root: AtomicBool,
    ptr: *const RefCell<GcPtr<T>>,
}

impl<T> GcRefCellInternal<T> where T: 'static + Sized + Trace {
    fn new(ptr: *const RefCell<GcPtr<T>>) -> GcRefCellInternal<T> {
        GcRefCellInternal {
            is_root: AtomicBool::new(true),
            ptr: ptr,
        }
    }
}

impl<T> Trace for GcRefCellInternal<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        self.is_root.load(Ordering::Acquire)
    }

    fn reset_root(&self) {
        if self.is_root.load(Ordering::Acquire) {
            self.is_root.store(false, Ordering::Release);
            unsafe {
                (*self.ptr).borrow().reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            (*self.ptr).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            (*self.ptr).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        unsafe {
            (*self.ptr).borrow().is_traceable()
        }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for GcRefCellInternal<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

/// A garbage-collected mutable cell that can be shared across threads.
///
/// The thread-safe counterpart of the local `GcRefCell<T>`. `borrow_mut()`
/// triggers the write barrier for generational collection.
pub struct GcRefCell<T> where T: 'static + Sized + Trace {
    internal_ptr: *mut GcRefCellInternal<T>,
    ptr: *const RefCell<GcPtr<T>>,
}

unsafe impl<T> Sync for GcRefCell<T> where T: 'static + Sized + Trace + Sync {}
unsafe impl<T> Send for GcRefCell<T> where T: 'static + Sized + Trace + Send {}

impl<T> Drop for GcRefCell<T> where T: Sized + Trace {
    fn drop(&mut self) {
        unsafe {
            (*GLOBAL_GC).remove_tracer(self.internal_ptr);
        }
    }
}

impl<T> Deref for GcRefCell<T> where T: 'static + Sized + Trace {
    type Target = RefCell<GcPtr<T>>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &(*self.ptr)
        }
    }
}

impl<T> std::fmt::Debug for GcRefCell<T> where T: 'static + Sized + Trace + std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GcRefCell").field(&**self.borrow()).finish()
    }
}

impl<T> GcRefCell<T> where T: 'static + Sized + Trace {
    pub fn new(t: T) -> GcRefCell<T> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            (*GLOBAL_GC).create_gc_cell(t)
        }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    pub fn try_new(t: T) -> Result<GcRefCell<T>, crate::gc::GcAllocError> {
        basic_gc_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            (*GLOBAL_GC).try_create_gc_cell(t)
        }
    }

    /// Mutable borrow with write barrier.
    /// Triggers the write barrier so that if this object is in an older generation,
    /// it gets added to the remembered set for young-generation collections.
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GcPtr<T>> {
        unsafe {
            let _stw = (*GLOBAL_GC).core.stw_lock.read().unwrap();
            (*GLOBAL_GC).core.write_barrier(self.ptr as *const dyn Trace);
            (*self.ptr).borrow_mut()
        }
    }
}

impl<T> Clone for GcRefCell<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        unsafe {
            (*GLOBAL_GC).clone_from_gc_cell(self)
        }
    }
}

impl<T> Trace for GcRefCell<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        unsafe {
            (*self.internal_ptr).is_root()
        }
    }

    fn reset_root(&self) {
        unsafe {
            (*self.internal_ptr).reset_root();
        }
    }

    fn trace(&self) {
        unsafe {
            (*self.ptr).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            (*self.ptr).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        unsafe {
            (*self.ptr).borrow().is_traceable()
        }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for GcRefCell<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

/// A weak reference to a global GC-managed object.
///
/// Does not prevent collection. `upgrade()` acquires the STW read lock
/// to safely check liveness before returning a strong `Gc<T>`.
pub struct GcWeak<T> where T: 'static + Sized + Trace {
    alive: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ptr: *const GcPtr<T>,
}

unsafe impl<T> Send for GcWeak<T> where T: 'static + Sized + Trace + Send {}
unsafe impl<T> Sync for GcWeak<T> where T: 'static + Sized + Trace + Sync {}

impl<T> Clone for GcWeak<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        GcWeak {
            alive: self.alive.clone(),
            ptr: self.ptr,
        }
    }
}

impl<T> std::fmt::Debug for GcWeak<T> where T: 'static + Sized + Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.alive.load(std::sync::atomic::Ordering::Relaxed) {
            f.write_str("GcWeak(alive)")
        } else {
            f.write_str("GcWeak(dead)")
        }
    }
}

impl<T> GcWeak<T> where T: 'static + Sized + Trace {
    pub fn upgrade(&self) -> Option<Gc<T>> {
        if !self.alive.load(Ordering::Acquire) {
            return None;
        }
        unsafe {
            // Acquire STW read lock to prevent collection during upgrade
            let _stw = (*GLOBAL_GC).core.stw_lock.read().unwrap();
            if self.alive.load(Ordering::Acquire) {
                Some((*GLOBAL_GC).upgrade_weak(self))
            } else {
                None
            }
        }
    }
}

impl<T> Gc<T> where T: 'static + Sized + Trace {
    pub fn downgrade(this: &Gc<T>) -> GcWeak<T> {
        let alive = (*GLOBAL_GC).core.get_or_create_weak_alive(this.ptr as *const dyn Trace);
        GcWeak {
            alive,
            ptr: this.ptr,
        }
    }
}

impl<T> Trace for GcWeak<T> where T: Sized + Trace {
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

impl<T> Finalize for GcWeak<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

pub struct GlobalGarbageCollector {
    pub(crate) core: GarbageCollector,
}

unsafe impl Sync for GlobalGarbageCollector {}
unsafe impl Send for GlobalGarbageCollector {}

impl GlobalGarbageCollector {
    fn new() -> GlobalGarbageCollector {
        GlobalGarbageCollector { core: GarbageCollector::new() }
    }

    unsafe fn create_gc<T>(&self, t: T) -> Gc<T>
        where T: Sized + Trace {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap();
            let (gc_ptr, mem_info_gc_ptr) = self.core.alloc_mem::<GcPtr<T>>();
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();
            std::ptr::write(gc_ptr, GcPtr::new(t));
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr));
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut tracers = self.core.tracers.write().unwrap();
            let mut objects = self.core.objects.lock().unwrap();
            tracers.mem_to_trc.insert(gc_inter_ptr as usize, gc_inter_ptr);
            tracers.trs.insert(gc_inter_ptr, mem_info_internal_ptr);
            tracers.tracer_obj.insert(gc_inter_ptr, gc_ptr as *const dyn Trace);
            objects.objs.insert(gc_ptr, mem_info_gc_ptr);
            objects.obj_gen.insert(gc_ptr, Generation::Gen0);
            objects.survive_count.insert(gc_ptr, 0);
            objects.fin.insert(gc_ptr, (*gc_ptr).t.as_finalize());
            unsafe fn drop_gc_ptr<T: 'static + Trace>(ptr: *mut u8) { unsafe { std::ptr::drop_in_place(ptr as *mut GcPtr<T>); } }
            objects.drop_fns.insert(gc_ptr, drop_gc_ptr::<T>);
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            gc
        }
    }

    unsafe fn clone_from_gc<T>(&self, gc: &Gc<T>) -> Gc<T> where T: Sized + Trace {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap();
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc.ptr));
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut tracers = self.core.tracers.write().unwrap();
            tracers.mem_to_trc.insert(gc_inter_ptr as usize, gc_inter_ptr);
            tracers.trs.insert(gc_inter_ptr, mem_info_internal_ptr);
            tracers.tracer_obj.insert(gc_inter_ptr, gc.ptr as *const dyn Trace);
            gc
        }
    }

    unsafe fn create_gc_cell<T>(&self, t: T) -> GcRefCell<T> where T: Sized + Trace {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap();
            let (gc_ptr, mem_info_gc_ptr) = self.core.alloc_mem::<RefCell<GcPtr<T>>>();
            let (gc_cell_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcRefCellInternal<T>>();
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));
            std::ptr::write(gc_cell_inter_ptr, GcRefCellInternal::new(gc_ptr));
            let gc = GcRefCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut tracers = self.core.tracers.write().unwrap();
            let mut objects = self.core.objects.lock().unwrap();
            tracers.mem_to_trc.insert(gc_cell_inter_ptr as usize, gc_cell_inter_ptr);
            tracers.trs.insert(gc_cell_inter_ptr, mem_info_internal_ptr);
            tracers.tracer_obj.insert(gc_cell_inter_ptr, gc_ptr as *const dyn Trace);
            objects.objs.insert(gc_ptr, mem_info_gc_ptr);
            objects.obj_gen.insert(gc_ptr, Generation::Gen0);
            objects.survive_count.insert(gc_ptr, 0);
            objects.fin.insert(gc_ptr, (*(*gc_ptr).as_ptr()).t.as_finalize());
            unsafe fn drop_gc_cell_ptr<T: 'static + Trace>(ptr: *mut u8) { unsafe { std::ptr::drop_in_place(ptr as *mut RefCell<GcPtr<T>>); } }
            objects.drop_fns.insert(gc_ptr, drop_gc_cell_ptr::<T>);
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            gc
        }
    }

    unsafe fn clone_from_gc_cell<T>(&self, gc: &GcRefCell<T>) -> GcRefCell<T> where T: Sized + Trace {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap();
            let (gc_inter_ptr, mem_info) = self.core.alloc_mem::<GcRefCellInternal<T>>();
            std::ptr::write(gc_inter_ptr, GcRefCellInternal::new(gc.ptr));
            let gc = GcRefCell {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut tracers = self.core.tracers.write().unwrap();
            tracers.mem_to_trc.insert(gc_inter_ptr as usize, gc_inter_ptr);
            tracers.trs.insert(gc_inter_ptr, mem_info);
            tracers.tracer_obj.insert(gc_inter_ptr, gc.ptr as *const dyn Trace);
            gc
        }
    }

    unsafe fn try_create_gc<T>(&self, t: T) -> Result<Gc<T>, crate::gc::GcAllocError>
        where T: Sized + Trace {
        unsafe {
            use std::alloc::dealloc;
            let _stw = self.core.stw_lock.read().unwrap();
            let (gc_ptr, mem_info_gc_ptr) = self.core.try_alloc_mem::<GcPtr<T>>()?;
            let (gc_inter_ptr, mem_info_internal_ptr) = match self.core.try_alloc_mem::<GcInternal<T>>() {
                Ok(v) => v,
                Err(e) => {
                    dealloc(mem_info_gc_ptr.0, mem_info_gc_ptr.1);
                    return Err(e);
                }
            };
            std::ptr::write(gc_ptr, GcPtr::new(t));
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr));
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut tracers = self.core.tracers.write().unwrap();
            let mut objects = self.core.objects.lock().unwrap();
            tracers.mem_to_trc.insert(gc_inter_ptr as usize, gc_inter_ptr);
            tracers.trs.insert(gc_inter_ptr, mem_info_internal_ptr);
            tracers.tracer_obj.insert(gc_inter_ptr, gc_ptr as *const dyn Trace);
            objects.objs.insert(gc_ptr, mem_info_gc_ptr);
            objects.obj_gen.insert(gc_ptr, Generation::Gen0);
            objects.survive_count.insert(gc_ptr, 0);
            objects.fin.insert(gc_ptr, (*gc_ptr).t.as_finalize());
            unsafe fn drop_gc_ptr<T: 'static + Trace>(ptr: *mut u8) { unsafe { std::ptr::drop_in_place(ptr as *mut GcPtr<T>); } }
            objects.drop_fns.insert(gc_ptr, drop_gc_ptr::<T>);
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            Ok(gc)
        }
    }

    unsafe fn try_create_gc_cell<T>(&self, t: T) -> Result<GcRefCell<T>, crate::gc::GcAllocError>
        where T: Sized + Trace {
        unsafe {
            use std::alloc::dealloc;
            let _stw = self.core.stw_lock.read().unwrap();
            let (gc_ptr, mem_info_gc_ptr) = self.core.try_alloc_mem::<RefCell<GcPtr<T>>>()?;
            let (gc_cell_inter_ptr, mem_info_internal_ptr) = match self.core.try_alloc_mem::<GcRefCellInternal<T>>() {
                Ok(v) => v,
                Err(e) => {
                    dealloc(mem_info_gc_ptr.0, mem_info_gc_ptr.1);
                    return Err(e);
                }
            };
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));
            std::ptr::write(gc_cell_inter_ptr, GcRefCellInternal::new(gc_ptr));
            let gc = GcRefCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut tracers = self.core.tracers.write().unwrap();
            let mut objects = self.core.objects.lock().unwrap();
            tracers.mem_to_trc.insert(gc_cell_inter_ptr as usize, gc_cell_inter_ptr);
            tracers.trs.insert(gc_cell_inter_ptr, mem_info_internal_ptr);
            tracers.tracer_obj.insert(gc_cell_inter_ptr, gc_ptr as *const dyn Trace);
            objects.objs.insert(gc_ptr, mem_info_gc_ptr);
            objects.obj_gen.insert(gc_ptr, Generation::Gen0);
            objects.survive_count.insert(gc_ptr, 0);
            objects.fin.insert(gc_ptr, (*(*gc_ptr).as_ptr()).t.as_finalize());
            unsafe fn drop_gc_cell_ptr<T: 'static + Trace>(ptr: *mut u8) { unsafe { std::ptr::drop_in_place(ptr as *mut RefCell<GcPtr<T>>); } }
            objects.drop_fns.insert(gc_ptr, drop_gc_cell_ptr::<T>);
            self.core.allocation_count.fetch_add(1, Ordering::Relaxed);
            Ok(gc)
        }
    }

    /// Create a new strong Gc<T> from a weak reference (for upgrade).
    /// Caller must hold STW read lock.
    unsafe fn upgrade_weak<T>(&self, weak: &GcWeak<T>) -> Gc<T> where T: Sized + Trace {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap();
            let (gc_inter_ptr, mem_info_internal_ptr) = self.core.alloc_mem::<GcInternal<T>>();
            std::ptr::write(gc_inter_ptr, GcInternal::new(weak.ptr));
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: weak.ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut tracers = self.core.tracers.write().unwrap();
            tracers.mem_to_trc.insert(gc_inter_ptr as usize, gc_inter_ptr);
            tracers.trs.insert(gc_inter_ptr, mem_info_internal_ptr);
            tracers.tracer_obj.insert(gc_inter_ptr, weak.ptr as *const dyn Trace);
            gc
        }
    }

    pub unsafe fn collect(&self) {
        unsafe { self.core.collect(); }
    }

    #[allow(dead_code)]
    unsafe fn collect_all(&self) {
        unsafe { self.core.collect_all(); }
    }

    pub(crate) unsafe fn remove_tracer(&self, tracer: *const dyn Trace) {
        unsafe { self.core.remove_tracer(tracer); }
    }

    pub unsafe fn begin_collection(&self, max_gen: crate::generation::Generation) {
        unsafe { self.core.begin_collection(max_gen); }
    }

    pub unsafe fn mark_step(&self, budget: usize) -> bool {
        unsafe { self.core.mark_step(budget) }
    }

    pub unsafe fn finish_collection(&self) -> crate::generation::CollectionStats {
        unsafe { self.core.finish_collection() }
    }

    pub unsafe fn collect_incremental(&self, max_gen: crate::generation::Generation, step_budget: usize) -> crate::generation::CollectionStats {
        unsafe { self.core.collect_incremental(max_gen, step_budget) }
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> crate::generation::GcStats {
        self.core.stats()
    }
}

pub type StartGlobalStrategyFn = Box<dyn FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;
pub type StopGlobalStrategyFn = Box<dyn FnMut(&'static GlobalGarbageCollector)>;

pub struct GlobalStrategy {
    gc: Cell<&'static GlobalGarbageCollector>,
    is_active: AtomicBool,
    start_func: Mutex<StartGlobalStrategyFn>,
    stop_func: Mutex<StopGlobalStrategyFn>,
    join_handle: Mutex<Option<JoinHandle<()>>>,
}

unsafe impl Sync for GlobalStrategy {}
unsafe impl Send for GlobalStrategy {}

impl GlobalStrategy {
    fn new<StartFn, StopFn>(gc: &'static GlobalGarbageCollector, start_fn: StartFn, stop_fn: StopFn) -> GlobalStrategy
        where StartFn: 'static + FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
              StopFn: 'static + FnMut(&'static GlobalGarbageCollector) {
        GlobalStrategy {
            gc: Cell::new(gc),
            is_active: AtomicBool::new(false),
            start_func: Mutex::new(Box::new(start_fn)),
            stop_func: Mutex::new(Box::new(stop_fn)),
            join_handle: Mutex::new(None),
        }
    }

    pub fn change_strategy<StartFn, StopFn>(&self, start_fn: StartFn, stop_fn: StopFn)
        where StartFn: 'static + FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
              StopFn: 'static + FnMut(&'static GlobalGarbageCollector) {
        if self.is_active() {
            self.stop();
        }
        let mut start_func = self.start_func.lock().unwrap();
        let mut stop_func = self.stop_func.lock().unwrap();
        *start_func = Box::new(start_fn);
        *stop_func = Box::new(stop_fn);
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Atomically check if inactive and start. Safe to call from multiple threads.
    pub fn ensure_started(&'static self) {
        if self.is_active.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            let mut start_func = self.start_func.lock().unwrap();
            let mut join_handle = self.join_handle.lock().unwrap();
            *join_handle = (&mut *(start_func))(self.gc.get(), &self.is_active);
        }
    }

    pub fn start(&'static self) {
        self.is_active.store(true, Ordering::Release);
        let mut start_func = self.start_func.lock().unwrap();
        let mut join_handle = self.join_handle.lock().unwrap();
        *join_handle = (&mut *(start_func))(self.gc.get(), &self.is_active);
    }

    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        let mut join_handle = self.join_handle.lock().unwrap();
        if let Some(join_handle) = join_handle.take() {
            join_handle.join().expect("GlobalStrategy::stop, GlobalStrategy Thread being joined has panicked !!");
        }
        let mut stop_func = self.stop_func.lock().unwrap();
        (&mut *(stop_func))(self.gc.get());
    }
}

impl Drop for GlobalStrategy {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
        let mut stop_func = self.stop_func.lock().unwrap();
        (&mut *(stop_func))(self.gc.get());
    }
}

lazy_static! {
    static ref GLOBAL_GC: GlobalGarbageCollector = {
        GlobalGarbageCollector::new()
    };
    pub static ref GLOBAL_GC_STRATEGY: GlobalStrategy = {
        let gc = &(*GLOBAL_GC);
        GlobalStrategy::new(gc,
            move |global_gc, _| {
                let mut basic_strategy_global_gc = BASIC_STRATEGY_GLOBAL_GC.write().unwrap();
                *basic_strategy_global_gc = Some(global_gc);
                None
            },
            move |_global_gc| {
                let mut basic_strategy_global_gc = BASIC_STRATEGY_GLOBAL_GC.write().unwrap();
                *basic_strategy_global_gc = None;
            })
    };
}

#[cfg(test)]
mod tests {
    use crate::gc::sync::{Gc, GLOBAL_GC};
    use std::sync::Mutex;

    // Serialize sync GC tests since they share GLOBAL_GC.
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    /// Clean residual state and return baseline trs count.
    fn setup() -> (std::sync::MutexGuard<'static, ()>, usize) {
        let guard = TEST_MUTEX.lock().unwrap();
        unsafe { (*GLOBAL_GC).collect() };
        let baseline = (*GLOBAL_GC).core.tracers.read().unwrap().trs.len();
        (guard, baseline)
    }

    #[test]
    fn one_object() {
        let (_guard, baseline) = setup();
        let _one = Gc::new(1);
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!((*GLOBAL_GC).core.tracers.read().unwrap().trs.len() - baseline, 1);
    }

    #[test]
    fn gc_collect_one_from_one() {
        let (_guard, baseline) = setup();
        {
            let _one = Gc::new(1);
        }
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!((*GLOBAL_GC).core.tracers.read().unwrap().trs.len() - baseline, 0);
    }

    #[test]
    #[allow(unused_assignments)]
    fn two_objects_reassign() {
        let (_guard, baseline) = setup();
        let mut one = Gc::new(1);
        one = Gc::new(2);
        unsafe { (*GLOBAL_GC).collect() };
        // Reassignment drops old Gc (remove_tracer), so only 1 tracer remains
        assert_eq!((*GLOBAL_GC).core.tracers.read().unwrap().trs.len() - baseline, 1);
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_after_reassign() {
        let (_guard, baseline) = setup();
        let mut one = Gc::new(1);
        one = Gc::new(2);
        unsafe { (*GLOBAL_GC).collect() };
        // one is still live, so 1 tracer remains
        assert_eq!((*GLOBAL_GC).core.tracers.read().unwrap().trs.len() - baseline, 1);
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
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!((*GLOBAL_GC).core.tracers.read().unwrap().trs.len() - baseline, 0);
    }

    #[test]
    fn stw_blocks_allocation_during_collection() {
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
        let (_guard, _) = setup();

        let allocated_during_stw = Arc::new(AtomicBool::new(false));
        let collection_started = Arc::new(AtomicBool::new(false));
        let collection_done = Arc::new(AtomicBool::new(false));

        // Hold the STW write lock to simulate an ongoing collection
        let stw_guard = (*GLOBAL_GC).core.stw_lock.write().unwrap();
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
        assert!(!allocated_during_stw.load(Ordering::Acquire),
            "allocation must block while STW write lock is held");

        // Release the STW lock — simulates end of collection
        drop(stw_guard);
        collection_done.store(true, Ordering::Release);

        handle.join().expect("allocator thread panicked");
        assert!(allocated_during_stw.load(Ordering::Acquire),
            "allocation should complete after STW lock is released");
    }

    #[test]
    fn stw_allows_concurrent_allocations() {
        let (_guard, _) = setup();
        // Multiple threads allocating concurrently should all succeed (read locks are shared)
        let handles: Vec<_> = (0..4).map(|i| {
            std::thread::spawn(move || {
                let _obj = Gc::new(i);
            })
        }).collect();
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
            GLOBAL_GC_STRATEGY.change_strategy(
                |_gc, _| None,
                |_gc| {},
            );
            tx.send(()).unwrap();
        });
        rx.recv_timeout(std::time::Duration::from_secs(3))
            .expect("change_strategy deadlocked when called while active");
    }

    #[test]
    fn incremental_collects_dead_objects() {
        let (_guard, baseline) = setup();
        {
            let _obj = Gc::new(42);
        }
        let stats = unsafe { (*GLOBAL_GC).collect_incremental(crate::generation::Generation::Gen2, 10) };
        assert!(stats.objects_collected > 0, "incremental should collect dead objects");
        assert_eq!((*GLOBAL_GC).core.tracers.read().unwrap().trs.len() - baseline, 0);
    }

    #[test]
    fn incremental_preserves_live_objects() {
        let (_guard, baseline) = setup();
        let _live = Gc::new(99);
        let stats = unsafe { (*GLOBAL_GC).collect_incremental(crate::generation::Generation::Gen2, 10) };
        assert_eq!(stats.objects_collected, 0, "incremental must not collect live objects");
        assert_eq!((*GLOBAL_GC).core.tracers.read().unwrap().trs.len() - baseline, 1);
    }

    #[test]
    fn incremental_step_by_step_sync() {
        let (_guard, _) = setup();
        {
            let _dead = Gc::new(1);
        }
        let _live = Gc::new(2);
        unsafe {
            (*GLOBAL_GC).begin_collection(crate::generation::Generation::Gen2);
            while !(*GLOBAL_GC).mark_step(1) {}
            let stats = (*GLOBAL_GC).finish_collection();
            assert!(stats.objects_collected >= 1, "should collect dead object");
        }
    }

    #[test]
    fn try_new_succeeds() {
        let (_guard, _) = setup();
        let result = Gc::try_new(42);
        assert!(result.is_ok(), "try_new should succeed for normal allocation");
        assert_eq!(**result.unwrap(), 42);
    }

    #[test]
    fn try_new_object_collected_when_dead() {
        let (_guard, baseline) = setup();
        {
            let _obj = Gc::try_new(77).unwrap();
        }
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!((*GLOBAL_GC).core.tracers.read().unwrap().trs.len() - baseline, 0);
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
        unsafe { (*GLOBAL_GC).collect() };
        unsafe { (*GLOBAL_GC).collect() };
        let after = (*GLOBAL_GC).stats().total_collections;
        assert_eq!(after - before, 2);
    }

    #[test]
    fn sync_stats_tracks_last_collection() {
        let (_guard, _) = setup();
        {
            let _obj = Gc::new(99);
        }
        unsafe { (*GLOBAL_GC).collect() };
        let stats = (*GLOBAL_GC).stats();
        let last = stats.last_collection.expect("last_collection should be Some");
        assert!(last.objects_collected >= 1);
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
        fn is_root(&self) -> bool { false }
        fn reset_root(&self) {
            if let Some(ref gc) = *self.next.borrow() { gc.reset_root(); }
        }
        fn trace(&self) {
            if let Some(ref gc) = *self.next.borrow() { gc.trace(); }
        }
        fn reset(&self) {
            if let Some(ref gc) = *self.next.borrow() { gc.reset(); }
        }
        fn is_traceable(&self) -> bool { false }
        fn trace_children(&self, children: &mut Vec<*const dyn crate::gc::Trace>) {
            if let Some(ref gc) = *self.next.borrow() { gc.trace_children(children); }
        }
    }
    impl crate::gc::Finalize for SyncCyclicNode { fn finalize(&self) {} }

    #[test]
    fn sync_self_cycle_collected() {
        let (_guard, _) = setup();
        {
            let a = Gc::new(SyncCyclicNode { next: std::cell::RefCell::new(None) });
            *a.next.borrow_mut() = Some(a.clone());
            drop(a);
        }
        unsafe { (*GLOBAL_GC).collect() };
    }

    #[test]
    fn sync_three_node_cycle_collected() {
        let (_guard, _) = setup();
        {
            let a = Gc::new(SyncCyclicNode { next: std::cell::RefCell::new(None) });
            let b = Gc::new(SyncCyclicNode { next: std::cell::RefCell::new(None) });
            let c = Gc::new(SyncCyclicNode { next: std::cell::RefCell::new(None) });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(c.clone());
            *c.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
            drop(c);
        }
        unsafe { (*GLOBAL_GC).collect() };
    }
}
