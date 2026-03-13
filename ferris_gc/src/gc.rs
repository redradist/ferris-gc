use std::alloc::{alloc, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crate::generation::{Generation, CollectionStats, GcStats, MarkColor, CollectionPhase};
use std::thread::JoinHandle;

use crate::basic_gc_strategy::{basic_gc_strategy_start, BASIC_STRATEGY_LOCAL_GCS};

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
    fn is_root(&self) -> bool;
    fn reset_root(&self);
    fn trace(&self);
    fn reset(&self);
    fn is_traceable(&self) -> bool;
    /// Non-recursive child discovery for incremental tri-color marking.
    /// Pushes immediate GC-managed children (object pointers) onto `children`.
    fn trace_children(&self, _children: &mut Vec<*const dyn Trace>) {}
}

/// Destructor callback invoked by the collector before an object is deallocated.
///
/// Called during the sweep phase. The default `#[derive(Finalize)]` generates
/// an empty body — override only when you need explicit cleanup of non-GC
/// resources (file handles, GPU buffers, etc.).
pub trait Finalize {
    fn finalize(&self);
    fn as_finalize(&self) -> &dyn Finalize
        where Self: Sized {
        self
    }
}

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
pub struct Gc<T> where T: 'static + Sized + Trace {
    internal_ptr: *mut GcInternal<T>,
    ptr: *const GcPtr<T>,
}

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
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().create_gc(t)
        })
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    pub fn try_new(t: T) -> Result<Gc<T>, GcAllocError> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().try_create_gc(t)
        })
    }
}

impl<T> Clone for Gc<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().clone_from_gc(self)
        })
    }
}

impl<T> Drop for Gc<T> where T: Sized + Trace {
    fn drop(&mut self) {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().remove_tracer(self.internal_ptr);
        });
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

/// A garbage-collected mutable cell for thread-local use.
///
/// Provides interior mutability (`borrow()` / `borrow_mut()`) for GC-managed
/// objects. `borrow_mut()` triggers the **write barrier**, which records
/// cross-generation references so that young-generation collections can
/// discover pointers from old objects to young objects.
///
/// Thread-local only — for a cross-thread variant, use [`sync::GcRefCell<T>`].
pub struct GcRefCell<T> where T: 'static + Sized + Trace {
    internal_ptr: *mut GcRefCellInternal<T>,
    ptr: *const RefCell<GcPtr<T>>,
}

impl<T> Drop for GcRefCell<T> where T: Sized + Trace {
    fn drop(&mut self) {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().remove_tracer(self.internal_ptr);
        });
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
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().create_gc_cell(t)
        })
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    pub fn try_new(t: T) -> Result<GcRefCell<T>, GcAllocError> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().try_create_gc_cell(t)
        })
    }

    /// Mutable borrow with write barrier.
    /// Triggers the write barrier so that if this object is in an older generation,
    /// it gets added to the remembered set for young-generation collections.
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GcPtr<T>> {
        LOCAL_GC.with(|gc| {
            gc.borrow().core.write_barrier(self.ptr as *const dyn Trace);
        });
        unsafe { (*self.ptr).borrow_mut() }
    }
}

impl<T> Clone for GcRefCell<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().clone_from_gc_cell(self)
        })
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

/// A weak reference to a GC-managed object.
///
/// Does not prevent collection. Use [`upgrade()`](GcWeak::upgrade) to obtain
/// a strong `Gc<T>`. Returns `None` if the object has already been collected.
///
/// `GcWeak<T>` is `Send + Sync` when `T` is, so it can be shared across
/// threads to observe liveness of a thread-local object.
pub struct GcWeak<T> where T: 'static + Sized + Trace {
    alive: Arc<AtomicBool>,
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
        if self.alive.load(Ordering::Relaxed) {
            f.write_str("GcWeak(alive)")
        } else {
            f.write_str("GcWeak(dead)")
        }
    }
}

impl<T> GcWeak<T> where T: 'static + Sized + Trace {
    /// Try to upgrade this weak reference to a strong `Gc<T>`.
    /// Returns `None` if the object has been collected.
    pub fn upgrade(&self) -> Option<Gc<T>> {
        if !self.alive.load(Ordering::Acquire) {
            return None;
        }
        LOCAL_GC.with(|gc| {
            // Acquire STW read lock via raw pointer to avoid RefCell borrow conflict
            let gc_ptr = gc.as_ptr();
            let _stw = unsafe { (*gc_ptr).core.stw_lock.read().unwrap() };
            // Re-check alive under STW protection
            if self.alive.load(Ordering::Acquire) {
                Some(unsafe { gc.borrow_mut().upgrade_weak(self) })
            } else {
                None
            }
        })
    }
}

impl<T> Gc<T> where T: 'static + Sized + Trace {
    /// Create a weak reference to this GC-managed object.
    pub fn downgrade(this: &Gc<T>) -> GcWeak<T> {
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let alive = gc_ref.core.get_or_create_weak_alive(this.ptr as *const dyn Trace);
            GcWeak {
                alive,
                ptr: this.ptr,
            }
        })
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

/// Tracer-related maps, protected by a single RwLock.
pub(crate) struct TracerMaps {
    pub(crate) mem_to_trc: HashMap<usize, *const dyn Trace>,
    pub(crate) trs: HashMap<*const dyn Trace, (GcObjMem, Layout)>,
    pub(crate) tracer_obj: HashMap<*const dyn Trace, *const dyn Trace>,
}

/// Object-related maps, protected by a single Mutex.
pub(crate) struct ObjectMaps {
    pub(crate) objs: HashMap<*const dyn Trace, (GcObjMem, Layout)>,
    pub(crate) obj_gen: HashMap<*const dyn Trace, Generation>,
    pub(crate) survive_count: HashMap<*const dyn Trace, u32>,
    pub(crate) fin: HashMap<*const dyn Trace, *const dyn Finalize>,
    pub(crate) drop_fns: HashMap<*const dyn Trace, DropFn>,
    pub(crate) weak_alive: HashMap<*const dyn Trace, Arc<AtomicBool>>,
}

/// State for incremental tri-color marking.
pub(crate) struct IncrementalState {
    pub(crate) phase: CollectionPhase,
    pub(crate) max_gen: Generation,
    pub(crate) colors: HashMap<*const dyn Trace, MarkColor>,
    pub(crate) gray_stack: Vec<*const dyn Trace>,
}

impl IncrementalState {
    fn new() -> IncrementalState {
        IncrementalState {
            phase: CollectionPhase::Idle,
            max_gen: Generation::Gen0,
            colors: HashMap::new(),
            gray_stack: Vec::new(),
        }
    }
}

/// Shared GC bookkeeping used by both LocalGarbageCollector and GlobalGarbageCollector.
/// Maps are consolidated under fewer locks to reduce contention:
///   - 3 locks per new allocation (stw + tracers + objects), down from 9
///   - 2 locks per clone (stw + tracers), down from 4
pub(crate) struct GarbageCollector {
    pub(crate) tracers: RwLock<TracerMaps>,
    pub(crate) objects: Mutex<ObjectMaps>,
    pub(crate) allocation_count: AtomicUsize,
    pub(crate) remembered_set: Mutex<HashSet<*const dyn Trace>>,
    pub(crate) stw_lock: RwLock<()>,
    pub(crate) incremental: Mutex<IncrementalState>,
    pub(crate) total_collections: AtomicUsize,
    pub(crate) last_collection: Mutex<Option<CollectionStats>>,
}

unsafe impl Sync for GarbageCollector {}
unsafe impl Send for GarbageCollector {}

impl GarbageCollector {
    pub(crate) fn new() -> GarbageCollector {
        GarbageCollector {
            tracers: RwLock::new(TracerMaps {
                mem_to_trc: HashMap::new(),
                trs: HashMap::new(),
                tracer_obj: HashMap::new(),
            }),
            objects: Mutex::new(ObjectMaps {
                objs: HashMap::new(),
                obj_gen: HashMap::new(),
                survive_count: HashMap::new(),
                fin: HashMap::new(),
                drop_fns: HashMap::new(),
                weak_alive: HashMap::new(),
            }),
            allocation_count: AtomicUsize::new(0),
            remembered_set: Mutex::new(HashSet::new()),
            stw_lock: RwLock::new(()),
            incremental: Mutex::new(IncrementalState::new()),
            total_collections: AtomicUsize::new(0),
            last_collection: Mutex::new(None),
        }
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> GcStats {
        let tracers = self.tracers.read().unwrap();
        let objects = self.objects.lock().unwrap();
        let mut gen0 = 0usize;
        let mut gen1 = 0usize;
        let mut gen2 = 0usize;
        for g in objects.obj_gen.values() {
            match g {
                Generation::Gen0 => gen0 += 1,
                Generation::Gen1 => gen1 += 1,
                Generation::Gen2 => gen2 += 1,
            }
        }
        let heap_size: usize = objects.objs.values().map(|(_, layout)| layout.size()).sum();
        GcStats {
            heap_size,
            live_objects: objects.objs.len(),
            live_tracers: tracers.trs.len(),
            gen0_objects: gen0,
            gen1_objects: gen1,
            gen2_objects: gen2,
            total_collections: self.total_collections.load(Ordering::Relaxed),
            last_collection: self.last_collection.lock().unwrap().clone(),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
        }
    }

    pub(crate) unsafe fn alloc_mem<T>(&self) -> (*mut T, (GcObjMem, Layout)) where T: Sized {
        unsafe {
            let layout = Layout::new::<T>();
            let mem = alloc(layout);
            if mem.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let type_ptr: *mut T = mem as *mut _;
            (type_ptr, (mem, layout))
        }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` instead of aborting on OOM.
    pub(crate) unsafe fn try_alloc_mem<T>(&self) -> Result<(*mut T, (GcObjMem, Layout)), GcAllocError> where T: Sized {
        unsafe {
            let layout = Layout::new::<T>();
            let mem = alloc(layout);
            if mem.is_null() {
                return Err(GcAllocError);
            }
            let type_ptr: *mut T = mem as *mut _;
            Ok((type_ptr, (mem, layout)))
        }
    }

    /// Register a new object in Gen0.
    #[allow(dead_code)]
    pub(crate) fn register_object(&self, obj_ptr: *const dyn Trace) {
        let mut objects = self.objects.lock().unwrap();
        objects.obj_gen.insert(obj_ptr, Generation::Gen0);
        objects.survive_count.insert(obj_ptr, 0);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Register a tracer -> object mapping.
    #[allow(dead_code)]
    pub(crate) fn register_tracer_obj(&self, tracer_ptr: *const dyn Trace, obj_ptr: *const dyn Trace) {
        self.tracers.write().unwrap().tracer_obj.insert(tracer_ptr, obj_ptr);
    }

    /// Get or create the alive flag for an object (used by weak references).
    pub(crate) fn get_or_create_weak_alive(&self, obj_ptr: *const dyn Trace) -> Arc<AtomicBool> {
        let mut objects = self.objects.lock().unwrap();
        objects.weak_alive.entry(obj_ptr)
            .or_insert_with(|| Arc::new(AtomicBool::new(true)))
            .clone()
    }

    /// Write barrier: if the object is in Gen1+, add it to the remembered set
    /// so that young-generation collections trace through it.
    /// During incremental marking, also re-grays Black objects to maintain
    /// the tri-color invariant (no Black→White edges).
    pub(crate) fn write_barrier(&self, obj_ptr: *const dyn Trace) {
        let objects = self.objects.lock().unwrap();
        if let Some(g) = objects.obj_gen.get(&obj_ptr) {
            if *g > Generation::Gen0 {
                self.remembered_set.lock().unwrap().insert(obj_ptr);
            }
        }
        drop(objects);

        // During incremental marking, re-gray Black objects so their
        // new children are discovered in subsequent mark steps.
        let mut incr = self.incremental.lock().unwrap();
        if incr.phase == CollectionPhase::Marking {
            if let Some(color) = incr.colors.get_mut(&obj_ptr) {
                if *color == MarkColor::Black {
                    *color = MarkColor::Gray;
                    incr.gray_stack.push(obj_ptr);
                }
            }
        }
    }

    pub(crate) unsafe fn remove_tracer(&self, tracer: *const dyn Trace) {
        unsafe {
            let mut tracers = self.tracers.write().unwrap();
            if let Some(tracer) = tracers.mem_to_trc.remove(&(tracer.get_thin_ptr())) {
                tracers.tracer_obj.remove(&tracer);
                if let Some(del) = tracers.trs.remove(&tracer) {
                    dealloc(del.0, del.1);
                }
            }
        }
    }

    /// Full collection (all generations). Backwards-compatible with existing callers.
    pub unsafe fn collect(&self) {
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
                let _stw = self.stw_lock.write().unwrap();
                let mut tracers = self.tracers.write().unwrap();
                let mut objects = self.objects.lock().unwrap();
                let mut remembered_set = self.remembered_set.lock().unwrap();

                // Mark phase: trace from ALL roots (needed for correctness)
                for (gc_info, _) in &tracers.trs {
                    let tracer = &(**gc_info);
                    if tracer.is_root() {
                        tracer.trace();
                    }
                }

                // For partial collections, also trace from remembered set entries.
                // These are old-gen objects that were mutated and may reference young objects.
                if max_gen < Generation::Gen2 {
                    for &obj_ptr in remembered_set.iter() {
                        (&*obj_ptr).trace();
                    }
                }

                // Build set of objects in target generations
                let in_scope_objs: HashSet<*const dyn Trace> = objects.obj_gen.iter()
                    .filter(|(_, g)| **g <= max_gen)
                    .map(|(k, _)| *k)
                    .collect();

                stats.objects_scanned = in_scope_objs.len();

                // Identify in-scope tracers (their object is in target generations)
                let in_scope_tracers: HashSet<*const dyn Trace> = tracers.tracer_obj.iter()
                    .filter(|(_, obj_ptr)| in_scope_objs.contains(obj_ptr))
                    .map(|(k, _)| *k)
                    .collect();

                // Sweep tracers: only collect unreachable in-scope tracers
                let collected_tracers: Vec<_> = tracers.trs.iter()
                    .filter(|(ptr, _)| in_scope_tracers.contains(ptr) && !(&***ptr).is_traceable())
                    .map(|(k, _)| *k)
                    .collect();

                let mut tracer_deallocs = Vec::new();
                for tracer_ptr in &collected_tracers {
                    let del = tracers.trs.remove(tracer_ptr).unwrap();
                    tracers.mem_to_trc.remove(&tracer_ptr.get_thin_ptr());
                    tracers.tracer_obj.remove(tracer_ptr);
                    tracer_deallocs.push(del);
                }
                stats.tracers_collected = collected_tracers.len();

                // Sweep objects: only collect unreachable in-scope objects
                let collected_objects: Vec<_> = objects.objs.iter()
                    .filter(|(ptr, _)| in_scope_objs.contains(ptr) && !(&***ptr).is_traceable())
                    .map(|(k, _)| *k)
                    .collect();
                stats.objects_collected = collected_objects.len();

                // Reset ALL remaining tracers (not just in-scope)
                for (gc_info, _) in &tracers.trs {
                    let tracer = &(**gc_info);
                    tracer.reset();
                }

                // Reset remembered set entries that were traced in partial collections
                if max_gen < Generation::Gen2 {
                    for &obj_ptr in remembered_set.iter() {
                        (&*obj_ptr).reset();
                    }
                }

                // Remove collected objects from all maps
                let mut object_deallocs = Vec::new();
                for col in &collected_objects {
                    let del = objects.objs.remove(col).unwrap();
                    let finalizer = objects.fin.remove(col);
                    let drop_fn = objects.drop_fns.remove(col);
                    objects.obj_gen.remove(col);
                    objects.survive_count.remove(col);
                    // Remove collected objects from remembered set
                    remembered_set.remove(col);
                    // Invalidate weak references to this object
                    if let Some(alive) = objects.weak_alive.remove(col) {
                        alive.store(false, Ordering::Release);
                    }
                    object_deallocs.push((del, finalizer, drop_fn));
                }

                // On full collection, clear the entire remembered set
                if max_gen >= Generation::Gen2 {
                    remembered_set.clear();
                }

                // Promotion: surviving in-scope objects may be promoted
                let surviving_objs: Vec<*const dyn Trace> = in_scope_objs.iter()
                    .filter(|ptr| !collected_objects.contains(ptr))
                    .copied()
                    .collect();

                for obj_ptr in surviving_objs {
                    if let Some(cur_gen) = objects.obj_gen.get(&obj_ptr).copied() {
                        if cur_gen <= max_gen {
                            let count = objects.survive_count.entry(obj_ptr).or_insert(0);
                            *count += 1;
                            let should_promote = *count >= cur_gen.promotion_threshold();
                            if should_promote {
                                *count = 0;
                            }
                            // Release survive_count borrow before mutating obj_gen
                            if should_promote {
                                if let Some(next_gen) = cur_gen.next() {
                                    objects.obj_gen.insert(obj_ptr, next_gen);
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
            for (mem, layout) in tracer_deallocs {
                dealloc(mem, layout);
            }
            for ((mem, layout), finalizer, drop_fn) in object_deallocs {
                if let Some(f) = finalizer {
                    (*f).finalize();
                }
                if let Some(drop_fn) = drop_fn {
                    (drop_fn)(mem);
                }
                dealloc(mem, layout);
            }

            // Record diagnostics
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self.last_collection.lock().unwrap() = Some(stats);

            stats
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn collect_all(&self) {
        unsafe {
            let (tracer_deallocs, object_deallocs) = {
                // STW: block all mutator operations during cleanup
                let _stw = self.stw_lock.write().unwrap();
                let mut tracers = self.tracers.write().unwrap();
                let mut objects = self.objects.lock().unwrap();
                tracers.tracer_obj.clear();
                objects.obj_gen.clear();
                objects.survive_count.clear();
                self.remembered_set.lock().unwrap().clear();
                // Invalidate all weak references
                for (_, alive) in objects.weak_alive.drain() {
                    alive.store(false, Ordering::Release);
                }
                let tracer_deallocs: Vec<_> = tracers.trs.drain().collect::<Vec<_>>();
                for (k, _) in &tracer_deallocs {
                    tracers.mem_to_trc.remove(&k.get_thin_ptr());
                }
                let tracer_deallocs: Vec<_> = tracer_deallocs.into_iter().map(|(_, v)| v).collect();
                let obj_entries: Vec<_> = objects.objs.drain().collect();
                let object_deallocs: Vec<_> = obj_entries.into_iter().map(|(k, v)| {
                    let finalizer = objects.fin.remove(&k);
                    let drop_fn = objects.drop_fns.remove(&k);
                    (v, finalizer, drop_fn)
                }).collect();
                (tracer_deallocs, object_deallocs)
            };
            self.allocation_count.store(0, Ordering::Relaxed);
            for (mem, layout) in tracer_deallocs {
                dealloc(mem, layout);
            }
            for ((mem, layout), finalizer, drop_fn) in object_deallocs {
                if let Some(f) = finalizer {
                    (*f).finalize();
                }
                if let Some(drop_fn) = drop_fn {
                    (drop_fn)(mem);
                }
                dealloc(mem, layout);
            }
        }
    }

    /// Begin an incremental collection cycle.
    /// Short STW: snapshots roots, initializes tri-color marks (all in-scope objects White,
    /// root-reachable objects Gray), and sets phase to Marking.
    pub unsafe fn begin_collection(&self, max_gen: Generation) {
        let _stw = self.stw_lock.write().unwrap();
        let mut incr = self.incremental.lock().unwrap();
        let tracers = self.tracers.read().unwrap();
        let objects = self.objects.lock().unwrap();

        incr.phase = CollectionPhase::Marking;
        incr.max_gen = max_gen;
        incr.colors.clear();
        incr.gray_stack.clear();

        // Initialize all in-scope objects as White
        for (obj_ptr, g) in &objects.obj_gen {
            if *g <= max_gen {
                incr.colors.insert(*obj_ptr, MarkColor::White);
            }
        }

        // Gray objects reachable from root tracers
        for (tracer_ptr, _) in &tracers.trs {
            unsafe {
                let tracer = &(**tracer_ptr);
                if tracer.is_root() {
                    if let Some(&obj_ptr) = tracers.tracer_obj.get(tracer_ptr) {
                        if let Some(color) = incr.colors.get_mut(&obj_ptr) {
                            if *color == MarkColor::White {
                                *color = MarkColor::Gray;
                                incr.gray_stack.push(obj_ptr);
                            }
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
        let _stw = self.stw_lock.write().unwrap();
        let mut incr = self.incremental.lock().unwrap();

        let mut processed = 0;
        let mut children_buf = Vec::new();

        while processed < budget {
            let obj_ptr = match incr.gray_stack.pop() {
                Some(ptr) => ptr,
                None => break,
            };

            // Discover immediate children
            children_buf.clear();
            unsafe { (&*obj_ptr).trace_children(&mut children_buf); }

            // Gray any White children
            for &child_ptr in &children_buf {
                if let Some(color) = incr.colors.get_mut(&child_ptr) {
                    if *color == MarkColor::White {
                        *color = MarkColor::Gray;
                        incr.gray_stack.push(child_ptr);
                    }
                }
                // Children not in the colors map are out-of-scope (different gen)
                // or allocated during marking — they're safe from sweep.
            }

            // This object is now fully scanned
            incr.colors.insert(obj_ptr, MarkColor::Black);
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
                let _stw = self.stw_lock.write().unwrap();
                let mut incr = self.incremental.lock().unwrap();
                let mut tracers = self.tracers.write().unwrap();
                let mut objects = self.objects.lock().unwrap();
                let mut remembered_set = self.remembered_set.lock().unwrap();

                max_gen = incr.max_gen;
                incr.phase = CollectionPhase::Sweeping;

                // Re-gray any remembered-set entries that were already Black.
                // These objects were mutated between mark steps and may have new White children.
                for &obj_ptr in remembered_set.iter() {
                    if let Some(color) = incr.colors.get_mut(&obj_ptr) {
                        if *color == MarkColor::Black {
                            *color = MarkColor::Gray;
                            incr.gray_stack.push(obj_ptr);
                        }
                    }
                }

                // Also gray objects reachable from NEW root tracers (allocated during marking).
                // These tracers weren't present at begin_collection time.
                for (tracer_ptr, _) in &tracers.trs {
                    let tracer = &(**tracer_ptr);
                    if tracer.is_root() {
                        if let Some(&obj_ptr) = tracers.tracer_obj.get(tracer_ptr) {
                            // Objects not in colors map were allocated during marking — skip
                            if let Some(color) = incr.colors.get_mut(&obj_ptr) {
                                if *color == MarkColor::White {
                                    *color = MarkColor::Gray;
                                    incr.gray_stack.push(obj_ptr);
                                }
                            }
                        }
                    }
                }

                // Drain remaining gray (from re-graying + new roots)
                let mut children_buf = Vec::new();
                while let Some(obj_ptr) = incr.gray_stack.pop() {
                    children_buf.clear();
                    (&*obj_ptr).trace_children(&mut children_buf);
                    for &child_ptr in &children_buf {
                        if let Some(color) = incr.colors.get_mut(&child_ptr) {
                            if *color == MarkColor::White {
                                *color = MarkColor::Gray;
                                incr.gray_stack.push(child_ptr);
                            }
                        }
                    }
                    incr.colors.insert(obj_ptr, MarkColor::Black);
                }

                // Sweep: collect White objects
                let mut stats = CollectionStats {
                    generation: max_gen,
                    objects_scanned: incr.colors.len(),
                    objects_collected: 0,
                    objects_promoted: 0,
                    tracers_collected: 0,
                };

                let white_objects: Vec<*const dyn Trace> = incr.colors.iter()
                    .filter(|(_, color)| **color == MarkColor::White)
                    .map(|(ptr, _)| *ptr)
                    .collect();
                stats.objects_collected = white_objects.len();

                let white_set: HashSet<*const dyn Trace> = white_objects.iter().copied().collect();

                // Sweep dead tracers (those pointing to white objects)
                let dead_tracers: Vec<*const dyn Trace> = tracers.tracer_obj.iter()
                    .filter(|(_, obj_ptr)| white_set.contains(obj_ptr))
                    .map(|(k, _)| *k)
                    .collect();
                stats.tracers_collected = dead_tracers.len();

                let mut tracer_deallocs = Vec::new();
                for tracer_ptr in &dead_tracers {
                    if let Some(del) = tracers.trs.remove(tracer_ptr) {
                        tracers.mem_to_trc.remove(&tracer_ptr.get_thin_ptr());
                        tracers.tracer_obj.remove(tracer_ptr);
                        tracer_deallocs.push(del);
                    }
                }

                // Sweep dead objects
                let mut object_deallocs = Vec::new();
                for &obj_ptr in &white_objects {
                    if let Some(del) = objects.objs.remove(&obj_ptr) {
                        let finalizer = objects.fin.remove(&obj_ptr);
                        let drop_fn = objects.drop_fns.remove(&obj_ptr);
                        objects.obj_gen.remove(&obj_ptr);
                        objects.survive_count.remove(&obj_ptr);
                        remembered_set.remove(&obj_ptr);
                        if let Some(alive) = objects.weak_alive.remove(&obj_ptr) {
                            alive.store(false, Ordering::Release);
                        }
                        object_deallocs.push((del, finalizer, drop_fn));
                    }
                }

                // Clear remembered set on full collection
                if max_gen >= Generation::Gen2 {
                    remembered_set.clear();
                }

                // Promote surviving in-scope objects
                let surviving: Vec<*const dyn Trace> = incr.colors.iter()
                    .filter(|(_, color)| **color != MarkColor::White)
                    .map(|(ptr, _)| *ptr)
                    .collect();

                for obj_ptr in surviving {
                    if let Some(cur_gen) = objects.obj_gen.get(&obj_ptr).copied() {
                        if cur_gen <= max_gen {
                            let count = objects.survive_count.entry(obj_ptr).or_insert(0);
                            *count += 1;
                            let should_promote = *count >= cur_gen.promotion_threshold();
                            if should_promote {
                                *count = 0;
                            }
                            if should_promote {
                                if let Some(next_gen) = cur_gen.next() {
                                    objects.obj_gen.insert(obj_ptr, next_gen);
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
            for (mem, layout) in tracer_deallocs {
                dealloc(mem, layout);
            }
            for ((mem, layout), finalizer, drop_fn) in object_deallocs {
                if let Some(f) = finalizer {
                    (*f).finalize();
                }
                if let Some(drop_fn) = drop_fn {
                    (drop_fn)(mem);
                }
                dealloc(mem, layout);
            }

            // Record diagnostics
            self.total_collections.fetch_add(1, Ordering::Relaxed);
            *self.last_collection.lock().unwrap() = Some(stats);

            stats
        }
    }

    /// Convenience: run a complete incremental collection.
    /// Breaks the mark phase into steps of `step_budget` objects each,
    /// releasing the STW lock between steps to reduce pause times.
    pub unsafe fn collect_incremental(&self, max_gen: Generation, step_budget: usize) -> CollectionStats {
        unsafe {
            self.begin_collection(max_gen);
            loop {
                let done = self.mark_step(step_budget);
                if done { break; }
            }
            self.finish_collection()
        }
    }
}

pub struct LocalGarbageCollector {
    pub(crate) core: GarbageCollector,
}

unsafe impl Sync for LocalGarbageCollector {}
unsafe impl Send for LocalGarbageCollector {}

impl LocalGarbageCollector {
    fn new() -> LocalGarbageCollector {
        LocalGarbageCollector { core: GarbageCollector::new() }
    }

    unsafe fn create_gc<T>(&self, t: T) -> Gc<T>
        where T: Sized + Trace {
        unsafe {
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

    /// Fallible version of `create_gc`. Returns `Err(GcAllocError)` on OOM.
    unsafe fn try_create_gc<T>(&self, t: T) -> Result<Gc<T>, GcAllocError>
        where T: Sized + Trace {
        unsafe {
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

    /// Fallible version of `create_gc_cell`. Returns `Err(GcAllocError)` on OOM.
    unsafe fn try_create_gc_cell<T>(&self, t: T) -> Result<GcRefCell<T>, GcAllocError>
        where T: Sized + Trace {
        unsafe {
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
    unsafe fn upgrade_weak<T>(&self, weak: &GcWeak<T>) -> Gc<T> where T: Sized + Trace {
        unsafe {
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

    /// Begin an incremental collection cycle.
    pub unsafe fn begin_collection(&self, max_gen: Generation) {
        unsafe { self.core.begin_collection(max_gen); }
    }

    /// Process a batch of gray objects. Returns true when marking is complete.
    pub unsafe fn mark_step(&self, budget: usize) -> bool {
        unsafe { self.core.mark_step(budget) }
    }

    /// Finish incremental collection: sweep white objects, promote survivors.
    pub unsafe fn finish_collection(&self) -> CollectionStats {
        unsafe { self.core.finish_collection() }
    }

    /// Run a complete incremental collection with the given step budget.
    pub unsafe fn collect_incremental(&self, max_gen: Generation, step_budget: usize) -> CollectionStats {
        unsafe { self.core.collect_incremental(max_gen, step_budget) }
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> GcStats {
        self.core.stats()
    }
}

impl PartialEq for &LocalGarbageCollector {
    fn eq(&self, other: &Self) -> bool {
        ((*self) as *const LocalGarbageCollector) == ((*other) as *const LocalGarbageCollector)
    }
}

pub type StartLocalStrategyFn = Box<dyn FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;
pub type StopLocalStrategyFn = Box<dyn FnMut(&'static LocalGarbageCollector)>;

pub struct LocalStrategy {
    gc: Cell<&'static LocalGarbageCollector>,
    is_active: AtomicBool,
    start_func: RefCell<StartLocalStrategyFn>,
    stop_func: RefCell<StopLocalStrategyFn>,
    join_handle: RefCell<Option<JoinHandle<()>>>,
}

impl LocalStrategy {
    fn new<StartFn, StopFn>(gc: &'static LocalGarbageCollector, start_fn: StartFn, stop_fn: StopFn) -> LocalStrategy
        where StartFn: 'static + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
              StopFn: 'static + FnMut(&'static LocalGarbageCollector) {
        LocalStrategy {
            gc: Cell::new(gc),
            is_active: AtomicBool::new(false),
            start_func: RefCell::new(Box::new(start_fn)),
            stop_func: RefCell::new(Box::new(stop_fn)),
            join_handle: RefCell::new(None),
        }
    }

    pub fn change_strategy<StartFn, StopFn>(&self, start_fn: StartFn, stop_fn: StopFn)
        where StartFn: 'static + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
              StopFn: 'static + FnMut(&'static LocalGarbageCollector) {
        if self.is_active() {
            self.stop();
        }
        let _ = self.start_func.replace(Box::new(start_fn));
        let _ = self.stop_func.replace(Box::new(stop_fn));
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    pub fn start(&'static self) {
        self.is_active.store(true, Ordering::Release);
        self.join_handle.replace((&mut *(self.start_func.borrow_mut()))(self.gc.get(), &self.is_active));
    }

    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        if let Some(join_handle) = self.join_handle.borrow_mut().take() {
            join_handle.join().expect("LocalStrategy::stop, LocalStrategy Thread being joined has panicked !!");
        }
        (&mut *(self.stop_func.borrow_mut()))(self.gc.get());
    }
}

impl Drop for LocalStrategy {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
        (&mut *(self.stop_func.borrow_mut()))(self.gc.get());
    }
}

thread_local! {
    static LOCAL_GC: RefCell<LocalGarbageCollector> = RefCell::new(LocalGarbageCollector::new());
    pub static LOCAL_GC_STRATEGY: RefCell<LocalStrategy> = {
        LOCAL_GC.with(move |gc| {
            let gc = unsafe { &mut *gc.as_ptr() };
            RefCell::new(LocalStrategy::new(gc,
            move |local_gc, _| {
                let mut basic_strategy_local_gcs = BASIC_STRATEGY_LOCAL_GCS.write().unwrap();
                basic_strategy_local_gcs.push(local_gc);
                None
            },
            move |local_gc| {
                let mut basic_strategy_local_gcs = BASIC_STRATEGY_LOCAL_GCS.write().unwrap();
                let index = basic_strategy_local_gcs.iter().position(|&r| r == local_gc).unwrap();
                basic_strategy_local_gcs.remove(index);
            }))
        })
    };
}

#[cfg(test)]
mod tests {
    use crate::gc::{Gc, GcRefCell, LOCAL_GC};
    use crate::{Trace, Finalize};
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
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().trs.len());
        let _one = Gc::new(1);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().core.tracers.read().unwrap().trs.len() - baseline, 1);
        });
    }

    #[test]
    fn gc_collect_one_from_one() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().trs.len());
        {
            let _one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().core.tracers.read().unwrap().trs.len() - baseline, 0);
        });
    }

    #[test]
    #[allow(unused_assignments)]
    fn two_objects_reassign() {
        // Reassigning drops the old Gc (remove_tracer removes it from trs),
        // so only 1 tracer remains for the surviving Gc.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().trs.len());
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| {
            assert_eq!(gc.borrow().core.tracers.read().unwrap().trs.len() - baseline, 1);
        });
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_after_reassign() {
        // After reassign, one live Gc remains. collect() keeps live objects.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().trs.len());
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().core.tracers.read().unwrap().trs.len() - baseline, 1);
        });
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_two_from_two() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().trs.len());
        {
            let mut one = Gc::new(1);
            one = Gc::new(2);
            drop(one);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().core.tracers.read().unwrap().trs.len() - baseline, 0);
        });
    }

    #[test]
    fn mem_to_trc_cleaned_on_collect() {
        clean_gc_state();
        let baseline_trs = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().trs.len());
        let baseline_m2t = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().mem_to_trc.len());
        {
            let _one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().core.tracers.read().unwrap().trs.len() - baseline_trs, 0);
            assert_eq!(gc.borrow().core.tracers.read().unwrap().mem_to_trc.len() - baseline_m2t, 0);
        });
    }

    #[test]
    fn collect_from_another_thread() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().trs.len());
        let _one = Gc::new(42);
        LOCAL_GC.with(|gc| {
            let gc_ptr = gc.as_ptr();
            let gc_ref = unsafe { &*gc_ptr };
            std::thread::scope(|s| {
                s.spawn(|| {
                    unsafe { gc_ref.collect(); }
                });
            });
            assert_eq!(gc.borrow().core.tracers.read().unwrap().trs.len() - baseline, 1);
        });
    }

    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct DropCounter {
        _value: String,
    }

    impl Trace for DropCounter {
        fn is_root(&self) -> bool { false }
        fn reset_root(&self) {}
        fn trace(&self) {}
        fn reset(&self) {}
        fn is_traceable(&self) -> bool { false }
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
            let _obj = Gc::new(DropCounter { _value: String::from("hello") });
        }
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1, "Drop should be called during collect");
    }

    #[test]
    fn clone_from_registers_with_gc() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.tracers.read().unwrap().trs.len());
        let source = Gc::new(42);
        let mut target = Gc::new(99);
        target.clone_from(&source);
        LOCAL_GC.with(|gc| {
            let delta = gc.borrow().core.tracers.read().unwrap().trs.len() - baseline;
            // source + target + the new clone's tracer = at least 2 alive
            assert!(delta >= 2, "clone_from should register new tracer with GC, got {delta}");
        });
    }

    struct CyclicNode {
        next: std::cell::RefCell<Option<Gc<CyclicNode>>>,
    }

    impl Trace for CyclicNode {
        fn is_root(&self) -> bool { false }
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
        fn is_traceable(&self) -> bool { false }
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
            let objects = gc_ref.core.objects.lock().unwrap();
            assert!(objects.obj_gen.values().any(|g| *g == crate::generation::Generation::Gen0),
                "newly allocated objects should be in Gen0");
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
                unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            }
            // Verify object was promoted to Gen1
            {
                let objects = gc_ref.core.objects.lock().unwrap();
                assert!(objects.obj_gen.values().any(|g| *g == crate::generation::Generation::Gen1),
                    "object should be promoted to Gen1 after surviving 3 Gen0 collections");
            }

            let baseline_objs = gc_ref.core.objects.lock().unwrap().objs.len();
            // Gen0-only collection should not touch Gen1 objects
            unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            let after_objs = gc_ref.core.objects.lock().unwrap().objs.len();
            assert_eq!(baseline_objs, after_objs,
                "Gen0 collection must not collect Gen1 objects");
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
                unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            }
            // Still Gen0 after 2 survivals
            {
                let objects = gc_ref.core.objects.lock().unwrap();
                assert!(objects.obj_gen.values().all(|g| *g == crate::generation::Generation::Gen0),
                    "object should still be in Gen0 after 2 survivals");
            }

            // Third survival triggers promotion
            unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            let objects = gc_ref.core.objects.lock().unwrap();
            assert!(objects.obj_gen.values().any(|g| *g == crate::generation::Generation::Gen1),
                "object should be promoted to Gen1 after 3 survivals");
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
                unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            }
            // Now survive 5 Gen1 collections to promote to Gen2
            for _ in 0..5 {
                unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen1); }
            }
            let objects = gc_ref.core.objects.lock().unwrap();
            assert!(objects.obj_gen.values().any(|g| *g == crate::generation::Generation::Gen2),
                "object should be promoted to Gen2 after surviving Gen1 threshold");
        });
    }

    #[test]
    fn dead_object_collected_by_gen0() {
        clean_gc_state();
        {
            let _obj = Gc::new(99);
        }
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let baseline = gc_ref.core.objects.lock().unwrap().objs.len();
            unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            let after = gc_ref.core.objects.lock().unwrap().objs.len();
            assert!(after < baseline, "dead Gen0 object should be collected by Gen0 collection");
        });
    }

    #[test]
    fn allocation_count_tracks_new_objects() {
        clean_gc_state();
        let before = LOCAL_GC.with(|gc| gc.borrow().core.allocation_count.load(Ordering::Relaxed));
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let after = LOCAL_GC.with(|gc| gc.borrow().core.allocation_count.load(Ordering::Relaxed));
        assert_eq!(after - before, 2, "allocation_count should increment per new object");
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
                unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            }
            // Verify it's in Gen1 (check by value to avoid fat pointer comparison issues)
            let objects = gc_ref.core.objects.lock().unwrap();
            assert!(objects.obj_gen.values().any(|g| *g == crate::generation::Generation::Gen1),
                "cell should be promoted to Gen1 after 3 Gen0 collections");
        });

        // Remembered set should be empty before mutation
        LOCAL_GC.with(|gc| {
            assert!(gc.borrow().core.remembered_set.lock().unwrap().is_empty(),
                "remembered set should be empty before write barrier");
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
            let rs = gc_ref.core.remembered_set.lock().unwrap();
            assert!(!rs.is_empty(),
                "write barrier should add old-gen object to remembered set");
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
                unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            }
        });

        // Trigger write barrier (outside LOCAL_GC borrow)
        **cell.borrow_mut() = Some(Gc::new(99));

        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            assert!(!gc_ref.core.remembered_set.lock().unwrap().is_empty(),
                "remembered set should not be empty after write barrier");

            // Full collection (Gen2) should clear remembered set
            unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen2); }
            assert!(gc_ref.core.remembered_set.lock().unwrap().is_empty(),
                "remembered set should be cleared after full collection");
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
                unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
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
            let objs_before = gc_ref.core.objects.lock().unwrap().objs.len();
            unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            let objs_after = gc_ref.core.objects.lock().unwrap().objs.len();
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
            assert!(gc.borrow().core.remembered_set.lock().unwrap().is_empty(),
                "Gen0 objects should not be added to remembered set");
        });
    }

    #[test]
    fn cyclic_gc_collect_does_not_overflow() {
        // Create a cycle: a → b → a. Without cycle protection in trace()/reset(),
        // collect() will recurse infinitely and overflow the stack.
        let result = std::thread::Builder::new()
            .stack_size(256 * 1024) // small stack to detect infinite recursion quickly
            .spawn(|| {
                let a = Gc::new(CyclicNode { next: std::cell::RefCell::new(None) });
                let b = Gc::new(CyclicNode { next: std::cell::RefCell::new(Some(a.clone())) });
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
        assert!(result.is_ok(), "collect() with cyclic references must not stack overflow");
    }

    #[test]
    fn weak_upgrade_returns_some_while_alive() {
        clean_gc_state();
        let strong = Gc::new(42);
        let weak = Gc::downgrade(&strong);
        let upgraded = weak.upgrade();
        assert!(upgraded.is_some(), "upgrade should succeed while object is alive");
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
        assert!(weak.upgrade().is_none(),
            "upgrade should return None after object is collected");
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
        clean_gc_state();
        let weak = {
            let strong = Gc::new(123);
            Gc::downgrade(&strong)
        };
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.objects.lock().unwrap().objs.len());
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        let after = LOCAL_GC.with(|gc| gc.borrow().core.objects.lock().unwrap().objs.len());
        assert!(after < baseline, "weak reference should not prevent collection");
        assert!(weak.upgrade().is_none());
    }

    // --- Incremental tri-color collection tests ---

    #[test]
    fn incremental_collects_dead_objects() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.objects.lock().unwrap().objs.len());
        {
            let _obj = Gc::new(42);
        }
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let stats = unsafe { gc_ref.core.collect_incremental(crate::generation::Generation::Gen2, 10) };
            assert!(stats.objects_collected > 0, "incremental should collect dead objects");
            assert_eq!(gc_ref.core.objects.lock().unwrap().objs.len(), baseline);
        });
    }

    #[test]
    fn incremental_preserves_live_objects() {
        clean_gc_state();
        let _live = Gc::new(99);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let objs_before = gc_ref.core.objects.lock().unwrap().objs.len();
            let stats = unsafe { gc_ref.core.collect_incremental(crate::generation::Generation::Gen2, 10) };
            assert_eq!(stats.objects_collected, 0, "incremental must not collect live objects");
            assert_eq!(gc_ref.core.objects.lock().unwrap().objs.len(), objs_before);
        });
    }

    #[test]
    fn incremental_step_by_step() {
        // Verify that begin_collection + mark_step + finish_collection works
        clean_gc_state();
        {
            let _dead = Gc::new(1);
        }
        let _live = Gc::new(2);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            unsafe {
                gc_ref.core.begin_collection(crate::generation::Generation::Gen2);
                // Process one object at a time
                while !gc_ref.core.mark_step(1) {}
                let stats = gc_ref.core.finish_collection();
                assert!(stats.objects_collected >= 1, "should collect the dead object");
            }
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
            let objs_before = gc_ref.core.objects.lock().unwrap().objs.len();
            unsafe { gc_ref.core.collect_incremental(crate::generation::Generation::Gen2, 1); }
            let objs_after = gc_ref.core.objects.lock().unwrap().objs.len();
            assert_eq!(objs_before, objs_after,
                "child referenced from parent must survive incremental collection");
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
                unsafe { gc_ref.core.collect_incremental(crate::generation::Generation::Gen0, 10); }
            }
            let objects = gc_ref.core.objects.lock().unwrap();
            assert!(objects.obj_gen.values().any(|g| *g == crate::generation::Generation::Gen1),
                "incremental collection should promote survivors");
        });
    }

    #[test]
    fn incremental_phase_tracking() {
        clean_gc_state();
        let _obj = Gc::new(1);
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            // Idle before collection
            assert_eq!(gc_ref.core.incremental.lock().unwrap().phase,
                crate::generation::CollectionPhase::Idle);

            unsafe { gc_ref.core.begin_collection(crate::generation::Generation::Gen2); }
            assert_eq!(gc_ref.core.incremental.lock().unwrap().phase,
                crate::generation::CollectionPhase::Marking);

            while unsafe { !gc_ref.core.mark_step(1) } {}
            // Still marking until finish_collection
            assert_eq!(gc_ref.core.incremental.lock().unwrap().phase,
                crate::generation::CollectionPhase::Marking);

            unsafe { gc_ref.core.finish_collection(); }
            assert_eq!(gc_ref.core.incremental.lock().unwrap().phase,
                crate::generation::CollectionPhase::Idle);
        });
    }

    // --- OOM handling tests ---

    #[test]
    fn try_new_succeeds_for_normal_allocation() {
        clean_gc_state();
        let result = Gc::try_new(42);
        assert!(result.is_ok(), "try_new should succeed for normal allocation");
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
        let baseline = LOCAL_GC.with(|gc| gc.borrow().core.objects.lock().unwrap().objs.len());
        {
            let _obj = Gc::try_new(77).unwrap();
        }
        LOCAL_GC.with(|gc| unsafe {
            gc.borrow_mut().collect();
        });
        let after = LOCAL_GC.with(|gc| gc.borrow().core.objects.lock().unwrap().objs.len());
        assert_eq!(after, baseline, "try_new objects should be collected when dead");
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
        assert_eq!(stats.live_objects - baseline, 2, "stats should report 2 new live objects");
    }

    #[test]
    fn stats_reports_live_tracers() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().stats().live_tracers);
        let _a = Gc::new(1);
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert_eq!(stats.live_tracers - baseline, 1, "stats should report 1 new tracer");
    }

    #[test]
    fn stats_reports_heap_size() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().stats().heap_size);
        let _a = Gc::new(42i32);
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(stats.heap_size > baseline, "heap_size should increase after allocation");
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
                unsafe { gc_ref.core.collect_generation(crate::generation::Generation::Gen0); }
            }
        });
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(stats.gen1_objects >= 1, "promoted object should be in Gen1");
    }

    #[test]
    fn stats_tracks_total_collections() {
        clean_gc_state();
        let before = LOCAL_GC.with(|gc| gc.borrow().stats().total_collections);
        LOCAL_GC.with(|gc| unsafe { gc.borrow_mut().collect(); });
        LOCAL_GC.with(|gc| unsafe { gc.borrow_mut().collect(); });
        let after = LOCAL_GC.with(|gc| gc.borrow().stats().total_collections);
        assert_eq!(after - before, 2, "total_collections should increment per collect call");
    }

    #[test]
    fn stats_tracks_last_collection() {
        clean_gc_state();
        {
            let _obj = Gc::new(99);
        }
        LOCAL_GC.with(|gc| unsafe { gc.borrow_mut().collect(); });
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        let last = stats.last_collection.expect("last_collection should be Some after collect");
        assert!(last.objects_collected >= 1, "last collection should have collected dead object");
    }

    #[test]
    fn stats_reports_allocation_count() {
        clean_gc_state();
        // After clean_gc_state (which calls collect), allocation_count is reset
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let stats = LOCAL_GC.with(|gc| gc.borrow().stats());
        assert!(stats.allocation_count >= 2, "allocation_count should track new objects");
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
        LOCAL_GC.with(|gc| unsafe { gc.borrow_mut().collect(); });
        assert_eq!(format!("{:?}", weak), "GcWeak(dead)");
    }

    // --- Cycle tests ---

    #[test]
    fn self_cycle_collected() {
        // A single node pointing to itself should be collected without overflow.
        let result = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let a = Gc::new(CyclicNode { next: std::cell::RefCell::new(None) });
                *a.next.borrow_mut() = Some(a.clone());
                drop(a);
                LOCAL_GC.with(|gc| unsafe { gc.borrow_mut().collect(); });
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
                let a = Gc::new(CyclicNode { next: std::cell::RefCell::new(None) });
                let b = Gc::new(CyclicNode { next: std::cell::RefCell::new(None) });
                let c = Gc::new(CyclicNode { next: std::cell::RefCell::new(None) });
                *a.next.borrow_mut() = Some(b.clone());
                *b.next.borrow_mut() = Some(c.clone());
                *c.next.borrow_mut() = Some(a.clone());
                drop(a);
                drop(b);
                drop(c);
                LOCAL_GC.with(|gc| unsafe { gc.borrow_mut().collect(); });
            })
            .unwrap()
            .join();
        assert!(result.is_ok(), "3-node cycle must not stack overflow");
    }

    struct VecNode {
        children: std::cell::RefCell<Vec<Gc<VecNode>>>,
    }
    impl Trace for VecNode {
        fn is_root(&self) -> bool { false }
        fn reset_root(&self) {
            for gc in self.children.borrow().iter() { gc.reset_root(); }
        }
        fn trace(&self) {
            for gc in self.children.borrow().iter() { gc.trace(); }
        }
        fn reset(&self) {
            for gc in self.children.borrow().iter() { gc.reset(); }
        }
        fn is_traceable(&self) -> bool { false }
        fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
            for gc in self.children.borrow().iter() { gc.trace_children(children); }
        }
    }
    impl Finalize for VecNode { fn finalize(&self) {} }

    #[test]
    fn cycle_in_vec_no_overflow() {
        // Two nodes each holding the other in their Vec of children.
        let result = std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(|| {
                let a = Gc::new(VecNode { children: std::cell::RefCell::new(vec![]) });
                let b = Gc::new(VecNode { children: std::cell::RefCell::new(vec![]) });
                let b_clone = b.clone();
                a.children.borrow_mut().push(b_clone);
                let a_clone = a.clone();
                b.children.borrow_mut().push(a_clone);
                drop(a);
                drop(b);
                LOCAL_GC.with(|gc| unsafe { gc.borrow_mut().collect(); });
            })
            .unwrap()
            .join();
        assert!(result.is_ok(), "cycle in Vec must not stack overflow");
    }
}
