use std::alloc::{alloc, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

pub trait Trace: Finalize {
    fn is_root(&self) -> bool;
    fn reset_root(&self);
    fn trace(&self);
    fn reset(&self);
    fn is_traceable(&self) -> bool;
}

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
}

impl<T> Finalize for GcInternal<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

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
}

impl<T> Finalize for GcRefCellInternal<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

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
}

impl<T> Finalize for GcRefCell<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

type GcObjMem = *mut u8;

type DropFn = unsafe fn(*mut u8);

pub struct LocalGarbageCollector {
    mem_to_trc: RwLock<HashMap<usize, *const dyn Trace>>,
    trs: RwLock<HashMap<*const dyn Trace, (GcObjMem, Layout)>>,
    objs: Mutex<HashMap<*const dyn Trace, (GcObjMem, Layout)>>,
    fin: Mutex<HashMap<*const dyn Trace, *const dyn Finalize>>,
    drop_fns: Mutex<HashMap<*const dyn Trace, DropFn>>,
}

unsafe impl Sync for LocalGarbageCollector {}
unsafe impl Send for LocalGarbageCollector {}

impl LocalGarbageCollector {
    fn new() -> LocalGarbageCollector {
        LocalGarbageCollector {
            mem_to_trc: RwLock::new(HashMap::new()),
            trs: RwLock::new(HashMap::new()),
            objs: Mutex::new(HashMap::new()),
            fin: Mutex::new(HashMap::new()),
            drop_fns: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_objs(&self) -> &Mutex<HashMap<*const dyn Trace, (*mut u8, Layout)>> {
        &self.objs
    }

    unsafe fn create_gc<T>(&self, t: T) -> Gc<T>
        where T: Sized + Trace {
        unsafe {
            let (gc_ptr, mem_info_gc_ptr) = self.alloc_mem::<GcPtr<T>>();
            let (gc_inter_ptr, mem_info_internal_ptr) = self.alloc_mem::<GcInternal<T>>();
            std::ptr::write(gc_ptr, GcPtr::new(t));
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr));
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc_ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut mem_to_trc = self.mem_to_trc.write().unwrap();
            let mut trs = self.trs.write().unwrap();
            let mut objs = self.objs.lock().unwrap();
            let mut fin = self.fin.lock().unwrap();
            mem_to_trc.insert(gc_inter_ptr as usize, gc_inter_ptr);
            trs.insert(gc_inter_ptr, mem_info_internal_ptr);
            objs.insert(gc_ptr, mem_info_gc_ptr);
            fin.insert(gc_ptr, (*gc_ptr).t.as_finalize());
            unsafe fn drop_gc_ptr<T: 'static + Trace>(ptr: *mut u8) { unsafe { std::ptr::drop_in_place(ptr as *mut GcPtr<T>); } }
            self.drop_fns.lock().unwrap().insert(gc_ptr, drop_gc_ptr::<T>);
            gc
        }
    }

    unsafe fn clone_from_gc<T>(&self, gc: &Gc<T>) -> Gc<T> where T: Sized + Trace {
        unsafe {
            let (gc_inter_ptr, mem_info_internal_ptr) = self.alloc_mem::<GcInternal<T>>();
            std::ptr::write(gc_inter_ptr, GcInternal::new(gc.ptr));
            let gc = Gc {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut mem_to_trc = self.mem_to_trc.write().unwrap();
            let mut trs = self.trs.write().unwrap();
            mem_to_trc.insert(gc_inter_ptr as usize, gc_inter_ptr);
            trs.insert(gc_inter_ptr, mem_info_internal_ptr);
            gc
        }
    }

    unsafe fn create_gc_cell<T>(&self, t: T) -> GcRefCell<T> where T: Sized + Trace {
        unsafe {
            let (gc_ptr, mem_info_gc_ptr) = self.alloc_mem::<RefCell<GcPtr<T>>>();
            let (gc_cell_inter_ptr, mem_info_internal_ptr) = self.alloc_mem::<GcRefCellInternal<T>>();
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));
            std::ptr::write(gc_cell_inter_ptr, GcRefCellInternal::new(gc_ptr));
            let gc = GcRefCell {
                internal_ptr: gc_cell_inter_ptr,
                ptr: gc_ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut mem_to_trc = self.mem_to_trc.write().unwrap();
            let mut trs = self.trs.write().unwrap();
            let mut objs = self.objs.lock().unwrap();
            let mut fin = self.fin.lock().unwrap();
            mem_to_trc.insert(gc_cell_inter_ptr as usize, gc_cell_inter_ptr);
            trs.insert(gc_cell_inter_ptr, mem_info_internal_ptr);
            objs.insert(gc_ptr, mem_info_gc_ptr);
            fin.insert(gc_ptr, (*(*gc_ptr).as_ptr()).t.as_finalize());
            unsafe fn drop_gc_cell_ptr<T: 'static + Trace>(ptr: *mut u8) { unsafe { std::ptr::drop_in_place(ptr as *mut RefCell<GcPtr<T>>); } }
            self.drop_fns.lock().unwrap().insert(gc_ptr, drop_gc_cell_ptr::<T>);
            gc
        }
    }

    unsafe fn clone_from_gc_cell<T>(&self, gc: &GcRefCell<T>) -> GcRefCell<T> where T: Sized + Trace {
        unsafe {
            let (gc_inter_ptr, mem_info) = self.alloc_mem::<GcRefCellInternal<T>>();
            std::ptr::write(gc_inter_ptr, GcRefCellInternal::new(gc.ptr));
            let gc = GcRefCell {
                internal_ptr: gc_inter_ptr,
                ptr: gc.ptr,
            };
            (*(*gc.internal_ptr).ptr).reset_root();
            let mut mem_to_trc = self.mem_to_trc.write().unwrap();
            let mut trs = self.trs.write().unwrap();
            mem_to_trc.insert(gc_inter_ptr as usize, gc_inter_ptr);
            trs.insert(gc_inter_ptr, mem_info);
            gc
        }
    }

    unsafe fn alloc_mem<T>(&self) -> (*mut T, (GcObjMem, Layout)) where T: Sized {
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

    unsafe fn remove_tracer(&self, tracer: *const dyn Trace) {
        unsafe {
            let mut mem_to_trc = self.mem_to_trc.write().unwrap();
            let mut trs = self.trs.write().unwrap();
            if let Some(tracer) = mem_to_trc.remove(&(tracer.get_thin_ptr())) {
                if let Some(del) = trs.remove(&tracer) {
                    dealloc(del.0, del.1);
                }
            }
        }
    }

    pub unsafe fn collect(&self) {
        unsafe {
            // Phase 1-5: under locks, identify what to collect
            let (tracer_deallocs, object_deallocs) = {
                let mut mem_to_trc = self.mem_to_trc.write().unwrap();
                let mut trs = self.trs.write().unwrap();
                // Trace from roots
                for (gc_info, _) in &*trs {
                    let tracer = &(**gc_info);
                    if tracer.is_root() {
                        tracer.trace();
                    }
                }
                // Identify unreachable tracers
                let collected_tracers: Vec<_> = trs.iter()
                    .filter(|(gc_info, _)| !(&***gc_info).is_traceable())
                    .map(|(k, _)| *k)
                    .collect();
                // Remove collected tracers from maps, gather dealloc info
                let mut tracer_deallocs = Vec::new();
                for tracer_ptr in collected_tracers {
                    let del = trs.remove(&tracer_ptr).unwrap();
                    mem_to_trc.remove(&tracer_ptr.get_thin_ptr());
                    tracer_deallocs.push(del);
                }
                // Identify unreachable objects
                let mut objs = self.objs.lock().unwrap();
                let collected_objects: Vec<_> = objs.iter()
                    .filter(|(gc_info, _)| !(&***gc_info).is_traceable())
                    .map(|(k, _)| *k)
                    .collect();
                // Reset remaining tracers
                for (gc_info, _) in &*trs {
                    let tracer = &(**gc_info);
                    tracer.reset();
                }
                // Remove collected objects from maps, gather dealloc info
                let mut fin = self.fin.lock().unwrap();
                let mut drop_fns = self.drop_fns.lock().unwrap();
                let mut object_deallocs = Vec::new();
                for col in collected_objects {
                    let del = objs.remove(&col).unwrap();
                    let finalizer = fin.remove(&col);
                    let drop_fn = drop_fns.remove(&col);
                    object_deallocs.push((del, finalizer, drop_fn));
                }
                (tracer_deallocs, object_deallocs)
            };
            // Phase 6: all locks released — safe to call drop_in_place
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

    #[allow(dead_code)]
    unsafe fn collect_all(&self) {
        unsafe {
            let (tracer_deallocs, object_deallocs) = {
                let mut mem_to_trc = self.mem_to_trc.write().unwrap();
                let mut trs = self.trs.write().unwrap();
                let mut objs = self.objs.lock().unwrap();
                let mut fin = self.fin.lock().unwrap();
                let mut drop_fns = self.drop_fns.lock().unwrap();
                let tracer_deallocs: Vec<_> = trs.drain().map(|(k, v)| {
                    mem_to_trc.remove(&k.get_thin_ptr());
                    v
                }).collect();
                let object_deallocs: Vec<_> = objs.drain().map(|(k, v)| {
                    let finalizer = fin.remove(&k);
                    let drop_fn = drop_fns.remove(&k);
                    (v, finalizer, drop_fn)
                }).collect();
                (tracer_deallocs, object_deallocs)
            };
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
    use crate::gc::{Gc, LOCAL_GC};
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
        let baseline = LOCAL_GC.with(|gc| gc.borrow().trs.read().unwrap().len());
        let _one = Gc::new(1);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().trs.read().unwrap().len() - baseline, 1);
        });
    }

    #[test]
    fn gc_collect_one_from_one() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().trs.read().unwrap().len());
        {
            let _one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().trs.read().unwrap().len() - baseline, 0);
        });
    }

    #[test]
    #[allow(unused_assignments)]
    fn two_objects_reassign() {
        // Reassigning drops the old Gc (remove_tracer removes it from trs),
        // so only 1 tracer remains for the surviving Gc.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().trs.read().unwrap().len());
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| {
            assert_eq!(gc.borrow().trs.read().unwrap().len() - baseline, 1);
        });
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_after_reassign() {
        // After reassign, one live Gc remains. collect() keeps live objects.
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().trs.read().unwrap().len());
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().trs.read().unwrap().len() - baseline, 1);
        });
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_two_from_two() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().trs.read().unwrap().len());
        {
            let mut one = Gc::new(1);
            one = Gc::new(2);
            drop(one);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().trs.read().unwrap().len() - baseline, 0);
        });
    }

    #[test]
    fn mem_to_trc_cleaned_on_collect() {
        clean_gc_state();
        let baseline_trs = LOCAL_GC.with(|gc| gc.borrow().trs.read().unwrap().len());
        let baseline_m2t = LOCAL_GC.with(|gc| gc.borrow().mem_to_trc.read().unwrap().len());
        {
            let _one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow().trs.read().unwrap().len() - baseline_trs, 0);
            assert_eq!(gc.borrow().mem_to_trc.read().unwrap().len() - baseline_m2t, 0);
        });
    }

    #[test]
    fn collect_from_another_thread() {
        clean_gc_state();
        let baseline = LOCAL_GC.with(|gc| gc.borrow().trs.read().unwrap().len());
        let _one = Gc::new(42);
        LOCAL_GC.with(|gc| {
            let gc_ptr = gc.as_ptr();
            let gc_ref = unsafe { &*gc_ptr };
            std::thread::scope(|s| {
                s.spawn(|| {
                    unsafe { gc_ref.collect(); }
                });
            });
            assert_eq!(gc.borrow().trs.read().unwrap().len() - baseline, 1);
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
        let baseline = LOCAL_GC.with(|gc| gc.borrow().trs.read().unwrap().len());
        let source = Gc::new(42);
        let mut target = Gc::new(99);
        target.clone_from(&source);
        LOCAL_GC.with(|gc| {
            let delta = gc.borrow().trs.read().unwrap().len() - baseline;
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
}
