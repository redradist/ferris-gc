use std::alloc::{alloc, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;

use crate::gc::{Finalize, Trace};
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

impl<T> Gc<T> where T: Sized + Trace {
    pub fn new(t: T) -> Gc<T> {
        basic_gc_strategy_start();
        let global_strategy = &(*GLOBAL_GC_STRATEGY);
        if !global_strategy.is_active() {
            global_strategy.start();
        }
        unsafe {
            (*GLOBAL_GC).create_gc(t)
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
}

impl<T> Finalize for Gc<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

pub struct GcCellInternal<T> where T: 'static + Sized + Trace {
    is_root: AtomicBool,
    ptr: *const RefCell<GcPtr<T>>,
}

impl<T> GcCellInternal<T> where T: 'static + Sized + Trace {
    fn new(ptr: *const RefCell<GcPtr<T>>) -> GcCellInternal<T> {
        GcCellInternal {
            is_root: AtomicBool::new(true),
            ptr: ptr,
        }
    }
}

impl<T> Trace for GcCellInternal<T> where T: Sized + Trace {
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

impl<T> Finalize for GcCellInternal<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

pub struct GcRefCell<T> where T: 'static + Sized + Trace {
    internal_ptr: *mut GcCellInternal<T>,
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

impl<T> GcRefCell<T> where T: 'static + Sized + Trace {
    pub fn new(t: T) -> GcRefCell<T> {
        basic_gc_strategy_start();
        let global_strategy = &(*GLOBAL_GC_STRATEGY);
        if !global_strategy.is_active() {
            global_strategy.start();
        }
        unsafe {
            (*GLOBAL_GC).create_gc_cell(t)
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
}

impl<T> Finalize for GcRefCell<T> where T: Sized + Trace {
    fn finalize(&self) {}
}

type GcObjMem = *mut u8;
type DropFn = unsafe fn(*mut u8);

pub struct GlobalGarbageCollector {
    mem_to_trc: RwLock<HashMap<usize, *const dyn Trace>>,
    trs: RwLock<HashMap<*const dyn Trace, (GcObjMem, Layout)>>,
    objs: Mutex<HashMap<*const dyn Trace, (GcObjMem, Layout)>>,
    fin: Mutex<HashMap<*const dyn Trace, *const dyn Finalize>>,
    drop_fns: Mutex<HashMap<*const dyn Trace, DropFn>>,
}

unsafe impl Sync for GlobalGarbageCollector {}

unsafe impl Send for GlobalGarbageCollector {}

impl GlobalGarbageCollector {
    fn new() -> GlobalGarbageCollector {
        GlobalGarbageCollector {
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
            let (gc_cell_inter_ptr, mem_info_internal_ptr) = self.alloc_mem::<GcCellInternal<T>>();
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));
            std::ptr::write(gc_cell_inter_ptr, GcCellInternal::new(gc_ptr));
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
            let (gc_inter_ptr, mem_info) = self.alloc_mem::<GcCellInternal<T>>();
            std::ptr::write(gc_inter_ptr, GcCellInternal::new(gc.ptr));
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
            let tracer_thin_ptr = tracer as *const () as usize;
            if let Some(tracer) = mem_to_trc.remove(&tracer_thin_ptr) {
                if let Some(del) = trs.remove(&tracer) {
                    dealloc(del.0, del.1);
                }
            }
        }
    }

    pub unsafe fn collect(&self) {
        unsafe {
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
                // Remove collected tracers from maps
                let mut tracer_deallocs = Vec::new();
                for tracer_ptr in collected_tracers {
                    let del = trs.remove(&tracer_ptr).unwrap();
                    mem_to_trc.remove(&(tracer_ptr as *const () as usize));
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
                // Remove collected objects from maps
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
            // All locks released — safe to call drop_in_place
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
                    mem_to_trc.remove(&(k as *const () as usize));
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
        let baseline = (*GLOBAL_GC).trs.read().unwrap().len();
        (guard, baseline)
    }

    #[test]
    fn one_object() {
        let (_guard, baseline) = setup();
        let _one = Gc::new(1);
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!((*GLOBAL_GC).trs.read().unwrap().len() - baseline, 1);
    }

    #[test]
    fn gc_collect_one_from_one() {
        let (_guard, baseline) = setup();
        {
            let _one = Gc::new(1);
        }
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!((*GLOBAL_GC).trs.read().unwrap().len() - baseline, 0);
    }

    #[test]
    #[allow(unused_assignments)]
    fn two_objects_reassign() {
        let (_guard, baseline) = setup();
        let mut one = Gc::new(1);
        one = Gc::new(2);
        unsafe { (*GLOBAL_GC).collect() };
        // Reassignment drops old Gc (remove_tracer), so only 1 tracer remains
        assert_eq!((*GLOBAL_GC).trs.read().unwrap().len() - baseline, 1);
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
        assert_eq!((*GLOBAL_GC).trs.read().unwrap().len() - baseline, 1);
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
        assert_eq!((*GLOBAL_GC).trs.read().unwrap().len() - baseline, 0);
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
}
