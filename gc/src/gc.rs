pub mod sync;

use std::cell::{Cell, RefCell, Ref};
use std::collections::{VecDeque, LinkedList, BTreeMap, HashMap, HashSet, BTreeSet, BinaryHeap};
use std::alloc::{alloc, dealloc, Layout};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::sync::{RwLock, Mutex};
use core::time;
use std::thread;
use std::borrow::BorrowMut;

pub trait Trace : Finalizer {
    fn is_root(&self) -> bool;
    fn reset_root(&self);
    fn trace(&self);
    fn reset(&self);
    fn is_traceable(&self) -> bool;
}

pub trait Finalizer {
    fn finalize(&self);
    fn as_finalize(&self) -> &dyn Finalizer
        where Self: Sized {
        self
    }
}

macro_rules! primitive_types {
    ($($prm:ident),*) => {
        $(
            impl Trace for $prm {
                fn is_root(&self) -> bool {
                    unreachable!("is_root should never be called on primitive type !!");
                }
                fn reset_root(&self) {
                }
                fn trace(&self) {
                }
                fn reset(&self) {
                }
                fn is_traceable(&self) -> bool {
                    unreachable!("is_traceable should never be called on primitive type !!");
                }
            }

            impl Finalizer for $prm {
                fn finalize(&self) {
                }
            }
        )*
    };
}

primitive_types!(
    u8, i8, u16, i16, u32, i32, u64, i64, u128, i128,
    usize, isize,
    f32, f64,
    bool
);

macro_rules! std_types {
    ($($std:ident<T>),*) => {
        $(
            impl<T> Trace for $std<T> where T: 'static + Sized + Trace {
                fn is_root(&self) -> bool {
                    unreachable!("is_root should never be called on primitive type !!");
                }
                fn reset_root(&self) {
                    for child in self {
                        child.reset_root();
                    }
                }
                fn trace(&self) {
                    for child in self {
                        child.trace();
                    }
                }
                fn reset(&self) {
                    for child in self {
                        child.reset();
                    }
                }
                fn is_traceable(&self) -> bool {
                    unreachable!("is_traceable should never be called on primitive type !!");
                }
            }

            impl<T> Finalizer for $std<T> where T: 'static + Sized + Trace {
                fn finalize(&self) {
                }
            }
        )*
    };
}

std_types!(
    Vec<T>, VecDeque<T>, LinkedList<T>,
    HashSet<T>, BTreeSet<T>, BinaryHeap<T>
);

// , HashMap<T>, BTreeMap<T>

struct GcInfo {
    has_root: Cell<bool>,
}

impl GcInfo {
    fn new() -> GcInfo {
        GcInfo {
            has_root: Cell::new(false),
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
        self.info.has_root.set(true);
        self.t.trace();
    }

    fn reset(&self) {
        self.info.has_root.set(false);
        self.t.reset();
    }

    fn is_traceable(&self) -> bool {
        self.info.has_root.get()
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
        self.borrow().info.has_root.set(true);
        self.borrow().t.trace();
    }

    fn reset(&self) {
        self.borrow().info.has_root.set(false);
        self.borrow().t.reset();
    }

    fn is_traceable(&self) -> bool {
        self.borrow().info.has_root.get()
    }
}

impl<T> Finalizer for RefCell<GcPtr<T>> where T: Sized + Trace {
    fn finalize(&self) {
    }
}

impl<T> Finalizer for GcPtr<T> where T: Sized + Trace {
    fn finalize(&self) {
    }
}

pub struct GcInternal<T> where T: 'static + Sized + Trace {
    is_root: Cell<bool>,
    ptr: *const GcPtr<T>,
}

impl<T> GcInternal<T> where T: 'static + Sized + Trace {
    fn new(ptr: *const GcPtr<T>) -> GcInternal<T> {
        GcInternal {
            is_root: Cell::new(true),
            ptr: ptr,
        }
    }
}

impl<T> Trace for GcInternal<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        self.is_root.get()
    }

    fn reset_root(&self) {
        if self.is_root.get() {
            self.is_root.set(false);
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
        unreachable!();
    }
}

impl<T> Finalizer for GcInternal<T> where T: Sized + Trace {
    fn finalize(&self) {
    }
}

impl<T> Deref for GcInternal<T> where T: 'static + Sized + Trace {
    type Target = GcPtr<T>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &(*self.ptr)
        }
    }
}

pub struct Gc<T> where T: 'static + Sized + Trace {
    internal_ptr: *mut GcInternal<T>,
}

impl<T> Deref for Gc<T> where T: 'static + Sized + Trace {
    type Target = GcInternal<T>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &(*self.internal_ptr)
        }
    }
}

impl<T> Gc<T> where T: Sized + Trace {
    pub fn new<'a>(t: T) -> Gc<T> {
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

    fn clone_from(&mut self, source: &Self) {
        self.is_root.set(false);
        unsafe {
            (*self.internal_ptr).ptr = (*source.internal_ptr).ptr;
        }
    }
}

impl<T> Drop for Gc<T> where T: Sized + Trace {
    fn drop(&mut self) {
        println!("Gc::drop");
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
            (*self.internal_ptr).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            (*self.internal_ptr).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.is_root()
    }
}

impl<T> Finalizer for Gc<T> where T: Sized + Trace {
    fn finalize(&self) {
    }
}

pub struct GcCellInternal<T> where T: 'static + Sized + Trace {
    is_root: Cell<bool>,
    ptr: *const RefCell<GcPtr<T>>,
}

impl<T> GcCellInternal<T> where T: 'static + Sized + Trace {
    fn new(ptr: *const RefCell<GcPtr<T>>) -> GcCellInternal<T> {
        GcCellInternal {
            is_root: Cell::new(true),
            ptr: ptr,
        }
    }
}

impl<T> Trace for GcCellInternal<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        self.is_root.get()
    }

    fn reset_root(&self) {
        if self.is_root.get() {
            self.is_root.set(false);
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
        unreachable!();
    }
}

impl<T> Finalizer for GcCellInternal<T> where T: Sized + Trace {
    fn finalize(&self) {
    }
}

impl<T> Deref for GcCellInternal<T> where T: 'static + Sized + Trace {
    type Target = RefCell<GcPtr<T>>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &(*self.ptr)
        }
    }
}

pub struct GcCell<T> where T: 'static + Sized + Trace {
    internal_ptr: *mut GcCellInternal<T>,
}

impl<T> Drop for GcCell<T> where T: Sized + Trace {
    fn drop(&mut self) {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().remove_tracer(self.internal_ptr);
        });
    }
}

impl<T> Deref for GcCell<T> where T: 'static + Sized + Trace {
    type Target = GcCellInternal<T>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &(*self.internal_ptr)
        }
    }
}

impl<T> GcCell<T> where T: 'static + Sized + Trace {
    pub fn new<'a>(t: T) -> GcCell<T> {
        basic_gc_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| unsafe {
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

impl<T> Clone for GcCell<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        let gc = LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().clone_from_gc_cell(self)
        });
        unsafe {
            (*gc.internal_ptr).ptr = (*self.internal_ptr).ptr;
            (*gc.internal_ptr).is_root.set(true);
        }
        gc
    }

    fn clone_from(&mut self, source: &Self) {
        self.is_root.set(false);
        unsafe {
            (*self.internal_ptr).ptr = (*source.internal_ptr).ptr;
        }
    }
}

impl<T> Trace for GcCell<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        unsafe {
            (*self.ptr).borrow().is_root()
        }
    }

    fn reset_root(&self) {
        self.is_root.set(false);
        unsafe {
            (*self.ptr).borrow().reset_root();
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
        self.is_root()
    }
}

impl<T> Finalizer for GcCell<T> where T: Sized + Trace {
    fn finalize(&self) {
    }
}

type GcObjMem = *mut u8;

pub struct LocalGarbageCollector {
    trs: RwLock<HashMap<*const dyn Trace, (GcObjMem, Layout)>>,
    objs: Mutex<HashMap<*const dyn Trace, (GcObjMem, Layout)>>,
    fin: Mutex<HashMap<*const dyn Trace, *const dyn Finalizer>>,
}

unsafe impl Sync for LocalGarbageCollector {}
unsafe impl Send for LocalGarbageCollector {}

impl LocalGarbageCollector {
    fn new() -> LocalGarbageCollector {
        LocalGarbageCollector {
            trs: RwLock::new(HashMap::new()),
            objs: Mutex::new(HashMap::new()),
            fin: Mutex::new(HashMap::new()),
        }
    }

    unsafe fn create_gc<T>(&self, t: T) -> Gc<T>
        where T: Sized + Trace {
        let (gc_ptr, mem_info_gc_ptr) = self.alloc_mem::<GcPtr<T>>();
        let (gc_inter_ptr, mem_info_internal_ptr) = self.alloc_mem::<GcInternal<T>>();
        std::ptr::write(gc_ptr, GcPtr::new(t));
        std::ptr::write(gc_inter_ptr, GcInternal::new(gc_ptr));
        let gc = Gc {
            internal_ptr: gc_inter_ptr,
        };
        (*(*gc.internal_ptr).ptr).reset_root();
        let mut trs = self.trs.write().unwrap();
        let mut objs = self.objs.lock().unwrap();
        let mut fin = self.fin.lock().unwrap();
        trs.insert(gc_inter_ptr, mem_info_internal_ptr);
        objs.insert(gc_ptr, mem_info_gc_ptr);
        fin.insert(gc_ptr, (*gc_ptr).t.as_finalize());
        gc
    }

    unsafe fn clone_from_gc<T>(&self, gc: &Gc<T>) -> Gc<T> where T: Sized + Trace {
        let (gc_inter_ptr, mem_info_internal_ptr) = self.alloc_mem::<GcInternal<T>>();
        std::ptr::write(gc_inter_ptr, GcInternal::new(gc.ptr));
        let mut trs = self.trs.write().unwrap();
        trs.insert(gc_inter_ptr, mem_info_internal_ptr);
        let gc = Gc {
            internal_ptr: gc_inter_ptr,
        };
        (*(*gc.internal_ptr).ptr).reset_root();
        gc
    }

    unsafe fn create_gc_cell<T>(&self, t: T) -> GcCell<T> where T: Sized + Trace {
        let (gc_ptr, mem_info_gc_ptr) = self.alloc_mem::<RefCell<GcPtr<T>>>();
        let (gc_cell_inter_ptr, mem_info_internal_ptr) = self.alloc_mem::<GcCellInternal<T>>();
        std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));
        std::ptr::write(gc_cell_inter_ptr, GcCellInternal::new(gc_ptr));
        let gc = GcCell {
            internal_ptr: gc_cell_inter_ptr,
        };
        (*(*gc.internal_ptr).ptr).reset_root();
        let mut trs = self.trs.write().unwrap();
        let mut objs = self.objs.lock().unwrap();
        let mut fin = self.fin.lock().unwrap();
        trs.insert(gc_cell_inter_ptr, mem_info_internal_ptr);
        objs.insert(gc_ptr, mem_info_gc_ptr);
        fin.insert(gc_ptr, (*(*gc_ptr).as_ptr()).t.as_finalize());
        gc
    }

    unsafe fn clone_from_gc_cell<T>(&self, gc: &GcCell<T>) -> GcCell<T> where T: Sized + Trace {
        let (gc_inter_ptr, mem_info) = self.alloc_mem::<GcCellInternal<T>>();
        std::ptr::write(gc_inter_ptr, GcCellInternal::new(gc.ptr));
        let mut trs = self.trs.write().unwrap();
        trs.insert(gc_inter_ptr, mem_info);
        let gc = GcCell {
            internal_ptr: gc_inter_ptr,
        };
        (*(*gc.internal_ptr).ptr).reset_root();
        gc
    }

    unsafe fn alloc_mem<T>(&self) -> (*mut T, (GcObjMem, Layout)) where T: Sized {
        let layout = Layout::new::<T>();
        let mem = alloc(layout);
        let gc_inter_ptr: *mut T = mem as *mut _;
        (gc_inter_ptr, (mem, layout))
    }

    pub unsafe fn remove_tracer(&self, tracer: *const dyn Trace) {
        dbg!("remove_tracer(&self, tracer: *const dyn Trace)");
        let mut trs = self.trs.write().unwrap();
        let del = (&*trs)[&tracer];
        dealloc(del.0, del.1);
        trs.remove(&tracer);
    }

    pub unsafe fn collect(&self) {
        dbg!("Start collect ...");
        let mut collected_int_objects: Vec<*const dyn Trace> = Vec::new();
        let mut trs = self.trs.read().unwrap();
        for (gc_info, _) in &*trs {
            let tracer = &(**gc_info);
            if tracer.is_root() {
                tracer.trace();
            }
        }
        let mut collected_objects: Vec<*const dyn Trace> = Vec::new();
        let mut objs = self.objs.lock().unwrap();
        for (gc_info, _) in &*objs {
            let obj = &(**gc_info);
            if !obj.is_traceable() {
                collected_objects.push(*gc_info);
            }
        }
        for (gc_info, _) in &*trs {
            let tracer = &(**gc_info);
            tracer.reset();
        }
        dbg!("collected_objects: {}", collected_objects.len());
        let mut fin = self.fin.lock().unwrap();
        for col in collected_objects {
            let del = (&*objs)[&col];
            let finilizer = (&*fin)[&col];
            (*finilizer).finalize();
            dealloc(del.0, del.1);
            objs.remove(&col);
            fin.remove(&col);
        }
    }

    unsafe fn collect_all(&self) {
        dbg!("Start collect ...");
        let mut collected_int_objects: Vec<*const dyn Trace> = Vec::new();
        let mut trs = self.trs.write().unwrap();
        for (gc_info, _) in &*trs {
            collected_int_objects.push(*gc_info);
        }
        let mut collected_objects: Vec<*const dyn Trace> = Vec::new();
        let mut objs = self.objs.lock().unwrap();
        for (gc_info, _) in &*objs {
            collected_objects.push(*gc_info);
        }
        for col in collected_int_objects {
            self.remove_tracer(col);
        }
        dbg!("collected_objects: {}", collected_objects.len());
        let mut fin = self.fin.lock().unwrap();
        for col in collected_objects {
            let del = (&*objs)[&col];
            let finilizer = (&*fin)[&col];
            (*finilizer).finalize();
            dealloc(del.0, del.1);
            objs.remove(&col);
            fin.remove(&col);
        }
    }
}

impl Drop for LocalGarbageCollector {
    fn drop(&mut self) {
        dbg!("GarbageCollector::drop");
    }
}

impl PartialEq for &LocalGarbageCollector {
    fn eq(&self, other: &Self) -> bool {
        *self == *other
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
            join_handle: RefCell::new(None)
        }
    }

    pub fn change_strategy<StartFn, StopFn>(&self, start_fn: StartFn, stop_fn: StopFn)
        where StartFn: 'static + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
               StopFn: 'static + FnMut(&'static LocalGarbageCollector) {
        if self.is_active() {
            self.stop();
        }
        self.start_func.replace(Box::new(start_fn));
        self.stop_func.replace(Box::new(stop_fn));
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    pub fn start(&'static self) {
        dbg!("LocalStrategy::start");
        self.is_active.store(true, Ordering::Release);
        self.join_handle.replace((&mut *(self.start_func.borrow_mut()))(self.gc.get(), &self.is_active));
    }

    pub fn stop(&self) {
        dbg!("LocalStrategy::stop");
        self.is_active.store(false, Ordering::Release);
        if let Some(join_handle) = self.join_handle.borrow_mut().take()  {
            join_handle.join().expect("LocalStrategy::stop, LocalStrategy Thread being joined has panicked !!");
        }
        (&mut *(self.stop_func.borrow_mut()))(self.gc.get());
    }
}

impl Drop for LocalStrategy {
    fn drop(&mut self) {
        dbg!("LocalStrategy::drop");
        self.is_active.store(false, Ordering::Release);
    }
}

use crate::gc_strategy::{BASIC_STRATEGY_LOCAL_GCS, basic_gc_strategy_start};
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

    #[test]
    fn one_object() {
        let one = Gc::new(1);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().trs.read().unwrap().len(), 1);
        });
    }

    #[test]
    fn gc_collect_one_from_one() {
        {
            let one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().trs.read().unwrap().len(), 0);
        });
    }

    #[test]
    fn two_objects() {
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| {
            assert_eq!(gc.borrow_mut().trs.read().unwrap().len(), 2);
        });
    }

    #[test]
    fn gc_collect_one_from_two() {
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().trs.read().unwrap().len(), 0);
        });
    }

    #[test]
    fn gc_collect_two_from_two() {
        {
            let mut one = Gc::new(1);
            one = Gc::new(2);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().trs.read().unwrap().len(), 0);
        });
    }
}
