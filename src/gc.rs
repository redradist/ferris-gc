pub mod sync;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::alloc::{alloc, dealloc, Layout};

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
    f32, f64,
    bool
);

struct GcInfo {
    has_root: Cell<bool>,
}

pub struct GcPtr<T> where T: 'static + Sized + Trace {
    info: GcInfo,
    t: T,
}

impl<T> Deref for GcPtr<T> where T: 'static + Sized + Trace {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.t
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

impl<T> Finalizer for GcPtr<T> where T: Sized + Trace {
    fn finalize(&self) {
    }
}

pub struct GcInternal<T> where T: 'static + Sized + Trace {
    is_root: Cell<bool>,
    ptr: *const GcPtr<T>,
}

impl<T> Trace for GcInternal<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        self.is_root.get()
    }

    fn reset_root(&self) {
        self.is_root.set(false);
        if !self.ptr.is_null() {
            unsafe {
                (*self.ptr).reset_root();
            }
        }
    }

    fn trace(&self) {
        if !self.ptr.is_null() {
            unsafe {
                (*self.ptr).trace();
            }
        }
    }

    fn reset(&self) {
        if !self.ptr.is_null() {
            unsafe {
                (*self.ptr).reset();
            }
        }
    }

    fn is_traceable(&self) -> bool {
        if !self.ptr.is_null() {
            unsafe {
                (*self.ptr).is_traceable()
            }
        } else {
            true
        }
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
        LOCAL_GC_STRATEGY.with(|strategy| unsafe {
            if !strategy.borrow().is_active() {
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().create_gc(t)
        })
    }

    pub fn null() -> Gc<T> {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().null_gc()
        })
    }
}

impl<T> Clone for Gc<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        let gc = LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().null_gc()
        });
        unsafe {
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

impl<T> Drop for Gc<T> where T: Sized + Trace {
    fn drop(&mut self) {
        self.is_root.set(false);
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
        unsafe {
            (*self.internal_ptr).is_traceable()
        }
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

impl<T> Trace for GcCellInternal<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        self.is_root.get()
    }

    fn reset_root(&self) {
        self.is_root.set(false);
        if !self.ptr.is_null() {
            unsafe {
                (*self.ptr).borrow().reset_root();
            }
        }
    }

    fn trace(&self) {
        if !self.ptr.is_null() {
            unsafe {
                (*self.ptr).borrow().trace();
            }
        }
    }

    fn reset(&self) {
        if !self.ptr.is_null() {
            unsafe {
                (*self.ptr).borrow().reset();
            }
        }
    }

    fn is_traceable(&self) -> bool {
        if !self.ptr.is_null() {
            unsafe {
                (*self.ptr).borrow().is_traceable()
            }
        } else {
            true
        }
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
        self.is_root.set(false);
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

    pub fn null() -> GcCell<T> {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().null_gc_cell()
        })
    }
}

impl<T> Clone for GcCell<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        let gc = LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().null_gc_cell()
        });
        gc.is_root.set(true);
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
        unsafe {
            (*self.ptr).borrow().is_traceable()
        }
    }
}

impl<T> Finalizer for GcCell<T> where T: Sized + Trace {
    fn finalize(&self) {
    }
}

type GcObjMem = *mut u8;

pub struct GarbageCollector {
    vec: RefCell<HashMap<*const dyn Trace, ((GcObjMem, Layout), Option<(GcObjMem, Layout)>)>>,
    fin: RefCell<HashMap<*const dyn Trace, *const dyn Finalizer>>,
}

unsafe impl Sync for GarbageCollector {}
unsafe impl Send for GarbageCollector {}

impl GarbageCollector {
    fn new() -> GarbageCollector {
        GarbageCollector {
            vec: RefCell::new(HashMap::new()),
            fin: RefCell::new(HashMap::new()),
        }
    }

    unsafe fn create_gc<T>(&mut self, t: T) -> Gc<T>
        where T: Sized + Trace {
        let (gc_inter_ptr, mem_info_internal_ptr) = self.get_gc_inter_ptr::<T>();
        let (gc_ptr, mem_info_gc_ptr) = self.get_gc_ptr::<T>();
        let gc = Gc {
            internal_ptr: gc_inter_ptr,
        };
        (*gc_ptr).info.has_root.set(false);
        std::ptr::write(&mut (*gc_ptr).t, t);
        (*gc_ptr).t.reset_root();
        (*gc.internal_ptr).is_root.set(true);
        (*gc.internal_ptr).ptr = gc_ptr;
        self.register_root(gc.internal_ptr, (mem_info_internal_ptr, Some(mem_info_gc_ptr)));
        self.register_root_fin(gc.internal_ptr, (*gc_ptr).t.as_finalize());
        gc
    }

    unsafe fn null_gc<T>(&mut self) -> Gc<T> where T: Sized + Trace {
        let (gc_inter_ptr, mem_info) = self.get_gc_inter_ptr::<T>();
        (*gc_inter_ptr).ptr = std::ptr::null();
        self.register_root(gc_inter_ptr, (mem_info, None));
        Gc {
            internal_ptr: gc_inter_ptr,
        }
    }

    unsafe fn create_gc_cell<T>(&mut self, t: T) -> GcCell<T> where T: Sized + Trace {
        let (gc_cell_inter_ptr, mem_info_internal_ptr) = self.get_gc_cell_inter_ptr::<T>();
        let (gc_ptr, mem_info_gc_ptr) = self.get_ref_cell_gc_ptr::<T>();
        let gc = GcCell {
            internal_ptr: std::ptr::null_mut(),
        };
        (*gc_ptr).borrow().info.has_root.set(false);
        std::ptr::write(&mut (*gc_ptr).borrow_mut().t, t);
        (*gc_ptr).borrow_mut().t.reset_root();
        (*gc.internal_ptr).is_root.set(true);
        (*gc.internal_ptr).ptr = gc_ptr;
        self.register_root(gc.internal_ptr, (mem_info_internal_ptr, Some(mem_info_gc_ptr)));
        self.register_root_fin(gc.internal_ptr, (*gc_ptr).as_ptr());
        gc
    }

    unsafe fn null_gc_cell<T>(&mut self) -> GcCell<T> where T: Sized + Trace {
        let (gc_inter_ptr, mem_info) = self.get_gc_cell_inter_ptr::<T>();
        (*gc_inter_ptr).ptr = std::ptr::null();
        self.register_root(gc_inter_ptr, (mem_info, None));
        GcCell {
            internal_ptr: gc_inter_ptr,
        }
    }

    fn register_root(&mut self, root_ptr: *const dyn Trace, mem: ((GcObjMem, Layout), Option<(GcObjMem, Layout)>)) {
        self.vec.borrow_mut().insert(root_ptr, mem);
    }

    fn register_root_fin(&mut self, root_ptr: *const dyn Trace, fin_ptr: *const dyn Finalizer) {
        self.fin.borrow_mut().insert(root_ptr, fin_ptr);
    }

    unsafe fn get_gc_inter_ptr<T>(&mut self) -> (*mut GcInternal<T>, (GcObjMem, Layout)) where T: Sized + Trace {
        let layout = Layout::new::<GcInternal<T>>();
        let mem = alloc(layout);
        let gc_inter_ptr: *mut GcInternal<T> = mem as *mut _;
        (gc_inter_ptr, (mem, layout))
    }

    unsafe fn get_gc_ptr<T>(&mut self) -> (*mut GcPtr<T>, (GcObjMem, Layout)) where T: Sized + Trace {
        let layout = Layout::new::<GcPtr<T>>();
        let mem = alloc(layout);
        let gc_ptr: *mut GcPtr<T> = mem as *mut _;
        (gc_ptr, (mem, layout))
    }

    unsafe fn get_gc_cell_inter_ptr<T>(&mut self) -> (*mut GcCellInternal<T>, (GcObjMem, Layout)) where T: Sized + Trace {
        let layout = Layout::new::<GcInternal<T>>();
        let mem = alloc(layout);
        let gc_cell_inter_ptr: *mut GcCellInternal<T> = mem as *mut _;
        (gc_cell_inter_ptr, (mem, layout))
    }

    unsafe fn get_ref_cell_gc_ptr<T>(&mut self) -> (*mut RefCell<GcPtr<T>>, (*mut u8, Layout)) where T: Sized + Trace {
        let layout = Layout::new::<RefCell<GcPtr<T>>>();
        let mem = alloc(layout);
        let gc_ptr: *mut RefCell<GcPtr<T>> = mem as *mut _;
        (gc_ptr, (mem, layout))
    }

    pub unsafe fn collect(&self) {
        dbg!("Start collect ...");
        let mut collected_objects: Vec<*const dyn Trace> = Vec::new();
        for (gc_info, _) in &*self.vec.borrow() {
            let tracer = &(**gc_info);
            if tracer.is_root() {
                tracer.trace();
            }
        }
        for (gc_info, _) in &*self.vec.borrow() {
            let tracer = &(**gc_info);
            if !tracer.is_traceable() {
                collected_objects.push(*gc_info);
            } else {
                tracer.reset();
            }
        }
        dbg!("collected_objects: {}", collected_objects.len());
        for col in collected_objects {
            let del = (&*self.vec.borrow())[&col];
            let fin = (&*self.fin.borrow())[&col];
            (*fin).finalize();
            dealloc((del.0).0, (del.0).1);
            if let Some(t) = del.1 {
                dealloc(t.0, t.1);
            }
            self.vec.borrow_mut().remove(&col);
            self.fin.borrow_mut().remove(&col);
        }
    }
}

impl Drop for GarbageCollector {
    fn drop(&mut self) {
        dbg!("GarbageCollector::drop");
    }
}

pub type LocalStrategyFn = Box<dyn FnMut(&'static GarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;

pub struct LocalStrategy {
    gc: Cell<&'static GarbageCollector>,
    is_active: AtomicBool,
    func: RefCell<LocalStrategyFn>,
    join_handle: RefCell<Option<JoinHandle<()>>>,
}

impl LocalStrategy {
    fn new<F>(gc: &'static mut GarbageCollector, f: F) -> LocalStrategy
        where F: 'static + FnMut(&'static GarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> {
        LocalStrategy {
            gc: Cell::new(gc),
            is_active: AtomicBool::new(false),
            func: RefCell::new(Box::new(f)),
            join_handle: RefCell::new(None)
        }
    }

    pub fn new_from<F>(&self, f: F) -> LocalStrategy
        where F: 'static + FnMut(&'static GarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> {
        LocalStrategy {
            gc: Cell::new(self.gc.get()),
            is_active: AtomicBool::new(false),
            func: RefCell::new(Box::new(f)),
            join_handle: RefCell::new(None)
        }
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    pub fn start(&'static self) {
        dbg!("LocalStrategy::start");
        self.is_active.store(true, Ordering::Release);
        self.join_handle.replace((&mut *(self.func.borrow_mut()))(self.gc.get(), &self.is_active));
    }

    pub fn stop(&self) {
        dbg!("LocalStrategy::stop");
        dbg!("LocalStrategy::stop, is_active: {}", self.is_active.load(Ordering::Acquire));
        self.is_active.store(false, Ordering::Release);
        if let Some(join_handle) = self.join_handle.borrow_mut().take()  {
            // NOTE(redra): Crash due to destroying LocalStrategy from wrong thread
            // join_handle.join().expect("LocalStrategy::stop, LocalStrategy Thread being joined has panicked !!");
        }
    }
}

impl Drop for LocalStrategy {
    fn drop(&mut self) {
        dbg!("LocalStrategy::drop");
        self.stop();
    }
}

fn basic_local_strategy(gc: &'static GarbageCollector, is_work: &'static AtomicBool) -> Option<JoinHandle<()>> {
    Some(thread::spawn(move || {
        while is_work.load(Ordering::Acquire) {
            let ten_secs = time::Duration::from_secs(10);
            thread::sleep(ten_secs);
            unsafe {
                gc.collect();
            }
        }
        dbg!("Stop thread::spawn");
    }))
}

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::sync::RwLock;
use core::time;
use std::thread;
use std::borrow::BorrowMut;
thread_local! {
    static LOCAL_GC: RefCell<GarbageCollector> = RefCell::new(GarbageCollector::new());
    pub static LOCAL_GC_STRATEGY: RefCell<LocalStrategy> = {
        LOCAL_GC.with(move |gc| {
            let gc = unsafe { &mut *gc.as_ptr() };
            RefCell::new(LocalStrategy::new(gc, move |obj, sda| {
                basic_local_strategy(obj, sda)
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
            assert_eq!(gc.borrow_mut().vec.borrow().len(), 1);
        });
    }

    #[test]
    fn gc_collect_one_from_one() {
        {
            let one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().vec.borrow().len(), 0);
        });
    }

    #[test]
    fn two_objects() {
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| {
            assert_eq!(gc.borrow_mut().vec.borrow().len(), 2);
        });
    }

    #[test]
    fn gc_collect_one_from_two() {
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().vec.borrow().len(), 0);
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
            assert_eq!(gc.borrow_mut().vec.borrow().len(), 0);
        });
    }
}
