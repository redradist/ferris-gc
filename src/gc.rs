pub mod sync;

use std::cell::Cell;
use std::collections::HashMap;
use std::alloc::{alloc, dealloc, Layout};

pub trait Trace {
    fn is_root(&self) -> bool;
    fn reset_root(&self);
    fn trace(&self);
    fn reset(&self);
    fn is_traceable(&self) -> bool;
}

pub trait Finalizer {
    fn finalize(&self);
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
    pub fn new(t: T) -> Gc<T> {
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
        let tracer = self.internal_ptr;
        LOCAL_GC.with(move |gc| {
            gc.borrow_mut().unregister_root(tracer);
        });
        unsafe {
            (*self.internal_ptr).ptr = (*source.internal_ptr).ptr;
        }
    }
}

impl<T> Drop for Gc<T> where T: Sized + Trace {
    fn drop(&mut self) {
        LOCAL_GC.with(move |gc| {
            gc.borrow_mut().unregister_root(self.internal_ptr);
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
        unsafe {
            (*self.internal_ptr).is_traceable()
        }
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
        LOCAL_GC.with(move |gc| {
            gc.borrow_mut().unregister_root(self.internal_ptr);
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

impl<T> GcCell<T> where T: Sized + Trace {
    pub fn new(t: T) -> GcCell<T> {
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
        let tracer = self.internal_ptr;
        LOCAL_GC.with(move |gc| {
            gc.borrow_mut().unregister_root(tracer);
        });
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

type GcObjMem = *mut u8;

pub struct GarbageCollector {
    vec: HashMap<*const dyn Trace, ((GcObjMem, Layout), (GcObjMem, Layout))>,
}

impl GarbageCollector {
    fn new() -> GarbageCollector {
        GarbageCollector {
            vec: HashMap::new(),
        }
    }

    pub fn strategy(&mut self) {
        // std::thread::spawn(move || unsafe {
        //     self.collect();
        // });
    }

    unsafe fn create_gc<T>(&mut self, t: T) -> Gc<T>
        where T: Sized + Trace {
        let (gc_inter_ptr, mem_info_internal_ptr) = self.get_gc_inter_ptr::<T>();
        let (gc_ptr, mem_info_gc_ptr) = self.get_gc_ptr::<T>();
        let gc = Gc {
            internal_ptr: gc_inter_ptr,
        };
        (*gc_ptr).info.has_root.set(false);
        (*gc_ptr).t = t;
        (*gc.internal_ptr).ptr = gc_ptr;
        gc.reset_root();
        (*gc.internal_ptr).is_root.set(true);
        self.register_root(gc.internal_ptr, (mem_info_internal_ptr, mem_info_gc_ptr));
        gc
    }

    unsafe fn null_gc<T>(&mut self) -> Gc<T> where T: Sized + Trace {
        let (gc_inter_ptr, mem_info) = self.get_gc_inter_ptr::<T>();
        (*gc_inter_ptr).ptr = std::ptr::null();
        Gc {
            internal_ptr: gc_inter_ptr,
        }
    }

    unsafe fn create_gc_cell<T>(&mut self, t: T) -> GcCell<T> where T: Sized + Trace {
        let (gc_cell_inter_ptr, mem_info_internal_ptr) = self.get_gc_cell_inter_ptr::<T>();
        let (gc_ptr, mem_info_gc_ptr) = self.get_ref_cell_gc_ptr::<T>();
        let gc = GcCell {
            internal_ptr: gc_cell_inter_ptr,
        };
        (*gc_ptr).borrow().info.has_root.set(false);
        (*gc_ptr).borrow_mut().t = t;
        (*gc.internal_ptr).ptr = gc_ptr;
        gc.reset_root();
        (*gc.internal_ptr).is_root.set(true);
        self.register_root(gc.internal_ptr, (mem_info_internal_ptr, mem_info_gc_ptr));
        gc
    }

    unsafe fn null_gc_cell<T>(&mut self) -> GcCell<T> where T: Sized + Trace {
        let (gc_inter_ptr, mem_info) = self.get_gc_cell_inter_ptr::<T>();
        (*gc_inter_ptr).ptr = std::ptr::null();
        GcCell {
            internal_ptr: gc_inter_ptr,
        }
    }

    fn register_root(&mut self, root_ptr: *const dyn Trace, mem: ((GcObjMem, Layout), (GcObjMem, Layout))) {
        self.vec.insert(root_ptr, mem);
    }

    fn unregister_root(&mut self, root_ptr: *const dyn Trace) {
        self.vec.remove(&root_ptr);
    }

    unsafe fn get_gc_inter_ptr<T>(&mut self) -> (*mut GcInternal<T>, (GcObjMem, Layout)) where T: Sized + Trace {
        let layout = Layout::new::<GcInternal<T>>();
        let mem = alloc(layout);
        let gc_inter_ptr: *mut GcInternal<T> = std::ptr::read(mem as *mut _);
        (gc_inter_ptr, (mem, layout))
    }

    unsafe fn get_gc_ptr<T>(&mut self) -> (*mut GcPtr<T>, (GcObjMem, Layout)) where T: Sized + Trace {
        let layout = Layout::new::<GcPtr<T>>();
        let mem = alloc(layout);
        let gc_ptr: *mut GcPtr<T> = std::ptr::read(mem as *mut _);
        (gc_ptr, (mem, layout))
    }

    unsafe fn get_gc_cell_inter_ptr<T>(&mut self) -> (*mut GcCellInternal<T>, (GcObjMem, Layout)) where T: Sized + Trace {
        let layout = Layout::new::<GcInternal<T>>();
        let mem = alloc(layout);
        let gc_cell_inter_ptr: *mut GcCellInternal<T> = std::ptr::read(mem as *mut _);
        (gc_cell_inter_ptr, (mem, layout))
    }

    unsafe fn get_ref_cell_gc_ptr<T>(&mut self) -> (*const RefCell<GcPtr<T>>, (*mut u8, Layout)) where T: Sized + Trace {
        let layout = Layout::new::<RefCell<GcPtr<T>>>();
        let mem = alloc(layout);
        let gc_ptr: *const RefCell<GcPtr<T>> = std::ptr::read(mem as *const _);
        (gc_ptr, (mem, layout))
    }

    unsafe fn collect(&mut self) {
        let mut collected_objects: Vec<*const dyn Trace> = Vec::new();
        for (gc_info, _) in &self.vec {
            let tracer = &(**gc_info);
            if tracer.is_root() {
                tracer.trace();
            }
        }
        for (gc_info, _) in &self.vec {
            let tracer = &(**gc_info);
            if !tracer.is_traceable() {
                collected_objects.push(*gc_info);
            } else {
                tracer.reset();
            }
        }
        for col in collected_objects {
            let del = self.vec[&col];
            dealloc((del.0).0, (del.0).1);
            dealloc((del.1).0, (del.1).1);
            self.vec.remove(&col);
        }
    }
}

// trait Strategy {
//     fn start(&mut self);
//     fn stop(&mut self);
// }
//
// struct BasicStrategy<F>
//     where F: Fn(),
//           F: Send + 'static {
//     is_active: AtomicBool,
//     task: Arc<F>,
//     task_join: Option<JoinHandle<()>>,
// }
//
// impl<F> BasicStrategy<F>
//     where F: Fn(),
//           F: Send + 'static {
//     fn new(f: Arc<F>) -> BasicStrategy<F> {
//         BasicStrategy {
//             is_active: AtomicBool::new(true),
//             task: f,
//             task_join: None,
//         }
//     }
// }
//
// impl<F> Strategy for BasicStrategy<F>
//     where F: Fn(),
//           F: Send + 'static {
//     fn start(&mut self) {
//         let handler = self.task.clone();
//         self.is_active.store(true, Ordering::Release);
//         self.task_join = Some(std::thread::spawn(move || {
//             (*handler)();
//         }));
//     }
//
//     fn stop(&mut self) {
//         self.is_active.store(false, Ordering::Release);
//         if let Some(task_join) = self.task_join {
//             task_join.join();
//         }
//     }
// }
//
// impl<F> Drop for BasicStrategy<F>
//     where F: Fn(),
//           F: Send + 'static {
//     fn drop(&mut self) {
//         self.stop();
//     }
// }

use std::cell::RefCell;
use std::ops::Deref;
use std::thread;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::rc::Rc;
use std::sync::Arc;
use std::borrow::BorrowMut;
thread_local! {
    pub static LOCAL_GC: RefCell<GarbageCollector> = RefCell::new(GarbageCollector::new());
    // static STRATEGY: Cell<BasicStrategy<Arc<&'static dyn Fn()>>> = {
    //     LOCAL_GC.with(move |gc| {
    //         Cell::new(BasicStrategy::new(Arc::new(move || {
    //
    //         })))
    //     })
    // };
}

#[cfg(test)]
mod tests {
    use crate::gc::{Gc, LOCAL_GC};

    #[test]
    fn one_object() {
        let one = Gc::new(1);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().vec.len(), 1);
        });
    }

    #[test]
    fn gc_collect_one_from_one() {
        {
            let one = Gc::new(1);
        }
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().vec.len(), 0);
        });
    }

    #[test]
    fn two_objects() {
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| {
            assert_eq!(gc.borrow_mut().vec.len(), 2);
        });
    }

    #[test]
    fn gc_collect_one_from_two() {
        let mut one = Gc::new(1);
        one = Gc::new(2);
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().collect();
            assert_eq!(gc.borrow_mut().vec.len(), 0);
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
            assert_eq!(gc.borrow_mut().vec.len(), 0);
        });
    }
}
