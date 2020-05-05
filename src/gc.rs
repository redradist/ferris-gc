pub mod sync;

use std::cell::Cell;
use std::collections::HashMap;
use std::alloc::{alloc, Layout, dealloc};

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
    u8, i8, u16, i16, u32, i32, u64, i64, u128, i128
);

struct GcInfo {
    is_root: Cell<bool>,
    has_root: Cell<bool>,
}

struct GcPtr<T> where T: 'static + Sized + Trace {
    info: GcInfo,
    t: Option<T>,
}

impl<T> Trace for GcPtr<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        self.info.is_root.get()
    }

    fn reset_root(&self) {
        self.info.is_root.set(false);
        if let Some(t) = &self.t {
            t.reset_root();
        }
    }

    fn trace(&self) {
        self.info.has_root.set(true);
        if let Some(t) = &self.t {
            t.trace();
        }
    }

    fn reset(&self) {
        self.info.has_root.set(false);
        if let Some(t) = &self.t {
            t.reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.info.has_root.get()
    }
}

pub struct Gc<T> where T: 'static + Sized + Trace {
    ptr: *const GcPtr<T>,
}

impl<T> Clone for Gc<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        let gc = Gc {
            ptr: self.ptr
        };
        unsafe {
            (*gc.ptr).info.is_root.set(true);
        }
        gc
    }

    fn clone_from(&mut self, source: &Self) {
        let is_root_prev = self.is_root();
        self.ptr = source.ptr;
        unsafe {
            (*self.ptr).info.is_root.set(is_root_prev);
        }
    }
}

impl<T> Gc<T> where T: Sized + Trace {
    pub fn new(t: T) -> Gc<T> {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().create_gc(t)
        })
    }
}

impl<T> Drop for Gc<T> where T: Sized + Trace {
    fn drop(&mut self) {
        self.reset_root();
    }
}

impl<T> Trace for Gc<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        unsafe {
            (*self.ptr).is_root()
        }
    }

    fn reset_root(&self) {
        unsafe {
            (*self.ptr).reset_root();
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

pub struct GcCell<T> where T: 'static + Sized + Trace {
    ptr: *const RefCell<GcPtr<T>>,
}

impl<T> Clone for GcCell<T> where T: 'static + Sized + Trace {
    fn clone(&self) -> Self {
        unimplemented!()
    }

    fn clone_from(&mut self, source: &Self) {

    }
}

impl<T> GcCell<T> where T: Sized + Trace {
    pub fn new(t: T) -> GcCell<T> {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().create_gc_cell(t)
        })
    }
}

impl<T> Trace for GcCell<T> where T: Sized + Trace {
    fn is_root(&self) -> bool {
        unsafe {
            (*self.ptr).borrow().is_root()
        }
    }

    fn reset_root(&self) {
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

pub struct GarbageCollector {
    vec: HashMap<*const dyn Trace, (*mut u8, Layout)>
}

impl GarbageCollector {
    fn new() -> GarbageCollector {
        GarbageCollector { vec: HashMap::new() }
    }

    pub fn strategy() {

    }

    unsafe fn create_gc<T>(&mut self, t: T) -> Gc<T> where T: Sized + Trace {
        let gc = Gc {
            ptr: self.get_gc_ptr(t),
        };
        gc.reset_root();
        unsafe {
            (*gc.ptr).info.is_root.set(true);
        }
        gc
    }

    unsafe fn create_gc_cell<T>(&mut self, t: T) -> GcCell<T> where T: Sized + Trace {
        let gc_cell = GcCell {
            ptr: self.get_ref_cell_gc_ptr(t),
        };
        gc_cell.reset_root();
        unsafe {
            (*gc_cell.ptr).borrow().info.is_root.set(true);
        }
        gc_cell
    }

    unsafe fn get_gc_ptr<T>(&mut self, t: T) -> *mut GcPtr<T> where T: Sized + Trace {
        let layout = Layout::new::<GcPtr<T>>();
        let mem = alloc(layout);
        let gc_ptr: *mut GcPtr<T> = std::ptr::read(mem as *const _);
        (*gc_ptr).info.is_root.set(true);
        (*gc_ptr).info.has_root.set(false);
        (*gc_ptr).t = Some(t);
        self.vec.insert(gc_ptr, (mem, layout));
        gc_ptr
    }

    unsafe fn get_ref_cell_gc_ptr<T>(&mut self, t: T) -> *const RefCell<GcPtr<T>> where T: Sized + Trace {
        let layout = Layout::new::<RefCell<GcPtr<T>>>();
        let mem = alloc(layout);
        let gc_ptr: *const RefCell<GcPtr<T>> = std::ptr::read(mem as *const _);
        (*gc_ptr).borrow_mut().info.is_root.set(true);
        (*gc_ptr).borrow_mut().info.has_root.set(false);
        (*gc_ptr).borrow_mut().t = Some(t);
        self.vec.insert((*gc_ptr).as_ptr(), (mem, layout));
        gc_ptr
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
            dealloc(del.0, del.1);
            self.vec.remove(&col);
        }
    }
}

use std::cell::RefCell;
thread_local! {
    pub static LOCAL_GC: RefCell<GarbageCollector> = RefCell::new(GarbageCollector::new());
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
        LOCAL_GC.with(move |gc| unsafe {
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
