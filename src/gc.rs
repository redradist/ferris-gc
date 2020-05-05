pub mod sync;

use std::cell::Cell;
use std::collections::HashMap;
use std::alloc::{alloc, Layout, dealloc};

pub trait Trace {
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
                fn trace(&self) {
                }
                fn reset(&self) {
                }
                fn is_traceable(&self) -> bool {
                    true
                }
            }
        )*
    };
}

primitive_types!(
    u8, i8, u16, i16, u32, i32, u64, i64, u128, i128
);

struct GcInfo {
    has_root: Cell<bool>,
}

struct GcPtr<T> where T: 'static + Sized + Trace {
    ptr: GcInfo,
    t: T,
}

impl<T> Trace for GcPtr<T> where T: Sized + Trace {
    fn trace(&self) {
        self.ptr.has_root.set(true);
        self.t.trace();
    }

    fn reset(&self) {
        self.ptr.has_root.set(false);
        self.t.reset();
    }

    fn is_traceable(&self) -> bool {
        self.ptr.has_root.get()
    }
}

pub struct Gc<T> where T: 'static + Sized + Trace {
    ptr: *const GcPtr<T>,
}

impl<T> Gc<T> where T: Sized + Trace {
    pub fn new(t: T) -> Gc<T> {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().create_gc(t)
        })
    }
}

impl<T> Trace for Gc<T> where T: Sized + Trace {
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
    ptr: *mut GcPtr<T>,
}

impl<T> GcCell<T> where T: Sized + Trace {
    pub fn new(t: T) -> GcCell<T> {
        LOCAL_GC.with(move |gc| unsafe {
            gc.borrow_mut().create_gc_cell(t)
        })
    }
}

impl<T> Trace for GcCell<T> where T: Sized + Trace {
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
        Gc {
            ptr: self.get_gc_ptr(t),
        }
    }

    unsafe fn create_gc_cell<T>(&mut self, t: T) -> GcCell<T> where T: Sized + Trace {
        GcCell {
            ptr: self.get_gc_ptr(t),
        }
    }

    unsafe fn get_gc_ptr<T>(&mut self, t: T) -> *mut GcPtr<T> where T: Sized + Trace {
        let layout = Layout::new::<GcPtr<T>>();
        let mem = alloc(layout);
        let gc_ptr: *mut GcPtr<T> = std::ptr::read(mem as *const _);
        (*gc_ptr).t = t;
        self.vec.insert(gc_ptr, (mem, layout));
        gc_ptr
    }

    unsafe fn collect(&mut self) {
        let mut collected_objects = Vec::new();
        for (gc_info, _) in &self.vec {
            let tracer = &(**gc_info);
            if !tracer.is_traceable() {
                tracer.trace();
            }
        }
        for (gc_info, _) in &self.vec {
            let tracer = &(**gc_info);
            if !tracer.is_traceable() {
                collected_objects.push(gc_info);
            } else {
                tracer.reset();
            }
        }
        for col in collected_objects {
            let del = self.vec[col];
            dealloc(del.0, del.1);
        }
    }
}

use std::cell::RefCell;
thread_local! {
    pub static LOCAL_GC: RefCell<GarbageCollector> = RefCell::new(GarbageCollector::new());
}