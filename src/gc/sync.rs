use std::collections::HashMap;
use std::alloc::{alloc, Layout, dealloc};
use crate::gc::Trace;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use std::borrow::BorrowMut;

struct GcInfo {
    ref_count: AtomicUsize,
}

struct GcPtr<T> where T: 'static + Sized + Trace {
    ptr: GcInfo,
    t: T,
}

impl<T> Trace for GcPtr<T> where T: Sized + Trace {
    fn trace(&self) {
        self.ptr.ref_count.fetch_add(1, Ordering::AcqRel);
        self.t.trace();
    }

    fn reset(&self) {
        self.ptr.ref_count.fetch_sub(1, Ordering::AcqRel);
        self.t.reset();
    }

    fn is_traceable(&self) -> bool {
        self.ptr.ref_count.load(Ordering::Acquire) > 0
    }
}

pub struct Gc<T> where T: 'static + Sized + Trace {
    ptr: *const GcPtr<T>,
}

impl<T> Gc<T> where T: Sized + Trace {
    pub fn new(t: T) -> Gc<T> {
        unsafe {
            let mut writer = (*GLOBAL_GC).write().unwrap();
            writer.create_gc(t)
        }
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
        unsafe {
            let mut writer = (*GLOBAL_GC).write().unwrap();
            writer.create_gc_cell(t)
        }
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
    vec: RwLock<HashMap<*const dyn Trace, (*mut u8, Layout)>>
}

unsafe impl Sync for GarbageCollector {}
unsafe impl Send for GarbageCollector {}

impl GarbageCollector {
    fn new() -> GarbageCollector {
        GarbageCollector { vec: RwLock::new(HashMap::new()) }
    }

    unsafe fn create_gc<T>(&self, t: T) -> Gc<T> where T: Sized + Trace {
        Gc {
            ptr: self.get_gc_ptr(t),
        }
    }

    unsafe fn create_gc_cell<T>(&self, t: T) -> GcCell<T> where T: Sized + Trace {
        GcCell {
            ptr: self.get_gc_ptr(t),
        }
    }

    unsafe fn get_gc_ptr<T>(&self, t: T) -> *mut GcPtr<T> where T: Sized + Trace {
        let layout = Layout::new::<GcPtr<T>>();
        let mem = alloc(layout);
        let gc_ptr: *mut GcPtr<T> = std::ptr::read(mem as *const _);
        (*gc_ptr).t = t;
        let mut w_vec = self.vec.write().unwrap();
        (*w_vec).insert(gc_ptr, (mem, layout));
        gc_ptr
    }

    unsafe fn collect(&self) {
        let mut collected_objects: Vec<*const dyn Trace> = Vec::new();
        {
            let r_vec = self.vec.read().unwrap();
            for (gc_info, _) in &*r_vec {
                let tracer = &(**gc_info);
                if !tracer.is_traceable() {
                    tracer.trace();
                }
            }
            for (gc_info, _) in &*r_vec {
                let tracer = &(**gc_info);
                if !tracer.is_traceable() {
                    println!("Try to delete !!");
                    collected_objects.push(*gc_info);
                } else {
                    println!("Try to delete !!");
                    tracer.reset();
                }
            }
        }
        {
            println!("Try to delete !!");
            let mut w_vec = self.vec.write().unwrap();
            for col in collected_objects {
                let del = (*w_vec).get(&col).unwrap();
                dealloc(del.0, del.1);
                w_vec.remove(&col);
                println!("Delete object !!");
            }
        }
    }
}

lazy_static! {
    pub static ref GLOBAL_GC: RwLock<GarbageCollector> = {
        RwLock::new(GarbageCollector::new())
    };
}

#[cfg(test)]
mod tests {
    use crate::gc::sync::{Gc, GLOBAL_GC};

    #[test]
    fn it_works() {
        {
            let df = Gc::new(1);
        }
        let mut writer = (*GLOBAL_GC).write().unwrap();
        let mut global_gc = &(*writer);
        unsafe {
            writer.collect();
        }
        let vec = writer.vec.write().unwrap();
        assert_eq!(vec.len(), 0);
    }
}