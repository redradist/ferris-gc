use std::sync::{RwLock, Once};
use crate::gc::LocalGarbageCollector;
use crate::gc::sync::GlobalGarbageCollector;
use std::{thread, time};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::borrow::BorrowMut;
use std::sync::atomic::{AtomicBool, Ordering};

lazy_static! {
    pub static ref BASIC_STRATEGY_LOCAL_GCS: RwLock<Vec<&'static LocalGarbageCollector>> = {
        RwLock::new(Vec::new())
    };
    pub static ref BASIC_STRATEGY_GLOBAL_GC: RwLock<Option<&'static GlobalGarbageCollector>> = {
        RwLock::new(None)
    };
    pub static ref APPLICATION_ACTIVE: AtomicBool = {
        AtomicBool::new(true)
    };
    pub static ref BACKGROUND_THREADS: RwLock<Vec<JoinHandle<()>>> = {
        RwLock::new(Vec::new())
    };
}
static START_BASIC_GC_STRATEGY: Once = Once::new();

pub struct ApplicationCleanup;
impl Drop for ApplicationCleanup {
    fn drop(&mut self) {
        let app_active = &(*APPLICATION_ACTIVE);
        app_active.store(false, Ordering::Release);
        let mut bthreads = (&*BACKGROUND_THREADS).write().unwrap();
        while ((*bthreads).len() > 0) {
            let bthread = (*bthreads).pop().unwrap();
            bthread.join();
        }
    }
}

pub fn basic_gc_strategy_start() {
    START_BASIC_GC_STRATEGY.call_once(|| {
        let bthreads = &(*BACKGROUND_THREADS);
        bthreads.write().unwrap().push(
            thread::spawn(move || {
                let app_active = &(*APPLICATION_ACTIVE);
                while app_active.load(Ordering::Acquire) {
                    let ten_secs = time::Duration::from_millis(500);
                    thread::sleep(ten_secs);
                    {
                        if let Some(global_gc) = *(*BASIC_STRATEGY_GLOBAL_GC).read().unwrap() {
                            unsafe {
                                global_gc.collect();
                            }
                        }
                    }
                    {
                        let local_gcs_read_guard = (*BASIC_STRATEGY_LOCAL_GCS).read().unwrap();
                        let local_gcs = &(*local_gcs_read_guard);
                        unsafe {
                            for local_gc in local_gcs.into_iter() {
                                local_gc.collect();
                            }
                        }
                    }
                }
                if let Some(global_gc) = *(*BASIC_STRATEGY_GLOBAL_GC).read().unwrap() {
                    unsafe {
                        global_gc.collect();
                    }
                }
                let local_gcs_read_guard = (*BASIC_STRATEGY_LOCAL_GCS).read().unwrap();
                let local_gcs = &(*local_gcs_read_guard);
                unsafe {
                    for local_gc in local_gcs.into_iter() {
                        local_gc.collect_all();
                    }
                }
            })
        );
    });
}