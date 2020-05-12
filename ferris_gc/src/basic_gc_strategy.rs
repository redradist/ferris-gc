use std::sync::{RwLock, Once};
use crate::gc::LocalGarbageCollector;
use crate::gc::sync::GlobalGarbageCollector;
use std::{thread, time};

lazy_static! {
    pub static ref BASIC_STRATEGY_LOCAL_GCS: RwLock<Vec<&'static LocalGarbageCollector>> = {
        RwLock::new(Vec::new())
    };
    pub static ref BASIC_STRATEGY_GLOBAL_GC: RwLock<Option<&'static GlobalGarbageCollector>> = {
        RwLock::new(None)
    };
}
static START_BASIC_GC_STRATEGY: Once = Once::new();

pub fn basic_gc_strategy_start() {
    START_BASIC_GC_STRATEGY.call_once(|| {
        thread::spawn(move || {
            loop {
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
        });
    });
}