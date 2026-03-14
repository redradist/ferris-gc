use crate::gc::LocalGarbageCollector;
use crate::gc::sync::GlobalGarbageCollector;
use std::sync::{Once, RwLock};
use std::{thread, time};

use std::thread::JoinHandle;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// When set to `true`, `basic_gc_strategy_start()` becomes a no-op.
/// Used by `#[ferris_gc_main(strategy = "...")]` to prevent the basic
/// background thread from starting when a different strategy is configured.
pub static BASIC_STRATEGY_DISABLED: AtomicBool = AtomicBool::new(false);

/// Polling interval for the basic strategy background thread, in milliseconds.
/// Set this **before** the first allocation to change the default (500ms).
/// Used by `#[ferris_gc_main(poll_interval_ms = ...)]`.
pub static BASIC_POLL_INTERVAL_MS: AtomicU64 = AtomicU64::new(50);

lazy_static! {
    pub static ref BASIC_STRATEGY_LOCAL_GCS: RwLock<Vec<&'static LocalGarbageCollector>> =
        RwLock::new(Vec::new());
    pub static ref BASIC_STRATEGY_GLOBAL_GC: RwLock<Option<&'static GlobalGarbageCollector>> =
        RwLock::new(None);
    pub static ref APPLICATION_ACTIVE: AtomicBool = AtomicBool::new(true);
    pub static ref BACKGROUND_THREADS: RwLock<Vec<JoinHandle<()>>> = RwLock::new(Vec::new());
}
static START_BASIC_GC_STRATEGY: Once = Once::new();

pub struct ApplicationCleanup;
impl Drop for ApplicationCleanup {
    fn drop(&mut self) {
        let app_active = &(*APPLICATION_ACTIVE);
        app_active.store(false, Ordering::Release);
        let mut bthreads = (*BACKGROUND_THREADS).write().unwrap();
        while let Some(bthread) = (*bthreads).pop() {
            let _ = bthread.join();
        }
    }
}

pub fn basic_gc_strategy_start() {
    if BASIC_STRATEGY_DISABLED.load(Ordering::Acquire) {
        return;
    }
    START_BASIC_GC_STRATEGY.call_once(|| {
        let bthreads = &(*BACKGROUND_THREADS);
        bthreads.write().unwrap().push(thread::spawn(move || {
            let app_active = &(*APPLICATION_ACTIVE);
            while app_active.load(Ordering::Acquire) {
                let interval =
                    time::Duration::from_millis(BASIC_POLL_INTERVAL_MS.load(Ordering::Relaxed));
                thread::sleep(interval);
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
                        for local_gc in local_gcs.iter() {
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
                for local_gc in local_gcs.iter() {
                    local_gc.collect();
                }
            }
        }));
    });
}
