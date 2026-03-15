use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::gc::LocalGarbageCollector;
use crate::gc::sync::GlobalGarbageCollector;
use crate::generation::Generation;

/// Configuration for the threshold-based GC strategy.
pub struct ThresholdConfig {
    /// Number of allocations before triggering a Gen0 collection.
    pub gen0_threshold: usize,
    /// Number of Gen0 collections between each Gen1 collection.
    pub gen0_collections_per_gen1: u32,
    /// Number of Gen1 collections between each Gen2 collection.
    pub gen1_collections_per_gen2: u32,
    /// Polling interval for the background thread.
    pub poll_interval: Duration,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        ThresholdConfig {
            gen0_threshold: 100,
            gen0_collections_per_gen1: 5,
            gen1_collections_per_gen2: 5,
            poll_interval: Duration::from_millis(50),
        }
    }
}

/// Creates a strategy closure for a local threshold-based collector.
///
/// The returned closure, when called, spawns a background thread that polls
/// `allocation_count` and triggers generational collections based on the config.
#[allow(clippy::type_complexity)]
pub fn threshold_local_strategy(
    config: ThresholdConfig,
) -> impl FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> + 'static
{
    let config = std::sync::Arc::new(config);
    move |gc: &'static LocalGarbageCollector,
          is_active: &'static AtomicBool|
          -> Option<JoinHandle<()>> {
        let config = config.clone();
        Some(thread::spawn(move || {
            let mut gen0_count: u32 = 0;
            let mut gen1_count: u32 = 0;
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config.poll_interval);
                let allocs = gc.core.allocation_count.load(Ordering::Relaxed);
                if allocs >= config.gen0_threshold {
                    unsafe {
                        gc.core.collect_generation(Generation::Gen0);
                    }
                    gen0_count += 1;
                    if gen0_count >= config.gen0_collections_per_gen1 {
                        gen0_count = 0;
                        unsafe {
                            gc.core.collect_generation(Generation::Gen1);
                        }
                        gen1_count += 1;
                        if gen1_count >= config.gen1_collections_per_gen2 {
                            gen1_count = 0;
                            unsafe {
                                gc.core.collect_generation(Generation::Gen2);
                            }
                        }
                    }
                }
            }
            // Final full collection on shutdown
            unsafe {
                gc.core.collect_generation(Generation::Gen2);
            }
        }))
    }
}

/// Creates a strategy closure for a global threshold-based collector.
///
/// The returned closure, when called, spawns a background thread that polls
/// `allocation_count` and triggers generational collections based on the config.
#[allow(clippy::type_complexity)]
pub fn threshold_global_strategy(
    config: ThresholdConfig,
) -> impl FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> + 'static
{
    let config = std::sync::Arc::new(config);
    move |gc: &'static GlobalGarbageCollector,
          is_active: &'static AtomicBool|
          -> Option<JoinHandle<()>> {
        let config = config.clone();
        Some(thread::spawn(move || {
            let mut gen0_count: u32 = 0;
            let mut gen1_count: u32 = 0;
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config.poll_interval);
                let allocs = gc.core.allocation_count.load(Ordering::Relaxed);
                if allocs >= config.gen0_threshold {
                    unsafe {
                        gc.core.collect_generation(Generation::Gen0);
                    }
                    gen0_count += 1;
                    if gen0_count >= config.gen0_collections_per_gen1 {
                        gen0_count = 0;
                        unsafe {
                            gc.core.collect_generation(Generation::Gen1);
                        }
                        gen1_count += 1;
                        if gen1_count >= config.gen1_collections_per_gen2 {
                            gen1_count = 0;
                            unsafe {
                                gc.core.collect_generation(Generation::Gen2);
                            }
                        }
                    }
                }
            }
            // Final full collection on shutdown
            unsafe {
                gc.core.collect_generation(Generation::Gen2);
            }
        }))
    }
}
