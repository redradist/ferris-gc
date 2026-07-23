use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::gc::LocalGarbageCollector;
use crate::gc::sync::GlobalGarbageCollector;
use crate::generation::Generation;

/// Configuration for the background (.NET-style) GC strategy.
///
/// Gen0/Gen1 collections run in a **foreground** thread (short STW pauses),
/// while Gen2 collection runs **concurrently** in a dedicated background thread
/// using the snapshot-based concurrent marking infrastructure. This dramatically
/// reduces pause times for large heaps.
pub struct BackgroundConfig {
    /// Number of allocations before triggering a Gen0 collection (foreground).
    pub gen0_threshold: usize,
    /// Number of Gen0 collections between each Gen1 collection.
    pub gen0_collections_per_gen1: u32,
    /// Number of Gen1 collections between each Gen2 collection.
    pub gen1_collections_per_gen2: u32,
    /// Polling interval for the foreground strategy thread.
    pub poll_interval: Duration,
    /// Heap occupancy ratio that triggers background Gen2 marking.
    /// When `current_heap_size > peak_heap_size * occupancy_trigger`, start background Gen2.
    pub gen2_occupancy_trigger: f64,
    /// Budget per concurrent mark step (number of objects to scan).
    pub gen2_mark_step_budget: usize,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        BackgroundConfig {
            gen0_threshold: 100,
            gen0_collections_per_gen1: 5,
            gen1_collections_per_gen2: 5,
            poll_interval: Duration::from_millis(50),
            gen2_occupancy_trigger: 0.75,
            gen2_mark_step_budget: 64,
        }
    }
}

/// Creates a strategy closure for a local background collector.
///
/// The returned closure spawns **two** threads:
/// - **Foreground thread**: polls `allocation_count` and triggers Gen0/Gen1 STW collections.
/// - **Background Gen2 thread**: monitors heap occupancy and runs concurrent Gen2 collection
///   (`begin_concurrent_collection` → `concurrent_mark_step` loop → `finish_collection`).
///
/// Both threads share the same `is_active` flag and exit when it becomes `false`.
/// The foreground thread joins the background thread on exit.
#[allow(clippy::type_complexity)]
pub fn background_local_strategy(
    config: BackgroundConfig,
) -> impl FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> + 'static
{
    let config = Arc::new(config);

    move |gc: &'static LocalGarbageCollector,
          _is_active: &'static AtomicBool|
          -> Option<JoinHandle<()>> {
        // A thread-local GC's mutator path accesses `gc_maps` locklessly
        // (`gc_maps_unsync`) for speed, so collecting it from any thread
        // other than its owner is a data race on non-atomic
        // Cell/RefCell/TLAB state and segfaults under load. Concurrent /
        // background collection is therefore only sound for the thread-safe
        // global GC (`background_global_strategy`). For a local GC we fall
        // back to allocation-triggered collection on the owning thread
        // (`LocalGarbageCollector::maybe_collect`), honoring the configured
        // Gen0 threshold; no collector thread is spawned.
        gc.core
            .alloc_threshold
            .store(config.gen0_threshold, Ordering::Relaxed);
        None
    }
}

/// Creates a strategy closure for a global background collector.
///
/// Same two-thread architecture as the local variant, but operates on the
/// global (thread-safe) `GlobalGarbageCollector`.
#[allow(clippy::type_complexity)]
pub fn background_global_strategy(
    config: BackgroundConfig,
) -> impl FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> + 'static
{
    let config = Arc::new(config);

    move |gc: &'static GlobalGarbageCollector,
          is_active: &'static AtomicBool|
          -> Option<JoinHandle<()>> {
        let config_bg = config.clone();
        let config_fg = config.clone();

        // Spawn background Gen2 thread
        let bg_handle: Arc<Mutex<Option<JoinHandle<()>>>> = Arc::new(Mutex::new(None));
        let bg_handle_fg = bg_handle.clone();

        let bg_thread = thread::spawn(move || {
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config_bg.poll_interval);

                // Check heap occupancy to decide whether to trigger Gen2
                let current = gc.core.current_heap_size.load(Ordering::Relaxed);
                let peak = gc.core.peak_heap_size.load(Ordering::Relaxed);

                if peak > 0 && current as f64 > peak as f64 * config_bg.gen2_occupancy_trigger {
                    unsafe {
                        gc.core.begin_concurrent_collection(Generation::Gen2);
                    }

                    loop {
                        if !is_active.load(Ordering::Acquire) {
                            break;
                        }
                        let done = gc
                            .core
                            .concurrent_mark_step(config_bg.gen2_mark_step_budget);
                        if done {
                            break;
                        }
                        thread::yield_now();
                    }

                    unsafe {
                        gc.core.finish_collection();
                    }
                }
            }

            // Final concurrent Gen2 collection on shutdown
            unsafe {
                gc.core.begin_concurrent_collection(Generation::Gen2);
            }
            while !gc
                .core
                .concurrent_mark_step(config_bg.gen2_mark_step_budget)
            {}
            unsafe {
                gc.core.finish_collection();
            }
        });

        {
            let mut handle = bg_handle.lock().unwrap_or_else(|e| e.into_inner());
            *handle = Some(bg_thread);
        }

        let config_fg2 = config_fg.clone();

        Some(thread::spawn(move || {
            let mut gen0_count: u32 = 0;
            let mut gen1_count: u32 = 0;
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config_fg2.poll_interval);
                let allocs = gc.core.allocation_count.load(Ordering::Relaxed);
                if allocs >= config_fg2.gen0_threshold {
                    unsafe {
                        gc.core.collect_generation(Generation::Gen0);
                    }
                    gen0_count += 1;
                    if gen0_count >= config_fg2.gen0_collections_per_gen1 {
                        gen0_count = 0;
                        unsafe {
                            gc.core.collect_generation(Generation::Gen1);
                        }
                        gen1_count += 1;
                        if gen1_count >= config_fg2.gen1_collections_per_gen2 {
                            gen1_count = 0;
                        }
                    }
                }
            }
            unsafe {
                gc.core.collect_generation(Generation::Gen1);
            }
            // Join the background Gen2 thread before exiting
            let mut handle = bg_handle_fg.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(h) = handle.take() {
                let _ = h.join();
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = BackgroundConfig::default();
        assert_eq!(config.gen0_threshold, 100);
        assert_eq!(config.gen0_collections_per_gen1, 5);
        assert_eq!(config.gen1_collections_per_gen2, 5);
        assert_eq!(config.poll_interval, Duration::from_millis(50));
        assert!((config.gen2_occupancy_trigger - 0.75).abs() < f64::EPSILON);
        assert_eq!(config.gen2_mark_step_budget, 64);
    }

    #[test]
    fn custom_config() {
        let config = BackgroundConfig {
            gen0_threshold: 200,
            gen2_occupancy_trigger: 0.5,
            gen2_mark_step_budget: 128,
            ..Default::default()
        };
        assert_eq!(config.gen0_threshold, 200);
        assert!((config.gen2_occupancy_trigger - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.gen2_mark_step_budget, 128);
        // Defaults preserved
        assert_eq!(config.gen0_collections_per_gen1, 5);
        assert_eq!(config.gen1_collections_per_gen2, 5);
    }

    #[test]
    fn background_local_strategy_returns_closure() {
        let config = BackgroundConfig::default();
        let _strategy = background_local_strategy(config);
    }

    #[test]
    fn background_global_strategy_returns_closure() {
        let config = BackgroundConfig::default();
        let _strategy = background_global_strategy(config);
    }
}
