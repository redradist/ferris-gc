use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::gc::LocalGarbageCollector;
use crate::gc::sync::GlobalGarbageCollector;
use crate::generation::Generation;

/// Configuration for the G1 (Garbage First) GC strategy.
///
/// Combines G1-style region selection (collecting the dirtiest regions first)
/// with background Gen2 concurrent marking. The user specifies a target pause
/// time and the GC auto-tunes to meet it.
///
/// # Two-Thread Architecture
///
/// **Foreground thread** (short pauses):
/// - Polls `allocation_count` and triggers `collect_garbage_first(pause_target)`
///   when the young-gen threshold is reached. This performs a mixed collection,
///   sweeping the dirtiest regions within the time budget.
///
/// **Background Gen2 thread** (concurrent marking):
/// - Monitors heap occupancy and starts concurrent Gen2 mark cycles when it
///   exceeds `initiating_heap_occupancy_percent`. After the mark cycle completes,
///   the foreground's next `collect_garbage_first()` benefits from fresh mark
///   data for region prioritization.
pub struct G1Config {
    /// Target maximum pause time for collections.
    pub pause_target: Duration,
    /// Allocation count threshold for triggering young-gen collection.
    pub young_gen_threshold: usize,
    /// Heap occupancy ratio that triggers concurrent Gen2 marking.
    pub initiating_heap_occupancy_percent: f64,
    /// Budget per concurrent mark step (objects to scan).
    pub concurrent_mark_budget: usize,
    /// Polling interval for the strategy threads.
    pub poll_interval: Duration,
}

impl Default for G1Config {
    fn default() -> Self {
        G1Config {
            pause_target: Duration::from_millis(10),
            young_gen_threshold: 100,
            initiating_heap_occupancy_percent: 0.45,
            concurrent_mark_budget: 64,
            poll_interval: Duration::from_millis(50),
        }
    }
}

/// Creates start/stop closures for a local G1 strategy.
///
/// The start closure spawns **two** threads:
/// - **Foreground thread**: polls `allocation_count` and triggers
///   `collect_garbage_first(pause_target)` — a mixed collection that sweeps
///   the dirtiest regions within the time budget.
/// - **Background Gen2 thread**: monitors heap occupancy and runs concurrent
///   Gen2 collection (`begin_concurrent_collection` -> `concurrent_mark_step`
///   loop -> `finish_collection`).
///
/// Both threads share the same `is_active` flag and exit when it becomes `false`.
/// The background thread handle is stored in shared state so the stop closure
/// can join it.
#[allow(clippy::type_complexity)]
pub fn g1_local_start(
    config: G1Config,
) -> (
    impl FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> + 'static,
    impl FnMut(&'static LocalGarbageCollector) + 'static,
) {
    let config = Arc::new(config);
    let bg_handle: Arc<Mutex<Option<JoinHandle<()>>>> = Arc::new(Mutex::new(None));

    let config_start = config.clone();
    let bg_handle_start = bg_handle.clone();

    let start_fn = move |gc: &'static LocalGarbageCollector,
                         is_active: &'static AtomicBool|
          -> Option<JoinHandle<()>> {
        let config_bg = config_start.clone();

        // Spawn background Gen2 thread
        let bg_thread = thread::spawn(move || {
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config_bg.poll_interval);

                // Check heap occupancy to decide whether to trigger Gen2
                let current = gc.core.current_heap_size.load(Ordering::Relaxed);
                let peak = gc.core.peak_heap_size.load(Ordering::Relaxed);

                if peak > 0
                    && current as f64 > peak as f64 * config_bg.initiating_heap_occupancy_percent
                {
                    // Begin concurrent Gen2 collection (brief STW for snapshot)
                    unsafe {
                        gc.core.begin_concurrent_collection(Generation::Gen2);
                    }

                    // Concurrent mark steps -- NO STW lock needed
                    loop {
                        if !is_active.load(Ordering::Acquire) {
                            break;
                        }
                        let done = gc
                            .core
                            .concurrent_mark_step(config_bg.concurrent_mark_budget);
                        if done {
                            break;
                        }
                        // Yield between mark steps to allow foreground collections
                        thread::yield_now();
                    }

                    // Finish collection (brief STW for re-mark + sweep)
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
                .concurrent_mark_step(config_bg.concurrent_mark_budget)
            {}
            unsafe {
                gc.core.finish_collection();
            }
        });

        // Store background thread handle for later joining
        {
            let mut handle = bg_handle_start.lock().unwrap_or_else(|e| e.into_inner());
            *handle = Some(bg_thread);
        }

        let config_fg = config_start.clone();

        // Spawn foreground G1 thread -- returned as the main strategy handle
        Some(thread::spawn(move || {
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config_fg.poll_interval);
                let allocs = gc.core.allocation_count.load(Ordering::Relaxed);
                if allocs >= config_fg.young_gen_threshold {
                    // G1-style mixed collection: sweep dirtiest regions within
                    // the pause target budget.
                    unsafe {
                        gc.core.collect_garbage_first(config_fg.pause_target);
                    }
                }
            }
            // Final G1 collection on shutdown with a generous pause target
            unsafe {
                gc.core.collect_garbage_first(Duration::from_secs(10));
            }
        }))
    };

    let stop_fn = move |_gc: &'static LocalGarbageCollector| {
        // Join the background Gen2 thread
        let mut handle = bg_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = handle.take() {
            let _ = h.join();
        }
    };

    (start_fn, stop_fn)
}

/// Creates start/stop closures for a global G1 strategy.
///
/// Same two-thread architecture as the local variant, but operates on the
/// global (thread-safe) `GlobalGarbageCollector`.
#[allow(clippy::type_complexity)]
pub fn g1_global_start(
    config: G1Config,
) -> (
    impl FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> + 'static,
    impl FnMut(&'static GlobalGarbageCollector) + 'static,
) {
    let config = Arc::new(config);
    let bg_handle: Arc<Mutex<Option<JoinHandle<()>>>> = Arc::new(Mutex::new(None));

    let config_start = config.clone();
    let bg_handle_start = bg_handle.clone();

    let start_fn = move |gc: &'static GlobalGarbageCollector,
                         is_active: &'static AtomicBool|
          -> Option<JoinHandle<()>> {
        let config_bg = config_start.clone();

        // Spawn background Gen2 thread
        let bg_thread = thread::spawn(move || {
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config_bg.poll_interval);

                // Check heap occupancy to decide whether to trigger Gen2
                let current = gc.core.current_heap_size.load(Ordering::Relaxed);
                let peak = gc.core.peak_heap_size.load(Ordering::Relaxed);

                if peak > 0
                    && current as f64 > peak as f64 * config_bg.initiating_heap_occupancy_percent
                {
                    // Begin concurrent Gen2 collection (brief STW for snapshot)
                    unsafe {
                        gc.core.begin_concurrent_collection(Generation::Gen2);
                    }

                    // Concurrent mark steps -- NO STW lock needed
                    loop {
                        if !is_active.load(Ordering::Acquire) {
                            break;
                        }
                        let done = gc
                            .core
                            .concurrent_mark_step(config_bg.concurrent_mark_budget);
                        if done {
                            break;
                        }
                        // Yield between mark steps to allow foreground collections
                        thread::yield_now();
                    }

                    // Finish collection (brief STW for re-mark + sweep)
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
                .concurrent_mark_step(config_bg.concurrent_mark_budget)
            {}
            unsafe {
                gc.core.finish_collection();
            }
        });

        // Store background thread handle for later joining
        {
            let mut handle = bg_handle_start.lock().unwrap_or_else(|e| e.into_inner());
            *handle = Some(bg_thread);
        }

        let config_fg = config_start.clone();

        // Spawn foreground G1 thread -- returned as the main strategy handle
        Some(thread::spawn(move || {
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config_fg.poll_interval);
                let allocs = gc.core.allocation_count.load(Ordering::Relaxed);
                if allocs >= config_fg.young_gen_threshold {
                    // G1-style mixed collection: sweep dirtiest regions within
                    // the pause target budget.
                    unsafe {
                        gc.core.collect_garbage_first(config_fg.pause_target);
                    }
                }
            }
            // Final G1 collection on shutdown with a generous pause target
            unsafe {
                gc.core.collect_garbage_first(Duration::from_secs(10));
            }
        }))
    };

    let stop_fn = move |_gc: &'static GlobalGarbageCollector| {
        // Join the background Gen2 thread
        let mut handle = bg_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = handle.take() {
            let _ = h.join();
        }
    };

    (start_fn, stop_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = G1Config::default();
        assert_eq!(config.pause_target, Duration::from_millis(10));
        assert_eq!(config.young_gen_threshold, 100);
        assert!((config.initiating_heap_occupancy_percent - 0.45).abs() < f64::EPSILON);
        assert_eq!(config.concurrent_mark_budget, 64);
        assert_eq!(config.poll_interval, Duration::from_millis(50));
    }

    #[test]
    fn custom_config() {
        let config = G1Config {
            pause_target: Duration::from_millis(20),
            young_gen_threshold: 200,
            initiating_heap_occupancy_percent: 0.6,
            concurrent_mark_budget: 128,
            ..Default::default()
        };
        assert_eq!(config.pause_target, Duration::from_millis(20));
        assert_eq!(config.young_gen_threshold, 200);
        assert!((config.initiating_heap_occupancy_percent - 0.6).abs() < f64::EPSILON);
        assert_eq!(config.concurrent_mark_budget, 128);
        // Default preserved
        assert_eq!(config.poll_interval, Duration::from_millis(50));
    }

    #[test]
    fn g1_local_start_returns_closures() {
        let config = G1Config::default();
        let (_start_fn, _stop_fn) = g1_local_start(config);
        // Verify closures are created without panic
    }

    #[test]
    fn g1_global_start_returns_closures() {
        let config = G1Config::default();
        let (_start_fn, _stop_fn) = g1_global_start(config);
        // Verify closures are created without panic
    }
}
