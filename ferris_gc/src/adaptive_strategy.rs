use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::gc::LocalGarbageCollector;
use crate::gc::sync::GlobalGarbageCollector;
use crate::generation::Generation;

/// Configuration for the adaptive GC strategy.
///
/// Like ThresholdStrategy but automatically tunes `gen0_threshold` based on
/// collection effectiveness. High collection ratios (many dead objects) keep
/// or lower the threshold; low ratios (few dead objects) increase it.
pub struct AdaptiveConfig {
    /// Initial allocation threshold for Gen0 collection.
    pub initial_gen0_threshold: usize,
    /// Minimum allowed threshold (floor).
    pub min_threshold: usize,
    /// Maximum allowed threshold (ceiling).
    pub max_threshold: usize,
    /// Collection ratio above which threshold is decreased (objects_collected / objects_scanned).
    pub high_ratio: f64,
    /// Collection ratio below which threshold is increased.
    pub low_ratio: f64,
    /// Multiplicative factor for threshold adjustment.
    pub adjust_factor: f64,
    /// Number of Gen0 collections between each Gen1 collection.
    pub gen0_collections_per_gen1: u32,
    /// Number of Gen1 collections between each Gen2 collection.
    pub gen1_collections_per_gen2: u32,
    /// Polling interval for the background thread.
    pub poll_interval: Duration,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        AdaptiveConfig {
            initial_gen0_threshold: 100,
            min_threshold: 50,
            max_threshold: 10_000,
            high_ratio: 0.5,
            low_ratio: 0.1,
            adjust_factor: 1.5,
            gen0_collections_per_gen1: 5,
            gen1_collections_per_gen2: 5,
            poll_interval: Duration::from_millis(50),
        }
    }
}

fn adapt_threshold(config: &AdaptiveConfig, current: usize, collected: usize, scanned: usize) -> usize {
    if scanned == 0 {
        return current;
    }
    let ratio = collected as f64 / scanned as f64;
    let new = if ratio > config.high_ratio {
        // Many dead objects — keep threshold low (collect often)
        (current as f64 / config.adjust_factor) as usize
    } else if ratio < config.low_ratio {
        // Few dead objects — increase threshold (collect less often)
        (current as f64 * config.adjust_factor) as usize
    } else {
        current
    };
    new.clamp(config.min_threshold, config.max_threshold)
}

/// Creates start/stop closures for a local adaptive strategy.
pub fn adaptive_local_start(
    config: AdaptiveConfig,
) -> (
    impl FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> + 'static,
    impl FnMut(&'static LocalGarbageCollector) + 'static,
) {
    let config = std::sync::Arc::new(config);
    let config_start = config.clone();
    let start_fn = move |gc: &'static LocalGarbageCollector, is_active: &'static AtomicBool| -> Option<JoinHandle<()>> {
        let config = config_start.clone();
        Some(thread::spawn(move || {
            let mut threshold = config.initial_gen0_threshold;
            let mut gen0_count: u32 = 0;
            let mut gen1_count: u32 = 0;
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config.poll_interval);
                let allocs = gc.core.allocation_count.load(Ordering::Relaxed);
                if allocs >= threshold {
                    let stats = unsafe { gc.core.collect_generation(Generation::Gen0) };
                    threshold = adapt_threshold(&config, threshold, stats.objects_collected, stats.objects_scanned);
                    gen0_count += 1;
                    if gen0_count >= config.gen0_collections_per_gen1 {
                        gen0_count = 0;
                        unsafe { gc.core.collect_generation(Generation::Gen1); }
                        gen1_count += 1;
                        if gen1_count >= config.gen1_collections_per_gen2 {
                            gen1_count = 0;
                            unsafe { gc.core.collect_generation(Generation::Gen2); }
                        }
                    }
                }
            }
            // Final full collection on shutdown
            unsafe { gc.core.collect_generation(Generation::Gen2); }
        }))
    };
    let stop_fn = move |_gc: &'static LocalGarbageCollector| {};
    (start_fn, stop_fn)
}

/// Creates start/stop closures for a global adaptive strategy.
pub fn adaptive_global_start(
    config: AdaptiveConfig,
) -> (
    impl FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>> + 'static,
    impl FnMut(&'static GlobalGarbageCollector) + 'static,
) {
    let config = std::sync::Arc::new(config);
    let config_start = config.clone();
    let start_fn = move |gc: &'static GlobalGarbageCollector, is_active: &'static AtomicBool| -> Option<JoinHandle<()>> {
        let config = config_start.clone();
        Some(thread::spawn(move || {
            let mut threshold = config.initial_gen0_threshold;
            let mut gen0_count: u32 = 0;
            let mut gen1_count: u32 = 0;
            while is_active.load(Ordering::Acquire) {
                thread::sleep(config.poll_interval);
                let allocs = gc.core.allocation_count.load(Ordering::Relaxed);
                if allocs >= threshold {
                    let stats = unsafe { gc.core.collect_generation(Generation::Gen0) };
                    threshold = adapt_threshold(&config, threshold, stats.objects_collected, stats.objects_scanned);
                    gen0_count += 1;
                    if gen0_count >= config.gen0_collections_per_gen1 {
                        gen0_count = 0;
                        unsafe { gc.core.collect_generation(Generation::Gen1); }
                        gen1_count += 1;
                        if gen1_count >= config.gen1_collections_per_gen2 {
                            gen1_count = 0;
                            unsafe { gc.core.collect_generation(Generation::Gen2); }
                        }
                    }
                }
            }
            // Final full collection on shutdown
            unsafe { gc.core.collect_generation(Generation::Gen2); }
        }))
    };
    let stop_fn = move |_gc: &'static GlobalGarbageCollector| {};
    (start_fn, stop_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapt_threshold_high_ratio_decreases() {
        let config = AdaptiveConfig::default();
        // 80% collected → should decrease
        let new = adapt_threshold(&config, 100, 80, 100);
        assert!(new < 100, "high collection ratio should decrease threshold, got {new}");
        assert!(new >= config.min_threshold);
    }

    #[test]
    fn adapt_threshold_low_ratio_increases() {
        let config = AdaptiveConfig::default();
        // 5% collected → should increase
        let new = adapt_threshold(&config, 100, 5, 100);
        assert!(new > 100, "low collection ratio should increase threshold, got {new}");
        assert!(new <= config.max_threshold);
    }

    #[test]
    fn adapt_threshold_mid_ratio_unchanged() {
        let config = AdaptiveConfig::default();
        // 30% collected → in between, should stay the same
        let new = adapt_threshold(&config, 100, 30, 100);
        assert_eq!(new, 100, "mid-range ratio should keep threshold unchanged");
    }

    #[test]
    fn adapt_threshold_respects_min() {
        let config = AdaptiveConfig { min_threshold: 50, ..Default::default() };
        // Start at min, high ratio would try to decrease further
        let new = adapt_threshold(&config, 50, 50, 50);
        assert_eq!(new, 50, "threshold should not go below min_threshold");
    }

    #[test]
    fn adapt_threshold_respects_max() {
        let config = AdaptiveConfig { max_threshold: 200, ..Default::default() };
        // Start at max, low ratio would try to increase further
        let new = adapt_threshold(&config, 200, 0, 100);
        assert_eq!(new, 200, "threshold should not exceed max_threshold");
    }

    #[test]
    fn adapt_threshold_zero_scanned() {
        let config = AdaptiveConfig::default();
        let new = adapt_threshold(&config, 100, 0, 0);
        assert_eq!(new, 100, "zero scanned should leave threshold unchanged");
    }
}
