use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Serialize)]
pub struct BenchResult {
    pub benchmark: String,
    pub language: String,
    pub strategy: String,
    pub duration_ns: u128,
    pub duration_ms: f64,
    pub peak_heap_mb: f64,
    pub heap_in_use_mb: f64,
    pub num_gc: u32,
    pub total_gc_pause_ns: u128,
    pub total_gc_pause_ms: f64,
    pub max_gc_pause_ns: u128,
    pub max_gc_pause_ms: f64,
    pub throughput_ops_per_sec: f64,
}

impl BenchResult {
    pub fn print(&self) {
        let json = serde_json::to_string_pretty(self).unwrap();
        println!("{json}");
    }
}

/// GC stats collected from the collector + on_collection callback.
pub struct GcStats {
    pub num_gc: u32,
    pub total_gc_pause: Duration,
    pub max_gc_pause: Duration,
    pub heap_bytes: usize,
    pub peak_heap_bytes: usize,
    pub live_objects: usize,
}

// Global accumulators for on_collection callback
static GC_PAUSE_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static GC_PAUSE_MAX_NS: AtomicU64 = AtomicU64::new(0);

/// Install an on_collection callback on the thread-local GC
/// that accumulates pause durations.
pub fn install_local_gc_monitor() {
    ferris_gc::LOCAL_GC.with(|gc| {
        gc.borrow().set_on_collection(|stats| {
            let nanos = stats.duration.as_nanos() as u64;
            GC_PAUSE_TOTAL_NS.fetch_add(nanos, Ordering::Relaxed);
            GC_PAUSE_MAX_NS.fetch_max(nanos, Ordering::Relaxed);
        });
    });
}

/// Install an on_collection callback on the global (sync) GC
/// that accumulates pause durations.
pub fn install_sync_gc_monitor() {
    ferris_gc::sync::GLOBAL_GC.set_on_collection(|stats| {
        let nanos = stats.duration.as_nanos() as u64;
        GC_PAUSE_TOTAL_NS.fetch_add(nanos, Ordering::Relaxed);
        GC_PAUSE_MAX_NS.fetch_max(nanos, Ordering::Relaxed);
    });
}

/// Configure the local GC strategy based on a string name.
/// Must be called BEFORE any Gc::new() allocations.
/// Returns the strategy name actually used.
pub fn configure_local_strategy(strategy: &str) -> String {
    // Disable the default basic strategy
    ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);

    match strategy {
        "none" => {
            // No automatic collection — manual only
            "none".to_string()
        }
        "basic" => {
            // Re-enable basic strategy
            ferris_gc::BASIC_STRATEGY_DISABLED.store(false, std::sync::atomic::Ordering::Release);
            "basic".to_string()
        }
        "threshold" => {
            let config = ferris_gc::ThresholdConfig::default();
            let s = ferris_gc::threshold_local_strategy(config);
            ferris_gc::LOCAL_GC_STRATEGY.with(|strategy| {
                strategy.borrow_mut().set_strategy(s);
            });
            "threshold".to_string()
        }
        "adaptive" => {
            let config = ferris_gc::AdaptiveConfig::default();
            let s = ferris_gc::adaptive_local_strategy(config);
            ferris_gc::LOCAL_GC_STRATEGY.with(|strategy| {
                strategy.borrow_mut().set_strategy(s);
            });
            "adaptive".to_string()
        }
        "background" => {
            let config = ferris_gc::BackgroundConfig::default();
            let s = ferris_gc::background_local_strategy(config);
            ferris_gc::LOCAL_GC_STRATEGY.with(|strategy| {
                strategy.borrow_mut().set_strategy(s);
            });
            "background".to_string()
        }
        "g1" => {
            let config = ferris_gc::G1Config::default();
            let s = ferris_gc::g1_local_strategy(config);
            ferris_gc::LOCAL_GC_STRATEGY.with(|strategy| {
                strategy.borrow_mut().set_strategy(s);
            });
            "g1".to_string()
        }
        _ => {
            eprintln!("Unknown strategy '{strategy}', using 'none'");
            "none".to_string()
        }
    }
}

/// Configure the global (sync) GC strategy based on a string name.
pub fn configure_sync_strategy(strategy: &str) -> String {
    use ferris_gc::sync::GLOBAL_GC_STRATEGY;

    ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);

    match strategy {
        "none" => "none".to_string(),
        "basic" => {
            ferris_gc::BASIC_STRATEGY_DISABLED.store(false, std::sync::atomic::Ordering::Release);
            "basic".to_string()
        }
        "threshold" => {
            let config = ferris_gc::ThresholdConfig::default();
            let s = ferris_gc::threshold_global_strategy(config);
            GLOBAL_GC_STRATEGY.set_strategy(s);
            "threshold".to_string()
        }
        "adaptive" => {
            let config = ferris_gc::AdaptiveConfig::default();
            let s = ferris_gc::adaptive_global_strategy(config);
            GLOBAL_GC_STRATEGY.set_strategy(s);
            "adaptive".to_string()
        }
        "background" => {
            let config = ferris_gc::BackgroundConfig::default();
            let s = ferris_gc::background_global_strategy(config);
            GLOBAL_GC_STRATEGY.set_strategy(s);
            "background".to_string()
        }
        "g1" => {
            let config = ferris_gc::G1Config::default();
            let s = ferris_gc::g1_global_strategy(config);
            GLOBAL_GC_STRATEGY.set_strategy(s);
            "g1".to_string()
        }
        _ => {
            eprintln!("Unknown strategy '{strategy}', using 'none'");
            "none".to_string()
        }
    }
}

/// Parse --strategy=NAME from CLI args. Returns "basic" if not specified.
pub fn get_strategy() -> String {
    for arg in std::env::args() {
        if let Some(s) = arg.strip_prefix("--strategy=") {
            return s.to_string();
        }
    }
    "basic".to_string()
}

pub fn local_gc_stats() -> GcStats {
    ferris_gc::LOCAL_GC.with(|gc| {
        let stats = gc.borrow().stats();
        let total_ns = GC_PAUSE_TOTAL_NS.load(Ordering::Relaxed);
        let max_ns = GC_PAUSE_MAX_NS.load(Ordering::Relaxed);
        GcStats {
            num_gc: stats.total_collections as u32,
            total_gc_pause: Duration::from_nanos(total_ns),
            max_gc_pause: Duration::from_nanos(max_ns),
            heap_bytes: stats.heap_size,
            peak_heap_bytes: stats.peak_heap_size,
            live_objects: stats.live_objects,
        }
    })
}

pub fn sync_gc_stats() -> GcStats {
    let stats = ferris_gc::sync::GLOBAL_GC.stats();
    let total_ns = GC_PAUSE_TOTAL_NS.load(Ordering::Relaxed);
    let max_ns = GC_PAUSE_MAX_NS.load(Ordering::Relaxed);
    GcStats {
        num_gc: stats.total_collections as u32,
        total_gc_pause: Duration::from_nanos(total_ns),
        max_gc_pause: Duration::from_nanos(max_ns),
        heap_bytes: stats.heap_size,
        peak_heap_bytes: stats.peak_heap_size,
        live_objects: stats.live_objects,
    }
}

pub fn make_result(
    benchmark: &str,
    strategy: &str,
    n: usize,
    elapsed: Duration,
    stats: &GcStats,
) -> BenchResult {
    BenchResult {
        benchmark: benchmark.to_string(),
        language: "rust/ferris-gc".to_string(),
        strategy: strategy.to_string(),
        duration_ns: elapsed.as_nanos(),
        duration_ms: elapsed.as_secs_f64() * 1000.0,
        peak_heap_mb: stats.peak_heap_bytes as f64 / (1024.0 * 1024.0),
        heap_in_use_mb: stats.heap_bytes as f64 / (1024.0 * 1024.0),
        num_gc: stats.num_gc,
        total_gc_pause_ns: stats.total_gc_pause.as_nanos(),
        total_gc_pause_ms: stats.total_gc_pause.as_secs_f64() * 1000.0,
        max_gc_pause_ns: stats.max_gc_pause.as_nanos(),
        max_gc_pause_ms: stats.max_gc_pause.as_secs_f64() * 1000.0,
        throughput_ops_per_sec: n as f64 / elapsed.as_secs_f64(),
    }
}

pub fn get_n() -> usize {
    // First non-flag argument
    for arg in std::env::args().skip(1) {
        if !arg.starts_with("--") {
            if let Ok(n) = arg.parse() {
                return n;
            }
        }
    }
    100_000
}

/// Time a closure returning the elapsed duration.
pub fn timed<F: FnOnce() -> R, R>(f: F) -> (R, Duration) {
    let start = Instant::now();
    let r = f();
    (r, start.elapsed())
}
