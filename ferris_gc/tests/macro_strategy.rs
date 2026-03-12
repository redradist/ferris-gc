use ferris_gc::{Gc, Trace, Finalize};
use std::sync::Mutex;

struct Counter {
    _value: i32,
}

impl Trace for Counter {
    fn is_root(&self) -> bool { false }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool { false }
}

impl Finalize for Counter {
    fn finalize(&self) {}
}

// Serialize tests that modify global strategy state.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn macro_threshold_strategy_compiles_and_runs() {
    let _guard = TEST_MUTEX.lock().unwrap();
    // Simulate what #[ferris_gc_main(strategy = "threshold")] generates
    ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);
    ferris_gc::LOCAL_GC_STRATEGY.with(|s| {
        let (start, stop) = ferris_gc::threshold_local_start(ferris_gc::ThresholdConfig::default());
        s.borrow().change_strategy(start, stop);
    });
    {
        let (start, stop) = ferris_gc::threshold_global_start(ferris_gc::ThresholdConfig::default());
        ferris_gc::sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);
    }
    let _gc = Gc::new(Counter { _value: 2 });
}

#[test]
fn macro_adaptive_strategy_compiles_and_runs() {
    let _guard = TEST_MUTEX.lock().unwrap();
    // Simulate what #[ferris_gc_main(strategy = "adaptive")] generates
    ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);
    ferris_gc::LOCAL_GC_STRATEGY.with(|s| {
        let (start, stop) = ferris_gc::adaptive_local_start(ferris_gc::AdaptiveConfig::default());
        s.borrow().change_strategy(start, stop);
    });
    {
        let (start, stop) = ferris_gc::adaptive_global_start(ferris_gc::AdaptiveConfig::default());
        ferris_gc::sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);
    }
    let _gc = Gc::new(Counter { _value: 3 });
}
