//! Thread-safe garbage collection with sync::Gc<T>.

use ferris_gc::sync;
use ferris_gc::{Finalize, Trace};
use std::thread;

struct SharedData {
    value: i32,
}

impl Finalize for SharedData {
    fn finalize(&self) {}
}

impl Trace for SharedData {
    fn is_root(&self) -> bool {
        false
    }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool {
        false
    }
}

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;

    let data = sync::Gc::new(SharedData { value: 42 });

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let data = data.clone();
            thread::spawn(move || {
                println!("Thread {i}: value = {}", data.value);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Check GC stats
    let stats = sync::GLOBAL_GC.stats();
    println!("Live objects: {}", stats.live_objects);
    println!("Heap size: {} bytes", stats.heap_size);
}
