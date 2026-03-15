//! Basic usage of thread-local garbage collection.

use ferris_gc::{Finalize, Gc, Trace};

struct Point {
    x: f64,
    y: f64,
}

impl Finalize for Point {
    fn finalize(&self) {
        println!("Point({}, {}) finalized", self.x, self.y);
    }
}

impl Trace for Point {
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

    // Allocate GC-managed objects
    let p1 = Gc::new(Point { x: 1.0, y: 2.0 });
    let p2 = Gc::new(Point { x: 3.0, y: 4.0 });

    println!("p1 = ({}, {})", p1.x, p1.y);
    println!("p2 = ({}, {})", p2.x, p2.y);

    // Clone creates a new handle to the same object
    let p1_clone = p1.clone();
    println!("p1_clone = ({}, {})", p1_clone.x, p1_clone.y);

    // Objects are collected automatically when all handles are dropped.
    // The background strategy handles collection — no manual calls needed.
    drop(p1);
    drop(p1_clone);
    drop(p2);

    println!("All handles dropped — the strategy will collect them.");
}
