//! Demonstrates cycle collection using GcCell.

use ferris_gc::{Finalize, Gc, GcCell, Trace};

struct Node {
    name: String,
    next: GcCell<Option<Gc<Node>>>,
}

impl Finalize for Node {
    fn finalize(&self) {
        println!("Node '{}' finalized", self.name);
    }
}

impl Trace for Node {
    fn is_root(&self) -> bool {
        self.next.is_root()
    }
    fn reset_root(&self) {
        self.next.reset_root()
    }
    fn trace(&self) {
        self.next.trace()
    }
    fn reset(&self) {
        self.next.reset()
    }
    fn is_traceable(&self) -> bool {
        self.next.is_traceable()
    }
    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.next.trace_children(children);
    }
}

fn main() {
    let _cleanup = ferris_gc::ApplicationCleanup;

    // Create a cycle: A -> B -> A
    let a = Gc::new(Node {
        name: "A".into(),
        next: GcCell::new(None),
    });
    let b = Gc::new(Node {
        name: "B".into(),
        next: GcCell::new(Some(a.clone())),
    });
    **a.next.borrow_mut() = Some(b.clone());

    println!("Created cycle: A -> B -> A");

    // Drop all local handles
    drop(a);
    drop(b);

    // The cycle is detected and collected by the GC
    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });

    println!("Cycle collected!");
}
