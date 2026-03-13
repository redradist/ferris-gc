#![no_main]
use libfuzzer_sys::fuzz_target;
use ferris_gc::{Gc, GcRefCell, Trace, Finalize};


struct Node {
    next: GcRefCell<Option<Gc<Node>>>,
}

impl Finalize for Node {
    fn finalize(&self) {}
}

impl Trace for Node {
    fn is_root(&self) -> bool { self.next.is_root() }
    fn reset_root(&self) { self.next.reset_root() }
    fn trace(&self) { self.next.trace() }
    fn reset(&self) { self.next.reset() }
    fn is_traceable(&self) -> bool { self.next.is_traceable() }
    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.next.trace_children(children);
    }
}

fuzz_target!(|data: &[u8]| {
    let mut nodes: Vec<Gc<Node>> = Vec::new();

    for &byte in data {
        match byte % 5 {
            0 => {
                // Create new node
                nodes.push(Gc::new(Node { next: GcRefCell::new(None) }));
            }
            1 => {
                // Link two nodes (potentially creating a cycle)
                if nodes.len() >= 2 {
                    let from = byte as usize % nodes.len();
                    let to = byte.wrapping_mul(7) as usize % nodes.len();
                    let target = nodes[to].clone();
                    **nodes[from].next.borrow_mut() = Some(target);
                }
            }
            2 => {
                // Remove a node
                if !nodes.is_empty() {
                    let idx = byte as usize % nodes.len();
                    nodes.swap_remove(idx);
                }
            }
            3 => {
                // Break a link
                if !nodes.is_empty() {
                    let idx = byte as usize % nodes.len();
                    **nodes[idx].next.borrow_mut() = None;
                }
            }
            4 => {
                // Collect
                ferris_gc::LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut().collect();
                });
            }
            _ => unreachable!(),
        }
    }
    drop(nodes);
    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
});
