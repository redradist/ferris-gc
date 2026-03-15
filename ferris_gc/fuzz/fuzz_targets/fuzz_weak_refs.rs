#![no_main]
use libfuzzer_sys::fuzz_target;
use ferris_gc::{Gc, Trace, Finalize};

struct Val(u8);

impl Finalize for Val {
    fn finalize(&self) {}
}

impl Trace for Val {
    fn is_root(&self) -> bool { false }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool { false }
}

fuzz_target!(|data: &[u8]| {
    let mut strongs: Vec<Option<Gc<Val>>> = Vec::new();
    let mut weaks: Vec<ferris_gc::GcWeak<Val>> = Vec::new();

    for &byte in data {
        match byte % 7 {
            0 => {
                // Create a new Gc object
                strongs.push(Some(Gc::new(Val(byte))));
            }
            1 => {
                // Downgrade a strong ref to a weak ref
                if !strongs.is_empty() {
                    let idx = byte as usize % strongs.len();
                    if let Some(ref gc) = strongs[idx] {
                        weaks.push(Gc::downgrade(gc));
                    }
                }
            }
            2 => {
                // Upgrade a weak ref back to a strong ref
                if !weaks.is_empty() {
                    let idx = byte as usize % weaks.len();
                    if let Some(upgraded) = weaks[idx].upgrade() {
                        strongs.push(Some(upgraded));
                    }
                }
            }
            3 => {
                // Drop a strong reference (make object eligible for GC)
                if !strongs.is_empty() {
                    let idx = byte as usize % strongs.len();
                    strongs[idx] = None;
                }
            }
            4 => {
                // Check is_alive on a weak ref
                if !weaks.is_empty() {
                    let idx = byte as usize % weaks.len();
                    let _ = weaks[idx].is_alive();
                }
            }
            5 => {
                // Drop a weak ref
                if !weaks.is_empty() {
                    let idx = byte as usize % weaks.len();
                    weaks.swap_remove(idx);
                }
            }
            6 => {
                // Trigger collection and verify weak invariants
                ferris_gc::LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut().collect();
                });
                // After collection, any weak whose strong was dropped should
                // report not alive or return None on upgrade.
                for weak in &weaks {
                    if !weak.is_alive() {
                        assert!(
                            weak.upgrade().is_none(),
                            "dead weak must not upgrade"
                        );
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    // Final cleanup: drop all strong handles, collect, verify all weaks are dead
    strongs.clear();
    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    for weak in &weaks {
        assert!(
            weak.upgrade().is_none(),
            "all weaks should be dead after dropping all strongs and collecting"
        );
    }
    drop(weaks);
    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
});
