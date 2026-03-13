#![no_main]
use libfuzzer_sys::fuzz_target;
use ferris_gc::Gc;

fuzz_target!(|data: &[u8]| {
    let mut handles: Vec<Option<Gc<Vec<u8>>>> = Vec::new();

    for &byte in data {
        match byte % 4 {
            0 => {
                // Allocate a new GC object with some data
                let size = (byte as usize / 4).min(64);
                let v = vec![byte; size];
                handles.push(Some(Gc::new(v)));
            }
            1 => {
                // Drop a handle (make object eligible for GC)
                if !handles.is_empty() {
                    let idx = byte as usize % handles.len();
                    handles[idx] = None;
                }
            }
            2 => {
                // Clone a handle
                if !handles.is_empty() {
                    let idx = byte as usize % handles.len();
                    if let Some(ref gc) = handles[idx] {
                        handles.push(Some(gc.clone()));
                    }
                }
            }
            3 => {
                // Trigger collection
                ferris_gc::LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut().collect();
                });
            }
            _ => unreachable!(),
        }
    }
    // Final cleanup
    drop(handles);
    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
});
