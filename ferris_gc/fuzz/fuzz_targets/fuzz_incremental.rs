#![no_main]
use libfuzzer_sys::fuzz_target;
use ferris_gc::{Gc, Generation};

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 { return; }

    let count = ((data[0] as usize) << 8 | data[1] as usize) % 10000 + 1;
    let budget = (data[2] as usize).max(1);

    let mut handles: Vec<Option<Gc<i32>>> = Vec::with_capacity(count);
    for i in 0..count {
        handles.push(Some(Gc::new(i as i32)));
    }

    // Drop some objects based on remaining data
    for &byte in &data[3..] {
        if !handles.is_empty() {
            let idx = byte as usize % handles.len();
            handles[idx] = None;
        }
    }

    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._collect_incremental(Generation::Gen2, budget);
    });

    drop(handles);
    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow()._collect();
    });
});
