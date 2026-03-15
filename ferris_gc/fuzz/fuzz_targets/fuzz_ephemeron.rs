#![no_main]
use libfuzzer_sys::fuzz_target;
use ferris_gc::{Gc, Trace, Finalize, EphemeronTable};

struct Key(u8);

impl Finalize for Key {
    fn finalize(&self) {}
}

impl Trace for Key {
    fn is_root(&self) -> bool { false }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool { false }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

fuzz_target!(|data: &[u8]| {
    let mut keys: Vec<Option<Gc<Key>>> = Vec::new();
    let mut table: EphemeronTable<Key, u8> = EphemeronTable::new();

    for &byte in data {
        match byte % 8 {
            0 => {
                // Create a new key
                keys.push(Some(Gc::new(Key(byte))));
            }
            1 => {
                // Insert into ephemeron table (with duplicate check)
                if !keys.is_empty() {
                    let idx = byte as usize % keys.len();
                    if let Some(ref key) = keys[idx] {
                        table.insert(key, byte);
                    }
                }
            }
            2 => {
                // Insert unique into ephemeron table (no duplicate check)
                if !keys.is_empty() {
                    let idx = byte as usize % keys.len();
                    if let Some(ref key) = keys[idx] {
                        table.insert_unique(key, byte);
                    }
                }
            }
            3 => {
                // Look up a key in the table
                if !keys.is_empty() {
                    let idx = byte as usize % keys.len();
                    if let Some(ref key) = keys[idx] {
                        let _ = table.get(key);
                    }
                }
            }
            4 => {
                // Remove a key from the table
                if !keys.is_empty() {
                    let idx = byte as usize % keys.len();
                    if let Some(ref key) = keys[idx] {
                        let _ = table.remove(key);
                    }
                }
            }
            5 => {
                // Drop a key (make it eligible for GC)
                if !keys.is_empty() {
                    let idx = byte as usize % keys.len();
                    keys[idx] = None;
                }
            }
            6 => {
                // Collect and cleanup the table
                ferris_gc::LOCAL_GC.with(|gc| unsafe {
                    gc.borrow_mut().collect();
                });
                table.cleanup();
                // After cleanup, the len should only count live entries
                let live_count = table.len();
                let iter_count = table.iter().count();
                assert_eq!(
                    live_count, iter_count,
                    "len() and iter().count() must agree after cleanup"
                );
            }
            7 => {
                // Shrink the table and verify consistency
                table.shrink();
                let values: Vec<_> = table.values().collect();
                assert!(
                    values.len() <= table.len(),
                    "values count must not exceed len"
                );
            }
            _ => unreachable!(),
        }
    }

    // Final cleanup: drop all keys, collect, cleanup table, verify empty
    keys.clear();
    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
    table.cleanup();
    assert_eq!(
        table.len(),
        0,
        "table should be empty after all keys dropped and collected"
    );
    assert!(table.is_empty());
    ferris_gc::LOCAL_GC.with(|gc| unsafe {
        gc.borrow_mut().collect();
    });
});
