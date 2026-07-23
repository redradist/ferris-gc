//! Minimal FxHash implementation for integer-keyed internal maps.
//!
//! The GC hot paths key several maps by `usize` (raw thin pointers) and
//! `ObjectId` (a `u64` newtype): `ptr_to_object`, `weak_alive_map`, the
//! incremental `colors`/`edges` maps, the card table, and the dying-address
//! registry. The standard-library default hasher is SipHash, which is
//! DoS-resistant but comparatively slow for small integer keys — it is pure
//! overhead here since these keys are never attacker-controlled hash inputs.
//!
//! FxHash (the hasher rustc itself uses internally) multiplies each machine
//! word by a fixed odd constant with a rotate, giving a couple of ALU ops per
//! key instead of the full SipHash round. This is a drop-in `BuildHasher`
//! replacement; the maps stay `std::collections::HashMap`.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// Multiplicative constant (odd, well-mixed) taken from rustc's FxHasher.
const SEED64: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// A fast, non-cryptographic hasher for integer keys.
///
/// Not DoS-resistant — only use for internal maps whose keys are pointers or
/// object ids the GC controls, never for untrusted external input.
#[derive(Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ i).wrapping_mul(SEED64);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        // Consume 8 bytes at a time, then the tail. Rarely used for our keys
        // (the integer write_* methods below cover them), but required for a
        // correct Hasher impl.
        while bytes.len() >= 8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[..8]);
            self.add(u64::from_le_bytes(buf));
            bytes = &bytes[8..];
        }
        if !bytes.is_empty() {
            let mut buf = [0u8; 8];
            buf[..bytes.len()].copy_from_slice(bytes);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // Extra multiply spreads the low bits that the map's modulo would key on.
        self.hash.wrapping_mul(SEED64)
    }
}

/// `BuildHasher` for [`FxHasher`], usable as the `S` parameter of the std maps.
pub(crate) type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// `HashMap` specialized to the fast integer hasher.
pub(crate) type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// `HashSet` specialized to the fast integer hasher.
pub(crate) type FxHashSet<K> = HashSet<K, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_roundtrips() {
        let mut m: FxHashMap<u64, u32> = FxHashMap::default();
        for i in 0..1000u64 {
            m.insert(i.wrapping_mul(2654435761), i as u32);
        }
        for i in 0..1000u64 {
            assert_eq!(m.get(&i.wrapping_mul(2654435761)), Some(&(i as u32)));
        }
        assert_eq!(m.len(), 1000);
    }

    #[test]
    fn set_roundtrips() {
        let mut s: FxHashSet<usize> = FxHashSet::default();
        for i in 0..500usize {
            s.insert(i * 7);
        }
        for i in 0..500usize {
            assert!(s.contains(&(i * 7)));
        }
        assert!(!s.contains(&3)); // 3 is not a multiple of 7
    }

    #[test]
    fn distinct_keys_mostly_distinct_hashes() {
        // Sanity: consecutive ids should not all collide to one bucket-ish value.
        let h = |x: u64| {
            let mut hasher = FxHasher::default();
            hasher.write_u64(x);
            hasher.finish()
        };
        let a = h(1);
        let b = h(2);
        let c = h(3);
        assert!(a != b && b != c && a != c);
    }
}
