#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Generation {
    Gen0 = 0,
    Gen1 = 1,
    Gen2 = 2,
}

impl Generation {
    pub fn next(self) -> Option<Generation> {
        match self {
            Generation::Gen0 => Some(Generation::Gen1),
            Generation::Gen1 => Some(Generation::Gen2),
            Generation::Gen2 => None,
        }
    }

    /// How many collections an object must survive in this generation before promotion.
    pub fn promotion_threshold(self) -> u32 {
        match self {
            Generation::Gen0 => 3,
            Generation::Gen1 => 5,
            Generation::Gen2 => u32::MAX, // never promote from Gen2
        }
    }
}

#[allow(dead_code)]
pub struct CollectionStats {
    pub generation: Generation,
    pub objects_scanned: usize,
    pub objects_collected: usize,
    pub objects_promoted: usize,
    pub tracers_collected: usize,
}
