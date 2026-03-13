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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkColor {
    White,
    Gray,
    Black,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionPhase {
    Idle,
    Marking,
    Sweeping,
}

#[derive(Debug, Clone, Copy)]
pub struct CollectionStats {
    pub generation: Generation,
    pub objects_scanned: usize,
    pub objects_collected: usize,
    pub objects_promoted: usize,
    pub tracers_collected: usize,
}

/// Snapshot of GC diagnostics.
#[derive(Debug, Clone)]
pub struct GcStats {
    /// Total bytes allocated for GC-managed objects (object data only, not tracers).
    pub heap_size: usize,
    /// Number of live GC-managed objects.
    pub live_objects: usize,
    /// Number of live tracers (Gc/GcRefCell handles pointing to objects).
    pub live_tracers: usize,
    /// Number of objects per generation.
    pub gen0_objects: usize,
    pub gen1_objects: usize,
    pub gen2_objects: usize,
    /// Total number of collections performed since GC creation.
    pub total_collections: usize,
    /// Stats from the most recent collection, if any.
    pub last_collection: Option<CollectionStats>,
    /// Allocations since last Gen0 collection.
    pub allocation_count: usize,
}
