/// Region identifier for region-based collection.
/// Objects are assigned to a region on allocation. Individual regions can be
/// collected independently, reducing pause times by limiting the scope of each collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub u32);

/// Generational tier for the GC's generational collection strategy.
/// Objects start in Gen0 (nursery) and are promoted to higher generations
/// as they survive collections, reducing collection frequency for long-lived objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Generation {
    /// Nursery generation — newly allocated objects. Collected most frequently.
    Gen0 = 0,
    /// Middle generation — objects that survived several Gen0 collections.
    Gen1 = 1,
    /// Tenured generation — long-lived objects. Collected least frequently.
    Gen2 = 2,
}

impl Generation {
    /// Return the next generation, or `None` if already at the oldest (Gen2).
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

/// Tri-color marking state for incremental/concurrent collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkColor {
    /// Not yet reached — candidate for collection.
    White,
    /// Discovered but children not yet scanned.
    Gray,
    /// Fully scanned — reachable and alive.
    Black,
}

/// Current phase of an incremental or concurrent collection cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionPhase {
    /// No collection in progress.
    Idle,
    /// Mark phase — tracing object graph.
    Marking,
    /// Sweep phase — reclaiming unreachable objects.
    Sweeping,
}

/// Statistics from a single collection cycle.
#[derive(Debug, Clone, Copy)]
pub struct CollectionStats {
    /// The highest generation included in this collection.
    pub generation: Generation,
    /// Number of objects examined during the mark phase.
    pub objects_scanned: usize,
    /// Number of unreachable objects deallocated.
    pub objects_collected: usize,
    /// Number of surviving objects promoted to the next generation.
    pub objects_promoted: usize,
    /// Number of tracer handles deallocated (pointing to collected objects).
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
