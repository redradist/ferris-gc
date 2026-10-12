use crate::sharded_map::Counter;
use std::cell::{Cell, UnsafeCell};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::basic_strategy::{BASIC_STRATEGY_GLOBAL_GC, basic_strategy_start};
use crate::gc::{
    CompactLayout, Finalize, GarbageCollector, GcObjMem, ObjectEntry, ObjectEntryRef, SyncObjects,
    Trace, TracerList,
};
use crate::slot_map::ObjectId;

/// Layout of one sync object's block `[A (object) | ObjectEntry]`, and the
/// entry's offset in it.
#[inline]
fn object_block_layout<A>() -> (std::alloc::Layout, usize) {
    let (ae, off_e) = std::alloc::Layout::new::<A>()
        .extend(std::alloc::Layout::new::<ObjectEntry>())
        .expect("sync object block layout");
    (ae.pad_to_align(), off_e)
}

/// One allocation for a new sync object and its `ObjectEntry`.
///
/// Handles count themselves in the object header, so no per-handle tracer is
/// allocated (this used to be four mallocs per object: object, first
/// handle's tracer, boxed `ObjectEntry`, boxed tracer-list node). The entry is
/// marked `ENTRY_EMBEDDED` and freed together with the object. The block
/// comes from the per-thread pool (`crate::pool`) when it fits a size class,
/// otherwise from the system allocator (see `alloc_sync_mem`).
///
/// Returns `None` when the allocator fails.
///
/// # Safety
/// Both returned pointers are uninitialized and must be written before use.
#[inline]
unsafe fn alloc_object_block<A>() -> Option<(*mut A, *mut ObjectEntry, std::alloc::Layout)> {
    let (layout, off_e) = object_block_layout::<A>();
    // SAFETY: the layout is non-zero-sized (A holds a GcInfo) and well-formed.
    let p = unsafe { crate::gc::alloc_sync_mem(layout) };
    if p.is_null() {
        return None;
    }
    // SAFETY: the offset lies inside the block, from Layout::extend.
    unsafe { Some((p as *mut A, p.add(off_e) as *mut ObjectEntry, layout)) }
}

/// A sync collector object: `GcPtr<T>` or `GcCellBox<T>`, both headed by a
/// `GcInfo`.
trait SyncObject: Trace + 'static {
    /// The header of the object at `this`.
    ///
    /// # Safety
    /// `this` must point at an initialized object.
    unsafe fn info_ptr(this: *mut Self) -> *mut GcInfo;
}

impl<T: 'static + Sized + Trace> SyncObject for GcPtr<T> {
    #[inline]
    unsafe fn info_ptr(this: *mut Self) -> *mut GcInfo {
        unsafe { &raw mut (*this).info }
    }
}

impl<T: 'static + Sized + Trace> SyncObject for GcCellBox<T> {
    #[inline]
    unsafe fn info_ptr(this: *mut Self) -> *mut GcInfo {
        unsafe { &raw mut (*this).info }
    }
}

/// Convenience alias for an optional thread-safe GC pointer.
pub type OptGc<T> = Option<Gc<T>>;
/// Convenience alias for an optional thread-safe GC interior-mutable cell.
pub type OptGcCell<T> = Option<GcCell<T>>;

/// Type-safe region identifier for the **global (thread-safe)** garbage collector.
/// Returned by [`GlobalGarbageCollector::new_region`]. Cannot be used with
/// the thread-local GC — the compiler will reject the mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyncRegionId(crate::generation::RegionId);

impl SyncRegionId {
    /// Return the raw numeric id (for logging / diagnostics).
    pub fn id(self) -> u32 {
        (self.0).0
    }
}

/// Every sync object starts with this header (see `crate::gc::SyncGcInfo`).
use crate::gc::SyncGcInfo as GcInfo;

/// Count one more handle to the object headed by `info`. Relaxed, as in
/// `Arc::clone`: the new handle comes from one that already keeps the object
/// alive. Aborts rather than let the count wrap.
#[inline]
fn add_handle(info: &GcInfo) {
    let old = info.handle_count.fetch_add(1, Ordering::Relaxed);
    if old > i32::MAX as u32 {
        std::process::abort();
    }
}

/// Drop one handle to the object at `obj`, headed by `info`. The last handle
/// frees the object.
///
/// # Safety
/// The handle being dropped must point at `obj`, and `info` must be its
/// header.
#[inline]
unsafe fn drop_handle(obj: *const u8, info: *const GcInfo) {
    // During a sweep's destructor phase this handle may live inside a dead
    // object and point at another object of the same dead batch - even at
    // the object whose destructor is running. The collector owns those
    // objects and already removed them from the map; touching the header
    // here could be undefined behaviour, so the drop is a no-op.
    if GLOBAL_GC.core.is_dying(obj as usize) {
        return;
    }
    unsafe {
        // Read the id first: once the count is decremented, another thread
        // or a collection may free the object.
        let object_id = (*info).object_id;
        if (*info).handle_count.fetch_sub(1, Ordering::Release) == 1 {
            std::sync::atomic::fence(Ordering::Acquire);
            GLOBAL_GC.core.release_object(object_id);
        }
    }
}

/// Root discovery's cascade for one handle inside a scanned object: this
/// handle is not a root, so take it off the target's root count, and walk the
/// target once. Only the global collector's root discovery sets the context;
/// anything else (allocation, or a thread-local collection reaching a sync
/// handle stored in a local object) is a no-op.
///
/// # Safety
/// `obj` must be a live sync object headed by `info`.
#[inline]
unsafe fn cascade_handle(obj: &dyn Trace, info: &GcInfo) {
    let visited = crate::gc::SYNC_CASCADE.get();
    if visited.is_null() {
        return;
    }
    let rc = info.root_count.load(Ordering::Relaxed);
    info.root_count
        .store(rc.saturating_sub(1), Ordering::Relaxed);
    // root_ref_count is the cycle guard: 0 = unvisited, 1 = visited.
    if info.root_ref_count.load(Ordering::Relaxed) == 0 {
        info.root_ref_count.store(1, Ordering::Relaxed);
        // Recorded so root discovery clears the guard even when the target
        // is out of the collection's scope.
        unsafe {
            (*visited).push(&info.root_ref_count as *const AtomicUsize as *const Cell<usize>);
        }
        obj.reset_root();
    }
}

/// Internal pointer wrapper used as the `Deref` target of `sync::Gc<T>` and `sync::GcCell<T>`.
/// This type is `pub` because it appears in the public `Deref` interface, but users
/// should not need to name it directly — access `T` via double-deref (`**gc`).
#[doc(hidden)]
pub struct GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    info: GcInfo,
    t: T,
}

impl<T> GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    fn new(t: T) -> GcPtr<T> {
        GcPtr {
            info: GcInfo::new(),
            t,
        }
    }
}

impl<T> Deref for GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.t
    }
}

impl<T> DerefMut for GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.t
    }
}

impl<T> Trace for GcPtr<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        unreachable!("is_root on GcPtr is unreachable !!");
    }

    fn reset_root(&self) {
        self.t.reset_root();
    }

    fn trace(&self) {
        // Guard: only recurse into children on first trace (breaks cycles)
        let prev = self.info.root_ref_count.fetch_add(1, Ordering::AcqRel);
        if prev == 0 {
            self.t.trace();
        }
    }

    fn reset(&self) {
        // Guard: only recurse into children on last reset (breaks cycles)
        let prev = self.info.root_ref_count.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.t.reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.info.root_ref_count.load(Ordering::Acquire) > 0
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.t.trace_children(children);
    }

    fn clear_trace(&self) {
        self.info.root_ref_count.store(0, Ordering::Release);
    }
}

/// The object behind a [`GcCell`]: GC header, a reader-writer lock, and the
/// value that lock guards.
///
/// The header lives outside the lock and the value, so the collector can
/// update it while a user holds a borrow, without aliasing the user's `&mut`.
/// The collector never blocks on the lock: it reads the value under
/// `try_read`. If a `borrow_mut` holds the lock, the collector counts the
/// scan as blocked (see `GarbageCollector::blocked_scans`), and that
/// collection frees nothing.
pub(crate) struct GcCellBox<T>
where
    T: 'static + Sized + Trace,
{
    info: GcInfo,
    lock: RwLock<()>,
    value: UnsafeCell<T>,
}

impl<T> GcCellBox<T>
where
    T: 'static + Sized + Trace,
{
    fn new(t: T) -> GcCellBox<T> {
        GcCellBox {
            info: GcInfo::new(),
            lock: RwLock::new(()),
            value: UnsafeCell::new(t),
        }
    }

    /// Run `f` on the value unless a `borrow_mut` holds it. Never blocks.
    #[inline]
    fn try_with_value<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let _guard = match self.lock.try_read() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        // SAFETY: the read lock excludes every writer.
        Some(f(unsafe { &*self.value.get() }))
    }

    /// [`Self::try_with_value`] for marking. A value it cannot read leaves the
    /// current collection's mark incomplete, so that collection frees nothing.
    #[inline]
    fn scan_value(&self, f: impl FnOnce(&T)) {
        if self.try_with_value(f).is_none() {
            GLOBAL_GC.core.blocked_scans.fetch_add(1, Ordering::AcqRel);
        }
    }
}

impl<T> Trace for GcCellBox<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        unreachable!("is_root on GcCellBox is unreachable !!");
    }

    fn reset_root(&self) {
        // Skipping a value that is mutably borrowed leaves its handles counted
        // as roots, which only keeps more alive.
        self.try_with_value(|t| t.reset_root());
    }

    fn trace(&self) {
        let prev = self.info.root_ref_count.fetch_add(1, Ordering::AcqRel);
        if prev == 0 {
            self.scan_value(|t| t.trace());
        }
    }

    fn reset(&self) {
        let prev = self.info.root_ref_count.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.scan_value(|t| t.reset());
        }
    }

    fn is_traceable(&self) -> bool {
        self.info.root_ref_count.load(Ordering::Acquire) > 0
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.scan_value(|t| t.trace_children(children));
    }

    fn clear_trace(&self) {
        self.info.root_ref_count.store(0, Ordering::Release);
    }
}

impl<T> Finalize for GcCellBox<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

impl<T> Finalize for GcPtr<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A garbage-collected smart pointer that can be shared across threads.
///
/// One pointer wide. Like `Arc`, a clone adds to an atomic handle count in the
/// object's header and a drop subtracts from it; the last drop frees the
/// object at once, and the global collector frees unreachable cycles.
/// Allocation and collection synchronize through the collector's
/// stop-the-world lock: mutators hold it for reading, a collection for
/// writing.
///
/// Dereferences to `T` (via `GcPtr<T>`).
pub struct Gc<T>
where
    T: 'static + Sized + Trace,
{
    ptr: *const GcPtr<T>,
}

/// # Safety
/// Handles count themselves atomically in the object header, and the global
/// GC protects all structural mutations with Mutex/RwLock. Every handle (and
/// the collector) reaches the same value, so moving or sharing a handle
/// shares `&T` across threads and may drop `T` on another thread: like
/// `Arc<T>`, both need `T: Send + Sync`.
unsafe impl<T> Sync for Gc<T> where T: 'static + Sized + Trace + Send + Sync {}
unsafe impl<T> Send for Gc<T> where T: 'static + Sized + Trace + Send + Sync {}

impl<T> Deref for Gc<T>
where
    T: 'static + Sized + Trace,
{
    type Target = GcPtr<T>;

    /// Safe without holding the STW lock because as long as this `Gc<T>` handle
    /// exists, the object is reachable (root) and cannot be collected.
    fn deref(&self) -> &Self::Target {
        // SAFETY: Pointer is valid for the lifetime of this Gc handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { &(*self.ptr) }
    }
}

impl<T> std::fmt::Debug for Gc<T>
where
    T: 'static + Sized + Trace + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Gc").field(&***self).finish()
    }
}

impl<T> Gc<T>
where
    T: Sized + Trace,
{
    /// The object's id in the global collector.
    #[inline]
    pub(crate) fn object_id(&self) -> ObjectId {
        // SAFETY: the object is alive while this handle exists.
        unsafe { (*self.ptr).info.object_id }
    }

    /// Allocate a new thread-safe GC-managed object on the global collector.
    /// Starts the background collection strategy if not already active.
    pub fn new(t: T) -> Gc<T>
    where
        T: Send + Sync,
    {
        basic_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).create_gc(t, region)
        }
    }

    /// Allocate a new thread-safe GC-managed object in the specified region.
    pub fn new_in(t: T, region: SyncRegionId) -> Gc<T>
    where
        T: Send + Sync,
    {
        basic_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).create_gc(t, region.0) }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<Gc<T>, crate::gc::GcAllocError>
    where
        T: Send + Sync,
    {
        basic_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).try_create_gc(t, region)
        }
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(t: T, region: SyncRegionId) -> Result<Gc<T>, crate::gc::GcAllocError>
    where
        T: Send + Sync,
    {
        basic_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).try_create_gc(t, region.0) }
    }
}

impl<T> Clone for Gc<T>
where
    T: 'static + Sized + Trace,
{
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: this handle keeps the object alive.
        add_handle(unsafe { &(*self.ptr).info });
        Gc { ptr: self.ptr }
    }
}

impl<T> Drop for Gc<T>
where
    T: Sized + Trace,
{
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `ptr` is this handle's object; its header is `info`.
        unsafe { drop_handle(self.ptr as *const u8, &raw const (*self.ptr).info) };
    }
}

impl<T> Trace for Gc<T>
where
    T: Sized + Trace,
{
    /// Roots are counted per object (`SyncGcInfo::root_count`), not flagged
    /// per handle.
    fn is_root(&self) -> bool {
        false
    }

    fn reset_root(&self) {
        // SAFETY: the object is alive while this handle exists.
        unsafe { cascade_handle(&*self.ptr, &(*self.ptr).info) }
    }

    fn trace(&self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: Null check above ensures the pointer is valid.
        unsafe { (*self.ptr).trace() }
    }

    fn reset(&self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: Null check above ensures the pointer is valid.
        unsafe { (*self.ptr).reset() }
    }

    fn is_traceable(&self) -> bool {
        if self.ptr.is_null() {
            return false;
        }
        // SAFETY: Null check above ensures the pointer is valid.
        unsafe { (*self.ptr).is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for Gc<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A garbage-collected mutable cell that can be shared across threads.
///
/// The thread-safe counterpart of the local `GcCell<T>`, with reader-writer
/// lock semantics: any number of [`borrow`](Self::borrow)s or one
/// [`borrow_mut`](Self::borrow_mut) at a time, across all threads and clones.
/// A borrow that conflicts with one held elsewhere waits for it; the `try_`
/// variants return `None` instead. A conflicting borrow on the thread that
/// already holds the cell deadlocks, as with `RwLock`.
///
/// `borrow_mut()` triggers the write barrier for generational collection. The
/// collector never waits for a borrow: a collection that finds a cell mutably
/// borrowed frees nothing and is retried later.
pub struct GcCell<T>
where
    T: 'static + Sized + Trace,
{
    ptr: *const GcCellBox<T>,
}

// SAFETY: every handle (and the collector) reaches the same value, so moving
// or sharing a handle shares `&T` across threads (T: Sync) and lets any
// thread take `&mut T` or drop the value (T: Send). The value is only reached
// under the cell's lock.
unsafe impl<T> Sync for GcCell<T> where T: 'static + Sized + Trace + Send + Sync {}
unsafe impl<T> Send for GcCell<T> where T: 'static + Sized + Trace + Send + Sync {}

/// A shared borrow of a [`GcCell`]'s value, from [`GcCell::borrow`].
pub struct GcCellRef<'a, T: 'static> {
    value: &'a T,
    _guard: RwLockReadGuard<'a, ()>,
}

impl<T> Deref for GcCellRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for GcCellRef<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

/// A mutable borrow of a [`GcCell`]'s value, from [`GcCell::borrow_mut`].
pub struct GcCellRefMut<'a, T: 'static> {
    value: &'a mut T,
    _guard: RwLockWriteGuard<'a, ()>,
}

impl<T> Deref for GcCellRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<T> DerefMut for GcCellRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for GcCellRefMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

impl<T> Drop for GcCell<T>
where
    T: Sized + Trace,
{
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `ptr` is this handle's object; its header is `info`.
        unsafe { drop_handle(self.ptr as *const u8, &raw const (*self.ptr).info) };
    }
}

impl<T> std::fmt::Debug for GcCell<T>
where
    T: 'static + Sized + Trace + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.try_borrow() {
            Some(value) => f.debug_tuple("GcCell").field(&*value).finish(),
            None => f.write_str("GcCell(<borrowed>)"),
        }
    }
}

impl<T> GcCell<T>
where
    T: 'static + Sized + Trace,
{
    /// Allocate a new thread-safe GC-managed interior-mutable cell on the global collector.
    pub fn new(t: T) -> GcCell<T>
    where
        T: Send + Sync,
    {
        basic_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).create_gc_cell(t, region)
        }
    }

    /// Allocate a new thread-safe GC-managed interior-mutable cell in the specified region.
    pub fn new_in(t: T, region: SyncRegionId) -> GcCell<T>
    where
        T: Send + Sync,
    {
        basic_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).create_gc_cell(t, region.0) }
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<GcCell<T>, crate::gc::GcAllocError>
    where
        T: Send + Sync,
    {
        basic_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe {
            let region = GLOBAL_GC.core.current_region();
            (*GLOBAL_GC).try_create_gc_cell(t, region)
        }
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(t: T, region: SyncRegionId) -> Result<GcCell<T>, crate::gc::GcAllocError>
    where
        T: Send + Sync,
    {
        basic_strategy_start();
        GLOBAL_GC_STRATEGY.ensure_started();
        unsafe { (*GLOBAL_GC).try_create_gc_cell(t, region.0) }
    }

    /// The object's id in the global collector.
    #[inline]
    pub(crate) fn object_id(&self) -> ObjectId {
        self.cell().info.object_id
    }

    #[inline]
    fn cell(&self) -> &GcCellBox<T> {
        // SAFETY: the object is alive while this handle exists.
        unsafe { &*self.ptr }
    }

    /// Shared borrow of the value. Waits while another thread holds a
    /// [`borrow_mut`](Self::borrow_mut).
    pub fn borrow(&self) -> GcCellRef<'_, T> {
        let cell = self.cell();
        let guard = cell.lock.read().unwrap_or_else(|e| e.into_inner());
        GcCellRef {
            // SAFETY: the read lock excludes every writer.
            value: unsafe { &*cell.value.get() },
            _guard: guard,
        }
    }

    /// Shared borrow, or `None` while a [`borrow_mut`](Self::borrow_mut) is held.
    pub fn try_borrow(&self) -> Option<GcCellRef<'_, T>> {
        let cell = self.cell();
        let guard = match cell.lock.try_read() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        Some(GcCellRef {
            // SAFETY: the read lock excludes every writer.
            value: unsafe { &*cell.value.get() },
            _guard: guard,
        })
    }

    /// Mutable borrow with write barrier. Waits while any other borrow of the
    /// cell is held.
    ///
    /// Triggers the write barrier so that if this object is in an older
    /// generation, its card is marked dirty for young-generation collections.
    pub fn borrow_mut(&self) -> GcCellRefMut<'_, T> {
        loop {
            if let Some(borrow) = self.try_borrow_mut() {
                return borrow;
            }
            // Wait for the other borrows to end while holding no GC lock, so
            // a collection is never held up by a thread waiting here.
            drop(self.cell().lock.write().unwrap_or_else(|e| e.into_inner()));
        }
    }

    /// Mutable borrow with write barrier, or `None` while any other borrow of
    /// the cell is held. May wait for a running collection to finish.
    pub fn try_borrow_mut(&self) -> Option<GcCellRefMut<'_, T>> {
        let cell = self.cell();
        // Under the STW read lock, so no collection is running while the cell
        // is locked here: a collection holds the STW write lock throughout,
        // so the only mutable borrows it can meet are ones taken before it
        // started - and those it detects (see `GcCellBox`).
        let _stw = GLOBAL_GC
            .core
            .stw_lock
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let guard = match cell.lock.try_write() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        GLOBAL_GC
            .core
            .write_barrier(self.object_id(), self.ptr as *const dyn Trace);
        Some(GcCellRefMut {
            // SAFETY: the write lock excludes every other borrow.
            value: unsafe { &mut *cell.value.get() },
            _guard: guard,
        })
    }
}

impl<T> Clone for GcCell<T>
where
    T: 'static + Sized + Trace,
{
    #[inline]
    fn clone(&self) -> Self {
        add_handle(&self.cell().info);
        GcCell { ptr: self.ptr }
    }
}

impl<T> Trace for GcCell<T>
where
    T: Sized + Trace,
{
    /// Roots are counted per object, not flagged per handle (see `Gc`).
    fn is_root(&self) -> bool {
        false
    }

    fn reset_root(&self) {
        // SAFETY: the object is alive while this handle exists.
        unsafe { cascade_handle(self.cell(), &self.cell().info) }
    }

    fn trace(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
            (*self.ptr).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        // SAFETY: Pointer is valid for the lifetime of this GcCell handle; the GC guarantees the allocation is not freed while any handle exists.
        unsafe { (*self.ptr).is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.ptr as *const dyn Trace);
    }
}

impl<T> Finalize for GcCell<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A weak reference to a global GC-managed object.
///
/// Does not prevent collection. `upgrade()` acquires the STW read lock
/// to safely check liveness before returning a strong `Gc<T>`.
pub struct GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    alive: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ptr: *const GcPtr<T>,
    object_id: ObjectId,
}

// SAFETY: GcWeak holds only an Arc<AtomicBool> (thread-safe) and a raw pointer
// that is only dereferenced through the Mutex-protected global GC during upgrade().
unsafe impl<T> Send for GcWeak<T> where T: 'static + Sized + Trace + Send + Sync {}
unsafe impl<T> Sync for GcWeak<T> where T: 'static + Sized + Trace + Send + Sync {}

impl<T> Clone for GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        GcWeak {
            alive: self.alive.clone(),
            ptr: self.ptr,
            object_id: self.object_id,
        }
    }
}

impl<T> std::fmt::Debug for GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.alive.load(std::sync::atomic::Ordering::Relaxed) {
            f.write_str("GcWeak(alive)")
        } else {
            f.write_str("GcWeak(dead)")
        }
    }
}

impl<T> GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    /// Returns `true` if the referenced object is still alive (not yet collected).
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

    /// Attempt to upgrade the weak reference to a strong `Gc<T>`.
    /// Returns `None` if the object has been collected.
    pub fn upgrade(&self) -> Option<Gc<T>> {
        if !self.alive.load(Ordering::Acquire) {
            return None;
        }
        unsafe {
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            // Acquire STW read lock to prevent collection during upgrade
            let _stw = GLOBAL_GC
                .core
                .stw_lock
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if self.alive.load(Ordering::Acquire) {
                (*GLOBAL_GC).upgrade_weak(self)
            } else {
                None
            }
        }
    }
}

impl<T> Gc<T>
where
    T: 'static + Sized + Trace,
{
    /// Create a weak reference that does not prevent collection.
    pub fn downgrade(this: &Gc<T>) -> GcWeak<T> {
        let object_id = this.object_id();
        let alive = GLOBAL_GC.core.get_or_create_weak_alive(object_id);
        GcWeak {
            alive,
            ptr: this.ptr,
            object_id,
        }
    }
}

impl<T> Trace for GcWeak<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        unreachable!("is_root should never be called on GcWeak !!");
    }
    fn reset_root(&self) {}
    fn trace(&self) {}
    fn reset(&self) {}
    fn is_traceable(&self) -> bool {
        unreachable!("is_traceable should never be called on GcWeak !!");
    }
}

impl<T> Finalize for GcWeak<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// Thread-safe global garbage collector.
/// Shared across all threads via the `GLOBAL_GC` lazy_static.
/// Internally delegates to `GarbageCollector` which protects all state
/// with Mutex, RwLock, and atomic operations.
pub struct GlobalGarbageCollector {
    pub(crate) core: GarbageCollector<SyncObjects>,
}

// SAFETY: Delegates to GarbageCollector which protects all state with Mutex/atomics.
unsafe impl Sync for GlobalGarbageCollector {}
unsafe impl Send for GlobalGarbageCollector {}

#[allow(dead_code)]
impl GlobalGarbageCollector {
    fn new() -> GlobalGarbageCollector {
        GlobalGarbageCollector {
            core: GarbageCollector::with_shards(crate::sharded_map::MAX_SHARDS),
        }
    }

    /// Allocate `obj` with its `ObjectEntry` in one block and register it in
    /// this thread's shard. Its header already counts the one handle the
    /// caller is about to create. `None` if allocation failed (`obj` is
    /// dropped).
    unsafe fn register_object<A: SyncObject>(
        &self,
        obj: A,
        region: crate::generation::RegionId,
    ) -> Option<*const A> {
        unsafe {
            let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
            let (obj_ptr, oe_ptr, block_layout) = alloc_object_block::<A>()?;
            std::ptr::write(obj_ptr, obj);
            let info = A::info_ptr(obj_ptr);
            let root_ref_count_offset =
                ((&raw const (*info).root_ref_count) as usize - obj_ptr as usize) as u16;
            std::ptr::write(
                oe_ptr,
                ObjectEntry {
                    ptr: obj_ptr as *const dyn Trace,
                    // Frees the whole block, entry included.
                    mem: GcObjMem::Mem,
                    layout: CompactLayout::from_layout(block_layout),
                    gen_survive_region: region.0 & 0xFFFF,
                    tracers: TracerList::Empty,
                    handle_count: 0,
                    root_count: 0,
                    root_ref_count_offset,
                    entry_block: crate::gc::ENTRY_EMBEDDED,
                },
            );
            // Shard-scoped registration: insert into this thread's own shard so
            // concurrent allocation does not serialize on one global lock.
            // ptr_to_object, region stats, and card table are populated lazily
            // (during marking/promotion), not on the allocation hot path.
            let obj_id = self.core.with_own_shard(|shard_map, shard| {
                let obj_id = SyncObjects::tag_key(shard_map.insert(ObjectEntryRef(oe_ptr)), shard);
                // Before the shard unlocks: from then on the collector sees it.
                (&raw mut (*info).object_id).write(obj_id);
                obj_id
            });
            self.core.track_alloc(block_layout.size());
            self.core.note_new_object_during_marking(obj_id);
            self.core.allocation_count.add(1);
            Some(obj_ptr)
        }
    }

    unsafe fn create_gc<T>(&self, t: T, region: crate::generation::RegionId) -> Gc<T>
    where
        T: Sized + Trace,
    {
        let ptr = unsafe { self.register_object(GcPtr::new(t), region) }
            .unwrap_or_else(|| std::alloc::handle_alloc_error(object_block_layout::<GcPtr<T>>().0));
        Gc { ptr }
    }

    unsafe fn create_gc_cell<T>(&self, t: T, region: crate::generation::RegionId) -> GcCell<T>
    where
        T: Sized + Trace,
    {
        let ptr = unsafe { self.register_object(GcCellBox::new(t), region) }.unwrap_or_else(|| {
            std::alloc::handle_alloc_error(object_block_layout::<GcCellBox<T>>().0)
        });
        GcCell { ptr }
    }

    /// Fallible allocation: `GcAllocError` instead of aborting. No emergency
    /// collection - the STW read lock is held while allocating, and a
    /// collection takes the write lock, so a retry could only deadlock.
    unsafe fn try_create_gc<T>(
        &self,
        t: T,
        region: crate::generation::RegionId,
    ) -> Result<Gc<T>, crate::gc::GcAllocError>
    where
        T: Sized + Trace,
    {
        let ptr = unsafe { self.register_object(GcPtr::new(t), region) }
            .ok_or(crate::gc::GcAllocError)?;
        Ok(Gc { ptr })
    }

    /// Fallible cell allocation; see [`Self::try_create_gc`].
    unsafe fn try_create_gc_cell<T>(
        &self,
        t: T,
        region: crate::generation::RegionId,
    ) -> Result<GcCell<T>, crate::gc::GcAllocError>
    where
        T: Sized + Trace,
    {
        let ptr = unsafe { self.register_object(GcCellBox::new(t), region) }
            .ok_or(crate::gc::GcAllocError)?;
        Ok(GcCell { ptr })
    }

    unsafe fn upgrade_weak<T>(&self, weak: &GcWeak<T>) -> Option<Gc<T>>
    where
        T: Sized + Trace,
    {
        // No collection runs while the STW read lock is held, and the
        // object's shard lock excludes the paths that free it: both
        // `release_object` (with weak refs present) and a sweep clear `alive`
        // before freeing, under that lock. So seeing `alive` under it means
        // the object is still there.
        let _stw = self.core.stw_lock.read().unwrap_or_else(|e| e.into_inner());
        self.core.with_shard_of(weak.object_id, |_shard| {
            if !weak.alive.load(Ordering::Acquire) {
                return None;
            }
            // SAFETY: the object is alive (see above).
            let info = unsafe { &(*weak.ptr).info };
            let mut n = info.handle_count.load(Ordering::Relaxed);
            loop {
                // Zero: the last handle was dropped and the object is being
                // freed; it cannot come back.
                if n == 0 {
                    return None;
                }
                match info.handle_count.compare_exchange_weak(
                    n,
                    n + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Some(Gc { ptr: weak.ptr }),
                    Err(current) => n = current,
                }
            }
        })
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect(&self) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.collect();
        }
    }

    #[allow(dead_code)]
    unsafe fn collect_all(&self) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.collect_all();
        }
    }

    /// Parallel collection: serial mark phase, parallel sweep/dealloc using rayon.
    /// Collects objects in generations 0..=max_gen.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn collect_parallel(
        &self,
        max_gen: crate::generation::Generation,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_parallel(max_gen) }
    }

    /// Collection with parallel mark AND parallel sweep using rayon.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn collect_parallel_mark(
        &self,
        max_gen: crate::generation::Generation,
    ) -> crate::generation::CollectionStats {
        unsafe { self.core.collect_parallel_mark(max_gen) }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn begin_collection(&self, max_gen: crate::generation::Generation) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.begin_collection(max_gen);
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn mark_step(&self, budget: usize) -> bool {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.mark_step(budget) }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn finish_collection(&self) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.finish_collection() }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_incremental(
        &self,
        max_gen: crate::generation::Generation,
        step_budget: usize,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_incremental(max_gen, step_budget) }
    }

    /// Begin concurrent collection: short STW to snapshot roots + edges.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn begin_concurrent_collection(
        &self,
        max_gen: crate::generation::Generation,
    ) {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core.begin_concurrent_collection(max_gen);
        }
    }

    /// Process gray objects using edge snapshot. NO STW lock — safe for concurrent use.
    pub(crate) fn concurrent_mark_step(&self, budget: usize) -> bool {
        self.core.concurrent_mark_step(budget)
    }

    /// Run a complete concurrent collection.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_concurrent(
        &self,
        max_gen: crate::generation::Generation,
        step_budget: usize,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_concurrent(max_gen, step_budget) }
    }

    /// Time-budgeted incremental mark step. See `GarbageCollector::mark_step_timed`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn mark_step_timed(&self, max_duration: Duration) -> bool {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.mark_step_timed(max_duration) }
    }

    /// Run a complete incremental collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_incremental_timed(
        &self,
        max_gen: crate::generation::Generation,
        max_step_duration: Duration,
    ) -> crate::generation::CollectionStats {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core
                .collect_incremental_timed(max_gen, max_step_duration)
        }
    }

    /// Time-budgeted concurrent mark step. NO STW lock — safe for concurrent use.
    pub(crate) fn concurrent_mark_step_timed(&self, max_duration: Duration) -> bool {
        self.core.concurrent_mark_step_timed(max_duration)
    }

    /// Run a complete concurrent collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_concurrent_timed(
        &self,
        max_gen: crate::generation::Generation,
        max_step_duration: Duration,
    ) -> crate::generation::CollectionStats {
        unsafe {
            // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
            self.core
                .collect_concurrent_timed(max_gen, max_step_duration)
        }
    }

    /// Create a new region. Future allocations go into this region.
    pub fn new_region(&self) -> SyncRegionId {
        SyncRegionId(self.core.new_region())
    }

    /// Get the current allocation region.
    pub fn current_region(&self) -> SyncRegionId {
        SyncRegionId(self.core.current_region())
    }

    /// Collect only objects in the specified region.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_region(
        &self,
        region: SyncRegionId,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract that no GC-managed references are in use.
        unsafe { self.core.collect_region(region.0) }
    }

    /// G1-style "Garbage First" collection. Marks all objects, then sweeps
    /// regions with the highest garbage ratio first, stopping when the elapsed
    /// time reaches `pause_target`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are held
    /// across this call (stop-the-world requirement).
    pub(crate) unsafe fn collect_garbage_first(
        &self,
        pause_target: std::time::Duration,
    ) -> crate::generation::CollectionStats {
        // SAFETY: Caller upholds the safety contract.
        unsafe { self.core.collect_garbage_first(pause_target) }
    }

    /// Return per-region liveness statistics without running a collection.
    pub fn region_stats(&self) -> Vec<crate::generation::RegionStats> {
        self.core.region_stats()
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> crate::generation::GcStats {
        self.core.stats()
    }

    /// Set custom promotion thresholds.
    pub fn set_promotion_config(&self, config: crate::generation::PromotionConfig) {
        self.core.set_promotion_config(config);
    }

    /// Get the current promotion config.
    pub fn promotion_config(&self) -> crate::generation::PromotionConfig {
        self.core.promotion_config()
    }

    /// Register a callback invoked after each collection cycle.
    pub fn set_on_collection(
        &self,
        callback: impl Fn(&crate::generation::CollectionStats) + Send + Sync + 'static,
    ) {
        self.core.set_on_collection(callback);
    }

    /// Remove the collection callback.
    pub fn clear_on_collection(&self) {
        self.core.clear_on_collection();
    }

    /// Compact the heap by copying all live objects into a contiguous buffer.
    /// Runs a full collection first, then relocates surviving objects.
    ///
    /// Returns the number of objects compacted.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    /// Acquires STW write lock internally.
    pub(crate) unsafe fn compact(&self) -> usize {
        unsafe { self.core.compact() }
    }
}

// ---- Feature-gated public access for benchmarks and integration tests ----

#[cfg(feature = "_internal")]
impl GlobalGarbageCollector {
    pub unsafe fn _collect(&self) {
        unsafe { self.collect() }
    }
    pub unsafe fn _collect_incremental(
        &self,
        max_gen: crate::generation::Generation,
        step_budget: usize,
    ) -> crate::generation::CollectionStats {
        unsafe { self.collect_incremental(max_gen, step_budget) }
    }
    pub unsafe fn _collect_incremental_timed(
        &self,
        max_gen: crate::generation::Generation,
        max_step_duration: Duration,
    ) -> crate::generation::CollectionStats {
        unsafe { self.collect_incremental_timed(max_gen, max_step_duration) }
    }
    pub unsafe fn _collect_concurrent_timed(
        &self,
        max_gen: crate::generation::Generation,
        max_step_duration: Duration,
    ) -> crate::generation::CollectionStats {
        unsafe { self.collect_concurrent_timed(max_gen, max_step_duration) }
    }
    pub unsafe fn _collect_region(
        &self,
        region: SyncRegionId,
    ) -> crate::generation::CollectionStats {
        unsafe { self.collect_region(region) }
    }
    pub unsafe fn _compact(&self) -> usize {
        unsafe { self.compact() }
    }
}

// ---- SyncRegionId helpers for global (sync) GC ----

impl SyncRegionId {
    /// Allocate a new `sync::Gc<T>` in this region (global collector).
    pub fn gc<T: 'static + Sized + Trace + Send + Sync>(self, t: T) -> Gc<T> {
        Gc::new_in(t, self)
    }

    /// Fallible `sync::Gc<T>` allocation in this region (global collector).
    pub fn try_gc<T: 'static + Sized + Trace + Send + Sync>(
        self,
        t: T,
    ) -> Result<Gc<T>, crate::gc::GcAllocError> {
        Gc::try_new_in(t, self)
    }

    /// Allocate a new `sync::GcCell<T>` in this region (global collector).
    pub fn gc_cell<T: 'static + Sized + Trace + Send + Sync>(self, t: T) -> GcCell<T> {
        GcCell::new_in(t, self)
    }

    /// Fallible `sync::GcCell<T>` allocation in this region (global collector).
    pub fn try_gc_cell<T: 'static + Sized + Trace + Send + Sync>(
        self,
        t: T,
    ) -> Result<GcCell<T>, crate::gc::GcAllocError> {
        GcCell::try_new_in(t, self)
    }
}

/// Callback type for a global collection strategy.
pub type GlobalStrategyFn =
    Box<dyn FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;

/// Pluggable collection strategy for the global (thread-safe) GC.
/// Controls when and how garbage collection runs. Use `set_strategy` to swap at runtime.
pub struct GlobalStrategy {
    gc: Cell<&'static GlobalGarbageCollector>,
    is_active: AtomicBool,
    strategy_func: Mutex<GlobalStrategyFn>,
    join_handle: Mutex<Option<JoinHandle<()>>>,
}

// SAFETY: The Cell<&'static GlobalGarbageCollector> is only written during initialization
// (before any thread can access it). All other fields use Mutex or atomics.
unsafe impl Sync for GlobalStrategy {}
unsafe impl Send for GlobalStrategy {}

impl GlobalStrategy {
    fn new<StrategyFn>(
        gc: &'static GlobalGarbageCollector,
        strategy_fn: StrategyFn,
    ) -> GlobalStrategy
    where
        StrategyFn: 'static
            + FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
    {
        GlobalStrategy {
            gc: Cell::new(gc),
            is_active: AtomicBool::new(false),
            strategy_func: Mutex::new(Box::new(strategy_fn)),
            join_handle: Mutex::new(None),
        }
    }

    /// Replace the current collection strategy. Stops the current strategy first if active.
    pub fn set_strategy<StrategyFn>(&self, strategy_fn: StrategyFn)
    where
        StrategyFn: 'static
            + FnMut(&'static GlobalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
    {
        if self.is_active() {
            self.stop();
        }
        let mut strategy_func = self.strategy_func.lock().unwrap_or_else(|e| e.into_inner());
        *strategy_func = Box::new(strategy_fn);
    }

    /// Returns `true` if the strategy's background collection is currently running.
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Atomically check if inactive and start. Safe to call from multiple threads.
    #[inline]
    pub fn ensure_started(&'static self) {
        // Called on every allocation from every thread. A plain load keeps the
        // flag's cache line shared; a compare_exchange takes it exclusive even
        // when it fails, so every allocation would bounce it between cores.
        if self.is_active.load(Ordering::Acquire) {
            return;
        }
        if self
            .is_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let mut strategy_func = self.strategy_func.lock().unwrap_or_else(|e| e.into_inner());
            let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
            *join_handle = (*(strategy_func))(self.gc.get(), &self.is_active);
        }
    }

    /// Start the collection strategy (e.g., spawn a background collection thread).
    pub fn start(&'static self) {
        self.is_active.store(true, Ordering::Release);
        let mut strategy_func = self.strategy_func.lock().unwrap_or_else(|e| e.into_inner());
        let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
        *join_handle = (*(strategy_func))(self.gc.get(), &self.is_active);
    }

    /// Stop the collection strategy and join any background thread.
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        let mut join_handle = self.join_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(join_handle) = join_handle.take() {
            join_handle
                .join()
                .expect("GlobalStrategy::stop: strategy thread panicked");
        }
    }
}

impl Drop for GlobalStrategy {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }
}

lazy_static! {
    pub static ref GLOBAL_GC: GlobalGarbageCollector = GlobalGarbageCollector::new();
    pub static ref GLOBAL_GC_STRATEGY: GlobalStrategy = {
        let gc = &(*GLOBAL_GC);
        GlobalStrategy::new(gc, move |global_gc, _| {
            let mut basic_strategy_global_gc = BASIC_STRATEGY_GLOBAL_GC
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *basic_strategy_global_gc = Some(global_gc);
            None
        })
    };
}

#[cfg(test)]
mod tests {
    use crate::gc::sync::{GLOBAL_GC, Gc, GcCell};
    use std::sync::Mutex;

    // Serialize sync GC tests since they share GLOBAL_GC.
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    /// Live handles to one object (its header's count).
    fn handles_of(e: &crate::gc::ObjectEntryRef) -> usize {
        // SAFETY: entries of the global collector are sync objects, valid
        // while the maps are locked.
        unsafe {
            e.sync_info()
                .handle_count
                .load(std::sync::atomic::Ordering::Relaxed) as usize
        }
    }

    /// Clean residual state and return the baseline handle count.
    fn setup() -> (std::sync::MutexGuard<'static, ()>, usize) {
        let guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        let baseline = (*GLOBAL_GC)
            .core
            .lock_gc_maps()
            .objects
            .values()
            .map(|e| handles_of(e))
            .sum::<usize>();
        (guard, baseline)
    }

    #[test]
    fn one_object() {
        let (_guard, baseline) = setup();
        let _one = Gc::new(1);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            1
        );
    }

    #[test]
    fn gc_collect_one_from_one() {
        let (_guard, baseline) = setup();
        {
            let _one = Gc::new(1);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            0
        );
    }

    #[test]
    #[allow(unused_assignments)]
    fn two_objects_reassign() {
        let (_guard, baseline) = setup();
        let mut one = Gc::new(1);
        one = Gc::new(2);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        // Reassignment drops old Gc (remove_tracer), so only 1 tracer remains
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            1
        );
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_after_reassign() {
        let (_guard, baseline) = setup();
        let mut one = Gc::new(1);
        one = Gc::new(2);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        // one is still live, so 1 tracer remains
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            1
        );
        drop(one);
    }

    #[test]
    #[allow(unused_assignments)]
    fn gc_collect_two_from_two() {
        let (_guard, baseline) = setup();
        {
            let mut one = Gc::new(1);
            one = Gc::new(2);
            drop(one);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            0
        );
    }

    #[test]
    fn stw_blocks_allocation_during_collection() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let (_guard, _) = setup();

        let allocated_during_stw = Arc::new(AtomicBool::new(false));
        let collection_started = Arc::new(AtomicBool::new(false));
        let collection_done = Arc::new(AtomicBool::new(false));

        // Hold the STW write lock to simulate an ongoing collection
        let stw_guard = (*GLOBAL_GC)
            .core
            .stw_lock
            .write()
            .unwrap_or_else(|e| e.into_inner());
        collection_started.store(true, Ordering::Release);

        let alloc_flag = allocated_during_stw.clone();
        let started = collection_started.clone();
        let _done = collection_done.clone();

        let handle = std::thread::spawn(move || {
            // Wait for collection to be in progress
            while !started.load(Ordering::Acquire) {}
            // Try to allocate — should block on STW read lock
            let _obj = Gc::new(42);
            alloc_flag.store(true, Ordering::Release);
        });

        // Give the allocator thread time to block
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Allocation should NOT have completed while we hold the write lock
        assert!(
            !allocated_during_stw.load(Ordering::Acquire),
            "allocation must block while STW write lock is held"
        );

        // Release the STW lock — simulates end of collection
        drop(stw_guard);
        collection_done.store(true, Ordering::Release);

        handle.join().expect("allocator thread panicked");
        assert!(
            allocated_during_stw.load(Ordering::Acquire),
            "allocation should complete after STW lock is released"
        );
    }

    #[test]
    fn stw_allows_concurrent_allocations() {
        let (_guard, _) = setup();
        // Multiple threads allocating concurrently should all succeed (read locks are shared)
        let handles: Vec<_> = (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    let _obj = Gc::new(i);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("concurrent allocation thread panicked");
        }
    }

    #[test]
    fn set_strategy_while_active_does_not_deadlock() {
        let (_guard, _) = setup();
        let _gc = Gc::new(1); // ensures strategy is started
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use crate::gc::sync::GLOBAL_GC_STRATEGY;
            GLOBAL_GC_STRATEGY.set_strategy(|_gc, _| None);
            tx.send(()).unwrap();
        });
        rx.recv_timeout(std::time::Duration::from_secs(3))
            .expect("set_strategy deadlocked when called while active");
    }

    #[test]
    fn incremental_collects_dead_objects() {
        // RC hybrid frees non-cyclic objects immediately. Verify object is gone.
        let (_guard, baseline) = setup();
        {
            let _obj = Gc::new(42);
        }
        // Object already freed by RC
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            0
        );
        // Incremental still works correctly
        let _stats =
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            unsafe { (*GLOBAL_GC).collect_incremental(crate::generation::Generation::Gen2, 10) };
    }

    #[test]
    fn incremental_preserves_live_objects() {
        let (_guard, baseline) = setup();
        let _live = Gc::new(99);
        let stats =
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            unsafe { (*GLOBAL_GC).collect_incremental(crate::generation::Generation::Gen2, 10) };
        assert_eq!(
            stats.objects_collected, 0,
            "incremental must not collect live objects"
        );
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            1
        );
    }

    #[test]
    fn incremental_step_by_step_sync() {
        // With RC hybrid, dead objects are already freed. Test with live object.
        let (_guard, _) = setup();
        let _live = Gc::new(2);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe {
            (*GLOBAL_GC).begin_collection(crate::generation::Generation::Gen2);
            while !(*GLOBAL_GC).mark_step(1) {}
            let stats = (*GLOBAL_GC).finish_collection();
            assert_eq!(
                stats.objects_collected, 0,
                "live object must not be collected"
            );
        }
    }

    #[test]
    fn try_new_succeeds() {
        let (_guard, _) = setup();
        let result = Gc::try_new(42);
        assert!(
            result.is_ok(),
            "try_new should succeed for normal allocation"
        );
        assert_eq!(**result.unwrap(), 42);
    }

    #[test]
    fn try_new_object_collected_when_dead() {
        let (_guard, baseline) = setup();
        {
            let _obj = Gc::try_new(77).unwrap();
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            0
        );
    }

    // --- Diagnostics API tests ---

    #[test]
    fn sync_stats_reports_live_objects() {
        let (_guard, _) = setup();
        let baseline = (*GLOBAL_GC).stats().live_objects;
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let stats = (*GLOBAL_GC).stats();
        assert_eq!(stats.live_objects - baseline, 2);
    }

    #[test]
    fn sync_stats_reports_heap_size() {
        let (_guard, _) = setup();
        let baseline = (*GLOBAL_GC).stats().heap_size;
        let _a = Gc::new(42i32);
        let stats = (*GLOBAL_GC).stats();
        assert!(stats.heap_size > baseline);
    }

    #[test]
    fn sync_stats_tracks_total_collections() {
        let (_guard, _) = setup();
        let before = (*GLOBAL_GC).stats().total_collections;
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        let after = (*GLOBAL_GC).stats().total_collections;
        // Use >= because the background basic-strategy thread may collect concurrently.
        assert!(
            after - before >= 2,
            "expected at least 2 collections, got {}",
            after - before
        );
    }

    #[test]
    fn sync_stats_tracks_last_collection() {
        let (_guard, _) = setup();
        // With RC hybrid, dead objects are already freed. Just verify stats recording.
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
        let stats = (*GLOBAL_GC).stats();
        assert!(
            stats.last_collection.is_some(),
            "last_collection should be Some after collect"
        );
    }

    #[test]
    fn sync_stats_reports_allocation_count() {
        let (_guard, baseline) = setup();
        let _a = Gc::new(1);
        let _b = Gc::new(2);
        let stats = (*GLOBAL_GC).stats();
        // allocation_count can be reset by background strategy calling collect(),
        // so check live_objects instead — we hold both Gc references alive.
        let live: usize = (*GLOBAL_GC)
            .core
            .lock_gc_maps()
            .objects
            .values()
            .map(|e| handles_of(e))
            .sum();
        assert!(
            live - baseline >= 2,
            "expected at least 2 new handles, got {}",
            live - baseline
        );
        // Stats should still report some live objects
        assert!(stats.live_objects >= 2);
    }

    // --- Debug impl tests ---

    #[test]
    fn sync_debug_gc_prints_value() {
        let (_guard, _) = setup();
        let gc = Gc::new(42);
        assert_eq!(format!("{:?}", gc), "Gc(42)");
    }

    #[test]
    fn sync_debug_gc_weak() {
        let (_guard, _) = setup();
        let strong = Gc::new(1);
        let weak = Gc::downgrade(&strong);
        assert_eq!(format!("{:?}", weak), "GcWeak(alive)");
    }

    // --- Cycle tests ---

    struct SyncCyclicNode {
        next: GcCell<Option<Gc<SyncCyclicNode>>>,
    }
    impl crate::gc::Trace for SyncCyclicNode {
        fn is_root(&self) -> bool {
            false
        }
        fn reset_root(&self) {
            self.next.reset_root();
        }
        fn trace(&self) {
            self.next.trace();
        }
        fn reset(&self) {
            self.next.reset();
        }
        fn is_traceable(&self) -> bool {
            false
        }
        fn trace_children(&self, children: &mut Vec<*const dyn crate::gc::Trace>) {
            self.next.trace_children(children);
        }
    }
    impl crate::gc::Finalize for SyncCyclicNode {
        fn finalize(&self) {}
    }

    #[test]
    fn sync_self_cycle_collected() {
        let (_guard, _) = setup();
        {
            let a = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            *a.next.borrow_mut() = Some(a.clone());
            drop(a);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
    }

    #[test]
    fn sync_three_node_cycle_collected() {
        let (_guard, _) = setup();
        {
            let a = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            let b = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            let c = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(c.clone());
            *c.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
            drop(c);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe { (*GLOBAL_GC).collect() };
    }

    /// Stops the basic strategy's background thread from collecting until
    /// dropped, for tests that assert what one explicit collection frees (the
    /// background collector could free the garbage first). Taking the write
    /// lock also waits out a background collection already running.
    struct PauseBackgroundCollector(Option<&'static super::GlobalGarbageCollector>);

    impl PauseBackgroundCollector {
        fn new() -> Self {
            let mut gc = crate::basic_strategy::BASIC_STRATEGY_GLOBAL_GC
                .write()
                .unwrap_or_else(|e| e.into_inner());
            PauseBackgroundCollector(gc.take())
        }
    }

    impl Drop for PauseBackgroundCollector {
        fn drop(&mut self) {
            *crate::basic_strategy::BASIC_STRATEGY_GLOBAL_GC
                .write()
                .unwrap_or_else(|e| e.into_inner()) = self.0;
        }
    }

    #[test]
    fn sync_cycle_collection_reports_tracers_collected() {
        let (_guard, _) = setup();
        let _paused = PauseBackgroundCollector::new();
        {
            let a = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            let b = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        let stats = unsafe {
            (*GLOBAL_GC)
                .core
                .collect_generation(crate::generation::Generation::Gen2)
        };
        assert!(
            stats.objects_collected >= 2,
            "dead 2-cycle must be collected (got {})",
            stats.objects_collected
        );
        // Regression: the sweep stopped counting (and freeing) the GcInternal
        // tracer allocations that dead cycle members hold on each other.
        assert!(
            stats.tracers_collected >= 2,
            "cycle members' remaining handles must be counted (got {})",
            stats.tracers_collected
        );
    }

    // --- Concurrent marking tests (Strategy 21) ---

    #[test]
    fn sync_concurrent_preserves_live_objects() {
        let (_guard, baseline) = setup();
        let _live = Gc::new(42);
        let stats =
            // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
            unsafe { (*GLOBAL_GC).collect_concurrent(crate::generation::Generation::Gen2, 10) };
        assert_eq!(
            stats.objects_collected, 0,
            "concurrent must not collect live objects"
        );
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .lock_gc_maps()
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>()
                - baseline,
            1
        );
    }

    #[test]
    fn sync_concurrent_step_by_step() {
        let (_guard, _) = setup();
        let _live = Gc::new(99);
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        unsafe {
            (*GLOBAL_GC).begin_concurrent_collection(crate::generation::Generation::Gen2);
            while !(*GLOBAL_GC).concurrent_mark_step(1) {}
            let stats = (*GLOBAL_GC).finish_collection();
            assert_eq!(stats.objects_collected, 0, "live objects must survive");
        }
    }

    #[test]
    fn sync_concurrent_preserves_live_graph() {
        // The sync concurrent collector marks via an edge snapshot built with
        // trace_children (begin_concurrent_collection). A live *graph* (not just
        // leaf objects) must survive a full begin -> mark_step* -> finish cycle.
        // Regression: a type with GC children but no trace_children would have
        // its children swept (see Trace::trace_children docs).
        use crate::gc::{Finalize, Trace};
        struct Node {
            next: Option<Gc<Node>>,
        }
        impl Trace for Node {
            fn is_root(&self) -> bool {
                false
            }
            fn reset_root(&self) {
                if let Some(n) = &self.next {
                    n.reset_root();
                }
            }
            fn trace(&self) {
                if let Some(n) = &self.next {
                    n.trace();
                }
            }
            fn reset(&self) {
                if let Some(n) = &self.next {
                    n.reset();
                }
            }
            fn is_traceable(&self) -> bool {
                false
            }
            fn trace_children(&self, c: &mut Vec<*const dyn Trace>) {
                if let Some(n) = &self.next {
                    n.trace_children(c);
                }
            }
        }
        impl Finalize for Node {
            fn finalize(&self) {}
        }

        let (_guard, _) = setup();
        // Build an immutable chain of 50 nodes; the head keeps all reachable.
        let mut head = Gc::new(Node { next: None });
        for _ in 0..50 {
            head = Gc::new(Node { next: Some(head) });
        }
        // Full concurrent cycle with a tiny step budget (many mark steps).
        let stats =
            // SAFETY: GLOBAL_GC is initialized once via lazy_static, valid for 'static.
            unsafe { (*GLOBAL_GC).collect_concurrent(crate::generation::Generation::Gen2, 4) };
        assert_eq!(
            stats.objects_collected, 0,
            "concurrent collection must not free the live chain"
        );
        fn len(n: &Gc<Node>) -> usize {
            1 + n.next.as_ref().map(len).unwrap_or(0)
        }
        assert_eq!(len(&head), 51, "live graph lost nodes under concurrent GC");
    }

    // --- Region-based collection tests (Strategy 22) ---

    #[test]
    fn sync_region_assignment() {
        let (_guard, _) = setup();
        let _obj = Gc::new(1);
        let region = (*GLOBAL_GC).core.current_region();
        let gc_maps = (*GLOBAL_GC).core.lock_gc_maps();
        assert!(
            gc_maps.objects.values().any(|e| e.region() == region),
            "object should be assigned to current region"
        );
    }

    #[test]
    fn sync_collect_region() {
        let (_guard, _) = setup();
        let _paused = PauseBackgroundCollector::new();
        // Create cycle in a new region
        let region = (*GLOBAL_GC).new_region();
        {
            let a = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            let b = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            drop(a);
            drop(b);
        }
        // SAFETY: GLOBAL_GC is initialized once via lazy_static and remains valid for 'static.
        let stats = unsafe { (*GLOBAL_GC).collect_region(region) };
        assert!(
            stats.objects_collected >= 2,
            "should collect cycle in target region"
        );
    }

    // --- RC hybrid tests (Strategy 23) ---

    #[test]
    fn sync_rc_immediate_dealloc() {
        let (_guard, baseline) = setup();
        {
            let _obj = Gc::new(42);
        }
        // RC should have freed it — check tracers are back to baseline
        let gc_maps = (*GLOBAL_GC).core.lock_gc_maps();
        assert_eq!(
            gc_maps
                .objects
                .values()
                .map(|e| handles_of(e))
                .sum::<usize>(),
            baseline,
            "RC should free object immediately"
        );
    }

    #[test]
    fn sync_rc_clone_keeps_alive() {
        let (_guard, _) = setup();
        let a = Gc::new(99);
        let b = a.clone();
        let baseline = (*GLOBAL_GC).core.lock_gc_maps().objects.len();
        drop(a);
        let after = (*GLOBAL_GC).core.lock_gc_maps().objects.len();
        assert_eq!(after, baseline, "object should survive when clone exists");
        assert_eq!(**b, 99);
    }

    // --- Sharded object map (thread-affine shards) ---

    fn live_objects() -> usize {
        (*GLOBAL_GC).core.lock_gc_maps().objects.len()
    }

    #[test]
    fn sharded_allocations_from_threads_use_distinct_shards_and_free() {
        use crate::slot_map::SlotKey;
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let per_thread: Vec<Vec<Gc<u64>>> = (0..4u64)
            .map(|t| {
                std::thread::spawn(move || {
                    (0..64u64)
                        .map(|i| Gc::new(t * 1000 + i))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        // Each allocating thread owns its own shard - that affinity is what
        // removes the contention.
        let shards: std::collections::HashSet<usize> = per_thread
            .iter()
            .map(|v| v[0].object_id().shard())
            .collect();
        assert_eq!(
            shards.len(),
            4,
            "each allocating thread must own a distinct shard"
        );
        for v in &per_thread {
            let s = v[0].object_id().shard();
            assert!(v.iter().all(|g| g.object_id().shard() == s));
        }
        // Every object is registered and addressable through its tagged id.
        assert_eq!(live_objects(), before + 256);
        for (t, v) in per_thread.iter().enumerate() {
            for (i, g) in v.iter().enumerate() {
                assert_eq!(***g, t as u64 * 1000 + i as u64);
            }
        }
        // Clone a handle whose object lives in another thread's shard, then drop
        // everything on this thread: cross-shard RC-hybrid frees must land.
        let extra = per_thread[2][5].clone();
        drop(per_thread);
        assert_eq!(live_objects(), before + 1);
        assert_eq!(**extra, 2005);
        drop(extra);
        assert_eq!(live_objects(), before);
    }

    #[test]
    fn sharded_drop_on_foreign_thread_frees_immediately() {
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let v: Vec<Gc<u64>> = (0..32u64).map(Gc::new).collect();
        assert_eq!(live_objects(), before + 32);
        // Dropped on a thread that owns a different shard.
        std::thread::spawn(move || drop(v)).join().unwrap();
        assert_eq!(live_objects(), before);
    }

    #[test]
    fn weak_refs_route_drops_through_full_lock_and_then_recover() {
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let strong = Gc::new(7u64);
        let weak = Gc::downgrade(&strong);
        assert!(
            (*GLOBAL_GC)
                .core
                .weak_alive_len
                .load(std::sync::atomic::Ordering::Acquire)
                > 0,
            "creating a weak ref must move drops off the shard-scoped fast path"
        );
        assert_eq!(**weak.upgrade().expect("object is alive"), 7);
        drop(strong);
        assert!(
            weak.upgrade().is_none(),
            "weak must not upgrade a freed object"
        );
        assert_eq!(live_objects(), before);
        // The slow drop re-synced the hint, so later drops are fast again.
        assert_eq!(
            (*GLOBAL_GC)
                .core
                .weak_alive_len
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
    }

    // --- One block per object: [object | first tracer | ObjectEntry] ---

    #[test]
    fn first_handle_dropped_before_clone_keeps_object_block_alive() {
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let first = Gc::new(vec![1u64, 2, 3]);
        let clone = first.clone();
        // The first handle's tracer lives inside the object's block; dropping
        // that handle must not free the block while the clone is alive.
        drop(first);
        assert_eq!(live_objects(), before + 1);
        assert_eq!(***clone, vec![1, 2, 3]);
        // Last handle gone: the whole block (object, both tracers' records,
        // entry) is released.
        drop(clone);
        assert_eq!(live_objects(), before);
    }

    #[test]
    fn many_clones_dropped_in_mixed_order_free_the_block_once() {
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let a = Gc::new(7u64);
        let clones: Vec<_> = (0..5).map(|_| a.clone()).collect();
        drop(a);
        for (i, c) in clones.into_iter().enumerate().rev() {
            assert_eq!(**c, 7, "clone {i} must still read the live object");
            drop(c);
        }
        assert_eq!(live_objects(), before);
    }

    #[test]
    fn try_new_allocates_one_block() {
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let g = Gc::try_new(11u64).expect("allocation succeeds");
        let c = GcCell::try_new(12u64).expect("allocation succeeds");
        assert_eq!(**g, 11);
        assert_eq!(*c.borrow(), 12);
        assert_eq!(live_objects(), before + 2);
        drop(g);
        drop(c);
        assert_eq!(live_objects(), before);
    }

    /// 2 KiB payload: too large for the per-thread pool.
    struct Big([u64; 256]);
    /// Over-aligned payload: the pool only serves 16-byte alignment.
    #[repr(align(64))]
    struct Aligned(u64);

    macro_rules! leaf_trace {
        ($($t:ty),*) => {$(
            impl crate::gc::Trace for $t {
                fn is_root(&self) -> bool { false }
                fn reset_root(&self) {}
                fn trace(&self) {}
                fn reset(&self) {}
                fn is_traceable(&self) -> bool { false }
            }
            impl crate::gc::Finalize for $t {
                fn finalize(&self) {}
            }
        )*};
    }
    leaf_trace!(Big, Aligned);

    #[test]
    fn pool_and_system_blocks_free_from_any_thread() {
        use super::{GcCellBox, GcPtr, object_block_layout};
        let (_guard, _baseline) = setup();
        let before = live_objects();
        // Small objects come from the allocating thread's pool, large and
        // over-aligned ones from the system allocator.
        assert!(crate::pool::serves(object_block_layout::<GcPtr<u64>>().0));
        assert!(crate::pool::serves(
            object_block_layout::<GcCellBox<u64>>().0
        ));
        assert!(!crate::pool::serves(object_block_layout::<GcPtr<Big>>().0));
        assert!(!crate::pool::serves(
            object_block_layout::<GcPtr<Aligned>>().0
        ));
        let n = if cfg!(miri) { 6 } else { 300 };
        // The allocating thread exits before any of its objects die.
        let (small, cells, big, aligned) = std::thread::spawn(move || {
            (
                (0..n).map(|i| Gc::new(i as u64)).collect::<Vec<_>>(),
                (0..n).map(|i| GcCell::new(i as u64)).collect::<Vec<_>>(),
                (0..3).map(|i| Gc::new(Big([i; 256]))).collect::<Vec<_>>(),
                (0..3).map(|i| Gc::new(Aligned(i))).collect::<Vec<_>>(),
            )
        })
        .join()
        .unwrap();
        assert_eq!(live_objects(), before + 2 * n + 6);
        // Clones (their tracers from a second thread's pool) outlive the
        // originals, which die here, on a third thread.
        let clones = std::thread::scope(|s| {
            s.spawn(|| {
                (
                    small.iter().map(Gc::clone).collect::<Vec<_>>(),
                    cells.iter().map(GcCell::clone).collect::<Vec<_>>(),
                    big.iter().map(Gc::clone).collect::<Vec<_>>(),
                    aligned.iter().map(Gc::clone).collect::<Vec<_>>(),
                )
            })
            .join()
            .unwrap()
        });
        drop((small, cells, big, aligned));
        let (small, cells, big, aligned) = clones;
        for (i, g) in small.iter().enumerate() {
            assert_eq!(***g, i as u64);
        }
        for (i, c) in cells.iter().enumerate() {
            assert_eq!(*c.borrow(), i as u64);
        }
        for (i, g) in big.iter().enumerate() {
            assert!(g.0.iter().all(|&v| v == i as u64));
        }
        for (i, g) in aligned.iter().enumerate() {
            assert_eq!(g.0, i as u64);
            assert_eq!((&***g as *const Aligned).addr() % 64, 0);
        }
        assert_eq!(live_objects(), before + 2 * n + 6);
        // The last handles die on yet another thread, which then allocates
        // again (adopting a retired heap or its pages).
        std::thread::spawn(move || {
            drop((small, cells, big, aligned));
            let g = Gc::new(7u64);
            assert_eq!(**g, 7);
        })
        .join()
        .unwrap();
        assert_eq!(live_objects(), before);
    }

    // --- GcCell: reader-writer semantics, no races with the collector ---

    #[test]
    fn gccell_borrows_while_background_collector_runs() {
        // Repro for the old RefCell flag race: the basic strategy's collector
        // thread reads the cell while this thread borrows it.
        let (_guard, _baseline) = setup();
        let cells: Vec<GcCell<u64>> = (0..6).map(GcCell::new).collect();
        for _ in 0..if cfg!(miri) { 400 } else { 2000 } {
            for (i, c) in cells.iter().enumerate() {
                assert_eq!(*c.borrow(), i as u64);
            }
        }
    }

    #[test]
    fn gccell_try_borrow_and_debug_follow_rwlock_rules() {
        let (_guard, _baseline) = setup();
        let c = GcCell::new(5u64);
        let r = c.borrow();
        assert!(c.try_borrow().is_some(), "shared borrows coexist");
        assert!(c.try_borrow_mut().is_none(), "no writer while read");
        assert_eq!(format!("{:?}", c), "GcCell(5)");
        drop(r);
        let mut w = c.borrow_mut();
        *w += 1;
        assert!(c.try_borrow().is_none(), "no reader while written");
        assert!(c.try_borrow_mut().is_none());
        assert_eq!(format!("{:?}", c), "GcCell(<borrowed>)");
        drop(w);
        assert_eq!(*c.borrow(), 6);
    }

    #[test]
    fn gccell_borrow_mut_from_many_threads_is_exclusive() {
        let (_guard, _baseline) = setup();
        let threads = 4;
        let per = if cfg!(miri) { 20 } else { 2_000 };
        let counter = GcCell::new(0u64);
        std::thread::scope(|s| {
            for _ in 0..threads {
                let counter = counter.clone();
                s.spawn(move || {
                    for _ in 0..per {
                        let mut v = counter.borrow_mut();
                        // A non-atomic read-modify-write: lost updates would
                        // show up in the total.
                        let old = *v;
                        std::hint::black_box(&old);
                        *v = old + 1;
                    }
                });
            }
        });
        assert_eq!(*counter.borrow(), (threads * per) as u64);
    }

    #[test]
    fn gccell_borrow_mut_waits_for_a_reader_on_another_thread() {
        let (_guard, _baseline) = setup();
        let c = GcCell::new(1u64);
        let released = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let reader = c.clone();
            let flag = released.clone();
            s.spawn(move || {
                let r = reader.borrow();
                held_tx.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(20));
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                drop(r);
            });
            held_rx.recv().unwrap();
            let mut w = c.borrow_mut();
            assert!(
                released.load(std::sync::atomic::Ordering::SeqCst),
                "borrow_mut returned while another thread still read the cell"
            );
            *w = 2;
        });
        assert_eq!(*c.borrow(), 2);
    }

    #[test]
    fn collection_during_borrow_mut_frees_nothing() {
        let (_guard, _baseline) = setup();
        // A child reachable only through the cell, with its handle already
        // reset to non-root by a collection.
        let cell = GcCell::new(None::<Gc<u64>>);
        *cell.borrow_mut() = Some(Gc::new(7));
        unsafe { (*GLOBAL_GC).collect() };

        let held = cell.borrow_mut();
        // Garbage made while the borrow is held: a dead 2-cycle. (Allocating
        // under a borrow_mut must not deadlock.)
        let weak = {
            let a = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            let b = Gc::new(SyncCyclicNode {
                next: GcCell::new(None),
            });
            *a.next.borrow_mut() = Some(b.clone());
            *b.next.borrow_mut() = Some(a.clone());
            Gc::downgrade(&a)
        };
        // The collector cannot read the borrowed cell, so it must not free
        // anything - in particular not the child, which `held` reaches.
        let stats = unsafe {
            (*GLOBAL_GC)
                .core
                .collect_generation(crate::generation::Generation::Gen2)
        };
        assert_eq!(stats.objects_collected, 0);
        assert!(weak.is_alive(), "garbage kept while the mark is incomplete");
        assert_eq!(***held.as_ref().unwrap(), 7);
        drop(held);

        // Once the borrow ends, a collection frees the garbage again.
        unsafe { (*GLOBAL_GC).collect() };
        assert!(!weak.is_alive());
        assert_eq!(***cell.borrow().as_ref().unwrap(), 7);
    }

    #[test]
    fn incremental_cycle_keeps_a_blocked_scan_across_a_full_collection() {
        use crate::generation::Generation;
        let (_guard, _baseline) = setup();
        let cell = GcCell::new(None::<Gc<u64>>);
        *cell.borrow_mut() = Some(Gc::new(7));
        unsafe { (*GLOBAL_GC).collect() };

        let held = cell.borrow_mut();
        unsafe {
            GLOBAL_GC.core.begin_collection(Generation::Gen2);
            // Marks the (rooted) cell but cannot scan its value.
            while !GLOBAL_GC.core.mark_step(usize::MAX) {}
        }
        drop(held);
        // A full collection between the steps scans the cell fine and must not
        // hide the incremental cycle's blocked scan from its finish.
        unsafe { (*GLOBAL_GC).collect() };
        let stats = unsafe { GLOBAL_GC.core.finish_collection() };
        assert_eq!(stats.objects_collected, 0);
        assert_eq!(***cell.borrow().as_ref().unwrap(), 7);
    }

    // --- One-word handles counted in the object header ---

    static COUNTED_DROPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// Counts its drops, to check an object is freed exactly once.
    struct Counted(u64);
    impl Drop for Counted {
        fn drop(&mut self) {
            COUNTED_DROPS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    leaf_trace!(Counted);

    fn header_handles<T: crate::gc::Trace>(g: &Gc<T>) -> u32 {
        unsafe {
            (*g.ptr)
                .info
                .handle_count
                .load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[test]
    fn handles_are_one_pointer_wide() {
        assert_eq!(std::mem::size_of::<Gc<u64>>(), std::mem::size_of::<usize>());
        assert_eq!(
            std::mem::size_of::<GcCell<u64>>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<Option<Gc<u64>>>(),
            2 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn clone_and_drop_count_handles_in_the_header() {
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let drops = COUNTED_DROPS.load(std::sync::atomic::Ordering::SeqCst);
        let g = Gc::new(Counted(5));
        assert_eq!(header_handles(&g), 1);
        let c1 = g.clone();
        let c2 = c1.clone();
        assert_eq!(header_handles(&g), 3);
        assert_eq!(c2.0, 5);
        drop(c1);
        assert_eq!(header_handles(&g), 2);
        drop(g);
        assert_eq!(live_objects(), before + 1, "a handle is left");
        drop(c2);
        assert_eq!(live_objects(), before, "the last drop frees the object");
        assert_eq!(
            COUNTED_DROPS.load(std::sync::atomic::Ordering::SeqCst),
            drops + 1
        );
    }

    #[test]
    fn clones_dropped_on_many_threads_free_the_object_once() {
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let drops = COUNTED_DROPS.load(std::sync::atomic::Ordering::SeqCst);
        let rounds = if cfg!(miri) { 4 } else { 200 };
        for _ in 0..rounds {
            let g = Gc::new(Counted(7));
            let cell = GcCell::new(g.clone());
            std::thread::scope(|s| {
                for _ in 0..4 {
                    let mine = g.clone();
                    let cell = cell.clone();
                    s.spawn(move || {
                        let more: Vec<_> = (0..8).map(|_| mine.clone()).collect();
                        assert!(more.iter().all(|h| h.0 == 7));
                        let from_cell = cell.borrow().clone();
                        drop(more);
                        drop(from_cell);
                    });
                }
            });
            drop(cell);
            assert_eq!(header_handles(&g), 1);
            drop(g);
        }
        assert_eq!(live_objects(), before);
        assert_eq!(
            COUNTED_DROPS.load(std::sync::atomic::Ordering::SeqCst),
            drops + rounds
        );
    }

    #[test]
    fn weak_upgrade_racing_the_last_drop_never_resurrects() {
        let (_guard, _baseline) = setup();
        let before = live_objects();
        let drops = COUNTED_DROPS.load(std::sync::atomic::Ordering::SeqCst);
        let rounds = if cfg!(miri) { 3 } else { 100 };
        for _ in 0..rounds {
            let g = Gc::new(Counted(9));
            let weak = Gc::downgrade(&g);
            std::thread::scope(|s| {
                let weak = &weak;
                s.spawn(move || {
                    // Upgrades that succeed must see a live value; once one
                    // fails, the object is gone for good.
                    let mut failed = false;
                    for _ in 0..50 {
                        match weak.upgrade() {
                            Some(strong) => {
                                assert!(!failed, "upgrade succeeded after the object died");
                                assert_eq!(strong.0, 9);
                            }
                            None => failed = true,
                        }
                    }
                });
                drop(g);
            });
            assert!(weak.upgrade().is_none());
            assert!(!weak.is_alive());
        }
        assert_eq!(live_objects(), before);
        assert_eq!(
            COUNTED_DROPS.load(std::sync::atomic::Ordering::SeqCst),
            drops + rounds
        );
    }

    #[test]
    fn sync_handle_inside_a_local_object_is_a_root() {
        // A thread-local collection walks the local object's fields, which
        // include the sync handle; that must not discount it as an interior
        // handle of the sync object.
        struct Holder {
            s: Gc<u64>,
        }
        impl crate::gc::Trace for Holder {
            fn is_root(&self) -> bool {
                false
            }
            fn reset_root(&self) {
                self.s.reset_root();
            }
            fn trace(&self) {
                self.s.trace();
            }
            fn reset(&self) {
                self.s.reset();
            }
            fn is_traceable(&self) -> bool {
                false
            }
            fn trace_children(&self, children: &mut Vec<*const dyn crate::gc::Trace>) {
                self.s.trace_children(children);
            }
        }
        impl crate::gc::Finalize for Holder {
            fn finalize(&self) {}
        }
        let (_guard, _baseline) = setup();
        let local = crate::gc::Gc::new(Holder { s: Gc::new(41) });
        crate::gc::LOCAL_GC.with(|gc| unsafe { gc.borrow_mut().collect() });
        let _paused = PauseBackgroundCollector::new();
        let stats = unsafe {
            (*GLOBAL_GC)
                .core
                .collect_generation(crate::generation::Generation::Gen2)
        };
        assert_eq!(stats.objects_collected, 0);
        assert_eq!(**local.s, 41);
        assert_eq!(header_handles(&local.s), 1);
    }
}
