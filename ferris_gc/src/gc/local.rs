//! The thread-local collector: its handles (`Gc`, `GcCell`, `GcWeak`),
//! `LocalGarbageCollector`, its strategy, and the `LOCAL_GC` thread-local.
use super::*;

/// Type-safe region identifier for the **thread-local** garbage collector.
/// Returned by [`LocalGarbageCollector::new_region`]. Cannot be used with
/// the global (`sync`) GC — the compiler will reject the mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalRegionId(RegionId);

impl LocalRegionId {
    /// Return the raw numeric id (for logging / diagnostics).
    pub fn id(self) -> u32 {
        (self.0).0
    }
}

/// Convenience alias for an optional GC-managed pointer.
pub type OptGc<T> = Option<Gc<T>>;
/// Convenience alias for an optional GC-managed interior-mutable cell.
pub type OptGcCell<T> = Option<GcCell<T>>;

pub(crate) struct LocalGcInfo {
    pub(crate) root_ref_count: Cell<usize>,
    /// Root count for collection cascade. Stored here (inside GcPtr, TLAB-allocated)
    /// so that Gc<T>::reset_root() can decrement it directly via self.ptr without
    /// going through the SlotMap. After cascade, copied back to ObjectEntry.root_count.
    pub(crate) root_count: Cell<u32>,
}

impl LocalGcInfo {
    #[inline]
    fn new() -> LocalGcInfo {
        LocalGcInfo {
            root_ref_count: Cell::new(0),
            root_count: Cell::new(0),
        }
    }
}

/// Internal pointer wrapper used as the `Deref` target of `Gc<T>` and `GcCell<T>`.
/// This type is `pub` because it appears in the public `Deref` interface, but users
/// should not need to name it directly — access `T` via double-deref (`**gc`).
#[doc(hidden)]
pub struct GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    info: LocalGcInfo,
    object_id: ObjectId,
    t: T,
}

impl<T> GcPtr<T>
where
    T: 'static + Sized + Trace,
{
    #[inline]
    fn new(t: T) -> GcPtr<T> {
        GcPtr {
            info: LocalGcInfo::new(),
            object_id: ObjectId::ZERO,
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
        let prev = self.info.root_ref_count.get();
        self.info.root_ref_count.set(prev + 1);
        if prev == 0 {
            self.t.trace();
        }
    }

    fn reset(&self) {
        // Guard: only recurse into children on last reset (breaks cycles)
        let prev = self.info.root_ref_count.get();
        self.info.root_ref_count.set(prev - 1);
        if prev == 1 {
            self.t.reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.info.root_ref_count.get() > 0
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.t.trace_children(children);
    }

    fn clear_trace(&self) {
        self.info.root_ref_count.set(0);
    }
}

impl<T> Trace for RefCell<GcPtr<T>>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        unreachable!("is_root on GcPtr is unreachable !!");
    }

    fn reset_root(&self) {
        // Conservative skip when the cell is mutably borrowed (a cascade can
        // run mid-allocation via maybe_collect while a user RefMut is live);
        // the object then keeps its root state and survives this cycle.
        // See GcCell::reset_root.
        if let Ok(borrow) = self.try_borrow() {
            borrow.t.reset_root();
        }
    }

    fn trace(&self) {
        let prev = self.borrow().info.root_ref_count.get();
        self.borrow().info.root_ref_count.set(prev + 1);
        if prev == 0 {
            self.borrow().t.trace();
        }
    }

    fn reset(&self) {
        let prev = self.borrow().info.root_ref_count.get();
        self.borrow().info.root_ref_count.set(prev - 1);
        if prev == 1 {
            self.borrow().t.reset();
        }
    }

    fn is_traceable(&self) -> bool {
        self.borrow().info.root_ref_count.get() > 0
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        self.borrow().t.trace_children(children);
    }

    fn clear_trace(&self) {
        self.borrow().info.root_ref_count.set(0);
    }
}

impl<T> Finalize for RefCell<GcPtr<T>>
where
    T: Sized + Trace,
{
    fn finalize(&self) {
        self.borrow().t.finalize();
    }
}

impl<T> Finalize for GcPtr<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {
        self.t.finalize();
    }
}

/// A garbage-collected smart pointer for thread-local use.
///
/// `Gc<T>` is **not** `Send`/`Sync` — it is bound to the thread that
/// created it. For cross-thread GC pointers, use [`sync::Gc<T>`].
///
/// Dereferences to `T` (via `GcPtr<T>`), so `*gc` gives you `&T`.
///
/// # Allocation
///
/// Created via [`Gc::new`] (infallible) or [`Gc::try_new`] (fallible).
/// Objects are automatically registered with the thread-local collector.
///
/// # Collection
///
/// The background strategy thread calls `collect()` periodically. You can
/// also trigger collection manually via the `LocalGarbageCollector`.
/// Unreachable cycles are detected and collected.
/// `Gc<T>` is 8 bytes: a packed `NonZeroUsize` where bit 0 = `is_root` flag
/// and bits 1.. = `*const GcPtr<T>` (always ≥4-byte aligned, so bit 0 is free).
/// `Option<Gc<T>>` is 16 bytes: the `Cell` wrapper opts out of the
/// `NonZeroUsize` niche (see the `size_gc_is_compact` test).
/// `object_id` is stored inside `GcPtr<T>` on the heap.
///
/// `Gc<T>` is `!Send` and `!Sync`: it is bound to the collector of the thread
/// that allocated it, and dropping or cloning it on another thread would
/// corrupt both threads' collectors.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<ferris_gc::Gc<i32>>();
/// ```
pub struct Gc<T>
where
    T: 'static + Sized + Trace,
{
    /// Packed pointer: bit 0 = is_root, bits 1.. = pointer to GcPtr<T>.
    ptr: Cell<NonZeroUsize>,
    /// `*const T` keeps `Gc<T>` `!Send`/`!Sync` (thread-local collector).
    _marker: PhantomData<*const T>,
}

impl<T> Gc<T>
where
    T: 'static + Sized + Trace,
{
    #[inline(always)]
    fn gc_ptr(&self) -> *const GcPtr<T> {
        (self.ptr.get().get() & !1usize) as *const GcPtr<T>
    }

    #[inline(always)]
    fn is_root_flag(&self) -> bool {
        self.ptr.get().get() & 1 != 0
    }

    #[inline(always)]
    fn set_root_flag(&self, root: bool) {
        let raw = self.ptr.get().get();
        let new = if root { raw | 1 } else { raw & !1 };
        // SAFETY: gc_ptr is never null, so new is never 0.
        self.ptr.set(unsafe { NonZeroUsize::new_unchecked(new) });
    }

    #[inline(always)]
    fn object_id(&self) -> ObjectId {
        unsafe { (*self.gc_ptr()).object_id }
    }

    #[inline(always)]
    fn from_parts(gc_ptr: *const GcPtr<T>, is_root: bool) -> Self {
        let raw = gc_ptr as usize | (is_root as usize);
        Gc {
            // SAFETY: gc_ptr is a valid non-null allocation.
            ptr: Cell::new(unsafe { NonZeroUsize::new_unchecked(raw) }),
            _marker: PhantomData,
        }
    }
}

impl<T> Deref for Gc<T>
where
    T: 'static + Sized + Trace,
{
    type Target = GcPtr<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: packed pointer (with bit 0 masked off) is valid for the lifetime of this Gc.
        unsafe { &*self.gc_ptr() }
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
    /// Allocate a new GC-managed object on the thread-local collector.
    /// Starts the background collection strategy if not already active.
    pub fn new(t: T) -> Gc<T> {
        basic_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LocalGarbageCollector::allocate(move |gc| unsafe {
            let region = gc.core.current_region();
            gc.create_gc(t, region)
        })
    }

    /// Allocate a new GC-managed object in the specified region.
    pub fn new_in(t: T, region: LocalRegionId) -> Gc<T> {
        basic_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LocalGarbageCollector::allocate(move |gc| unsafe { gc.create_gc(t, region.0) })
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<Gc<T>, GcAllocError> {
        basic_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LocalGarbageCollector::allocate(move |gc| unsafe {
            let region = gc.core.current_region();
            gc.try_create_gc(t, region)
        })
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(t: T, region: LocalRegionId) -> Result<Gc<T>, GcAllocError> {
        basic_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LocalGarbageCollector::allocate(move |gc| unsafe { gc.try_create_gc(t, region.0) })
    }
}

impl<T> Clone for Gc<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().clone_from_gc(self) })
    }
}

impl<T> Drop for Gc<T>
where
    T: Sized + Trace,
{
    fn drop(&mut self) {
        let is_root = self.is_root_flag();
        let gc_ptr = self.gc_ptr();
        // SAFETY: Thread-local access ensures single-threaded operation.
        // Re-entrant drops (from RC hybrid dealloc cascading through inner Gc fields)
        // are safe because gc_maps borrows don't overlap: drop_handle_inner finishes
        // its &mut GcMaps borrow before the dealloc triggers nested drops.
        //
        // The dying check MUST precede the object_id read: during a sweep
        // dealloc phase this handle may live inside a dead object and point at
        // another dead object of the same batch — or at the very object whose
        // drop_in_place is running. Dereferencing it there is UB (freed memory
        // or a protected exclusive borrow); the collector already removed those
        // entries from gc_maps, so the drop is a no-op.
        let _ = LOCAL_GC.try_with(move |gc| unsafe {
            let local = &*gc.as_ptr();
            if local.core.is_dying(gc_ptr as usize) {
                return;
            }
            let object_id = (*gc_ptr).object_id;
            local.drop_handle(object_id, is_root);
        });
    }
}

impl<T> Trace for Gc<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        self.is_root_flag()
    }

    #[inline(always)]
    fn reset_root(&self) {
        let tagged = GC_MAPS_CTX.get();
        if tagged.is_null() {
            return;
        }
        unsafe {
            let gc_ptr = &*self.gc_ptr();
            if gc_ctx_is_collecting(tagged) {
                // Collection mode: decrement LocalGcInfo.root_count directly (no SlotMap lookup).
                // root_ref_count as cycle guard: 0 = unvisited, 1 = visited.
                let rc = gc_ptr.info.root_count.get();
                gc_ptr.info.root_count.set(rc.saturating_sub(1));
                if gc_ptr.info.root_ref_count.get() == 0 {
                    gc_ptr.info.root_ref_count.set(1);
                    // Record the visit so discover_roots phase 3 clears this
                    // guard even when the target is out of collection scope.
                    (*gc_ctx_ptr(tagged)).push(&gc_ptr.info.root_ref_count as *const Cell<usize>);
                    gc_ptr.reset_root();
                }
            } else if self.is_root_flag() {
                // Allocation mode: just flip is_root and cascade.
                // ObjectEntry.root_count is recomputed from handle_count in discover_roots.
                self.set_root_flag(false);
                gc_ptr.reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            (*self.gc_ptr()).trace();
        }
    }

    fn reset(&self) {
        unsafe {
            (*self.gc_ptr()).reset();
        }
    }

    fn is_traceable(&self) -> bool {
        unsafe { (*self.gc_ptr()).is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.gc_ptr() as *const dyn Trace);
    }

    unsafe fn relocate(&self, old_ptr: *const u8, new_ptr: *const u8) {
        if self.gc_ptr() as *const u8 == old_ptr {
            // Preserve is_root flag when updating the pointer.
            let is_root = self.is_root_flag();
            let raw = new_ptr as usize | (is_root as usize);
            self.ptr.set(unsafe { NonZeroUsize::new_unchecked(raw) });
        }
    }
}

impl<T> Finalize for Gc<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A garbage-collected mutable cell for thread-local use.
///
/// Provides interior mutability (`borrow()` / `borrow_mut()`) for GC-managed
/// objects. `borrow_mut()` triggers the **write barrier**, which records
/// cross-generation references so that young-generation collections can
/// discover pointers from old objects to young objects.
///
/// Thread-local only — for a cross-thread variant, use [`sync::GcCell<T>`].
/// `GcCell<T>` is 8 bytes: packed `NonZeroUsize` (bit 0 = `is_root`,
/// bits 1.. = `*const RefCell<GcPtr<T>>`). Same layout as `Gc<T>`.
///
/// Like [`Gc<T>`], `GcCell<T>` is `!Send` and `!Sync`.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<ferris_gc::GcCell<i32>>();
/// ```
pub struct GcCell<T>
where
    T: 'static + Sized + Trace,
{
    ptr: Cell<NonZeroUsize>,
    /// `*const T` keeps `GcCell<T>` `!Send`/`!Sync` (thread-local collector).
    _marker: PhantomData<*const T>,
}

impl<T> GcCell<T>
where
    T: 'static + Sized + Trace,
{
    #[inline(always)]
    fn refcell_ptr(&self) -> *const RefCell<GcPtr<T>> {
        (self.ptr.get().get() & !1usize) as *const RefCell<GcPtr<T>>
    }

    #[inline(always)]
    fn is_root_flag(&self) -> bool {
        self.ptr.get().get() & 1 != 0
    }

    #[inline(always)]
    fn set_root_flag(&self, root: bool) {
        let raw = self.ptr.get().get();
        let new = if root { raw | 1 } else { raw & !1 };
        self.ptr.set(unsafe { NonZeroUsize::new_unchecked(new) });
    }

    #[inline(always)]
    fn object_id(&self) -> ObjectId {
        unsafe { (*(*self.refcell_ptr()).as_ptr()).object_id }
    }

    #[inline(always)]
    fn from_parts(refcell_ptr: *const RefCell<GcPtr<T>>, is_root: bool) -> Self {
        let raw = refcell_ptr as usize | (is_root as usize);
        GcCell {
            ptr: Cell::new(unsafe { NonZeroUsize::new_unchecked(raw) }),
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for GcCell<T>
where
    T: Sized + Trace,
{
    fn drop(&mut self) {
        let is_root = self.is_root_flag();
        let refcell_ptr = self.refcell_ptr();
        // SAFETY: Thread-local access ensures single-threaded operation.
        // Re-entrant drops (from RC hybrid dealloc cascading through inner Gc fields)
        // are safe because gc_maps borrows don't overlap: drop_handle_inner finishes
        // its &mut GcMaps borrow before the dealloc triggers nested drops.
        //
        // The dying check MUST precede the object_id read — see Gc::drop.
        let _ = LOCAL_GC.try_with(move |gc| unsafe {
            let local = &*gc.as_ptr();
            if local.core.is_dying(refcell_ptr as usize) {
                return;
            }
            let object_id = (*(*refcell_ptr).as_ptr()).object_id;
            local.drop_handle(object_id, is_root);
        });
    }
}

impl<T> Deref for GcCell<T>
where
    T: 'static + Sized + Trace,
{
    type Target = RefCell<GcPtr<T>>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: packed pointer (with bit 0 masked off) is valid for the lifetime of this GcCell.
        unsafe { &*self.refcell_ptr() }
    }
}

impl<T> std::fmt::Debug for GcCell<T>
where
    T: 'static + Sized + Trace + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GcCell").field(&**self.borrow()).finish()
    }
}

impl<T> GcCell<T>
where
    T: 'static + Sized + Trace,
{
    /// Allocate a new GC-managed interior-mutable cell on the thread-local collector.
    /// The contained value can be borrowed mutably via `borrow_mut()`.
    pub fn new(t: T) -> GcCell<T> {
        basic_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LocalGarbageCollector::allocate(move |gc| unsafe {
            let region = gc.core.current_region();
            gc.create_gc_cell(t, region)
        })
    }

    /// Allocate a new GC-managed interior-mutable cell in the specified region.
    pub fn new_in(t: T, region: LocalRegionId) -> GcCell<T> {
        basic_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LocalGarbageCollector::allocate(move |gc| unsafe { gc.create_gc_cell(t, region.0) })
    }

    /// Fallible allocation. Returns `Err(GcAllocError)` if memory is exhausted.
    /// On OOM, triggers an emergency GC collection and retries once before failing.
    pub fn try_new(t: T) -> Result<GcCell<T>, GcAllocError> {
        basic_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LocalGarbageCollector::allocate(move |gc| unsafe {
            let region = gc.core.current_region();
            gc.try_create_gc_cell(t, region)
        })
    }

    /// Fallible allocation in a specified region.
    pub fn try_new_in(t: T, region: LocalRegionId) -> Result<GcCell<T>, GcAllocError> {
        basic_strategy_start();
        LOCAL_GC_STRATEGY.with(|strategy| {
            if !strategy.borrow().is_active() {
                // SAFETY: Single-threaded access via thread_local ensures no aliasing.
                let strategy = unsafe { &mut *strategy.as_ptr() };
                strategy.start();
            }
        });
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LocalGarbageCollector::allocate(move |gc| unsafe { gc.try_create_gc_cell(t, region.0) })
    }

    /// Mutable borrow with write barrier.
    /// Triggers the write barrier so that if this object is in an older generation,
    /// its card is marked dirty in the card table for young-generation collections.
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GcPtr<T>> {
        let refcell_ptr = self.refcell_ptr();
        let object_id = self.object_id();
        LOCAL_GC.with(|gc| {
            gc.borrow()
                .core
                .write_barrier(object_id, refcell_ptr as *const dyn Trace);
        });
        // SAFETY: packed pointer (with bit 0 masked off) is valid for the lifetime of this GcCell.
        unsafe { (*refcell_ptr).borrow_mut() }
    }
}

impl<T> Clone for GcCell<T>
where
    T: 'static + Sized + Trace,
{
    fn clone(&self) -> Self {
        // SAFETY: Thread-local access ensures single-threaded borrow.
        LOCAL_GC.with(move |gc| unsafe { gc.borrow_mut().clone_from_gc_cell(self) })
    }
}

impl<T> Trace for GcCell<T>
where
    T: Sized + Trace,
{
    fn is_root(&self) -> bool {
        self.is_root_flag()
    }

    #[inline(always)]
    fn reset_root(&self) {
        let tagged = GC_MAPS_CTX.get();
        if tagged.is_null() {
            return;
        }
        unsafe {
            // try_borrow, NOT raw as_ptr(): the cascade can run while the user
            // holds a live RefMut (maybe_collect fires inside Gc::new, and the
            // allocation-mode cascade runs on every create/clone). Touching the
            // data through a fresh reference would alias that RefMut — UB under
            // Stacked Borrows. Skipping is conservative: the target keeps its
            // root count / root flag and simply survives this cycle.
            let Ok(borrow) = (*self.refcell_ptr()).try_borrow() else {
                return;
            };
            let gc_ptr: &GcPtr<T> = &borrow;
            if gc_ctx_is_collecting(tagged) {
                let rc = gc_ptr.info.root_count.get();
                gc_ptr.info.root_count.set(rc.saturating_sub(1));
                if gc_ptr.info.root_ref_count.get() == 0 {
                    gc_ptr.info.root_ref_count.set(1);
                    // Record the visit so discover_roots phase 3 clears this
                    // guard even when the target is out of collection scope.
                    (*gc_ctx_ptr(tagged)).push(&gc_ptr.info.root_ref_count as *const Cell<usize>);
                    gc_ptr.t.reset_root();
                }
            } else if self.is_root_flag() {
                // Allocation mode: just flip is_root and cascade.
                // ObjectEntry.root_count is recomputed from handle_count in discover_roots.
                self.set_root_flag(false);
                gc_ptr.t.reset_root();
            }
        }
    }

    fn trace(&self) {
        unsafe {
            (*self.refcell_ptr()).borrow().trace();
        }
    }

    fn reset(&self) {
        unsafe {
            (*self.refcell_ptr()).borrow().reset();
        }
    }

    fn is_traceable(&self) -> bool {
        unsafe { (*self.refcell_ptr()).borrow().is_traceable() }
    }

    fn trace_children(&self, children: &mut Vec<*const dyn Trace>) {
        children.push(self.refcell_ptr() as *const dyn Trace);
    }

    unsafe fn relocate(&self, old_ptr: *const u8, new_ptr: *const u8) {
        if self.refcell_ptr() as *const u8 == old_ptr {
            let is_root = self.is_root_flag();
            let raw = new_ptr as usize | (is_root as usize);
            self.ptr.set(unsafe { NonZeroUsize::new_unchecked(raw) });
        }
    }
}

impl<T> Finalize for GcCell<T>
where
    T: Sized + Trace,
{
    fn finalize(&self) {}
}

/// A weak reference to a GC-managed object.
///
/// Does not prevent collection. Use [`upgrade()`](GcWeak::upgrade) to obtain
/// a strong `Gc<T>`. Returns `None` if the object has already been collected.
///
/// `GcWeak<T>` is `Send + Sync` when `T` is, so it can be shared across
/// threads to observe liveness of a thread-local object.
pub struct GcWeak<T>
where
    T: 'static + Sized + Trace,
{
    alive: Arc<AtomicBool>,
    ptr: *const GcPtr<T>,
    object_id: ObjectId,
}

// SAFETY: GcWeak contains only an Arc<AtomicBool> (thread-safe) and a raw pointer
// to the GcPtr (never written through). The pointer is only used for upgrade(),
// which goes through the GC's Mutex-protected maps. Send/Sync bounds on T
// ensure the underlying data is safe to share.
unsafe impl<T> Send for GcWeak<T> where T: 'static + Sized + Trace + Send {}
unsafe impl<T> Sync for GcWeak<T> where T: 'static + Sized + Trace + Sync {}

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
        if self.alive.load(Ordering::Relaxed) {
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

    /// Try to upgrade this weak reference to a strong `Gc<T>`.
    /// Returns `None` if the object has been collected.
    pub fn upgrade(&self) -> Option<Gc<T>> {
        if !self.alive.load(Ordering::Acquire) {
            return None;
        }
        LOCAL_GC.with(|gc| {
            // Acquire STW read lock via raw pointer to avoid RefCell borrow conflict
            let gc_ptr = gc.as_ptr();
            let _stw = unsafe {
                // SAFETY: gc_ptr is valid for the duration of this thread_local closure; raw pointer avoids RefCell borrow conflict with the subsequent borrow_mut.
                (*gc_ptr)
                    .core
                    .stw_lock
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
            };
            // Re-check alive under STW protection
            if self.alive.load(Ordering::Acquire) {
                // SAFETY: Thread-local access ensures single-threaded borrow; alive check above guarantees the object has not been collected.
                Some(unsafe { gc.borrow_mut().upgrade_weak(self) })
            } else {
                None
            }
        })
    }
}

impl<T> Gc<T>
where
    T: 'static + Sized + Trace,
{
    /// Create a weak reference to this GC-managed object.
    pub fn downgrade(this: &Gc<T>) -> GcWeak<T> {
        let object_id = this.object_id();
        LOCAL_GC.with(|gc| {
            let gc_ref = gc.borrow();
            let alive = gc_ref.core.get_or_create_weak_alive(object_id);
            GcWeak {
                alive,
                ptr: this.gc_ptr(),
                object_id,
            }
        })
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

/// Thread-local garbage collector wrapper.
/// Each thread gets its own `LocalGarbageCollector` via the `LOCAL_GC` thread-local.
/// Provides allocation (`create_gc`), collection (`collect`), and incremental/concurrent
/// collection methods. All operations are single-threaded (no cross-thread sharing).
/// Allocation threshold for triggering automatic Gen0 collection on the owning thread.
/// Local GCs collect on allocation (not from a background thread) to avoid data races
/// on non-atomic Cell/RefCell internals.
pub(crate) const LOCAL_GC_ALLOC_THRESHOLD: usize = 1_000;

/// Thread-local garbage collector.
///
/// Each thread gets its own instance via the [`LOCAL_GC`] thread-local.
/// Provides allocation, collection, and diagnostic methods. All operations
/// are single-threaded — for cross-thread GC, use [`sync::GlobalGarbageCollector`].
///
/// Collection is triggered automatically after every `LOCAL_GC_ALLOC_THRESHOLD`
/// allocations, or manually via `collect()` / `collect_generation()`.
pub struct LocalGarbageCollector {
    pub(crate) core: GarbageCollector,
    /// Thread-local allocation buffer for fast bump-pointer allocation.
    /// Lazily initialized on first allocation.
    pub(crate) tlab: Option<crate::tlab::Tlab>,
}

// SAFETY: Delegates to GarbageCollector which protects all state with Mutex/atomics.
// The Tlab is only accessed from the owning thread (thread_local).
unsafe impl Sync for LocalGarbageCollector {}
unsafe impl Send for LocalGarbageCollector {}

#[allow(dead_code)]
impl LocalGarbageCollector {
    fn new() -> LocalGarbageCollector {
        LocalGarbageCollector {
            core: GarbageCollector::new(),
            tlab: None,
        }
    }

    /// Try to allocate A and B in a single TLAB bump (combined allocation).
    /// Returns `Some((ptr_a, ptr_b, combined_mem, combined_layout))` on success,
    /// or `None` if the TLAB cannot satisfy the combined layout.
    ///
    /// # Safety
    /// Both returned pointers must be initialized before use.
    #[inline]
    unsafe fn try_alloc_combined_tlab<A: Sized, B: Sized>(
        &mut self,
    ) -> Option<(*mut A, *mut B, GcObjMem, Layout)> {
        let layout_a = Layout::new::<A>();
        let layout_b = Layout::new::<B>();
        let (combined, offset_b) = layout_a.extend(layout_b).ok()?;
        let combined = combined.pad_to_align();

        let tlab_result = if let Some(ref mut tlab) = self.tlab {
            tlab.alloc_or_grow(combined)
        } else {
            let mut tlab = crate::tlab::Tlab::new()?;
            let r = tlab.alloc(combined);
            self.tlab = Some(tlab);
            r
        };
        let (ptr, block) = tlab_result?;
        let ptr_a = ptr as *mut A;
        // SAFETY: offset_b is within the combined allocation.
        let ptr_b = unsafe { ptr.add(offset_b) } as *mut B;
        Some((ptr_a, ptr_b, GcObjMem::Tlab(block), combined))
    }

    /// Allocate an ObjectEntry in the TLAB. Returns a raw pointer to uninitialized
    /// memory and the TlabBlock pointer for later release.
    /// Falls back to system allocator if TLAB is unavailable.
    ///
    /// # Safety
    /// The returned pointer must be initialized via ptr::write before use.
    #[inline]
    unsafe fn alloc_entry_tlab(&mut self) -> (*mut ObjectEntry, *mut crate::tlab::TlabBlock) {
        let layout = Layout::new::<ObjectEntry>();

        if let Some(ref mut tlab) = self.tlab {
            if let Some((ptr, block)) = tlab.alloc_or_grow(layout) {
                return (ptr as *mut ObjectEntry, block);
            }
        } else if let Some(mut tlab) = crate::tlab::Tlab::new() {
            if let Some((ptr, block)) = tlab.alloc(layout) {
                self.tlab = Some(tlab);
                return (ptr as *mut ObjectEntry, block);
            }
            self.tlab = Some(tlab);
        }

        // Fall back to system allocator
        let ptr = unsafe { alloc(layout) } as *mut ObjectEntry;
        (ptr, std::ptr::null_mut())
    }

    /// Try to allocate memory from the TLAB. Falls back to system allocator if
    /// the TLAB is exhausted or cannot be created.
    ///
    /// Returns `(typed_ptr, (GcObjMem, Layout))` like `GarbageCollector::alloc_mem`.
    ///
    /// # Safety
    /// The returned pointer must be initialized before use.
    #[inline]
    unsafe fn alloc_mem_tlab<T: Sized>(&mut self) -> (*mut T, (GcObjMem, Layout)) {
        let layout = Layout::new::<T>();

        // Try TLAB allocation first
        if let Some(ref mut tlab) = self.tlab {
            if let Some((ptr, block)) = tlab.alloc_or_grow(layout) {
                let type_ptr: *mut T = ptr as *mut T;
                return (type_ptr, (GcObjMem::Tlab(block), layout));
            }
        } else {
            // Lazily initialize TLAB
            if let Some(mut tlab) = crate::tlab::Tlab::new() {
                if let Some((ptr, block)) = tlab.alloc(layout) {
                    let type_ptr: *mut T = ptr as *mut T;
                    self.tlab = Some(tlab);
                    return (type_ptr, (GcObjMem::Tlab(block), layout));
                }
                self.tlab = Some(tlab);
            }
        }

        // Fall back to system allocator
        // SAFETY: Delegates to alloc_mem which upholds its own safety invariants.
        unsafe { self.core.alloc_mem::<T>() }
    }

    /// Fallible TLAB allocation with GC retry.
    ///
    /// # Safety
    /// The returned pointer must be initialized before use.
    unsafe fn try_alloc_mem_tlab<T: Sized>(
        &mut self,
    ) -> Result<(*mut T, (GcObjMem, Layout)), GcAllocError> {
        let layout = Layout::new::<T>();

        // Try TLAB allocation first
        if let Some(ref mut tlab) = self.tlab {
            if let Some((ptr, block)) = tlab.alloc_or_grow(layout) {
                let type_ptr: *mut T = ptr as *mut T;
                return Ok((type_ptr, (GcObjMem::Tlab(block), layout)));
            }
        } else {
            // Lazily initialize TLAB
            if let Some(mut tlab) = crate::tlab::Tlab::new() {
                if let Some((ptr, block)) = tlab.alloc(layout) {
                    let type_ptr: *mut T = ptr as *mut T;
                    self.tlab = Some(tlab);
                    return Ok((type_ptr, (GcObjMem::Tlab(block), layout)));
                }
                self.tlab = Some(tlab);
            }
        }

        // Fall back to system allocator with GC retry
        // SAFETY: Delegates to try_alloc_mem_with_gc which upholds its own safety invariants.
        unsafe { self.core.try_alloc_mem_with_gc::<T>() }
    }

    /// Check allocation count and trigger Gen0 collection if threshold exceeded
    /// and a collection could free anything (see
    /// `skip_collection_without_garbage`). The threshold adapts based on
    /// collection efficiency: if a collection frees little garbage, the
    /// threshold doubles to avoid wasteful scans of a mostly-live heap. If a
    /// collection frees a lot, the threshold shrinks (down to 1K) to reclaim
    /// memory promptly.
    ///
    /// # Safety
    /// No borrow of the collector may be live: see [`Self::allocate`].
    #[inline]
    unsafe fn maybe_collect(gc: &RefCell<Self>) {
        // Only the cheap trigger checks stay inline. Every collection entry
        // point is kept out of line: inlining the collector into the allocation
        // fast path bloated it enough that LLVM stopped inlining Gc::new into its
        // callers (measured +4-6% instructions on alloc/churn).
        // SAFETY: only reads; a collection that is due takes a shared borrow.
        let this = unsafe { &*gc.as_ptr() };
        let budget = this.core.local_incremental_budget.load(Ordering::Relaxed);
        if budget > 0 {
            unsafe { gc.borrow().step_incremental(budget) };
            return;
        }

        // Classic full-collect path.
        let count = this.core.allocation_count.load(Ordering::Relaxed);
        let threshold = this.core.alloc_threshold.load(Ordering::Relaxed);
        if count >= threshold {
            unsafe { gc.borrow().collect_gen0_and_adapt() };
        }
    }

    /// Run an allocation on this thread's collector, then the allocation
    /// trigger. The trigger runs only once the allocation's `&mut` borrow
    /// has ended: a collection drops dead objects, and the drops of the
    /// handles inside them borrow the collector again, which would alias
    /// that `&mut` (undefined behaviour under Stacked Borrows).
    #[inline(always)]
    fn allocate<R>(alloc: impl FnOnce(&mut LocalGarbageCollector) -> R) -> R {
        LOCAL_GC.with(|gc| {
            let allocated = alloc(&mut gc.borrow_mut());
            // SAFETY: the allocation's borrow ended above.
            unsafe { Self::maybe_collect(gc) };
            allocated
        })
    }

    /// On-thread incremental mode (set by the local background/g1 strategies):
    /// advance a Gen2 mark by one bounded step per allocation instead of a
    /// full STW collect. Runs on the owning thread, so the lockless mutator is
    /// never raced; pauses stay short because each step marks <= `budget`
    /// objects. Correctness of interleaving with allocation relies on
    /// grey-at-birth (note_new_object_during_marking) + the write barrier.
    #[inline(never)]
    unsafe fn step_incremental(&self, budget: usize) {
        match self.core.incremental_phase() {
            CollectionPhase::Idle => {
                let count = self.core.allocation_count.load(Ordering::Relaxed);
                let threshold = self.core.alloc_threshold.load(Ordering::Relaxed);
                if count >= threshold
                    && unsafe { !self.skip_collection_without_garbage(Generation::Gen2) }
                {
                    unsafe { self.core.begin_collection(Generation::Gen2) };
                }
            }
            CollectionPhase::Marking => {
                let done = unsafe { self.core.mark_step(budget) };
                if done {
                    let stats = unsafe { self.core.finish_collection() };
                    self.core.allocation_count.store(0, Ordering::Relaxed);
                    self.adapt_alloc_threshold(&stats);
                }
            }
            CollectionPhase::Sweeping => {}
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn collect_gen0_and_adapt(&self) {
        if unsafe { self.skip_collection_without_garbage(Generation::Gen0) } {
            return;
        }
        let stats = unsafe { self.core.collect_generation(Generation::Gen0) };
        self.adapt_alloc_threshold(&stats);
    }

    /// The allocation trigger fired for a collection of generations
    /// `0..=max_gen`. If no handle has been dropped while its object kept
    /// others since the last collection of that scope, the scope holds no
    /// garbage (see `GcMaps::garbage_possible`) and the collection would
    /// free nothing: skip it, and treat it like a collection that freed
    /// nothing (restart the trigger's count and grow its threshold). Also
    /// trim `gen0_ids` of objects already freed by their last handle's drop,
    /// which collections otherwise do. Returns whether it was skipped.
    unsafe fn skip_collection_without_garbage(&self, max_gen: Generation) -> bool {
        // SAFETY: thread-local collector, no concurrent access.
        let gc_maps = unsafe { self.core.gc_maps_unsync() };
        if gc_maps.garbage_possible(max_gen) {
            return false;
        }
        self.core.allocation_count.store(0, Ordering::Relaxed);
        self.grow_alloc_threshold(0);
        // Trimming when more than half the ids are stale keeps it amortized
        // O(1) per freed object.
        if gc_maps.gen0_ids.len() > 2 * gc_maps.objects.len() {
            let objects = &gc_maps.objects;
            gc_maps.gen0_ids.retain(|&id| objects.get(id).is_some());
        }
        true
    }

    /// GOGC-style adaptive threshold: grow when almost nothing was collected
    /// (don't re-collect until the live set has roughly doubled), shrink when a
    /// lot was garbage (collect sooner).
    fn adapt_alloc_threshold(&self, stats: &CollectionStats) {
        if stats.objects_scanned == 0 {
            return;
        }
        let threshold = self.core.alloc_threshold.load(Ordering::Relaxed);
        let collected_ratio = stats.objects_collected as f64 / stats.objects_scanned as f64;
        if collected_ratio < 0.05 {
            self.grow_alloc_threshold(stats.objects_scanned);
        } else if collected_ratio > 0.25 {
            let new = (threshold / 2).max(LOCAL_GC_ALLOC_THRESHOLD);
            self.core.alloc_threshold.store(new, Ordering::Relaxed);
        }
    }

    /// Almost nothing was garbage: double the threshold, and make it at
    /// least the `scanned` live set, so the next collection waits until the
    /// live set has roughly doubled.
    fn grow_alloc_threshold(&self, scanned: usize) {
        let threshold = self.core.alloc_threshold.load(Ordering::Relaxed);
        let new = scanned.max(threshold * 2);
        self.core.alloc_threshold.store(new, Ordering::Relaxed);
    }

    // Forced: master's LLVM inlined this into Gc::new; once GarbageCollector
    // became generic (two monomorphizations) it stopped, costing ~11% of
    // churn instructions in call overhead.
    #[inline(always)]
    unsafe fn create_gc<T>(&mut self, t: T, region: RegionId) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            // Double allocation: [GcPtr<T> | ObjectEntry] in one TLAB bump.
            // When combined, obj_mem is Inline (no separate dealloc for GcPtr),
            // oe_block holds the TLAB block (releasing it frees both).
            let (gc_ptr, oe_ptr, obj_mem, obj_layout, oe_block);
            if let Some((a, b, mem, _combined_layout)) =
                self.try_alloc_combined_tlab::<GcPtr<T>, ObjectEntry>()
            {
                gc_ptr = a;
                oe_ptr = b;
                // Extract the TLAB block from the combined mem for entry_block.
                oe_block = match mem {
                    GcObjMem::Tlab(block) => block,
                    _ => std::ptr::null_mut(),
                };
                obj_mem = GcObjMem::Inline;
                obj_layout = Layout::new::<GcPtr<T>>();
            } else {
                let (a, (mem_a, layout_a)) = self.alloc_mem_tlab::<GcPtr<T>>();
                gc_ptr = a;
                obj_mem = mem_a;
                obj_layout = layout_a;
                let (c, block) = self.alloc_entry_tlab();
                oe_ptr = c;
                oe_block = block;
            }
            std::ptr::write(gc_ptr, GcPtr::new(t));

            let root_ref_count_offset = (&(*gc_ptr).info.root_ref_count as *const Cell<usize>
                as usize
                - gc_ptr as usize) as u16;
            std::ptr::write(
                oe_ptr,
                ObjectEntry {
                    ptr: gc_ptr as *const dyn Trace,
                    mem: obj_mem,
                    layout: CompactLayout::from_layout(obj_layout),
                    gen_survive_region: region.0 & 0xFFFF,
                    handle_count: 1,
                    root_count: 1,
                    root_ref_count_offset,
                    entry_block: oe_block,
                },
            );
            let gc_maps = self.core.gc_maps_unsync();
            let obj_id = gc_maps.objects.insert(ObjectEntryRef(oe_ptr));
            gc_maps.gen0_ids.push(obj_id);
            self.core.track_alloc(obj_layout.size());

            // Store object_id in GcPtr so Gc<T> can retrieve it via pointer deref.
            (*gc_ptr).object_id = obj_id;

            // If an incremental mark cycle is in progress, grey this object so
            // the colors-based sweep in finish_collection keeps it.
            self.core.note_new_object_during_marking(obj_id);

            let gc = Gc::from_parts(gc_ptr, true);
            // Set context for reset_root cascade to decrement child root_counts.
            let gc_maps_raw = gc_maps as *mut GcMaps;
            GC_MAPS_CTX.set(&raw mut (*gc_maps_raw).cascade_visited);
            (*gc_ptr).reset_root();
            GC_MAPS_CTX.set(std::ptr::null_mut());
            self.core.allocation_count.add(1);
            gc
        }
    }

    unsafe fn clone_from_gc<T>(&self, gc: &Gc<T>) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let object_id = gc.object_id();
            let gc_ptr = gc.gc_ptr();
            let gc_maps = self.core.gc_maps_unsync();
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.handle_count += 1;
                entry.root_count += 1;
            }
            let new_gc = Gc::from_parts(gc_ptr, true);
            let gc_maps_raw = gc_maps as *mut GcMaps;
            GC_MAPS_CTX.set(&raw mut (*gc_maps_raw).cascade_visited);
            (*gc_ptr).reset_root();
            GC_MAPS_CTX.set(std::ptr::null_mut());
            new_gc
        }
    }

    unsafe fn create_gc_cell<T>(&mut self, t: T, region: RegionId) -> GcCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            // Double allocation: [RefCell<GcPtr<T>> | ObjectEntry] in one TLAB bump.
            // When combined, obj_mem is Inline (no separate dealloc for RefCell<GcPtr>),
            // oe_block holds the TLAB block (releasing it frees both).
            let (gc_ptr, oe_ptr, obj_mem, obj_layout, oe_block);
            if let Some((a, b, mem, _combined_layout)) =
                self.try_alloc_combined_tlab::<RefCell<GcPtr<T>>, ObjectEntry>()
            {
                gc_ptr = a;
                oe_ptr = b;
                oe_block = match mem {
                    GcObjMem::Tlab(block) => block,
                    _ => std::ptr::null_mut(),
                };
                obj_mem = GcObjMem::Inline;
                obj_layout = Layout::new::<RefCell<GcPtr<T>>>();
            } else {
                let (a, (mem_a, layout_a)) = self.alloc_mem_tlab::<RefCell<GcPtr<T>>>();
                gc_ptr = a;
                obj_mem = mem_a;
                obj_layout = layout_a;
                let (c, block) = self.alloc_entry_tlab();
                oe_ptr = c;
                oe_block = block;
            }
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let root_ref_count_offset = (&(*(*gc_ptr).as_ptr()).info.root_ref_count
                as *const Cell<usize> as usize
                - gc_ptr as usize) as u16;
            std::ptr::write(
                oe_ptr,
                ObjectEntry {
                    ptr: gc_ptr as *const dyn Trace,
                    mem: obj_mem,
                    layout: CompactLayout::from_layout(obj_layout),
                    gen_survive_region: region.0 & 0xFFFF,
                    handle_count: 1,
                    root_count: 1,
                    root_ref_count_offset,
                    entry_block: oe_block,
                },
            );
            let gc_maps = self.core.gc_maps_unsync();
            let obj_id = gc_maps.objects.insert(ObjectEntryRef(oe_ptr));
            gc_maps.gen0_ids.push(obj_id);
            self.core.track_alloc(obj_layout.size());

            // Store object_id in GcPtr so GcCell<T> can retrieve it via pointer deref.
            (*(*gc_ptr).as_ptr()).object_id = obj_id;

            // If an incremental mark cycle is in progress, grey this object so
            // the colors-based sweep in finish_collection keeps it.
            self.core.note_new_object_during_marking(obj_id);

            let gc = GcCell::from_parts(gc_ptr, true);
            // Set context for reset_root cascade to decrement child root_counts.
            let gc_maps_raw = gc_maps as *mut GcMaps;
            GC_MAPS_CTX.set(&raw mut (*gc_maps_raw).cascade_visited);
            (*gc_ptr).reset_root();
            GC_MAPS_CTX.set(std::ptr::null_mut());
            self.core.allocation_count.add(1);
            gc
        }
    }

    unsafe fn clone_from_gc_cell<T>(&self, gc: &GcCell<T>) -> GcCell<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let object_id = gc.object_id();
            let refcell_ptr = gc.refcell_ptr();
            let gc_maps = self.core.gc_maps_unsync();
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.handle_count += 1;
                entry.root_count += 1;
            }
            let new_gc = GcCell::from_parts(refcell_ptr, true);
            let gc_maps_raw = gc_maps as *mut GcMaps;
            GC_MAPS_CTX.set(&raw mut (*gc_maps_raw).cascade_visited);
            (*refcell_ptr).reset_root();
            GC_MAPS_CTX.set(std::ptr::null_mut());
            new_gc
        }
    }

    /// Fallible version of `create_gc`. Returns `Err(GcAllocError)` on OOM.
    unsafe fn try_create_gc<T>(&mut self, t: T, region: RegionId) -> Result<Gc<T>, GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            // Double allocation: [GcPtr<T> | ObjectEntry] in one TLAB bump.
            // When combined, obj_mem is Inline (no separate dealloc for GcPtr),
            // oe_block holds the TLAB block (releasing it frees both).
            let (gc_ptr, oe_ptr, obj_mem, obj_layout, oe_block);
            if let Some((a, b, mem, _combined_layout)) =
                self.try_alloc_combined_tlab::<GcPtr<T>, ObjectEntry>()
            {
                gc_ptr = a;
                oe_ptr = b;
                oe_block = match mem {
                    GcObjMem::Tlab(block) => block,
                    _ => std::ptr::null_mut(),
                };
                obj_mem = GcObjMem::Inline;
                obj_layout = Layout::new::<GcPtr<T>>();
            } else {
                let (a, (mem_a, layout_a)) = self.try_alloc_mem_tlab::<GcPtr<T>>()?;
                gc_ptr = a;
                obj_mem = mem_a;
                obj_layout = layout_a;
                let (c, block) = self.alloc_entry_tlab();
                oe_ptr = c;
                oe_block = block;
            }
            std::ptr::write(gc_ptr, GcPtr::new(t));

            let root_ref_count_offset = (&(*gc_ptr).info.root_ref_count as *const Cell<usize>
                as usize
                - gc_ptr as usize) as u16;
            std::ptr::write(
                oe_ptr,
                ObjectEntry {
                    ptr: gc_ptr as *const dyn Trace,
                    mem: obj_mem,
                    layout: CompactLayout::from_layout(obj_layout),
                    gen_survive_region: region.0 & 0xFFFF,
                    handle_count: 1,
                    root_count: 1,
                    root_ref_count_offset,
                    entry_block: oe_block,
                },
            );
            let gc_maps = self.core.gc_maps_unsync();
            let obj_id = gc_maps.objects.insert(ObjectEntryRef(oe_ptr));
            gc_maps.gen0_ids.push(obj_id);
            self.core.track_alloc(obj_layout.size());

            // Store object_id in GcPtr so Gc<T> can retrieve it via pointer deref.
            (*gc_ptr).object_id = obj_id;

            // If an incremental mark cycle is in progress, grey this object so
            // the colors-based sweep in finish_collection keeps it.
            self.core.note_new_object_during_marking(obj_id);

            let gc = Gc::from_parts(gc_ptr, true);
            // Set context for reset_root cascade to decrement child root_counts.
            let gc_maps_raw = gc_maps as *mut GcMaps;
            GC_MAPS_CTX.set(&raw mut (*gc_maps_raw).cascade_visited);
            (*gc_ptr).reset_root();
            GC_MAPS_CTX.set(std::ptr::null_mut());
            self.core.allocation_count.add(1);
            Ok(gc)
        }
    }

    /// Fallible version of `create_gc_cell`. Returns `Err(GcAllocError)` on OOM.
    unsafe fn try_create_gc_cell<T>(
        &mut self,
        t: T,
        region: RegionId,
    ) -> Result<GcCell<T>, GcAllocError>
    where
        T: Sized + Trace,
    {
        unsafe {
            // Double allocation: [RefCell<GcPtr<T>> | ObjectEntry] in one TLAB bump.
            // When combined, obj_mem is Inline (no separate dealloc for RefCell<GcPtr>),
            // oe_block holds the TLAB block (releasing it frees both).
            let (gc_ptr, oe_ptr, obj_mem, obj_layout, oe_block);
            if let Some((a, b, mem, _combined_layout)) =
                self.try_alloc_combined_tlab::<RefCell<GcPtr<T>>, ObjectEntry>()
            {
                gc_ptr = a;
                oe_ptr = b;
                oe_block = match mem {
                    GcObjMem::Tlab(block) => block,
                    _ => std::ptr::null_mut(),
                };
                obj_mem = GcObjMem::Inline;
                obj_layout = Layout::new::<RefCell<GcPtr<T>>>();
            } else {
                let (a, (mem_a, layout_a)) = self.try_alloc_mem_tlab::<RefCell<GcPtr<T>>>()?;
                gc_ptr = a;
                obj_mem = mem_a;
                obj_layout = layout_a;
                let (c, block) = self.alloc_entry_tlab();
                oe_ptr = c;
                oe_block = block;
            }
            std::ptr::write(gc_ptr, RefCell::new(GcPtr::new(t)));

            let root_ref_count_offset = (&(*(*gc_ptr).as_ptr()).info.root_ref_count
                as *const Cell<usize> as usize
                - gc_ptr as usize) as u16;
            std::ptr::write(
                oe_ptr,
                ObjectEntry {
                    ptr: gc_ptr as *const dyn Trace,
                    mem: obj_mem,
                    layout: CompactLayout::from_layout(obj_layout),
                    gen_survive_region: region.0 & 0xFFFF,
                    handle_count: 1,
                    root_count: 1,
                    root_ref_count_offset,
                    entry_block: oe_block,
                },
            );
            let gc_maps = self.core.gc_maps_unsync();
            let obj_id = gc_maps.objects.insert(ObjectEntryRef(oe_ptr));
            gc_maps.gen0_ids.push(obj_id);
            self.core.track_alloc(obj_layout.size());

            // Store object_id in GcPtr so GcCell<T> can retrieve it via pointer deref.
            (*(*gc_ptr).as_ptr()).object_id = obj_id;

            // If an incremental mark cycle is in progress, grey this object so
            // the colors-based sweep in finish_collection keeps it.
            self.core.note_new_object_during_marking(obj_id);

            let gc = GcCell::from_parts(gc_ptr, true);
            // Set context for reset_root cascade to decrement child root_counts.
            let gc_maps_raw = gc_maps as *mut GcMaps;
            GC_MAPS_CTX.set(&raw mut (*gc_maps_raw).cascade_visited);
            (*gc_ptr).reset_root();
            GC_MAPS_CTX.set(std::ptr::null_mut());
            self.core.allocation_count.add(1);
            Ok(gc)
        }
    }

    /// Create a new strong Gc<T> from a weak reference (for upgrade).
    unsafe fn upgrade_weak<T>(&self, weak: &GcWeak<T>) -> Gc<T>
    where
        T: Sized + Trace,
    {
        unsafe {
            let object_id = weak.object_id;
            let gc_maps = self.core.gc_maps_unsync();
            if let Some(entry) = gc_maps.objects.get_mut(object_id) {
                entry.handle_count += 1;
                entry.root_count += 1;
            }
            let gc = Gc::from_parts(weak.ptr, true);
            let gc_maps_raw = gc_maps as *mut GcMaps;
            GC_MAPS_CTX.set(&raw mut (*gc_maps_raw).cascade_visited);
            (*weak.ptr).reset_root();
            GC_MAPS_CTX.set(std::ptr::null_mut());
            gc
        }
    }

    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect(&self) {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect().
        unsafe {
            self.core.collect();
        }
    }

    /// Collect objects up to the specified generation.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_generation(&self, max_gen: Generation) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_generation().
        unsafe { self.core.collect_generation(max_gen) }
    }

    #[allow(dead_code)]
    unsafe fn collect_all(&self) {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_all().
        unsafe {
            self.core.collect_all();
        }
    }

    /// Parallel collection: serial mark phase, parallel sweep/dealloc using rayon.
    /// Collects objects in generations 0..=max_gen.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn collect_parallel(&self, max_gen: Generation) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_parallel().
        unsafe { self.core.collect_parallel(max_gen) }
    }

    /// Collection with parallel mark AND parallel sweep using rayon.
    /// Parallelizes root tracing across worker threads for faster marking
    /// on large heaps with many roots.
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn collect_parallel_mark(&self, max_gen: Generation) -> CollectionStats {
        unsafe { self.core.collect_parallel_mark(max_gen) }
    }

    #[inline]
    pub(crate) unsafe fn drop_handle(&self, object_id: ObjectId, is_root: bool) {
        // SAFETY: Thread-local GC, no concurrent access; caller provides valid object_id.
        unsafe {
            self.core.drop_handle_unsync(object_id, is_root);
        }
    }

    /// Begin an incremental collection cycle.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn begin_collection(&self, max_gen: Generation) {
        // SAFETY: Caller upholds the safety contract; delegates to core.begin_collection().
        unsafe {
            self.core.begin_collection(max_gen);
        }
    }

    /// Process a batch of gray objects. Returns true when marking is complete.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn mark_step(&self, budget: usize) -> bool {
        // SAFETY: Caller upholds the safety contract; delegates to core.mark_step().
        unsafe { self.core.mark_step(budget) }
    }

    /// Finish incremental collection: sweep white objects, promote survivors.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn finish_collection(&self) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.finish_collection().
        unsafe { self.core.finish_collection() }
    }

    /// Run a complete incremental collection with the given step budget.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_incremental(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_incremental().
        unsafe { self.core.collect_incremental(max_gen, step_budget) }
    }

    /// Begin a concurrent collection: short STW to snapshot roots + edges.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn begin_concurrent_collection(&self, max_gen: Generation) {
        // SAFETY: Caller upholds the safety contract; delegates to core.begin_concurrent_collection().
        unsafe {
            self.core.begin_concurrent_collection(max_gen);
        }
    }

    /// Process gray objects using edge snapshot. NO STW lock — safe for concurrent use.
    pub(crate) fn concurrent_mark_step(&self, budget: usize) -> bool {
        self.core.concurrent_mark_step(budget)
    }

    /// Run a complete concurrent collection (snapshot → concurrent mark → STW sweep).
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_concurrent(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_concurrent().
        unsafe { self.core.collect_concurrent(max_gen, step_budget) }
    }

    /// Time-budgeted incremental mark step. See `GarbageCollector::mark_step_timed`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn mark_step_timed(&self, max_duration: Duration) -> bool {
        // SAFETY: Caller upholds the safety contract; delegates to core.mark_step_timed().
        unsafe { self.core.mark_step_timed(max_duration) }
    }

    /// Run a complete incremental collection with time-budgeted steps.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_incremental_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_incremental_timed().
        unsafe {
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
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_concurrent_timed().
        unsafe {
            self.core
                .collect_concurrent_timed(max_gen, max_step_duration)
        }
    }

    /// Create a new region. Future allocations go into this region.
    pub fn new_region(&self) -> LocalRegionId {
        LocalRegionId(self.core.new_region())
    }

    /// Get the current allocation region.
    pub fn current_region(&self) -> LocalRegionId {
        LocalRegionId(self.core.current_region())
    }

    /// Collect only objects in the specified region.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are used during collection.
    pub(crate) unsafe fn collect_region(&self, region: LocalRegionId) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.collect_region().
        unsafe { self.core.collect_region(region.0) }
    }

    /// G1-style "Garbage First" collection. Marks all objects, then sweeps
    /// regions with the highest garbage ratio first, stopping when the elapsed
    /// time reaches `pause_target`.
    ///
    /// # Safety
    /// The caller must ensure no references to GC-managed objects are held
    /// across this call (stop-the-world requirement).
    pub(crate) unsafe fn collect_garbage_first(&self, pause_target: Duration) -> CollectionStats {
        // SAFETY: Caller upholds the safety contract; delegates to core.
        unsafe { self.core.collect_garbage_first(pause_target) }
    }

    /// Return per-region liveness statistics without running a collection.
    pub fn region_stats(&self) -> Vec<crate::generation::RegionStats> {
        self.core.region_stats()
    }

    /// Return a snapshot of current GC diagnostics.
    pub fn stats(&self) -> GcStats {
        self.core.stats()
    }

    /// Set custom promotion thresholds (how many collections an object must survive
    /// before being promoted to the next generation).
    pub fn set_promotion_config(&self, config: PromotionConfig) {
        self.core.set_promotion_config(config);
    }

    /// Get the current promotion config.
    pub fn promotion_config(&self) -> PromotionConfig {
        self.core.promotion_config()
    }

    /// Register a callback invoked after each collection cycle.
    pub fn set_on_collection(&self, callback: impl Fn(&CollectionStats) + Send + Sync + 'static) {
        self.core.set_on_collection(callback);
    }

    /// Remove the collection callback.
    pub fn clear_on_collection(&self) {
        self.core.clear_on_collection();
    }

    /// Compact the heap by copying all live objects into a contiguous buffer.
    /// Runs a full collection first, then relocates surviving objects for
    /// improved cache locality and reduced fragmentation.
    ///
    /// Returns the number of objects compacted.
    ///
    /// **Currently disabled**: always returns 0 without moving any objects
    /// (see `GarbageCollector::compact` for what re-enabling requires).
    ///
    /// # Safety
    /// Must not be called while any GC references are being dereferenced.
    pub(crate) unsafe fn compact(&self) -> usize {
        unsafe { self.core.compact() }
    }
}

// ---- Feature-gated public access for benchmarks and integration tests ----

#[cfg(feature = "_internal")]
impl LocalGarbageCollector {
    pub unsafe fn _collect(&self) {
        unsafe { self.collect() }
    }
    pub unsafe fn _collect_generation(&self, max_gen: Generation) -> CollectionStats {
        unsafe { self.collect_generation(max_gen) }
    }
    pub unsafe fn _collect_incremental(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        unsafe { self.collect_incremental(max_gen, step_budget) }
    }
    pub unsafe fn _collect_incremental_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        unsafe { self.collect_incremental_timed(max_gen, max_step_duration) }
    }
    pub unsafe fn _collect_concurrent(
        &self,
        max_gen: Generation,
        step_budget: usize,
    ) -> CollectionStats {
        unsafe { self.collect_concurrent(max_gen, step_budget) }
    }
    pub unsafe fn _collect_concurrent_timed(
        &self,
        max_gen: Generation,
        max_step_duration: Duration,
    ) -> CollectionStats {
        unsafe { self.collect_concurrent_timed(max_gen, max_step_duration) }
    }
    pub unsafe fn _collect_region(&self, region: LocalRegionId) -> CollectionStats {
        unsafe { self.collect_region(region) }
    }
    pub unsafe fn _collect_garbage_first(&self, pause_target: Duration) -> CollectionStats {
        unsafe { self.collect_garbage_first(pause_target) }
    }
    pub unsafe fn _compact(&self) -> usize {
        unsafe { self.compact() }
    }
}

// ---- LocalRegionId helpers for thread-local GC ----

impl LocalRegionId {
    /// Allocate a new `Gc<T>` in this region (thread-local collector).
    pub fn gc<T: 'static + Sized + Trace>(self, t: T) -> Gc<T> {
        Gc::new_in(t, self)
    }

    /// Fallible `Gc<T>` allocation in this region (thread-local collector).
    pub fn try_gc<T: 'static + Sized + Trace>(self, t: T) -> Result<Gc<T>, GcAllocError> {
        Gc::try_new_in(t, self)
    }

    /// Allocate a new `GcCell<T>` in this region (thread-local collector).
    pub fn gc_cell<T: 'static + Sized + Trace>(self, t: T) -> GcCell<T> {
        GcCell::new_in(t, self)
    }

    /// Fallible `GcCell<T>` allocation in this region (thread-local collector).
    pub fn try_gc_cell<T: 'static + Sized + Trace>(self, t: T) -> Result<GcCell<T>, GcAllocError> {
        GcCell::try_new_in(t, self)
    }
}

impl PartialEq for &LocalGarbageCollector {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(
            *self as *const LocalGarbageCollector,
            *other as *const LocalGarbageCollector,
        )
    }
}

/// Callback type for a collection strategy. Receives the GC and an "active" flag;
/// may spawn a background thread and return its `JoinHandle`.
pub type LocalStrategyFn =
    Box<dyn FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>>;

/// Pluggable collection strategy for the thread-local GC.
/// Controls when and how garbage collection runs (e.g., periodic background thread,
/// allocation-triggered, or manual). Use `set_strategy` to swap at runtime.
///
/// Internally stores a raw pointer to the `LocalGarbageCollector` to avoid
/// Stacked Borrows violations during thread-local destruction ordering.
pub struct LocalStrategy {
    gc: Cell<*const LocalGarbageCollector>,
    is_active: AtomicBool,
    strategy_func: RefCell<LocalStrategyFn>,
    join_handle: RefCell<Option<JoinHandle<()>>>,
}

impl LocalStrategy {
    fn new<StrategyFn>(
        gc_ptr: *const LocalGarbageCollector,
        strategy_fn: StrategyFn,
    ) -> LocalStrategy
    where
        StrategyFn: 'static
            + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
    {
        LocalStrategy {
            gc: Cell::new(gc_ptr),
            is_active: AtomicBool::new(false),
            strategy_func: RefCell::new(Box::new(strategy_fn)),
            join_handle: RefCell::new(None),
        }
    }

    /// Replace the current collection strategy. Stops the current strategy first if active.
    pub fn set_strategy<StrategyFn>(&self, strategy_fn: StrategyFn)
    where
        StrategyFn: 'static
            + FnMut(&'static LocalGarbageCollector, &'static AtomicBool) -> Option<JoinHandle<()>>,
    {
        if self.is_active() {
            self.stop();
        }
        let _ = self.strategy_func.replace(Box::new(strategy_fn));
    }

    /// Returns `true` if the strategy's background collection is currently running.
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Start the collection strategy (e.g., spawn a background collection thread).
    pub fn start(&'static self) {
        self.is_active.store(true, Ordering::Release);
        // SAFETY: gc pointer is valid for the lifetime of the thread-local and is only
        // accessed from the owning thread.
        let gc_ref = unsafe { &*self.gc.get() };
        self.join_handle
            .replace((*(self.strategy_func.borrow_mut()))(
                gc_ref,
                &self.is_active,
            ));
    }

    /// Stop the collection strategy and join any background thread.
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        if let Some(join_handle) = self.join_handle.borrow_mut().take() {
            join_handle
                .join()
                .expect("LocalStrategy::stop: strategy thread panicked");
        }
    }
}

impl Drop for LocalStrategy {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
    }
}

thread_local! {
    pub static LOCAL_GC: RefCell<LocalGarbageCollector> = RefCell::new(LocalGarbageCollector::new());
    pub static LOCAL_GC_STRATEGY: RefCell<LocalStrategy> = {
        LOCAL_GC.with(move |gc| {
            // Use as_ptr() directly to get a raw pointer without creating an intermediate
            // reference, avoiding Stacked Borrows provenance issues during thread-local
            // destruction ordering.
            let gc_ptr = gc.as_ptr() as *const LocalGarbageCollector;
            // NOTE: Local GCs use allocation-triggered collection on the owning thread
            // instead of registering with the background thread. Collecting from a
            // background thread would be a data race because thread-local objects use
            // non-atomic Cell/RefCell internals. The background thread is only used
            // for the global/sync GC which uses atomic operations.
            RefCell::new(LocalStrategy::new(gc_ptr, move |_local_gc, _| {
                None
            }))
        })
    };
}
