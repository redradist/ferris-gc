//! Per-thread size-class pool for the global (`sync`) collector's memory.
//!
//! With the shared locks and counters gone, the system allocator had become
//! the largest cost of a `sync::Gc` allocation: glibc's `malloc`/`free` paths
//! (no fast bin above 128 bytes, coalescing on every free, heap trimming)
//! took about half the instructions of an allocate/drop pair. Freed blocks
//! are retained here instead and reused by the thread that owns them.
//!
//! # Layout
//!
//! Memory comes in [`PAGE_SIZE`] pages aligned to their own size, so the page
//! of any block is found by masking its address. A page serves one size class
//! (multiples of [`GRANULE`] up to [`MAX_BLOCK`]) and is owned by one [`Heap`].
//! Each thread that allocates owns one heap.
//!
//! # Freeing
//!
//! - **On the owning thread:** the block goes on the page's `free` list.
//!   Plain stores, no atomics.
//! - **From any other thread** (handles dropped elsewhere, the collector's
//!   sweep): the block is pushed with a CAS onto the page's `xfree` stack.
//!   The owner takes that stack over in one swap when its own lists run dry.
//!
//! A page with no free block is unlinked from its heap and marked full by
//! setting the low [`FULL`] bit of `xfree` (only possible while the stack is
//! empty). The remote free that clears the bit pushes the page onto the
//! heap's `rescued` stack, so the owner reuses it without scanning full
//! pages. Push and full-state change are one CAS, so a remote free never
//! touches a page after the block it freed is visible to the owner: the owner
//! may release the page from that moment on.
//!
//! # Lifetime
//!
//! Heaps are never freed. When a thread exits, its heap (with every page
//! still holding live objects) goes on a global list and is adopted by the
//! next thread that allocates. A page whose blocks are all free is released:
//! kept in a small per-heap cache of empty pages, beyond that returned to the
//! system allocator.

use std::alloc::{Layout, alloc, dealloc};
use std::cell::Cell;
use std::ptr::{self, NonNull};
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Size (and alignment) of one page.
const PAGE_SIZE: usize = 64 * 1024;
/// Size-class step; also the alignment every block gets.
const GRANULE: usize = 16;
/// Largest block the pool serves; bigger requests use the system allocator.
const MAX_BLOCK: usize = 1024;
const CLASSES: usize = MAX_BLOCK / GRANULE;
/// Empty pages a heap keeps for reuse before returning them to the system.
const EMPTY_PAGES_KEPT: usize = 32;

const PAGE_LAYOUT: Layout = match Layout::from_size_align(PAGE_SIZE, PAGE_SIZE) {
    Ok(layout) => layout,
    Err(_) => panic!("invalid pool page layout"),
};

/// Tag bit in `Page::xfree`: the page is full and not linked in its heap.
/// Blocks are `GRANULE`-aligned, so the bit is never part of a node address.
const FULL: usize = 1;

/// A free block's first word links it into a free list.
struct FreeNode {
    next: *mut FreeNode,
}

/// Page header, at the start of every page.
#[repr(C)]
struct Page {
    // ---- Read or written by other threads ----
    /// Owning heap. Written only when the page is first created; a remote
    /// thread reads it while holding a live block of the page.
    heap: *mut Heap,
    /// Blocks freed by other threads, plus the [`FULL`] tag bit.
    xfree: AtomicPtr<FreeNode>,
    /// Link in the owner's `rescued` stack; written by the remote thread that
    /// clears [`FULL`], read by the owner after taking the stack.
    rescue_next: *mut Page,
    // ---- Owner thread only ----
    free: *mut FreeNode,
    /// Blocks handed out and not yet back on `free`.
    used: usize,
    /// Offset of the first never-used block.
    bump: usize,
    block_size: usize,
    class: usize,
    next: *mut Page,
    prev: *mut Page,
    /// Linked in `Heap::avail` (otherwise full, or waiting in `rescued`).
    linked: bool,
}

/// Blocks start after the header, at a `GRANULE` boundary.
const HEADER: usize = (size_of::<Page>() + GRANULE - 1) & !(GRANULE - 1);

/// One thread's pages.
///
/// Only the owning thread touches the plain fields; other threads only push
/// onto `rescued`. All access goes through raw places (never `&mut Heap`), so
/// the owner's accesses do not alias a remote thread's `&AtomicPtr`.
struct Heap {
    /// Full pages that remote frees made usable again.
    rescued: AtomicPtr<Page>,
    /// Per class: pages that may have free blocks. The head is allocated from.
    avail: [*mut Page; CLASSES],
    /// Empty pages kept for reuse, linked through `Page::next`.
    empty: *mut Page,
    empty_count: usize,
}

struct HeapPtr(*mut Heap);
// SAFETY: a heap on the free list has no owner; the list's mutex hands it
// from the exiting thread to the adopting one.
unsafe impl Send for HeapPtr {}

/// Heaps of exited threads, waiting for a new owner.
static FREE_HEAPS: Mutex<Vec<HeapPtr>> = Mutex::new(Vec::new());

/// This thread's heap; retired (put on [`FREE_HEAPS`]) when the thread exits.
struct HeapSlot(Cell<*mut Heap>);

impl Drop for HeapSlot {
    fn drop(&mut self) {
        let heap = self.0.get();
        if !heap.is_null() {
            // SAFETY: the heap is owned by this (exiting) thread.
            unsafe { heap_retire(heap) };
        }
    }
}

thread_local! {
    static HEAP: HeapSlot = const { HeapSlot(Cell::new(ptr::null_mut())) };
}

/// Size class serving `layout`, or `None` if the pool does not serve it.
#[inline]
fn class_of(layout: Layout) -> Option<usize> {
    let size = layout.size();
    if size == 0 || size > MAX_BLOCK || layout.align() > GRANULE {
        return None;
    }
    Some((size - 1) / GRANULE)
}

/// Allocate a block for `layout` from this thread's heap.
///
/// Returns `None` if the pool does not serve the layout (too large or
/// over-aligned), the system allocator failed, or the thread is past its
/// thread-local destructors. Callers then fall back to the system allocator.
/// A returned block must be released with [`free`].
#[inline]
pub(crate) fn alloc_block(layout: Layout) -> Option<NonNull<u8>> {
    let class = class_of(layout)?;
    HEAP.try_with(|slot| {
        let mut heap = slot.0.get();
        if heap.is_null() {
            heap = heap_acquire();
            slot.0.set(heap);
        }
        // SAFETY: `heap` is this thread's heap.
        unsafe { heap_alloc(heap, class) }
    })
    .ok()
    .flatten()
}

/// Return a block obtained from [`alloc_block`]. Any thread may call this.
///
/// # Safety
/// `block` must come from [`alloc_block`] and not have been freed already;
/// nothing may access it afterwards.
#[inline]
pub(crate) unsafe fn free_block(block: *mut u8) {
    let page: *mut Page = block.map_addr(|a| a & !(PAGE_SIZE - 1)).cast();
    let node: *mut FreeNode = block.cast();
    unsafe {
        // SAFETY: the block is live, so its page is too, and `heap` was
        // written before the block was handed out.
        let heap = (*page).heap;
        let mine = HEAP.try_with(|slot| slot.0.get() == heap).unwrap_or(false);
        if mine {
            local_free(heap, page, node);
        } else {
            remote_free(heap, page, node);
        }
    }
}

fn heap_acquire() -> *mut Heap {
    if let Some(HeapPtr(heap)) = FREE_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        return heap;
    }
    Box::into_raw(Box::new(Heap {
        rescued: AtomicPtr::new(ptr::null_mut()),
        avail: [ptr::null_mut(); CLASSES],
        empty: ptr::null_mut(),
        empty_count: 0,
    }))
}

/// Give up a heap on thread exit: return what is free to the system and
/// leave the rest for the next thread.
unsafe fn heap_retire(heap: *mut Heap) {
    unsafe {
        drain_rescued(heap);
        for class in 0..CLASSES {
            let mut page = (*heap).avail[class];
            while !page.is_null() {
                let next = (*page).next;
                collect_remote(page);
                if (*page).used == 0 {
                    unlink(heap, page);
                    dealloc(page.cast(), PAGE_LAYOUT);
                }
                page = next;
            }
        }
        while !(*heap).empty.is_null() {
            let page = (*heap).empty;
            (*heap).empty = (*page).next;
            dealloc(page.cast(), PAGE_LAYOUT);
        }
        (*heap).empty_count = 0;
    }
    FREE_HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(HeapPtr(heap));
}

#[inline]
unsafe fn heap_alloc(heap: *mut Heap, class: usize) -> Option<NonNull<u8>> {
    unsafe {
        let page = (*heap).avail[class];
        if !page.is_null() {
            if let Some(block) = page_take(page) {
                return Some(block);
            }
        }
        heap_alloc_slow(heap, class)
    }
}

/// Take a block from the page's own free list, else from its unused tail.
#[inline]
unsafe fn page_take(page: *mut Page) -> Option<NonNull<u8>> {
    unsafe {
        let node = (*page).free;
        if !node.is_null() {
            (*page).free = (*node).next;
            (*page).used += 1;
            return Some(NonNull::new_unchecked(node.cast()));
        }
        let offset = (*page).bump;
        if offset + (*page).block_size <= PAGE_SIZE {
            (*page).bump = offset + (*page).block_size;
            (*page).used += 1;
            return Some(NonNull::new_unchecked(page.cast::<u8>().add(offset)));
        }
        None
    }
}

#[cold]
#[inline(never)]
unsafe fn heap_alloc_slow(heap: *mut Heap, class: usize) -> Option<NonNull<u8>> {
    unsafe {
        drain_rescued(heap);
        loop {
            let page = (*heap).avail[class];
            if page.is_null() {
                break;
            }
            collect_remote(page);
            if let Some(block) = page_take(page) {
                return Some(block);
            }
            // Exhausted: retire it as full. The tag can only be set while no
            // remote free is pending; if one is, take it and keep the page.
            unlink(heap, page);
            if (*page)
                .xfree
                .compare_exchange(
                    ptr::null_mut(),
                    ptr::without_provenance_mut(FULL),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                link_front(heap, page);
            }
        }
        let page = new_page(heap, class)?;
        link_front(heap, page);
        page_take(page)
    }
}

/// Move the remotely freed blocks onto the page's own free list.
/// Only for linked pages, whose `xfree` never carries the [`FULL`] tag.
#[inline]
unsafe fn collect_remote(page: *mut Page) {
    unsafe {
        let head = (*page).xfree.swap(ptr::null_mut(), Ordering::Acquire);
        if head.is_null() {
            return;
        }
        let mut tail = head;
        let mut count = 1;
        while !(*tail).next.is_null() {
            tail = (*tail).next;
            count += 1;
        }
        (*tail).next = (*page).free;
        (*page).free = head;
        (*page).used -= count;
    }
}

/// Re-link the full pages that remote frees made usable again.
unsafe fn drain_rescued(heap: *mut Heap) {
    unsafe {
        let mut page = (*heap).rescued.swap(ptr::null_mut(), Ordering::Acquire);
        while !page.is_null() {
            let next = (*page).rescue_next;
            link_second(heap, page);
            page = next;
        }
    }
}

unsafe fn new_page(heap: *mut Heap, class: usize) -> Option<*mut Page> {
    unsafe {
        let page = if !(*heap).empty.is_null() {
            // A released page: `xfree` is empty and untagged, `heap` is set.
            let page = (*heap).empty;
            (*heap).empty = (*page).next;
            (*heap).empty_count -= 1;
            page
        } else {
            let page: *mut Page = alloc(PAGE_LAYOUT).cast();
            if page.is_null() {
                return None;
            }
            ptr::write(
                page,
                Page {
                    heap,
                    xfree: AtomicPtr::new(ptr::null_mut()),
                    rescue_next: ptr::null_mut(),
                    free: ptr::null_mut(),
                    used: 0,
                    bump: 0,
                    block_size: 0,
                    class: 0,
                    next: ptr::null_mut(),
                    prev: ptr::null_mut(),
                    linked: false,
                },
            );
            page
        };
        (*page).class = class;
        (*page).block_size = (class + 1) * GRANULE;
        (*page).bump = HEADER;
        (*page).free = ptr::null_mut();
        (*page).used = 0;
        Some(page)
    }
}

/// Owner-thread free.
#[inline]
unsafe fn local_free(heap: *mut Heap, page: *mut Page, node: *mut FreeNode) {
    unsafe {
        (*node).next = (*page).free;
        (*page).free = node;
        (*page).used -= 1;
        if !(*page).linked {
            // Full page: reopen it, unless a remote free cleared the tag
            // first - then it is on its way through `rescued`.
            if (*page)
                .xfree
                .compare_exchange(
                    ptr::without_provenance_mut(FULL),
                    ptr::null_mut(),
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                link_second(heap, page);
            }
        } else if (*page).used == 0 && (*heap).avail[(*page).class] != page {
            unlink(heap, page);
            release_page(heap, page);
        }
    }
}

/// Free from a thread that does not own the page. After the CAS publishes
/// the block, the page is not touched again unless this thread also cleared
/// [`FULL`] - and a full page is unlinked, so the owner cannot release it
/// until it comes back through `rescued`.
unsafe fn remote_free(heap: *mut Heap, page: *mut Page, node: *mut FreeNode) {
    unsafe {
        let xfree = &(*page).xfree;
        let mut old = xfree.load(Ordering::Relaxed);
        loop {
            (*node).next = old.map_addr(|a| a & !FULL);
            match xfree.compare_exchange_weak(old, node, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => break,
                Err(current) => old = current,
            }
        }
        if old.addr() & FULL != 0 {
            let rescued = &(*heap).rescued;
            let mut head = rescued.load(Ordering::Relaxed);
            loop {
                (*page).rescue_next = head;
                match rescued.compare_exchange_weak(head, page, Ordering::Release, Ordering::Relaxed)
                {
                    Ok(_) => break,
                    Err(current) => head = current,
                }
            }
        }
    }
}

/// Keep an empty page for reuse, or return it to the system.
unsafe fn release_page(heap: *mut Heap, page: *mut Page) {
    unsafe {
        if (*heap).empty_count < EMPTY_PAGES_KEPT {
            (*page).next = (*heap).empty;
            (*heap).empty = page;
            (*heap).empty_count += 1;
        } else {
            dealloc(page.cast(), PAGE_LAYOUT);
        }
    }
}

#[inline]
unsafe fn link_front(heap: *mut Heap, page: *mut Page) {
    unsafe {
        let head = (*heap).avail[(*page).class];
        (*page).prev = ptr::null_mut();
        (*page).next = head;
        if !head.is_null() {
            (*head).prev = page;
        }
        (*heap).avail[(*page).class] = page;
        (*page).linked = true;
    }
}

/// Link behind the current allocation page, so that page stays current (and
/// this one can be released once empty).
#[inline]
unsafe fn link_second(heap: *mut Heap, page: *mut Page) {
    unsafe {
        let head = (*heap).avail[(*page).class];
        if head.is_null() {
            link_front(heap, page);
            return;
        }
        let next = (*head).next;
        (*page).prev = head;
        (*page).next = next;
        if !next.is_null() {
            (*next).prev = page;
        }
        (*head).next = page;
        (*page).linked = true;
    }
}

#[inline]
unsafe fn unlink(heap: *mut Heap, page: *mut Page) {
    unsafe {
        let (prev, next) = ((*page).prev, (*page).next);
        if prev.is_null() {
            (*heap).avail[(*page).class] = next;
        } else {
            (*prev).next = next;
        }
        if !next.is_null() {
            (*next).prev = prev;
        }
        (*page).prev = ptr::null_mut();
        (*page).next = ptr::null_mut();
        (*page).linked = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::mpsc;
    use std::thread;

    /// Blocks of one 16-byte-class page, minus the header.
    const PER_PAGE_64: usize = (PAGE_SIZE - HEADER) / 64;

    fn page_of(p: *mut u8) -> usize {
        p.addr() & !(PAGE_SIZE - 1)
    }

    fn layout(size: usize) -> Layout {
        Layout::from_size_align(size, 8).unwrap()
    }

    struct SendPtr(*mut u8);
    unsafe impl Send for SendPtr {}

    /// Pages linked for `class` in this thread's heap, plus its empty cache.
    fn heap_pages(class: usize) -> (usize, usize) {
        HEAP.with(|slot| unsafe {
            let heap = slot.0.get();
            let mut linked = 0;
            let mut page = (*heap).avail[class];
            while !page.is_null() {
                linked += 1;
                page = (*page).next;
            }
            (linked, (*heap).empty_count)
        })
    }

    #[test]
    fn serves_small_aligned_layouts_only() {
        assert!(class_of(layout(0)).is_none());
        assert_eq!(class_of(layout(1)), Some(0));
        assert_eq!(class_of(layout(16)), Some(0));
        assert_eq!(class_of(layout(17)), Some(1));
        assert_eq!(class_of(layout(MAX_BLOCK)), Some(CLASSES - 1));
        assert!(class_of(layout(MAX_BLOCK + 1)).is_none());
        assert!(class_of(Layout::from_size_align(64, 32).unwrap()).is_none());
        assert!(alloc_block(layout(MAX_BLOCK + 1)).is_none());
    }

    #[test]
    fn blocks_are_distinct_aligned_and_reused() {
        thread::spawn(|| {
            let n = PER_PAGE_64 * 2 + 3;
            let blocks: Vec<*mut u8> = (0..n)
                .map(|i| {
                    let p = alloc_block(layout(64)).unwrap().as_ptr();
                    assert_eq!(p.addr() % GRANULE, 0);
                    unsafe { p.write_bytes(i as u8, 64) };
                    p
                })
                .collect();
            let set: HashSet<usize> = blocks.iter().map(|p| p.addr()).collect();
            assert_eq!(set.len(), n);
            let pages: HashSet<usize> = blocks.iter().map(|&p| page_of(p)).collect();
            for (i, p) in blocks.iter().enumerate() {
                assert_eq!(unsafe { p.add(63).read() }, i as u8);
            }
            for &p in &blocks {
                unsafe { free_block(p) };
            }
            // Only the current page stays linked; the others were released.
            let (linked, empty) = heap_pages(class_of(layout(64)).unwrap());
            assert_eq!(linked, 1);
            assert_eq!(empty, 2);
            // Freed pages are reused rather than requested again.
            let again: Vec<*mut u8> = (0..n).map(|_| alloc_block(layout(64)).unwrap().as_ptr()).collect();
            assert!(again.iter().all(|&p| pages.contains(&page_of(p))));
            for p in again {
                unsafe { free_block(p) };
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn remote_frees_return_to_the_owner() {
        let (to_owner, from_remote) = mpsc::channel::<()>();
        let (to_remote, blocks_rx) = mpsc::channel::<Vec<SendPtr>>();
        let remote = thread::spawn(move || {
            for p in blocks_rx.recv().unwrap() {
                unsafe { free_block(p.0) };
            }
            to_owner.send(()).unwrap();
        });
        thread::spawn(move || {
            // More than a page, so the first page is retired as full and must
            // come back through `rescued`.
            let n = PER_PAGE_64 + 5;
            let blocks: Vec<*mut u8> = (0..n).map(|_| alloc_block(layout(64)).unwrap().as_ptr()).collect();
            let pages: HashSet<usize> = blocks.iter().map(|&p| page_of(p)).collect();
            to_remote.send(blocks.into_iter().map(SendPtr).collect()).unwrap();
            from_remote.recv().unwrap();
            let again: Vec<*mut u8> = (0..n).map(|_| alloc_block(layout(64)).unwrap().as_ptr()).collect();
            assert!(again.iter().all(|&p| pages.contains(&page_of(p))));
            for p in again {
                unsafe { free_block(p) };
            }
        })
        .join()
        .unwrap();
        remote.join().unwrap();
    }

    #[test]
    fn heap_outlives_its_thread() {
        // Blocks allocated by a thread that has exited stay valid, can be
        // freed from anywhere, and the heap is adopted by later threads.
        let blocks: Vec<SendPtr> = thread::spawn(|| {
            (0..PER_PAGE_64 + 1)
                .map(|i| {
                    let p = alloc_block(layout(48)).unwrap().as_ptr();
                    unsafe { p.write(i as u8) };
                    SendPtr(p)
                })
                .collect()
        })
        .join()
        .unwrap();
        for (i, p) in blocks.iter().enumerate() {
            assert_eq!(unsafe { p.0.read() }, i as u8);
        }
        for p in blocks {
            unsafe { free_block(p.0) };
        }
        thread::spawn(|| {
            let p = alloc_block(layout(48)).unwrap().as_ptr();
            unsafe { free_block(p) };
        })
        .join()
        .unwrap();
    }

    #[test]
    fn concurrent_cross_thread_churn() {
        let threads = 4;
        let per = if cfg!(miri) { 60 } else { 5_000 };
        let (txs, rxs): (Vec<_>, Vec<_>) = (0..threads).map(|_| mpsc::channel::<SendPtr>()).unzip();
        let handles: Vec<_> = rxs
            .into_iter()
            .enumerate()
            .map(|(t, rx)| {
                let txs = txs.clone();
                thread::spawn(move || {
                    for i in 0..per {
                        let size = 16 + (i % 5) * 40;
                        let p = alloc_block(layout(size)).unwrap().as_ptr();
                        unsafe { p.write_bytes(t as u8, size) };
                        // Every other block is freed by the next thread.
                        if i % 2 == 0 {
                            txs[(t + 1) % threads].send(SendPtr(p)).unwrap();
                        } else {
                            unsafe { free_block(p) };
                        }
                        while let Ok(q) = rx.try_recv() {
                            unsafe { free_block(q.0) };
                        }
                    }
                    rx
                })
            })
            .collect();
        drop(txs);
        let rxs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for rx in rxs {
            while let Ok(q) = rx.recv() {
                unsafe { free_block(q.0) };
            }
        }
    }
}
