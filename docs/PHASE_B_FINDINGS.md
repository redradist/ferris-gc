# Phase B (inline `ObjectEntry`) — WIP findings

> Status: **experimental, NOT merged.** This branch validates the cache-locality
> hypothesis from `PERFORMANCE_ARCHITECTURE.md` §3 Phase B but is a **trade-off,
> not a clean win** in its current (naive, full-inline) form.

## What changed

Before, the `objects` SlotMap stored an 8-byte `ObjectEntryRef` (a pointer) and
the 72-byte `ObjectEntry` lived in a separate TLAB allocation. Every mark/sweep
pass therefore chased a pointer into scattered TLAB memory — a cache miss per
object per pass.

This branch stores the `ObjectEntry` **by value** in the SlotMap's dense array:

- `SlotMap<ObjectId, ObjectEntry>` (was `SlotMap<ObjectId, ObjectEntryRef>`)
- `ObjectEntry` lost its `entry_block` field; `ObjectEntryRef` and the
  combined-TLAB allocation path are gone from the create sites.
- Mark/sweep now iterate the dense array directly (cache-friendly, no chase).

## Measurement (interleaved A/B, fat-LTO build, N=15, two runs, medians)

| Bench | duration | max GC pause | throughput | verdict |
|---|---|---|---|---|
| alloc | **+7–11%** | **+27–28%** | **+8–13%** | win |
| tree (2.1M) | **+16–18%** | **+27%** | **+19–21%** | win |
| churn | −20% | −30%/+28% (noisy) | −17% | regress |
| generational | −17–22% | −5% | −15–18% | regress |
| concurrent | −16–22% | −16–21% | −14–18% | regress |

(Positive = Phase B faster / lower pause. Both runs agreed on every sign except
churn's very small `max_gc_pause`.)

## Interpretation

The split is mechanistically clean and reproducible:

- **Mark/sweep-bound workloads win.** `tree` and `alloc` spend their time walking
  the object set; the dense-array layout removes the per-object pointer chase.
  `tree` is FerrisGC's worst gap vs Go (~6.6×) — a 27% pause cut there is the most
  valuable single result of the phase.
- **Alloc/free-churn-bound workloads regress.** `churn`, `generational`, and
  `concurrent` create and destroy short-lived objects rapidly. Storing the entry
  by value means every SlotMap insert copies 72 B into the array and every remove
  moves 72 B out — vs an 8-byte pointer before. That memory traffic dominates when
  the object barely outlives its allocation.

This is exactly the inline-vs-indirect tension the roadmap flagged
("Keep cold fields (layout, tracers) in the side entry").

## Why it isn't merged

Regressing three of five benchmarks by 15–22% fails the "don't merge a GC change
that makes things worse" bar. The naive full-inline moves *all* fields inline,
including the cold ones (`layout`, `mem`, `tracers`) that only the dealloc path
reads — so the churn path pays to copy bytes it never touches on the hot path.

## Next step — hot/cold split (the actually-mergeable Phase B)

Split `ObjectEntry` into:

- **Hot, inline in the dense array:** `ptr`, generation/region, `handle_count`,
  `root_count`, `root_ref_count_offset` — the fields mark/sweep touch. ~32 B, so
  the insert/remove copy roughly halves and the mark/sweep cache win is kept.
- **Cold, behind a pointer (ideally TLAB-combined with the object so dealloc
  touches one line):** `mem`, `layout`, `tracers`.

Caveat to verify: the sync GC touches `tracers` on the clone/drop path
(RC-hybrid), so pushing `tracers` cold risks re-introducing a chase on
`concurrent`. Measure that path specifically before committing to the layout.
