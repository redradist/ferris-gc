# Phase B (inline `ObjectEntry`) — WIP findings

> Status: **experimental, awaiting a merge decision.** This branch validates the
> cache-locality hypothesis from `PERFORMANCE_ARCHITECTURE.md` §3 Phase B. After
> shrinking the entry it is a **pause-vs-throughput trade**: the collector is
> strictly better everywhere (all GC-pause metrics down, big wins on GC-bound
> workloads) at the cost of some mutator alloc/free throughput on tiny-object
> churn. That last cost is **irreducible** — see §"Why not go smaller".

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

## Measurement (interleaved A/B, fat-LTO build, N=15, medians)

Two variants were measured against `master`:

- **v1 — full inline (80 B entry):** the whole `ObjectEntry` inline.
- **v2 — shrunk (72 B entry):** v1 plus removal of the dead `TracerList::Inline`
  variant (24 B → 16 B), which is never constructed. This is the current branch.

| Bench | metric | v1 (80 B) | **v2 (72 B)** |
|---|---|---|---|
| alloc | duration | +7–11% | **+8%** |
| alloc | max pause | +27–28% | **+28%** |
| tree (2.1M) | duration | +16–18% | **+19%** |
| tree | max pause | +27% | **+29%** |
| churn | duration | −20% | **−11%** |
| churn | max pause | mixed | **+25%** |
| generational | duration | −17–22% | **−19%** |
| generational | max pause | −5% | **+14%** |
| concurrent | duration | −16–22% | **−2%** (noise) |
| concurrent | max pause | −16–21% | **+22%** |

(Positive = Phase B faster / lower pause.) The shrink roughly **halved** the
churn regression and turned concurrent from −22% to noise, while **every GC-pause
metric is now green** and the tree/alloc wins held or improved.

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

The regression scales linearly with entry size (80 B → −20% churn, 72 B → −11%),
confirming the SlotMap insert/remove **copy** is the driver.

## Why not go smaller (the hot/cold split isn't worth it)

The obvious next step is a hot/cold split: keep only what mark/sweep touch inline
(`ptr` 16 B, `gen_survive_region` 4 B, `root_count` 4 B, `root_ref_count_offset`
2 B) and push `mem`/`layout`/`tracers` behind a pointer. But it **cannot reach an
all-green result**, for two independent reasons:

1. **A floor of ~40 B, not ~32 B.** The RC-hybrid free path
   (`remove_handle`) reads `handle_count` on *every* handle drop to decide
   whether to free. Pushing it cold adds a pointer chase to the hottest churn
   path — self-defeating — so it must stay inline. With `ptr` (16) + a cold
   pointer (8) + `gen_survive_region` (4) + `handle_count` (4) + `root_count` (4)
   + offset (2), the inline entry floors at ~40 B.

2. **The working set still overflows L1.** `generational_bench` (the stubborn
   −19%) holds ~1000–2000 live entries. At 40 B that's 40–80 KB — past a 32 KB
   L1 regardless. `master`'s 8 B pointer slot (8–16 KB) is the *only* layout that
   fits, and that is exactly the design Phase B trades away for the tree win.

So the churn/generational throughput cost is **irreducible** for tiny leaf
objects: the entry copy is inherently larger than `master`'s pointer copy. A
hot/cold split would shave the regression a few points while adding real
Miri/soundness risk (a cold pointer aliasing into a combined TLAB block,
combined-vs-fallback allocation lifetimes) and threatening the tree win with a
new indirection. Not a good trade.

## The actual decision

Phase B (72 B) is a **pause-vs-throughput trade**, not a pure win or a pure loss:

- **For:** every GC-pause metric improves; `tree` (the worst gap vs Go, ~6.6×)
  gains +19–29%; `alloc` gains +8–28%; `concurrent` duration is neutral with a
  +22% pause cut.
- **Against:** tiny-object mutator throughput regresses ~11% (churn) / ~19%
  (generational).

Whether to merge is a **product judgement** — lower, more predictable pauses at
some steady-state allocation throughput is a trade many production GCs take on
purpose, but it is the maintainer's call, not an automatic merge.
