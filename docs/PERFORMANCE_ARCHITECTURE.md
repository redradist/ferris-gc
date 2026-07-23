# FerrisGC Performance Architecture — Why Go/Java/.NET Are Faster, and What To Do

> Research-backed architectural roadmap. Sources: Go runtime source + design proposals
> (17503 hybrid barrier, 44167 pacer) + Rick Hudson ISMM 2018; .NET BOTR GC design +
> Maoni Stephens; HotSpot G1/ZGC/Shenandoah (Shipilev, Schatzl, JEP 439, ZGC TOPLAS).
> FerrisGC numbers are freshly measured on this machine (interleaved A/B, callgrind under
> the production fat-LTO build).

## 1. Where FerrisGC stands (measured)

Same machine, N=100k (tree = 2.1M nodes), FerrisGC with the two merged perf phases:

| Bench | Go | FerrisGC | Go faster |
|---|---|---|---|
| alloc | 3.1 ms | 13.9 ms | ~4.4× |
| churn | 1.5 ms | 3.1 ms | ~2.1× |
| generational | 1.5 ms | 3.4 ms | ~2.2× |
| tree (2.1M) | 88 ms | 583 ms | ~6.6× |
| concurrent | 0.9 ms | 68.7 ms | ~80× |

Memory (tree): Go 52 MB vs FerrisGC 128 MB (~2.5×). GC pause (tree): Go 0.35 ms
(concurrent) vs FerrisGC 380 ms (**stop-the-world**).

### fat-LTO profile of `churn` (production build)

- **`Gc::drop` ≈ 34%** — the RC-hybrid frees each object *immediately* on the mutator's
  critical path (SlotMap remove + card-table + `finalize`/`drop_in_place` vtables + TLAB
  block refcount). **No tracing GC does per-object work on "drop."**
- **Full collections ≈ 10%** — the basic strategy runs a full Gen2 STW every 50 ms even
  with a 1000-object live window.
- **Allocation** — TLAB bump is fast, but each object also pays a SlotMap insert, an
  `ObjectEntry` (behind a pointer → a cache miss on every mark/sweep touch), and a
  `reset_root` cascade.

**Conclusion: micro-optimization is exhausted.** fat-LTO (the one big merged win, +5–16%)
already extracted the low-level performance; two further micro-opts (root-list, TLS) came
back neutral-to-negative under fat-LTO. The remaining gap is **architectural**.

## 2. Why the others are fast — the convergent lessons

Three independently-researched runtimes agree on the same handful of ideas.

### A. Nothing heap-proportional runs stop-the-world
Go has **exactly two STW points**, both O(#threads), never O(heap): pause time is
*decoupled from heap size*. JVM's ZGC/Shenandoah and .NET's background gen2 do the same —
mark and sweep run concurrently; only a bounded root handshake is STW. **FerrisGC does a
full STW mark + sweep** — this is the 380 ms tree pause and the 80× concurrent gap.

### B. A write barrier makes concurrent marking sound *without* a full STW re-mark
This is the single most-cited mechanism.
- **Go**: hybrid Yuasa(deletion)+Dijkstra(insertion) barrier ⇒ stacks scanned **once**,
  never re-scanned. Cut worst-case STW from unbounded to <50 µs.
- **G1 / Shenandoah**: **SATB** (snapshot-at-the-beginning) pre-write barrier logs the
  *overwritten* reference; everything allocated during marking is implicitly live (a
  **TAMS** watermark). Final "remark" only drains the log + rescans roots — **O(roots +
  pending buffer), not O(heap)**.

FerrisGC *has* an incremental collector, but `finish_collection` **still does a full STW
re-mark via `trace()`**, which negates it. The barrier is the fix.

### C. A remembered set is the tax that makes generational collection cheap
.NET and G1 both: a **card table** (byte per ~512–2048 B) marked by a **post-write
barrier** records old→young pointers, so a young collection scans only dirty cards, never
the whole old heap. FerrisGC is generational (Gen0/1/2) and *has* a card table — but the
same barrier site should serve double duty (see §3).

### D. Lock-free thread-local bump allocation with cheap/no per-object metadata
- **Go**: per-P `mcache` over size-class spans; **no per-object header** — size/mark/type
  live in address-keyed side bitmaps, so allocation writes *only the object* and marking
  never dirties the object's cache line.
- **JVM/.NET**: per-thread TLAB / allocation context, ~3–6 inlined instructions; the
  object header (12–16 B) is *reused* for GC bits (mark/age/forwarding) — one cache line
  from the data.

FerrisGC has a TLAB (good) but its hot GC metadata sits in a **side `ObjectEntry` behind a
pointer** — a guaranteed cache miss per object per mark/sweep pass. Worst of both worlds:
a per-object header *and* a cache-missing side table.

### E. Allocation-triggered pacing, not a timer
Go/.NET size the next collection off live-set growth (GOGC/heap-goal) and start
concurrent work *early enough to finish before the goal*, with **mutator assists**
(allocators pay marking debt) as the safety valve. FerrisGC's basic strategy is a **50 ms
timer** — it over-collects idle heaps and under-responds to load spikes.

### F. FerrisGC's unique cost: RC-hybrid immediate free
None of Go/Java/.NET reclaim per-object on drop; they defer everything to bulk concurrent
sweep and never call `free()` per object. FerrisGC's RC-hybrid frees acyclic garbage
eagerly — which is *why its pauses could be short* (the collector only chases cycles) but
also **why `Gc::drop` is 34% of churn**. This is a genuine trade, not a bug; the question
is whether the drop path can be made much cheaper.

## 3. Roadmap for FerrisGC (prioritized by leverage / risk)

### Phase A — One dual-purpose write barrier  ⟵ highest leverage
Add a single pre-write barrier at the `GcCell` / `sync::GcCell` mutation choke point
(FerrisGC already funnels all interior mutation through these — a soundness advantage a
bare language lacks). It does **two** jobs at once:
1. **SATB**: if a mark cycle is active, push the *overwritten* `*const GcPtr<T>` onto a
   thread-local buffer. Combined with a **TAMS watermark per TLAB block** (mark anything
   bump-allocated above `top`-at-mark-start as implicitly live), this lets marking run
   concurrently and reduces the STW step to "drain buffers + rescan roots."
2. **Remembered set**: record old→young edges into the existing card table so Gen0
   collection is genuinely O(young).

Then rewrite `finish_collection` to **stop doing the full STW `trace()` re-mark** — drain
the SATB buffer and rescan roots only.
*Impact:* attacks the 380 ms pause and the 80× concurrent gap directly.
*Risk:* medium-high — soundness hinges on *every* reference store going through the
barrier (verify with Miri + a barrier-coverage audit); floating garbage is acceptable.

### Phase B — Kill the per-object metadata cache miss
Move the hot mark/generation/root bits **inline** (into `GcInfo`, already adjacent to the
object) or into an **address-keyed side bitmap** (Go-style), so mark/sweep passes stop
chasing the `ObjectEntry` pointer. Keep cold fields (layout, tracers) in the side entry.
*Impact:* the ~2.5× memory gap and the per-object cache miss dominating tree_bench passes.
*Risk:* medium — touches `ObjectEntry`/`SlotMap` layout; heavy Miri + test coverage.

### Phase C — Attack the RC-hybrid drop cost (34%)
Make `Gc::drop`'s fast path cheaper: batch/defer the SlotMap-remove + card-table-unregister
work, or skip the tracer-list drain for the thread-local path (local objects have empty
tracer lists). Measure whether deferring acyclic frees to a bulk pass (more Go-like) beats
eager free on churn without regressing pause/memory.
*Impact:* churn / alloc (the 2–4× gaps). *Risk:* medium.

### Phase D — Allocation-triggered pacer + assists
Replace the 50 ms timer with an allocation-volume trigger; fold non-heap roots into the
GOGC goal; if Phase A lands, add mutator assists so heap growth is self-limiting.
*Impact:* idle-heap waste + load-spike responsiveness. *Risk:* low (pure safe bookkeeping).

### Phase E — TLAB tuning
Adaptive per-thread TLAB sizing (EMA of allocation rate) + retire-vs-direct-alloc overflow
policy + block pooling (avoid allocator churn on exhaustion). *Risk:* low.

### Out of scope (for now) — concurrent relocation / compaction
ZGC/Shenandoah-style moving GC (forwarding word + load barrier in `Deref`) is the highest
value for fragmentation but the lowest feasibility in safe Rust — it *is* FerrisGC's
already-disabled `compact()` (relocate cascade + pinning + Stacked-Borrows). Revisit only
after A–E, and start from Shenandoah's forwarding-word model, not ZGC's colored pointers.

## 4. Recommended sequence
**A → B → C → D → E.** Phase A (the one write barrier) is the keystone: it unlocks both
low-pause concurrent marking *and* cheap generational collection from a single hook, and
directly attacks the two worst gaps (tree pause, concurrent). Each phase ships as its own
branch → PR → CI (incl. Miri) → merge, with interleaved-A/B benchmarks before/after.
