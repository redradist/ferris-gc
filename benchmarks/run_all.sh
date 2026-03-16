#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# FerrisGC vs Go GC — Comparative Benchmarks
#
# Usage:
#   ./run_all.sh                    # default N=100000, tree depth=18
#   ./run_all.sh 500000             # custom N
#   ./run_all.sh 500000 18          # custom N + tree depth
#   ./run_all.sh 100000 18 adaptive # single Rust strategy
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
N="${1:-100000}"
TREE_DEPTH="${2:-18}"
SINGLE_STRATEGY="${3:-}"
RESULTS_DIR="$SCRIPT_DIR/results"

# Strategies to benchmark (or just one if specified)
if [ -n "$SINGLE_STRATEGY" ]; then
    STRATEGIES=("$SINGLE_STRATEGY")
else
    STRATEGIES=("none" "basic" "threshold" "adaptive")
fi

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  FerrisGC vs Go GC — Comparative Benchmarks"
echo "============================================"
echo "  N = $N, Tree Depth = $TREE_DEPTH"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Results: $RESULTS_DIR/"
echo "============================================"
echo ""

# ---------- Build ----------

echo ">>> Building Rust benchmarks (release)..."
(cd "$SCRIPT_DIR/rust" && cargo build --release --quiet 2>&1)
RUST_BIN="$SCRIPT_DIR/rust/target/release"

echo ">>> Building Go benchmarks..."
GO_BIN="$RESULTS_DIR/go_bin"
mkdir -p "$GO_BIN"
for bench in alloc churn tree generational concurrent; do
    (cd "$SCRIPT_DIR/go" && go build -o "$GO_BIN/${bench}_bench" common.go "${bench}_bench.go")
done

echo ">>> Builds complete."
echo ""

# ---------- Run ----------

run_bench() {
    local name="$1"
    local arg="$2"

    echo "=== $name (N=$arg) ==="
    echo ""

    echo "  [Go]"
    "$GO_BIN/${name}_bench" "$arg" | tee "$RESULTS_DIR/${name}_go.json"
    echo ""

    for strat in "${STRATEGIES[@]}"; do
        echo "  [Rust strategy=$strat]"
        "$RUST_BIN/${name}_bench" "$arg" "--strategy=$strat" 2>/dev/null \
            | tee "$RESULTS_DIR/${name}_rust_${strat}.json"
        echo ""
    done
}

run_bench "alloc" "$N"
run_bench "churn" "$N"
run_bench "tree" "$TREE_DEPTH"
run_bench "generational" "$N"
run_bench "concurrent" "$N"

# ---------- Summary ----------

echo ""
echo "============================================"
echo "  Summary"
echo "============================================"

printf "%-14s  %-18s  %10s  %12s  %6s  %11s  %11s\n" \
    "Benchmark" "Lang/Strategy" "Time(ms)" "Throughput" "NumGC" "GCPause(ms)" "MaxPause(ms)"
printf "%-14s  %-18s  %10s  %12s  %6s  %11s  %11s\n" \
    "---------" "-------------" "--------" "----------" "-----" "-----------" "-----------"

for bench in alloc churn tree generational concurrent; do
    # Go result
    file="$RESULTS_DIR/${bench}_go.json"
    if [ -f "$file" ]; then
        dur=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['duration_ms']:.2f}\")" 2>/dev/null || echo "?")
        tp=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['throughput_ops_per_sec']:.0f}\")" 2>/dev/null || echo "?")
        ngc=$(python3 -c "import json; d=json.load(open('$file')); print(d['num_gc'])" 2>/dev/null || echo "?")
        gcp=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['total_gc_pause_ms']:.2f}\")" 2>/dev/null || echo "?")
        mgp=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['max_gc_pause_ms']:.2f}\")" 2>/dev/null || echo "?")
        printf "%-14s  %-18s  %10s  %12s  %6s  %11s  %11s\n" "$bench" "go" "$dur" "$tp" "$ngc" "$gcp" "$mgp"
    fi

    # Rust results per strategy
    for strat in "${STRATEGIES[@]}"; do
        file="$RESULTS_DIR/${bench}_rust_${strat}.json"
        if [ -f "$file" ]; then
            dur=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['duration_ms']:.2f}\")" 2>/dev/null || echo "?")
            tp=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['throughput_ops_per_sec']:.0f}\")" 2>/dev/null || echo "?")
            ngc=$(python3 -c "import json; d=json.load(open('$file')); print(d['num_gc'])" 2>/dev/null || echo "?")
            gcp=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['total_gc_pause_ms']:.2f}\")" 2>/dev/null || echo "?")
            mgp=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['max_gc_pause_ms']:.2f}\")" 2>/dev/null || echo "?")
            printf "%-14s  %-18s  %10s  %12s  %6s  %11s  %11s\n" "$bench" "rust/$strat" "$dur" "$tp" "$ngc" "$gcp" "$mgp"
        fi
    done
done

echo ""
echo "Done. Raw JSON results in: $RESULTS_DIR/"
