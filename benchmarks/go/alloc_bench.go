// Benchmark: mass allocation of N objects, measure time, memory, GC pauses.
//
// Usage: go run common.go alloc_bench.go [N]
package main

import (
	"runtime"
	"time"
)

type Node struct {
	Value int
	Next  *Node
}

func main() {
	n := mustGetN()

	// Force GC to get a clean baseline
	runtime.GC()

	start := time.Now()

	objects := make([]*Node, 0, n)
	for i := 0; i < n; i++ {
		objects = append(objects, &Node{Value: i})
	}

	// Prevent dead-code elimination
	sum := 0
	for _, obj := range objects {
		sum += obj.Value
	}
	_ = sum

	elapsed := time.Since(start)
	peakHeap, heapInUse, numGC, totalPause, maxPause := gcStats()

	printResult(BenchResult{
		Benchmark:    "alloc",
		Language:     "go",
		Duration:     elapsed,
		PeakHeapMB:   peakHeap,
		HeapInUseMB:  heapInUse,
		NumGC:        numGC,
		TotalGCPause: totalPause,
		MaxGCPause:   maxPause,
		Throughput:   float64(n) / elapsed.Seconds(),
	})
}
