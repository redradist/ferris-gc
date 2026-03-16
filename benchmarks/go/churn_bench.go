// Benchmark: alloc/drop churn — constantly create and discard objects.
// Measures GC throughput under sustained allocation pressure.
//
// Usage: go run common.go churn_bench.go [N]
package main

import (
	"runtime"
	"time"
)

type ChurnNode struct {
	Value int
}

func main() {
	n := mustGetN()

	runtime.GC()

	start := time.Now()

	// Keep a window of live objects, constantly replacing oldest
	windowSize := 1000
	window := make([]*ChurnNode, windowSize)
	totalAllocs := 0

	for i := 0; i < n; i++ {
		window[i%windowSize] = &ChurnNode{Value: i}
		totalAllocs++
	}

	// Prevent dead-code elimination
	sum := 0
	for _, obj := range window {
		if obj != nil {
			sum += obj.Value
		}
	}
	_ = sum

	elapsed := time.Since(start)
	peakHeap, heapInUse, numGC, totalPause, maxPause := gcStats()

	printResult(BenchResult{
		Benchmark:    "churn",
		Language:     "go",
		Duration:     elapsed,
		PeakHeapMB:   peakHeap,
		HeapInUseMB:  heapInUse,
		NumGC:        numGC,
		TotalGCPause: totalPause,
		MaxGCPause:   maxPause,
		Throughput:   float64(totalAllocs) / elapsed.Seconds(),
	})
}
