// Benchmark: generational stress — many short-lived objects + few long-lived.
// Tests how well the GC handles generational workloads.
//
// Usage: go run common.go generational_bench.go [N]
package main

import (
	"runtime"
	"time"
)

type GenNode struct {
	Value int
}

func main() {
	n := mustGetN()

	runtime.GC()

	// Long-lived objects (survive the whole benchmark)
	longLived := make([]*GenNode, 0, 1000)
	for i := 0; i < 1000; i++ {
		longLived = append(longLived, &GenNode{Value: i})
	}

	start := time.Now()

	// Create waves of short-lived objects
	totalAllocs := 0
	for wave := 0; wave < n/1000; wave++ {
		batch := make([]*GenNode, 0, 1000)
		for i := 0; i < 1000; i++ {
			batch = append(batch, &GenNode{Value: wave*1000 + i})
			totalAllocs++
		}
		// Read to prevent optimization
		sum := 0
		for _, obj := range batch {
			sum += obj.Value
		}
		_ = sum
		// batch goes out of scope — short-lived garbage
	}

	// Verify long-lived objects are still alive
	sum := 0
	for _, obj := range longLived {
		sum += obj.Value
	}
	_ = sum

	elapsed := time.Since(start)
	peakHeap, heapInUse, numGC, totalPause, maxPause := gcStats()

	printResult(BenchResult{
		Benchmark:    "generational",
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
