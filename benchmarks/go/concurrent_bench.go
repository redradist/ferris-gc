// Benchmark: concurrent allocation from multiple goroutines.
// Tests GC performance under contention.
//
// Usage: go run common.go concurrent_bench.go [N]
package main

import (
	"runtime"
	"sync"
	"time"
)

type ConcNode struct {
	Value int
}

func main() {
	n := mustGetN()
	numWorkers := runtime.NumCPU()
	if numWorkers > 8 {
		numWorkers = 8
	}
	perWorker := n / numWorkers

	runtime.GC()

	start := time.Now()

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			objects := make([]*ConcNode, 0, perWorker)
			for i := 0; i < perWorker; i++ {
				objects = append(objects, &ConcNode{Value: workerID*perWorker + i})
			}
			// Read to prevent optimization
			sum := 0
			for _, obj := range objects {
				sum += obj.Value
			}
			_ = sum
		}(w)
	}

	wg.Wait()

	elapsed := time.Since(start)
	totalAllocs := numWorkers * perWorker
	peakHeap, heapInUse, numGC, totalPause, maxPause := gcStats()

	printResult(BenchResult{
		Benchmark:    "concurrent",
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
