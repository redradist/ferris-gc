package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"time"
)

// BenchResult holds metrics from a single benchmark run.
type BenchResult struct {
	Benchmark    string        `json:"benchmark"`
	Language     string        `json:"language"`
	Duration     time.Duration `json:"duration_ns"`
	DurationMs   float64       `json:"duration_ms"`
	PeakHeapMB   float64       `json:"peak_heap_mb"`
	HeapInUseMB  float64       `json:"heap_in_use_mb"`
	NumGC        uint32        `json:"num_gc"`
	TotalGCPause time.Duration `json:"total_gc_pause_ns"`
	TotalGCMs    float64       `json:"total_gc_pause_ms"`
	MaxGCPause   time.Duration `json:"max_gc_pause_ns"`
	MaxGCPauseMs float64       `json:"max_gc_pause_ms"`
	Throughput   float64       `json:"throughput_ops_per_sec"`
}

func printResult(r BenchResult) {
	r.DurationMs = float64(r.Duration) / float64(time.Millisecond)
	r.TotalGCMs = float64(r.TotalGCPause) / float64(time.Millisecond)
	r.MaxGCPauseMs = float64(r.MaxGCPause) / float64(time.Millisecond)

	b, _ := json.MarshalIndent(r, "", "  ")
	fmt.Println(string(b))
}

func gcStats() (peakHeapMB, heapInUseMB float64, numGC uint32, totalPause, maxPause time.Duration) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	peakHeapMB = float64(m.HeapSys) / (1024 * 1024)
	heapInUseMB = float64(m.HeapInuse) / (1024 * 1024)
	numGC = m.NumGC

	for i := uint32(0); i < m.NumGC && i < 256; i++ {
		idx := (m.NumGC - 1 - i) % 256
		p := time.Duration(m.PauseNs[idx])
		totalPause += p
		if p > maxPause {
			maxPause = p
		}
	}
	return
}

func mustGetN() int {
	if len(os.Args) < 2 {
		return 100_000
	}
	n := 0
	fmt.Sscanf(os.Args[1], "%d", &n)
	if n <= 0 {
		n = 100_000
	}
	return n
}
