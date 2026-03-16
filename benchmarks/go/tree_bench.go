// Benchmark: build a binary tree of depth D, then drop and collect.
// Stresses the GC with pointer-heavy graph structures.
//
// Usage: go run common.go tree_bench.go [depth]
package main

import (
	"fmt"
	"os"
	"runtime"
	"time"
)

type TreeNode struct {
	Value int
	Left  *TreeNode
	Right *TreeNode
}

func buildTree(depth, value int) *TreeNode {
	if depth == 0 {
		return &TreeNode{Value: value}
	}
	return &TreeNode{
		Value: value,
		Left:  buildTree(depth-1, 2*value),
		Right: buildTree(depth-1, 2*value+1),
	}
}

func countNodes(node *TreeNode) int {
	if node == nil {
		return 0
	}
	return 1 + countNodes(node.Left) + countNodes(node.Right)
}

func sumTree(node *TreeNode) int64 {
	if node == nil {
		return 0
	}
	return int64(node.Value) + sumTree(node.Left) + sumTree(node.Right)
}

func main() {
	depth := 20
	if len(os.Args) >= 2 {
		fmt.Sscanf(os.Args[1], "%d", &depth)
	}

	runtime.GC()

	// Phase 1: Build tree
	start := time.Now()
	root := buildTree(depth, 1)
	buildTime := time.Since(start)

	nodeCount := countNodes(root)
	treeSum := sumTree(root)
	_ = treeSum

	// Phase 2: Drop tree and force GC
	root = nil
	gcStart := time.Now()
	runtime.GC()
	gcTime := time.Since(gcStart)

	totalTime := buildTime + gcTime
	peakHeap, heapInUse, numGC, totalPause, maxPause := gcStats()

	fmt.Printf("Tree depth=%d, nodes=%d\n", depth, nodeCount)
	fmt.Printf("Build time: %.2f ms\n", float64(buildTime)/float64(time.Millisecond))
	fmt.Printf("GC time:    %.2f ms\n", float64(gcTime)/float64(time.Millisecond))

	printResult(BenchResult{
		Benchmark:    "tree",
		Language:     "go",
		Duration:     totalTime,
		PeakHeapMB:   peakHeap,
		HeapInUseMB:  heapInUse,
		NumGC:        numGC,
		TotalGCPause: totalPause,
		MaxGCPause:   maxPause,
		Throughput:   float64(nodeCount) / totalTime.Seconds(),
	})
}
