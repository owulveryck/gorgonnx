package gorgonnx

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func toIntSlice(d []int64) []int {
	output := make([]int, len(d))
	for i, v := range d {
		output[i] = int(v)
	}
	return output
}

func (cg *computationGraph) getNode(nodeName string) (*gorgonia.Node, error) {
	n, ok := cg.db.Load(nodeName)
	if !ok {
		return nil, fmt.Errorf("Cannot node %v in the graph", nodeName)
	}
	return n.(*gorgonia.Node), nil
}
