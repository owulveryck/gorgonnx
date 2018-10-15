package gorgonnx

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func (cg *computationGraph) storeNode(name string, n *gorgonia.Node) error {
	_, ok := cg.db.Load(name)
	if ok {
		return fmt.Errorf("Warning a node named %v already exists", name)
	}
	cg.db.Store(name, n)
	cg.g.AddNode(n)
	return nil
}

func (cg *computationGraph) loadNode(nodeName string) (*gorgonia.Node, error) {
	n, ok := cg.db.Load(nodeName)
	if !ok {
		return nil, fmt.Errorf("Cannot find node %v in the graph", nodeName)
	}
	return n.(*gorgonia.Node), nil
}
