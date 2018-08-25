package gorgonnx

import (
	"fmt"

	onnx "github.com/owulveryck/onnx/go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor/tensonnx"
)

// graph is the internal representation of a graph.false
// it handles both structure (onnx and gorgonia) as well
// as a dictionnary of nodes
type graph struct {
	// db reference a Node by its name.
	// This is mandatory as NodeProto references the output node by its name
	db map[string]*gorgonia.Node
	g  *gorgonia.ExprGraph
	gx *onnx.GraphProto
}

func (gi *graph) addNode(name string, n *gorgonia.Node) error {
	gi.db[name] = n
	gi.g.AddNode(n)
	return nil
}

// getNodeByName from the database. Returns nil if not found
func (gi *graph) getNodeByName(name string) *gorgonia.Node {
	if n, ok := gi.db[name]; ok {
		return n
	}
	return nil
}

// NewGraph returns a new graph that is initialized with gx as its initial content.
func NewGraph(gx *onnx.GraphProto) (*gorgonia.ExprGraph, error) {
	g := &graph{
		db: make(map[string]*gorgonia.Node),
		gx: gx,
	}
	return g.parse(gx)
}

// Decode the graphproto and returns a gorgonia Graph
func (gi *graph) parse(gx *onnx.GraphProto) (*gorgonia.ExprGraph, error) {
	g := gorgonia.NewGraph(gorgonia.WithGraphName(gx.GetName()))
	for _, tensorProto := range gx.Initializer {
		name := tensorProto.GetName()
		t, err := tensonnx.NewTensor(tensorProto)
		if err != nil {
			return nil, err
		}
		n := g.AddNode(gorgonia.NewConstant(t, gorgonia.WithName(name)))
		gi.db[name] = n

	}
	// Process the inputs
	for _, valueInfo := range gx.Input {
		// Check if the name is not already present in the graph
		// as it may be an initializer (const)
		name := valueInfo.GetName()
		if _, ok := gi.db[name]; !ok {
			t, err := NewValue(valueInfo)
			if err != nil {
				return nil, err
			}

			// Adding node
			n := gorgonia.NodeFromAny(g, t, gorgonia.WithName(name))
			gi.db[name] = n
		}
	}
	// Process the nodes until the list is empty
	for len(gx.Node) != 0 {
		startingLen := len(gx.Node)
		for i, n := range gx.Node {
			// A node is addable to the graph, if all of its inputs is already in the node db
			isAddable := true
			for _, j := range n.Input {
				if _, ok := gi.db[j]; !ok {
					isAddable = false
					break
				}
			}
			if isAddable {
				err := gi.processNode(n)
				if err != nil {
					return nil, err
				}
				// The node has been processed, remove it from the list
				// https://github.com/golang/go/wiki/SliceTricks
				gx.Node[i] = gx.Node[len(gx.Node)-1]
				gx.Node[len(gx.Node)-1] = nil
				gx.Node = gx.Node[:len(gx.Node)-1]
				break
			}
		}
		if startingLen == len(gx.Node) {
			return g, fmt.Errorf("Endless loop, the graph may be broken")
		}
	}

	return g, nil
}
