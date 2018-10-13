package gorgonnx

import (
	"fmt"
	"sync"

	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// graph is the internal representation of a graph.false
// it handles both structure (onnx and gorgonia) as well
// as a dictionnary of nodes
type computationGraph struct {
	// db reference a Node by its name.
	// This is mandatory as NodeProto references the output node by its name
	db     sync.Map
	g      *gorgonia.ExprGraph
	gx     *onnx.GraphProto
	inputs []string
}

// GetOutputNodes returns the nodes that are the output of the graph
func GetOutputNodes(g *gorgonia.ExprGraph) gorgonia.Nodes {
	var output gorgonia.Nodes
	for _, n := range g.AllNodes() {
		if g.To(n.ID()).Len() == 0 {
			output = append(output, n)
		}
	}
	return output
}

// GetOutputGraphNodes returns the nodes that are the output of the graph and not orhpan Nodes
// This is avoid to return the nodes used for the reshape operator
func GetOutputGraphNodes(g *gorgonia.ExprGraph) gorgonia.Nodes {
	var output gorgonia.Nodes
	for _, n := range GetOutputNodes(g) {
		if g.From(n.ID()).Len() != 0 {
			output = append(output, n)
		}
	}
	return output
}

// NewGraph returns a new graph that is initialized with gx as its initial content.
func NewGraph(gx *onnx.GraphProto) (*gorgonia.ExprGraph, error) {
	g := &computationGraph{
		gx: gx,
	}
	return g.parse(gx)
}

// Decode the graphproto and returns a gorgonia Graph
func (cg *computationGraph) parse(gx *onnx.GraphProto) (*gorgonia.ExprGraph, error) {
	g := gorgonia.NewGraph(gorgonia.WithGraphName(gx.GetName()))
	cg.g = g
	for _, tensorProto := range gx.Initializer {
		name := tensorProto.GetName()
		t, err := tensorProto.Tensor()
		if err != nil {
			return nil, err
		}
		n := g.AddNode(gorgonia.NewConstant(t, gorgonia.WithName(name)))
		cg.storeNode(name, n)

	}
	// Process the inputs
	for _, valueInfo := range gx.Input {
		// Check if the name is not already present in the graph
		// as it may be an initializer (const)
		name := valueInfo.GetName()

		if _, err := cg.loadNode(name); err != nil {
			t, err := NewValue(valueInfo)
			if err != nil {
				return nil, err
			}

			// Adding node
			n := gorgonia.NodeFromAny(g, t, gorgonia.WithName(name))
			cg.inputs = append(cg.inputs, name)
			cg.storeNode(name, n)
		}
	}
	// Process the nodes until the list is empty
	for len(gx.Node) != 0 {
		startingLen := len(gx.Node)
		for i, n := range gx.Node {
			// A node is addable to the graph, if all of its inputs is already in the node db
			isAddable := true
			for _, j := range n.Input {
				if _, ok := cg.db.Load(j); !ok {
					isAddable = false
					break
				}
			}
			if isAddable {
				err := cg.processNode(n)
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
