package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/gorgonnx/onnx"
	"gorgonia.org/gorgonia"
)

var operators map[string]func(n *onnx.NodeProto) error

func init() {
	operators = make(map[string]func(n *onnx.NodeProto) error, 0)

}

// Decoder is a receiver
type Decoder struct {
	// db reference a Node by its name
	db map[string]*gorgonia.Node
	g  *gorgonia.ExprGraph
}

// NewDecoder returns a new decoder that reads a graph from g
func NewDecoder() *Decoder {
	d := &Decoder{
		db: make(map[string]*gorgonia.Node),
	}
	operators["Conv"] = d.convOp
	operators["Reshape"] = d.reshapeOp
	operators["Add"] = d.addOp
	operators["Relu"] = d.reluOp
	operators["MaxPool"] = d.maxPoolOp
	operators["MatMul"] = d.matMulOp

	return d
}

// Decode the graphproto and returns a gorgonia Graph
func (d *Decoder) Decode(gx *onnx.GraphProto) (*gorgonia.ExprGraph, error) {
	g := gorgonia.NewGraph(gorgonia.WithGraphName(gx.GetName()))
	d.g = g
	// Process the inputs
	for _, input := range gx.Input {
		_, err := d.add(input)
		if err != nil {
			return g, err
		}
	}
	for _, initializer := range gx.Initializer {
		_, err := d.addTensor(initializer)
		if err != nil {
			return g, err
		}

	}
	// Process the nodes until the list is empty
	for len(gx.Node) != 0 {
		startingLen := len(gx.Node)
		for i, n := range gx.Node {
			// A node is addable to the graph, if all of its inputs is already in the node db
			isAddable := true
			for _, j := range n.Input {
				_, ok := d.db[j]
				if !ok {
					isAddable = false
					break
				}
			}
			if isAddable {
				err := d.processNode(n)
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
			for i, n := range gx.Node {
				fmt.Printf("%v => %v\n", i, *n.Name)
			}
			return g, fmt.Errorf("Endless loop, the graph may be broken")
		}
	}
	return g, nil
}
