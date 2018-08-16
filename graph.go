package gorgonnx

import (
	"github.com/onnx/onnx"
	"gorgonia.org/gorgonia"
)

// Decoder is a receiver
type Decoder struct {
	// db reference a Node by its name
	db map[string]*gorgonia.Node
	g  *gorgonia.ExprGraph
}

// GetGraph returns a Gorgonia graph
func (d *Decoder) GetGraph() *gorgonia.ExprGraph {
	return d.g
}

// NewDecoder returns a new decoder that reads a graph from g
func NewDecoder() *Decoder {
	return &Decoder{
		db: make(map[string]*gorgonia.Node),
	}
}

// Decode the graphproto and returns a gorgonia Graph
func (d *Decoder) Decode(gx *onnx.GraphProto) (*gorgonia.ExprGraph, error) {
	g := gorgonia.NewGraph(gorgonia.WithGraphName(gx.GetName()))
	// Process the inputs
	for _, input := range gx.Input {
		_, err := d.Add(input)
		if err != nil {
			return g, err
		}
	}
	return g, nil
}
