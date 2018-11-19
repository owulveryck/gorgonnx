package tracer

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
)

type constantSubGraph struct {
	id           int64
	name         string
	subGraphType int
	graph.Directed
}

func (g constantSubGraph) DOTID() string { return g.name }

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g constantSubGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	// Create a special attribute "rank" to place the input at the same level in the graph

	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "rank",
			Value: `"max"`,
		},
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "style",
			Value: `"rounded,filled"`,
		},
		encoding.Attribute{
			Key:   "fontname",
			Value: "monospace",
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "record",
		},
		encoding.Attribute{
			Key:   "fillcolor",
			Value: "blue",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}