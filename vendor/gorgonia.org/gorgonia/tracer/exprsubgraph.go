package tracer

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
)

type exprSubGraph struct {
	name string
	graph.Directed
}

func (g exprSubGraph) DOTID() string { return "cluster_" + g.name }

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g exprSubGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	// Create a special attribute "rank" to place the input at the same level in the graph

	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "label",
			Value: g.name,
		},
		encoding.Attribute{
			Key:   "color",
			Value: "lightgray",
		},
		encoding.Attribute{
			Key:   "style",
			Value: "filled",
		},
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "style",
			Value: `"rounded,filled"`,
		},
		encoding.Attribute{
			Key:   "fillcolor",
			Value: "white",
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "Mrecord",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}
