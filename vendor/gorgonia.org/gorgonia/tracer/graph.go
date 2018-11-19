package tracer

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
)

// This structures handles the toplevel graph attributes
type dotGraph struct {
	graph.Directed
}

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g dotGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	// Create a special attribute "rank" to place the input at the same level in the graph

	graphAttributes := attributer{
		/*
			encoding.Attribute{
				Key:   "nodesep",
				Value: "1",
			},
		*/
		encoding.Attribute{
			Key:   "rankdir",
			Value: "TB",
		},
		/*
			encoding.Attribute{
				Key:   "ranksep",
				Value: `"1.5 equally"`,
			},
		*/
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "style",
			Value: "rounded",
		},
		encoding.Attribute{
			Key:   "fontsize",
			Value: "8",
		},
		encoding.Attribute{
			Key:   "fontname",
			Value: "monospace",
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "none",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}
