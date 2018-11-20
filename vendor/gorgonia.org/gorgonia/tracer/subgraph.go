package tracer

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/simple"
	"gorgonia.org/gorgonia"
)

const (
	exprGraphType = iota
	inputType
	constantType
)

/*
type subGraph struct {
	id           int64
	name         string
	subGraphType int
	graph.Directed
}

func (g subGraph) ID() int64 { return g.id }
func (g subGraph) Subgraph() graph.Graph {
	switch g.subGraphType {
	case constantType:
		return constantSubGraph(g)
	case inputType:
		return inputSubGraph(g)
	default:
		return exprSubGraph(g)
	}
}
func (g subGraph) DOTID() string { return g.name }
*/

type attributer []encoding.Attribute

func (a attributer) Attributes() []encoding.Attribute { return a }

func generateDotGraph(g *gorgonia.ExprGraph) graph.Graph {
	dg := simple.NewDirectedGraph()
	inputsG := simple.NewDirectedGraph()
	constantG := simple.NewDirectedGraph()
	exprG := simple.NewDirectedGraph()
	graph.Copy(dg, g)
	for _, n := range g.AllNodes() {
		currentCluster := n.NodeCluster()
		switch currentCluster {
		case gorgonia.ConstantCluster:
			constantG.AddNode(n)
		case gorgonia.InputCluster:
			inputsG.AddNode(n)
		default:
			exprG.AddNode(n)
		}
	}
	constantSubG := constantSubGraph{
		name:     "Constants",
		Directed: constantG,
	}
	inputsSubG := inputSubGraph{
		name:     "Inputs",
		Directed: inputsG,
	}
	exprSubG := exprSubGraph{
		name:     "ExprGraph",
		Directed: exprG,
	}
	return dotGraph{
		Directed: dg,
		subs: []dot.Graph{
			inputsSubG,
			constantSubG,
			exprSubG,
		},
	}
}
