package tracer

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/simple"
	"gorgonia.org/gorgonia"
)

const (
	exprGraphType = iota
	inputType
	constantType
)

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

type attributer []encoding.Attribute

func (a attributer) Attributes() []encoding.Attribute { return a }

func generateDotGraph(g *gorgonia.ExprGraph) graph.Graph {
	inputsG := simple.NewDirectedGraph()
	constantG := simple.NewDirectedGraph()
	exprG := simple.NewDirectedGraph()
	for _, n := range g.AllNodes() {
		switch n.NodeCluster() {
		case gorgonia.ConstantCluster:
			constantG.AddNode(n)
		case gorgonia.InputCluster:
			inputsG.AddNode(n)
		default:
			exprG.AddNode(n)

		}
	}
	dg := simple.NewDirectedGraph()
	constantSubG := subGraph{
		id:           dg.NewNode().ID(),
		name:         "Constants",
		subGraphType: constantType,
		Directed:     constantG,
	}
	dg.AddNode(constantSubG)
	inputsSubG := subGraph{
		id:           dg.NewNode().ID(),
		name:         "Inputs",
		subGraphType: inputType,
		Directed:     inputsG,
	}
	dg.AddNode(inputsSubG)
	exprSubG := subGraph{
		id:           dg.NewNode().ID(),
		name:         "ExprGraph",
		subGraphType: exprGraphType,
		Directed:     exprG,
	}
	dg.AddNode(exprSubG)
	nodes := g.Nodes()
	nodes.Reset()
	for nodes.Next() {
		u := nodes.Node()
		to := g.From(u.ID())
		for to.Next() {
			v := to.Node()
			exprG.SetEdge(exprG.NewEdge(u, v))
		}
	}
	return dotGraph{dg}
}
