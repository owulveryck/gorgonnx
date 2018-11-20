package tracer

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/simple"
	"gorgonia.org/gorgonia/debugger"
)

const (
	exprGraphType = iota
	inputType
	constantType
)

type attributer []encoding.Attribute

func (a attributer) Attributes() []encoding.Attribute { return a }

func generateDotGraph(g graph.Directed) graph.Graph {
	dg := simple.NewDirectedGraph()
	inputsG := simple.NewDirectedGraph()
	constantG := simple.NewDirectedGraph()
	exprG := simple.NewDirectedGraph()
	graph.Copy(dg, g)
	nodes := g.Nodes()
	for nodes.Next() {
		n := nodes.Node()
		if _, ok := n.(debugger.Grouper); ok {
			group := n.(debugger.Grouper).Group()
			switch group {
			case debugger.ConstantCluster:
				constantG.AddNode(n)
			case debugger.InputCluster:
				inputsG.AddNode(n)
			default:
				exprG.AddNode(n)
			}
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
