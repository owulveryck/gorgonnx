package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"strconv"

	"github.com/onnx/onnx"
	"gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/simple"
)

func main() {
	b, err := ioutil.ReadFile("mnist/model.onnx")
	if err != nil {
		log.Fatal(err)
	}
	model := new(onnx.ModelProto)
	err = model.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	graph := model.GetGraph()
	//fmt.Println("Graph Name: ", *graph.Name)

	// dict maintains a databse of the nodes
	dict := make(map[string]node, len(graph.Node)+len(graph.Input)+len(graph.Output))

	g := simple.NewDirectedGraph()
	id := 0
	for _, n := range graph.Node {
		fmt.Println(n)
		id++
		dict[*n.Name] = node{
			int64(id),
			"",
			n,
			nil,
		}
		g.AddNode(dict[*n.Name])
	}
	id++
	for _, n := range graph.Input {
		id++
		dict[*n.Name] = node{
			int64(id),
			"",
			nil,
			n,
		}
		g.AddNode(dict[*n.Name])
	}
	id++
	for _, n := range graph.Output {
		id++
		dict[*n.Name] = node{
			int64(id),
			"",
			nil,
			n,
		}
		g.AddNode(dict[*n.Name])
	}
	for _, n := range g.Nodes() {
		n := n.(node)
		if n.ID() == 0 {
			fmt.Println("DEBUG", n)
		}
		switch {
		case n.NodeProto != nil:
			for _, i := range n.NodeProto.Input {
				if dict[i].ID() == 0 {
					// If the node does not exist, add it (it may be an input or output)
					id := g.NewNode().ID()
					dict[i] = node{
						id:   int64(id),
						name: i,
					}
				}
				e := simple.Edge{
					F: dict[i],
					T: n,
				}
				g.SetEdge(e)
			}
			for _, i := range n.NodeProto.Output {
				if dict[i].ID() == 0 {
					// If the node does not exist, add it (it may be an input or output)
					id := g.NewNode().ID()
					dict[i] = node{
						id:   int64(id),
						name: i,
					}
				}
				e := simple.Edge{
					T: dict[i],
					F: n,
				}
				g.SetEdge(e)
			}
		default:
		}
	}
	b, err = dot.Marshal(g, *graph.Name, " ", " ", false)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))
}

type node struct {
	id             int64
	name           string
	NodeProto      *onnx.NodeProto
	ValueInfoProto *onnx.ValueInfoProto
}

func (n node) ID() int64 {
	return n.id
}

func (n node) DOTID() string {
	switch {
	case n.NodeProto != nil:
		return fmt.Sprintf(`"%v_%v"`, n.ID(), *n.NodeProto.Name)
	case n.ValueInfoProto != nil:
		return fmt.Sprintf(`"%v_%v"`, n.ID(), *n.ValueInfoProto.Name)
	default:
		return fmt.Sprintf(`"%v_%v"`, strconv.Itoa(int(n.ID())), n.name)
	}
}
