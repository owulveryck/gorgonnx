package gorgonnx

import (
	"fmt"

	onnx "github.com/owulveryck/onnx-go"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/simple"
)

type nodeType int64

const (
	nodeConstant = iota
	nodeInput
	nodeComputed
	nodeOperator
)

type node struct {
	id        int64
	Name      string
	Shape     []int64
	Operation string
	Type      nodeType
}

// ID of the node
func (n *node) ID() int64 {
	return n.id
}

// DOTID for graphbviz
func (n *node) DOTID() string {
	return n.Name
}

// Attributes defines graph.Node or graph.Edge values that can specify graph attributes.
func (n *node) Attributes() []encoding.Attribute {
	if n.Type == nodeOperator {
		return []encoding.Attribute{
			encoding.Attribute{
				Key:   "shape",
				Value: "record",
			},
			encoding.Attribute{
				Key:   "label",
				Value: fmt.Sprintf(`"{ %v | %v }"`, n.Operation, n.Name),
			},
		}
	}
	return []encoding.Attribute{
		encoding.Attribute{
			Key:   "style",
			Value: "rounded",
		},
	}
}

// graph is the internal representation of a graph.
// it handles both structure (onnx and gorgonia) as well
// as a dictionnary of nodes
type exprgraph struct {
	// db reference a Node by its name.
	// This is mandatory as NodeProto references the output node by its name
	db      map[string]*node
	digraph *simple.DirectedGraph
}

// NewGraph returns a new graph that is initialized with gx as its initial content.
func NewGraph(gx *onnx.GraphProto) (graph.Graph, error) {
	g := &exprgraph{
		db:      make(map[string]*node),
		digraph: simple.NewDirectedGraph(),
	}
	err := g.parse(gx)
	if err != nil {
		return nil, err
	}
	return g.digraph, nil
}

// Decode the graphproto and returns a gorgonia Graph
func (g *exprgraph) parse(gx *onnx.GraphProto) error {
	for _, tensorProto := range gx.Initializer {
		name := tensorProto.GetName()
		n := &node{
			id:    g.digraph.NewNode().ID(),
			Name:  name,
			Shape: tensorProto.Dims,
			Type:  nodeConstant,
		}
		g.digraph.AddNode(n)
		g.db[name] = n

	}
	// Process the inputs
	for _, valueInfo := range gx.Input {
		// Check if the name is not already present in the graph
		// as it may be an initializer (const)
		name := valueInfo.GetName()
		if _, ok := g.db[name]; !ok {
			// Exctract the tensor for clarity
			t := valueInfo.Type.Value.(*onnx.TypeProto_TensorType).TensorType
			// Get the dimensions of the tensor
			size := make([]int64, len(t.Shape.Dim))
			for i, dim := range t.Shape.Dim {
				dimValue, ok := dim.Value.(*onnx.TensorShapeProto_Dimension_DimValue)
				if !ok {
					// TODO: implement the TensorShapeProto_Dimension_DimParam type asertion
					return fmt.Errorf("Impossible type asertion, Only onnx.TensorShapeProto_Dimension_DimValue is implemented")
				}
				size[i] = int64(dimValue.DimValue)
			}

			n := &node{
				id:    g.digraph.NewNode().ID(),
				Name:  name,
				Shape: size,
				Type:  nodeInput,
			}
			g.digraph.AddNode(n)
			g.db[name] = n
		}
	}
	// Process the nodes until the list is empty
	for len(gx.Node) != 0 {
		startingLen := len(gx.Node)
		for i, n := range gx.Node {
			// A node is addable to the graph, if all of its inputs is already in the node db
			isAddable := true
			for _, j := range n.Input {
				if _, ok := g.db[j]; !ok {
					isAddable = false
					break
				}
			}
			if isAddable {
				err := g.processNode(n)
				if err != nil {
					return err
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
			return fmt.Errorf("Endless loop, the graph may be broken")
		}
	}
	return nil
}

func (g *exprgraph) processNode(nx *onnx.NodeProto) error {
	// Get the node from the database
	if len(nx.Output) != 1 {
		return fmt.Errorf("Operations with a single output node is supported")
	}
	n := &node{
		id:        g.digraph.NewNode().ID(),
		Name:      nx.Output[0],
		Type:      nodeOperator,
		Operation: nx.GetOpType(),
	}
	g.digraph.AddNode(n)
	g.db[n.Name] = n
	for _, i := range nx.Input {
		ni := g.db[i]
		// Now add edge
		g.digraph.SetEdge(g.digraph.NewEdge(ni, n))
	}
	return nil
}
