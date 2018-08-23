package gorgonnx

import (
	"github.com/owulveryck/gorgonnx/onnx"
	"gorgonia.org/gorgonia"
)

// add the value v to the graph g and return the added node
func (d *Decoder) addTensor(tx *onnx.TensorProto, constant bool) (*gorgonia.Node, error) {
	t, err := NewTensor(tx)
	if err != nil {
		return nil, err
	}
	var n *gorgonia.Node
	if constant {
		n = gorgonia.NewConstant(t, gorgonia.WithName(*tx.Name))
		n = d.g.AddNode(n)
	} else {

		n = gorgonia.NodeFromAny(d.g, t, gorgonia.WithName(*tx.Name))
	}
	// TODO: check if NAme is empty (according to the SPEC it must be filled but anyway)
	d.db[tx.GetName()] = n
	return n, nil
}
