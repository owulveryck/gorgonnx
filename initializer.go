package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/gorgonnx/onnx"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// newValue returns a Gorgonia compatible value from a onnx.ValueInfoProto structure
// By now, it will return a tensor.Tensor
func newTensor(tx *onnx.TensorProto) (tensor.Tensor, error) {
	// Get the data type
	dt, err := toDtype(tx.DataType)
	if err != nil {
		return nil, err
	}
	var size = make([]int, len(tx.Dims))
	for i := range tx.Dims {
		size[i] = int(tx.Dims[i])
	}
	var backing interface{}
	switch dt {
	case tensor.Float32:
		if len(tx.FloatData) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		backing = tx.FloatData
	case tensor.Float64:
		if len(tx.DoubleData) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		backing = tx.DoubleData
	case tensor.Int64:
		if len(tx.Int64Data) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		backing = tx.Int64Data
	case tensor.Int32:
		if len(tx.Int32Data) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		backing = tx.Int32Data
	default:
		return nil, fmt.Errorf("Backend not yet implemented")

	}

	return tensor.New(tensor.WithShape(size...), tensor.WithBacking(backing), tensor.Of(dt)), nil
}

// add the value v to the graph g and return the added node
func (d *Decoder) addTensor(tx *onnx.TensorProto) (*gorgonia.Node, error) {
	t, err := newTensor(tx)
	if err != nil {
		return nil, err
	}
	n := gorgonia.NodeFromAny(d.g, t, gorgonia.WithName(*tx.Name))
	// TODO: check if NAme is empty (according to the SPEC it must be filled but anyway)
	d.db[tx.GetName()] = n
	return n, nil
}
