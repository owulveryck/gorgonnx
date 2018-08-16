package gorgonnx

import (
	"fmt"

	"github.com/onnx/onnx"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// newValue returns a Gorgonia compatible value from a onnx.ValueInfoProto structure
// By now, it will return a tensor.Tensor
func newTensor(t *onnx.TensorProto) (tensor.Tensor, error) {
	// Get the data type
	dt, err := toDtype(t.DataType)
	if err != nil {
		return nil, err
	}
	var size = make([]int, len(t.Dims))
	for i := range t.Dims {
		size[i] = int(t.Dims[i])
	}
	var backing interface{}
	switch dt {
	case tensor.Float32:
		if len(t.FloatData) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		backing = t.FloatData
	case tensor.Float64:
		if len(t.DoubleData) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		backing = t.DoubleData
	case tensor.Int64:
		if len(t.Int64Data) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		backing = t.Int64Data
	case tensor.Int32:
		if len(t.Int32Data) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		backing = t.Int32Data
	default:
		return nil, fmt.Errorf("Backend not yet implemented")

	}

	return tensor.New(tensor.WithShape(size...), tensor.WithBacking(backing), tensor.Of(dt)), nil
}

// add the value v to the graph g and return the added node
func (d *Decoder) addTensor(v *onnx.TensorProto) (*gorgonia.Node, error) {
	val, err := newTensor(v)
	if err != nil {
		return nil, err
	}
	n := gorgonia.NodeFromAny(d.g, val, gorgonia.WithName(*v.Name))
	// TODO: check if NAme is empty (according to the SPEC it must be filled but anyway)
	d.db[v.GetName()] = n
	return n, nil
}
