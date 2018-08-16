package gorgonnx

import (
	"fmt"

	"github.com/onnx/onnx"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// newValue returns a Gorgonia compatible value from a onnx.ValueInfoProto structure
// By now, it will return a tensor.Tensor
func newValue(valueProto *onnx.ValueInfoProto) (gorgonia.Value, error) {
	// Exctract the tensor for clarity
	t := valueProto.Type.Value.(*onnx.TypeProto_TensorType).TensorType
	// Get the data type
	dt, err := toDtype(t.ElemType)
	if err != nil {
		return nil, err
	}
	// Get the dimensions of the tensor
	size := make([]int, len(t.Shape.Dim))

	for i, dim := range t.Shape.Dim {
		dimValue, ok := dim.Value.(*onnx.TensorShapeProto_Dimension_DimValue)
		if !ok {
			// TODO: implement the TensorShapeProto_Dimension_DimParam type asertion
			return nil, fmt.Errorf("Impossible type asertion, Only onnx.TensorShapeProto_Dimension_DimValue is implemented")
		}
		size[i] = int(dimValue.DimValue)
	}
	return tensor.New(tensor.WithShape(size...), tensor.Of(dt)), nil
}

func toDtype(t *onnx.TensorProto_DataType) (tensor.Dtype, error) {
	switch *t {
	case onnx.TensorProto_UNDEFINED:
		return tensor.Dtype{}, nil
	case onnx.TensorProto_FLOAT:
		// As defined in the spec, a float is a 32 floating-point value
		return tensor.Float32, nil
	case onnx.TensorProto_UINT8:
		return tensor.Uint8, nil
	case onnx.TensorProto_INT8:
		return tensor.Int8, nil
	case onnx.TensorProto_UINT16:
		return tensor.Uint16, nil
	case onnx.TensorProto_INT16:
		return tensor.Int16, nil
	case onnx.TensorProto_INT32:
		return tensor.Int32, nil
	case onnx.TensorProto_INT64:
		return tensor.Int64, nil
	case onnx.TensorProto_STRING:
		return tensor.String, nil
	case onnx.TensorProto_BOOL:
		return tensor.Bool, nil
		// Advanced types
	case onnx.TensorProto_FLOAT16:
		return tensor.Dtype{}, fmt.Errorf("Type not implemented: %v", t)
	case onnx.TensorProto_DOUBLE:
		// TODO see if Float64 can replace a double on all plateforms (or if it needs build tags)
		return tensor.Float64, nil
	case onnx.TensorProto_UINT32:
		return tensor.Uint32, nil
	case onnx.TensorProto_UINT64:
		return tensor.Uint64, nil
	case onnx.TensorProto_COMPLEX64:
		return tensor.Complex64, nil
	case onnx.TensorProto_COMPLEX128:
		return tensor.Complex128, nil
	}
	return tensor.Dtype{}, fmt.Errorf("Unknown input type: %v", t)
}

// add the value v to the graph g and return the added node
func (d *Decoder) add(v *onnx.ValueInfoProto) (*gorgonia.Node, error) {
	val, err := newValue(v)
	if err != nil {
		return nil, err
	}
	n := gorgonia.NodeFromAny(d.g, val, gorgonia.WithName(*v.Name))
	// TODO: check if NAme is empty (according to the SPEC it must be filled but anyway)
	d.db[v.GetName()] = n
	return n, nil
}
