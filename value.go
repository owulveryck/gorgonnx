package gorgonnx

import (
	"fmt"

	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// NewValue returns a Gorgonia compatible value from a onnx.ValueInfoProto structure
// By now, it will return a tensor.Tensor
func NewValue(valueProto *onnx.ValueInfoProto) (gorgonia.Value, error) {
	// Exctract the tensor for clarity
	t := valueProto.Type.Value.(*onnx.TypeProto_TensorType).TensorType
	// Get the data type
	dt, err := t.ElemType.Dtype()
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
