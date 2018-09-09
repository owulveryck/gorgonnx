package gorgonnx

import (
	"testing"

	onnx "github.com/owulveryck/onnx-go"
)

func TestNewValue(t *testing.T) {
	valueName := "MyInput"
	dataType := onnx.TensorProto_FLOAT
	testValue := &onnx.ValueInfoProto{
		Name: &valueName,
		Type: &onnx.TypeProto{
			Value: &onnx.TypeProto_TensorType{
				TensorType: &onnx.TypeProto_Tensor{
					ElemType: &dataType,
					Shape: &onnx.TensorShapeProto{
						Dim: []*onnx.TensorShapeProto_Dimension{
							&onnx.TensorShapeProto_Dimension{
								Value:                &onnx.TensorShapeProto_Dimension_DimValue{DimValue: 1},
								Denotation:           (*string)(nil),
								XXX_NoUnkeyedLiteral: struct{}{},
								XXX_unrecognized:     nil,
								XXX_sizecache:        0,
							},
							&onnx.TensorShapeProto_Dimension{
								Value:                &onnx.TensorShapeProto_Dimension_DimValue{DimValue: 2},
								Denotation:           (*string)(nil),
								XXX_NoUnkeyedLiteral: struct{}{},
								XXX_unrecognized:     nil,
								XXX_sizecache:        0,
							},
							&onnx.TensorShapeProto_Dimension{
								Value:                &onnx.TensorShapeProto_Dimension_DimValue{DimValue: 3},
								Denotation:           (*string)(nil),
								XXX_NoUnkeyedLiteral: struct{}{},
								XXX_unrecognized:     nil,
								XXX_sizecache:        0,
							},
							&onnx.TensorShapeProto_Dimension{
								Value:                &onnx.TensorShapeProto_Dimension_DimValue{DimValue: 4},
								Denotation:           (*string)(nil),
								XXX_NoUnkeyedLiteral: struct{}{},
								XXX_unrecognized:     nil,
								XXX_sizecache:        0,
							},
						},
						XXX_NoUnkeyedLiteral: struct{}{},
						XXX_unrecognized:     nil,
						XXX_sizecache:        0,
					},
					XXX_NoUnkeyedLiteral: struct{}{},
					XXX_unrecognized:     nil,
					XXX_sizecache:        0,
				},
			},
			Denotation:           (*string)(nil),
			XXX_NoUnkeyedLiteral: struct{}{},
			XXX_unrecognized:     nil,
			XXX_sizecache:        0,
		},
		DocString:            (*string)(nil),
		XXX_NoUnkeyedLiteral: struct{}{},
		XXX_unrecognized:     nil,
		XXX_sizecache:        0,
	}
	_, err := NewValue(testValue)
	if err != nil {
		t.Fatal(err)
	}
}
