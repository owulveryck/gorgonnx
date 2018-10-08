package gorgonnx

import (
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/tensonnx"
)

// This example is taken from https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
func TestConvOp(t *testing.T) {
	dataType := onnx.TensorProto_DataType(1)
	inputName := "x"
	kernelName := "w"
	input := &onnx.TensorProto{
		Dims:     []int64{1, 1, 5, 5},
		DataType: &dataType,
		Segment:  (*onnx.TensorProto_Segment)(nil),
		FloatData: []float32{
			0, 1, 2, 3, 4,
			5, 6, 7, 8, 9,
			10, 11, 12, 13, 14,
			15, 16, 17, 18, 19,
			20, 21, 22, 23, 24},
		Name: &inputName,
	}
	kernel := &onnx.TensorProto{
		Dims:     []int64{1, 1, 3, 3},
		DataType: &dataType,
		Segment:  (*onnx.TensorProto_Segment)(nil),
		FloatData: []float32{
			1, 1, 1,
			1, 1, 1,
			1, 1, 1},
		Name: &kernelName,
	}

	resultWithpadding := tensor.New(tensor.WithShape(1, 1, 5, 5), tensor.WithBacking([]float32{
		12, 21, 27, 33, 24,
		33, 54, 63, 72, 51,
		63, 99, 108, 117, 81,
		93, 144, 153, 162, 111,
		72, 111, 117, 123, 84}))
	/*
		resultWithoutpadding := tensor.New(tensor.WithShape(1, 1, 3, 3), tensor.WithBacking([]float32{
			54, 63, 72,
			99, 108, 117,
			144, 153, 162}))
	*/
	// First test with padding
	output := "y"
	outputs := []string{output}
	inputs := []string{inputName, kernelName}
	opName := "Conv2D"
	opType := "Conv"
	domain := ""
	docString := ""
	attrKernelShapeName := "kernel_shape"
	attrTypeInts := onnx.AttributeProto_AttributeType(7)
	attrTypeString := onnx.AttributeProto_AttributeType(3)
	attrStridesName := "strides"
	//attrGroupName := "group"
	//attrDilationsName := "dilations"
	kernelShape := &onnx.AttributeProto{
		Name: &attrKernelShapeName,
		Type: &attrTypeInts,
		Ints: []int64{3, 3},
	}
	strides := &onnx.AttributeProto{
		Name: &attrStridesName,
		Type: &attrTypeInts,
		Ints: []int64{1, 1},
	}
	/*
		attrPadsName := "pads"
		pad := &onnx.AttributeProto{
			Name: &attrPadsName,
			Type: &attrTypeInts,
			Ints: []int64{1, 1, 1, 1},
		}
	*/
	attrPadsName := "auto_pad"
	pad := &onnx.AttributeProto{
		Name: &attrPadsName,
		Type: &attrTypeString,
		S:    []byte(`SAME_UPPER`),
	}

	np := &onnx.NodeProto{
		Input:  inputs,
		Output: outputs,
		Name:   &opName,
		OpType: &opType,
		Domain: &domain,
		Attribute: []*onnx.AttributeProto{
			strides,
			kernelShape,
			pad,
		},
		DocString: &docString,
	}
	inputT, err := tensonnx.NewTensor(input)
	if err != nil {
		t.Fatal(err)
	}
	kernelT, err := tensonnx.NewTensor(kernel)
	if err != nil {
		t.Fatal(err)
	}
	g := computationGraph{
		db: make(map[string]*gorgonia.Node, 3),
		g:  gorgonia.NewGraph(),
	}
	inputN := gorgonia.NodeFromAny(g.g, inputT, gorgonia.WithName(inputName))
	kernelN := gorgonia.NodeFromAny(g.g, kernelT, gorgonia.WithName(kernelName))
	g.addNode(inputName, inputN)
	g.addNode(kernelName, kernelN)
	err = g.processNode(np)
	if err != nil {
		t.Fatal(err)
	}
	vm := gorgonia.NewTapeMachine(g.g)
	//vm := gorgonia.NewTapeMachine(g.g)
	err = vm.RunAll()
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, resultWithpadding.Data(), g.getNodeByName(output).Value().(tensor.Tensor).Data(), "Bad result for the convolution operator")
}
