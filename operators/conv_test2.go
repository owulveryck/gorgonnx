package operators

import (
	"testing"

	"github.com/owulveryck/gorgonnx"
	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TestConv ...
func TestConv(t *testing.T) {
	assert := assert.New(t)

	g := gorgonia.NewGraph()
	var op gorgonnx.Operator

	attribute0Name := "kernel_shape"
	attribute0Type := onnx.AttributeProto_AttributeType(7)
	attribute0 := &onnx.AttributeProto{
		Name: &attribute0Name,
		Type: &attribute0Type,
		Ints: []int64{3, 3},
	}
	attribute1Name := "pads"
	attribute1Type := onnx.AttributeProto_AttributeType(7)
	attribute1 := &onnx.AttributeProto{
		Name: &attribute1Name,
		Type: &attribute1Type,
		Ints: []int64{1, 1, 1, 1},
	}
	attribute2Name := "strides"
	attribute2Type := onnx.AttributeProto_AttributeType(7)
	attribute2 := &onnx.AttributeProto{
		Name: &attribute2Name,
		Type: &attribute2Type,
		Ints: []int64{2, 2},
	}
	attributes := []*onnx.AttributeProto{
		attribute0,
		attribute1,
		attribute2,
	}
	x := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 1, 7, 5),
			tensor.WithBacking([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34})))
	W := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 1, 3, 3),
			tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1})))
	yT := tensor.New(
		tensor.WithShape(1, 1, 4, 3),
		tensor.WithBacking([]float32{12, 27, 24, 63, 108, 81, 123, 198, 141, 112, 177, 124}))
	y := new(gorgonia.Node)
	err := op.Apply(
		[]*gorgonia.Node{
			x,
			W,
		},
		[]*gorgonia.Node{
			y,
		},
	)

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		f.Fatal(err)
	}
	assert.Equal(yT.Shape(), y.Shape(), "Tensors should be the same")
	assert.Equal(yT.Data(), y.Value().Data(), "Tensors should be the same")

}
