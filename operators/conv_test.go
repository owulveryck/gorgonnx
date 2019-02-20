package operators

import (
	"os"
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestConv_with_strides_padding_len_2(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Conv{}

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
		Ints: []int64{1, 1},
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

	if len(attributes) != 0 {
		err := op.Init(attributes)
		t.Logf("Info: operator %#v", op)
		if err != nil {
			_, ok := err.(*onnx.ErrNotImplemented)
			if ok && skip {
				t.Skip(err)
			}

			t.Fatal(err)
		}
	}

	x := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 1, 7, 5),
			tensor.WithBacking([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34})),
		gorgonia.WithName("x"))

	W := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 1, 3, 3),
			tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1})),
		gorgonia.WithName("W"))

	yT := tensor.New(
		tensor.WithShape(1, 1, 4, 3),
		tensor.WithBacking([]float32{12, 27, 24, 63, 108, 81, 123, 198, 141, 112, 177, 124}))
	y := new(gorgonia.Node)

	o, err := op.Apply(
		x, W,
	)
	if err != nil {
		_, ok := err.(*onnx.ErrNotImplemented)
		if ok && skip {
			t.Skip(err)
		}
		_, ok = err.(*gorgonia.ErrNotImplemented)
		if ok && skip {
			t.Skip(err)
		}

		t.Fatal(err)
	}

	y = o[0]

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	assert.Equal(yT.Shape(), y.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(yT.Data(), y.Value().Data(), 1e-5, "Tensors should be the same")

}

func TestConv_with_strides_auto_pad(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Conv{}

	attribute0Name := "kernel_shape"
	attribute0Type := onnx.AttributeProto_AttributeType(7)

	attribute0 := &onnx.AttributeProto{
		Name: &attribute0Name,
		Type: &attribute0Type,
		Ints: []int64{3, 3},
	}

	attribute1Name := "auto_pad"
	attribute1Type := onnx.AttributeProto_AttributeType(3)

	attribute1 := &onnx.AttributeProto{
		Name: &attribute1Name,
		Type: &attribute1Type,
		S:    []byte("SAME_UPPER"),
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

	if len(attributes) != 0 {
		err := op.Init(attributes)
		t.Logf("Info: operator %#v", op)
		if err != nil {
			_, ok := err.(*onnx.ErrNotImplemented)
			if ok && skip {
				t.Skip(err)
			}

			t.Fatal(err)
		}
	}

	x := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 1, 7, 5),
			tensor.WithBacking([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34})),
		gorgonia.WithName("x"))

	W := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 1, 3, 3),
			tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1})),
		gorgonia.WithName("W"))

	yT := tensor.New(
		tensor.WithShape(1, 1, 7, 5),
		tensor.WithBacking([]float32{0, 0, 0, 0, 0, 0, 1, 6, 7, 0, 0, 33, 63, 51, 0, 0, 93, 153, 111, 0, 0, 153, 243, 171, 0, 0, 61, 96, 67, 0, 0, 0, 0, 0, 0}))
	y := new(gorgonia.Node)

	o, err := op.Apply(
		x, W,
	)
	if err != nil {
		_, ok := err.(*onnx.ErrNotImplemented)
		if ok && skip {
			t.Skip(err)
		}
		_, ok = err.(*gorgonia.ErrNotImplemented)
		if ok && skip {
			t.Skip(err)
		}

		t.Fatal(err)
	}

	y = o[0]

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	assert.Equal(yT.Shape(), y.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(yT.Data(), y.Value().Data(), 1e-5, "Tensors should be the same")

}
