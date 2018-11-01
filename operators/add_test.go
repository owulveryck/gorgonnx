package operators

import (
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TestAdd_BCHW test the broadcasting for the mnist example with between a 4D and a 3D tensor
func TestAdd_BCHW(t *testing.T) {
	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Add{}

	attributes := []*onnx.AttributeProto{}

	if len(attributes) != 0 {
		err := op.Init(attributes)
		t.Logf("Info: operator %#v", op)
		if err != nil {
			t.Fatal(err)
		}
	}

	x := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 2, 3, 3),
			tensor.WithBacking([]float32{
				0, 1, 2,
				3, 4, 5,
				5, 6, 7,
				8, 9, 10,
				11, 12, 13,
				14, 15, 16,
			})),
		gorgonia.WithName("x"))

	y := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2, 1, 1),
			tensor.WithBacking([]float32{100, 100})),
		gorgonia.WithName("y"))

	sumT := tensor.New(
		tensor.WithShape(1, 2, 3, 3),
		tensor.WithBacking([]float32{
			100, 101, 102,
			103, 104, 105,
			105, 106, 107,
			108, 109, 110,
			111, 112, 113,
			114, 115, 116,
		}))
	sum := new(gorgonia.Node)

	o, err := op.Apply(
		x, y,
	)
	if err != nil {

		t.Fatal(err)
	}

	sum = o[0]

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	assert.Equal(sumT.Shape(), sum.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(sumT.Data(), sum.Value().Data(), 1e-5, "Tensors should be the same")

}
