package gorgonnx

import (
	"testing"

	onnx "github.com/owulveryck/onnx/go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/tensonnx"
)

func shapeEquals(a, b tensor.Tensor) bool {
	as := a.Shape()
	bs := b.Shape()
	if len(as) != len(bs) {
		return false
	}
	for i := range as {
		if as[i] != bs[i] {
			return false
		}
	}
	return true
}

func TestAddOp(t *testing.T) {
	input1 := "a"
	input2 := "b"
	output := "y"
	inputs := []string{input1, input2}
	outputs := []string{output}
	opName := "Plus"
	opType := "Add"
	domain := ""
	docString := ""
	dataType := onnx.TensorProto_DataType(1)
	simpleTest := &onnx.TensorProto{
		Dims:      []int64{2, 1, 1},
		DataType:  &dataType,
		Segment:   (*onnx.TensorProto_Segment)(nil),
		FloatData: []float32{100, 100},
		Name:      &input1,
	}
	simpleResult := tensor.New(tensor.WithShape(2, 1, 1), tensor.WithBacking([]int{101, 102}))
	broadcastTest := &onnx.TensorProto{
		Dims:     []int64{1, 2, 5, 5},
		DataType: &dataType,
		Segment:  (*onnx.TensorProto_Segment)(nil),
		FloatData: []float32{
			100, 100, 100, 100, 100,
			200, 200, 200, 200, 200,
			300, 300, 300, 300, 300,
			400, 400, 400, 400, 400,
			500, 500, 500, 500, 500,
			1000, 1000, 1000, 1000, 1000,
			2000, 2000, 2000, 2000, 2000,
			3000, 3000, 3000, 3000, 3000,
			4000, 4000, 4000, 4000, 4000,
			5000, 5000, 5000, 5000, 5000,
		},
		Name: &input1,
	}
	broadcastResult := tensor.New(tensor.WithShape(1, 2, 5, 5), tensor.WithBacking([]float32{
		101, 102, 103, 104, 105,
		201, 202, 203, 204, 205,
		301, 302, 303, 304, 305,
		401, 402, 403, 404, 405,
		501, 502, 503, 504, 505,
		1001, 1002, 1003, 1004, 1005,
		2001, 2002, 2003, 2004, 2005,
		3001, 3002, 3003, 3004, 3005,
		4001, 4002, 4003, 4004, 4005,
		5001, 5002, 5003, 5004, 5005,
	}))
	type test struct {
		input  *onnx.TensorProto
		output tensor.Tensor
		err    error
	}

	for _, unitTest := range []test{
		test{
			input:  simpleTest,
			output: simpleResult,
			err:    nil,
		},
		test{
			input:  broadcastTest,
			output: broadcastResult,
			err:    nil,
		},
	} {

		b := unitTest.input
		np := &onnx.NodeProto{
			Input:     inputs,
			Output:    outputs,
			Name:      &opName,
			OpType:    &opType,
			Domain:    &domain,
			Attribute: nil,
			DocString: &docString,
		}
		// Simple test...
		// Create the input values
		a := &onnx.TensorProto{
			Dims:      []int64{2, 1, 1},
			DataType:  &dataType,
			Segment:   (*onnx.TensorProto_Segment)(nil),
			FloatData: []float32{1, 2},
			Name:      &input1,
			DocString: (*string)(nil),
		}
		aa, err := tensonnx.NewTensor(a)
		if err != nil {
			t.Fatal(err)
		}
		bb, err := tensonnx.NewTensor(b)
		if err != nil {
			t.Fatal(err)
		}
		g := graph{
			db: make(map[string]*gorgonia.Node, 3),
			g:  gorgonia.NewGraph(),
		}
		na := gorgonia.NodeFromAny(g.g, aa, gorgonia.WithName(input1))
		nb := gorgonia.NodeFromAny(g.g, bb, gorgonia.WithName(input2))
		g.addNode(input1, na)
		g.addNode(input2, nb)
		err = g.addOp(np)
		if err != nil {
			t.Fatal(err)
		}
		vm := gorgonia.NewTapeMachine(g.g)
		err = vm.RunAll()
		if err != nil {
			t.Fatal(err)
		}
		if !shapeEquals(g.getNodeByName(output).Value().(tensor.Tensor), unitTest.output) {
			t.Fatal("Size mismatch")
		}
	}
}
