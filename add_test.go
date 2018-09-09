package gorgonnx

import (
	"testing"

	onnx "github.com/owulveryck/onnx-go"
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

func TestAddOp2(t *testing.T) {
	dataType := onnx.TensorProto_DataType(1)
	input1 := "a"
	simpleTest := &onnx.TensorProto{
		Dims:      []int64{1, 10},
		DataType:  &dataType,
		Segment:   (*onnx.TensorProto_Segment)(nil),
		FloatData: []float32{100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
		Name:      &input1,
	}
	simpleResult := tensor.New(tensor.WithShape(1, 10), tensor.WithBacking([]float32{101, 102, 103, 104, 105, 106, 107, 108, 109, 110}))
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
	} {
		input2 := "b"
		output := "y"
		outputs := []string{output}
		inputs := []string{input1, input2}
		opName := "Plus"
		opType := "Add"
		domain := ""
		docString := ""
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
			Dims:      []int64{1, 10},
			DataType:  &dataType,
			Segment:   (*onnx.TensorProto_Segment)(nil),
			FloatData: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
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
		err = g.processNode(np)
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
		for i, v := range unitTest.output.Data().([]float32) {
			if v != g.getNodeByName(output).Value().Data().([]float32)[i] {
				t.Log("Got: ", g.getNodeByName(output).Value())
				t.Log("Expeced: ", unitTest.output)
				t.FailNow()
			}
		}
	}
}
func TestAddOp(t *testing.T) {
	dataType := onnx.TensorProto_DataType(1)
	input1 := "a"
	simpleTest := &onnx.TensorProto{
		Dims:      []int64{2, 1, 1},
		DataType:  &dataType,
		Segment:   (*onnx.TensorProto_Segment)(nil),
		FloatData: []float32{100, 100},
		Name:      &input1,
	}
	simpleResult := tensor.New(tensor.WithShape(2, 1, 1), tensor.WithBacking([]float32{101, 102}))
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
		101, 101, 101, 101, 101,
		201, 201, 201, 201, 201,
		301, 301, 301, 301, 301,
		401, 401, 401, 401, 401,
		501, 501, 501, 501, 501,
		1002, 1002, 1002, 1002, 1002,
		2002, 2002, 2002, 2002, 2002,
		3002, 3002, 3002, 3002, 3002,
		4002, 4002, 4002, 4002, 4002,
		5002, 5002, 5002, 5002, 5002,
	}))
	type test struct {
		input  *onnx.TensorProto
		output tensor.Tensor
		err    error
	}

	for _, unitTest := range []test{
		test{
			input:  broadcastTest,
			output: broadcastResult,
			err:    nil,
		},
		test{
			input:  simpleTest,
			output: simpleResult,
			err:    nil,
		},
	} {
		input2 := "b"
		output := "y"
		outputs := []string{output}
		inputs := []string{input1, input2}
		opName := "Plus"
		opType := "Add"
		domain := ""
		docString := ""
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
		err = g.processNode(np)
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
		for i, v := range unitTest.output.Data().([]float32) {
			if v != g.getNodeByName(output).Value().Data().([]float32)[i] {
				t.Log("Got: ", g.getNodeByName(output).Value())
				t.Log("Expeced: ", unitTest.output)
				t.FailNow()
			}
		}
	}
}
