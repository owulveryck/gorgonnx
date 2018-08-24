package gorgonnx

import (
	"testing"

	onnx "github.com/owulveryck/onnx/go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor/tensonnx"
)

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
		Int32Data:  nil,
		StringData: nil,
		Int64Data:  nil,
		Name:       &input1,
		DocString:  (*string)(nil),
		RawData:    nil,
		DoubleData: nil,
		Uint64Data: nil,
	}

	for _, b := range []*onnx.TensorProto{
		broadcastTest,
	} {

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
			Dims:       []int64{2, 1, 1},
			DataType:   &dataType,
			Segment:    (*onnx.TensorProto_Segment)(nil),
			FloatData:  []float32{1, 2},
			Int32Data:  nil,
			StringData: nil,
			Int64Data:  nil,
			Name:       &input1,
			DocString:  (*string)(nil),
			RawData:    nil,
			DoubleData: nil,
			Uint64Data: nil,
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
	}
}
