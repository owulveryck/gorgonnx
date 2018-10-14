package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Conv operator
type Conv struct {
	Pads        []int64
	Dilations   []int64
	Group       int64
	KernelShape tensor.Shape
	Strides     []int64
}

func (c *Conv) Init(attrs []*onnx.AttributeProto) error {
	return nil
}

func (c *Conv) Apply(input []*gorgonia.Node, output []*gorgonia.Node) error {
	return nil
}
