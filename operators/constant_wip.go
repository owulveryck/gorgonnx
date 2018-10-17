package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Constant operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant
type Constant struct {
	name   string
	Tensor tensor.Tensor
}

// Init ...
func (o *Constant) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Constant"
	type attributes struct {
		Tensor tensor.Tensor `attributeName:"value" required:"true"`
	}
	attr := attributes{}
	err := onnx.UnmarshalAttributes(attrs, &attr)
	if err != nil {
		return err
	}
	o.Tensor = attr.Tensor
	return nil
}

// Apply ...
func (o *Constant) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 0 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 0,
			ActualInput:   len(input),
		}
	}
	n := gorgonia.NewConstant(o.Tensor)
	return []*gorgonia.Node{n}, nil
}
