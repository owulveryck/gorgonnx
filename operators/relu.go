package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Relu operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
type Relu struct {
	name string
}

// Init ...
func (o *Relu) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Relu"
	return nil
}

// Apply ...
func (o *Relu) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	return nil, nil
}
