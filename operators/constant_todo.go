package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Constant operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant
type Constant struct {
	name string
}

// Init ...
func (o *Constant) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Constant"
	return &onnx.ErrNotImplemented{
		Operator: o.name,
		Message:  "Not implemented yet",
	}
}

// Apply ...
func (o *Constant) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	/*
		if len(input) != 2 {
			return nil, &ErrBadArity{
				Operator:      o.name,
				ExpectedInput: 2,
				ActualInput:   len(input),
			}
		}
	*/
	return nil, &onnx.ErrNotImplemented{
		Operator: o.name,
		Message:  "Not implemented yet",
	}

}
