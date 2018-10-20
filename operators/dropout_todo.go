package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Dropout operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout
type Dropout struct {
	name string
}

// Init ...
func (o *Dropout) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Dropout"
	return &onnx.ErrNotImplemented{
		Operator: o.name,
		Message:  "Not implemented yet",
	}
}

// Apply ...
func (o *Dropout) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
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
