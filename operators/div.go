package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Div operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
type Div struct {
	name string
}

// Init ...
func (o *Div) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Div"
	return nil
}

// Apply ...
func (o *Div) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	return nil, nil
}
