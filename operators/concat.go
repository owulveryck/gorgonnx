package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Concat operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat
type Concat struct {
	name string
}

// Init ...
func (o *Concat) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Concat"
	return nil
}

// Apply ...
func (o *Concat) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	return nil, nil
}
