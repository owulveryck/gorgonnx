package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Maxpool operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Maxpool
type Maxpool struct {
	name string
}

// Init ...
func (o *Maxpool) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Maxpool"
	return nil
}

// Apply ...
func (o *Maxpool) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	return nil, nil
}
