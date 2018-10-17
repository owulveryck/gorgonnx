package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Mul operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul
type Mul struct {
	name string
}

// Init ...
func (m *Mul) Init(attrs []*onnx.AttributeProto) error {
	m.name = "Mul"
	return nil
}

// Apply ...
func (m *Mul) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      m.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	return nil, nil
}
