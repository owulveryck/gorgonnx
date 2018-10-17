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
// Warning this operator should be broadcastable
func (o *Div) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	if len(input[0].Shape()) != len(input[1].Shape()) {
		return nil, &onnx.ErrNotImplemented{
			Operator: o.name,
			Message:  "TODO: implement the broadcast",
		}
	}
	n, err := gorgonia.HadamardDiv(input[0], input[1])
	return []*gorgonia.Node{n}, err

}
