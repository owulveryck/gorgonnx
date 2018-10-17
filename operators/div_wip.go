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
	return nil, &onnx.ErrNotImplemented{
		Operator: o.name,
		Message:  "TODO: implement the broadcast",
	}
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	n, err := gorgonia.HadamardDiv(input[0], input[1])
	return []*gorgonia.Node{n}, err

}
