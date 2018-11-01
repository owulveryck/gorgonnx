package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Add operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
// Warning this operation is broadcastable
// See https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
//
// BUG(owulveryck): the broadcasting has to be implemented correctly in Gorgonia. see https://github.com/gorgonia/gorgonia/issues/223
type Add struct {
	name string
}

// Init is a noop as Add do not have any attribute
func (a *Add) Init(attrs []*onnx.AttributeProto) error {
	a.name = "Add"
	return nil
}

// Apply ...
func (a *Add) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      a.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	n, err := gorgonia.Add(input[0], input[1], 0)
	return []*gorgonia.Node{n}, err
}
