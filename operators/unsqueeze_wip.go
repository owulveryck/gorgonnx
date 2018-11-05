package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Unsqueeze operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
type Unsqueeze struct {
	name string
	Axes []int64
}

// Init ...
func (o *Unsqueeze) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Unsqueeze"
	type attributes struct {
		Axes []int64 `attributeName:"axes" required:"true"`
	}
	var attr attributes
	err := onnx.UnmarshalAttributes(attrs, &attr)
	if err != nil {
		return err
	}
	o.Axes = attr.Axes

	return nil
}

// Apply ...
func (o *Unsqueeze) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	/*
		if len(input) != 2 {
			return nil, &ErrlBadArity{
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
