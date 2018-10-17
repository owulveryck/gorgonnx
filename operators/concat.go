package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Concat operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat
type Concat struct {
	name string
	Axis int
}

// Init ...
func (o *Concat) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Concat"
	type attributes struct {
		Axis int64 `attributeName:"axis" required:"true"`
	}
	// Set the default values
	attr := attributes{
		Axis: 1,
	}
	err := onnx.UnmarshalAttributes(attrs, &attr)
	if err != nil {
		return err
	}
	o.Axis = int(attr.Axis)

	return nil
}

// Apply ...
func (o *Concat) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	n, err := gorgonia.Concat(o.Axis, input...)
	return []*gorgonia.Node{n}, err

}
