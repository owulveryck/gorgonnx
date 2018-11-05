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
	dims := make([]int, len(o.Axes)+input[0].Dims())
	for k := range dims {
		dims[k] = -1
	}
	for _, v := range o.Axes {
		dims[v] = 1
	}
	var index int
	for k, v := range dims {
		if v == -1 {
			index = k
			break
		}
	}
	for i := 0; i < input[0].Dims(); i++ {
		dims[i+index] = input[0].Shape()[i]
	}
	output, err := gorgonia.Reshape(input[0], dims)
	return []*gorgonia.Node{
			output,
		},
		err
}
