package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Reshape operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
type Reshape struct {
	name string
}

// Init is a noop as Reshape do not have any attribute
func (r *Reshape) Init(attrs []*onnx.AttributeProto) error {
	r.name = "Reshape"
	return nil
}

// Apply ...
func (r *Reshape) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      r.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	var data []int
	d, ok := input[1].Value().Data().([]int64)
	if ok {
		data = int64ToInt(d)
	} else {
		data = []int{int(input[1].Value().Data().(int64))}
	}

	n, err := gorgonia.Reshape(input[0], data)
	return []*gorgonia.Node{n}, err
}
