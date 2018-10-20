package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Matmul operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Matmul
type Matmul struct {
	name string
}

// Init ...
func (o *Matmul) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Matmul"
	return nil
}

// Apply ...
func (o *Matmul) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	n, err := gorgonia.Mul(input[0], input[1])
	return []*gorgonia.Node{n}, err

}
