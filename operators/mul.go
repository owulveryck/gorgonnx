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

// Init of the operator; the operator does not expect any attribute; therefore, any value of attrs is silently discarded
func (o *Mul) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Mul"
	return nil
}

// Apply the Hadamard Product to the input nodes. Broadcasting is computed and applied is needed
func (o *Mul) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	var bcastPattern gorgonia.BroadcastPattern
	x := input[0]
	y := input[1]
	switch {
	case len(x.Shape()) == 1 && len(y.Shape()) != 1:
		// Need left broadcasting
		// Make an educated guess: find the axis that has the same dimension
		// as x.Shape()[0] and broadcast on all axes of y but this one.
		var leftPattern []byte
		dims := make([]int, len(x.Shape()))
		for i := 0; i < len(y.Shape()); i++ {
			if y.Shape()[i] != x.Shape()[0] {
				dims[i] = 1
				leftPattern = append(leftPattern, byte(i))
			} else {
				dims[i] = x.Shape()[0]
			}
		}
		var err error
		x, err = gorgonia.Reshape(input[0], dims)
		if err != nil {
			return nil, err
		}
		bcastPattern = gorgonia.NewBroadcastPattern(leftPattern, nil)
	case len(y.Shape()) == 1 && len(x.Shape()) != 1:
		// Need right broadcasting
		var rightPattern []byte
		dims := make([]int, len(x.Shape()))
		for i := 0; i < len(x.Shape()); i++ {
			if x.Shape()[i] != y.Shape()[0] {
				dims[i] = 1
				rightPattern = append(rightPattern, byte(i))
			} else {
				dims[i] = y.Shape()[0]
			}
		}
		var err error
		y, err = gorgonia.Reshape(input[1], dims)
		if err != nil {
			return nil, err
		}
		bcastPattern = gorgonia.NewBroadcastPattern(nil, rightPattern)
	case len(y.Shape()) == 3 && len(x.Shape()) == 4:
		// Ugly hack for the mnist model
		dims := make([]int, 4)
		dims[0] = 1
		for i := 0; i < 3; i++ {
			dims[i+1] = input[1].Shape()[i]
		}
		var err error
		y, err = gorgonia.Reshape(input[1], dims)
		if err != nil {
			return nil, err
		}
		bcastPattern = gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3})
	case len(y.Shape()) == 4 && len(x.Shape()) == 3:
		// Ugly hack for the mnist model
		dims := make([]int, 4)
		dims[0] = 1
		for i := 0; i < 3; i++ {
			dims[i+1] = input[0].Shape()[i]
		}
		var err error
		x, err = gorgonia.Reshape(input[0], dims)
		if err != nil {
			return nil, err
		}
		bcastPattern = gorgonia.NewBroadcastPattern([]byte{0, 2, 3}, nil)
	}

	n, err := gorgonia.HadamardProd(x, y, bcastPattern)
	return []*gorgonia.Node{n}, err

}
