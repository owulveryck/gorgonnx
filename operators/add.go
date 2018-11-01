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
	n, err := gorgonia.Add(x, y, bcastPattern)
	return []*gorgonia.Node{n}, err
}
