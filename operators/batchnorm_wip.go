package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Batchnorm operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
type Batchnorm struct {
	name     string
	Epsilon  float64
	Momentum float64
	Spatial  int
}

// Init ...
func (o *Batchnorm) Init(attrs []*onnx.AttributeProto) error {
	o.name = "BatchNormalization"
	type attributes struct {
		Epsilon  float64 `attributeName:"epsilon"`
		Momentum float64 `attributeName:"momentum"`
		Spatial  int64   `attributeName:"spatial"`
	}
	attr := attributes{
		Epsilon:  1e-5,
		Momentum: 1e-9,
		Spatial:  1,
	}
	err := onnx.UnmarshalAttributes(attrs, &attr)
	if err != nil {
		return err
	}
	o.Epsilon = attr.Epsilon
	o.Momentum = attr.Momentum
	o.Spatial = int(attr.Spatial)

	return nil
}

// Apply ...
func (o *Batchnorm) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 5 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 5,
			ActualInput:   len(input),
		}
	}
	if len(input[0].Shape()) != 4 {
		return nil, &onnx.ErrNotImplemented{
			Operator: o.name,
		}
	}
	x := input[0]
	mean, err := gorgonia.Reshape(input[3], []int{1, input[3].Shape()[0], 1, 1})
	if err != nil {
		return nil, err
	}
	variance := input[4]
	var outputY *gorgonia.Node
	xb1, err := gorgonia.Sub(x, mean, gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3}))
	if err != nil {
		return nil, err
	}
	epsilon := gorgonia.NewTensor(x.Graph(),
		x.Dtype(),
		1,
		gorgonia.WithName("epsilon"),
		gorgonia.WithShape(1),
		gorgonia.WithValue(
			tensor.New(
				tensor.WithShape(1),
				tensor.WithBacking([]float32{float32(o.Epsilon)}),
			),
		),
	)
	if err != nil {
		return nil, err
	}
	xb2, err := gorgonia.Add(variance, epsilon, gorgonia.NewBroadcastPattern(nil, []byte{0}))
	if err != nil {
		return nil, err
	}
	xb2sqrt, err := gorgonia.Sqrt(xb2)
	if err != nil {
		return nil, err
	}
	xb2sqrt, err = gorgonia.Reshape(xb2sqrt, []int{1, xb2sqrt.Shape()[0], 1, 1})
	if err != nil {
		return nil, err
	}
	xbar, err := gorgonia.HadamardDiv(xb1, xb2sqrt, gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3}))
	if err != nil {
		return nil, err
	}
	scale, err := gorgonia.Reshape(input[1], []int{1, input[1].Shape()[0], 1, 1})
	if err != nil {
		return nil, err
	}
	bias, err := gorgonia.Reshape(input[2], []int{1, input[2].Shape()[0], 1, 1})
	if err != nil {
		return nil, err
	}
	if outputY, err = gorgonia.HadamardProd(scale, xbar, gorgonia.NewBroadcastPattern([]byte{0, 2, 3}, nil)); err != nil {
		return nil, err
	}
	if outputY, err = gorgonia.Add(outputY, bias, gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3})); err != nil {
		return nil, err
	}
	/*
		x := input[0]
		mean, ok := input[3].Value().(*tensor.Dense)
		if !ok {
			return nil, errors.New("mean's Value must be a *tensor.Dense")
		}
		variance, ok := input[4].Value().(*tensor.Dense)
		if !ok {
			return nil, errors.New("variance's Value must be a *tensor.Dense")
		}
		op := &gorgonia.BatchNormOp{
			Momentum: o.Momentum,
			Epsilon:  o.Epsilon,
			Mean:     mean,
			Variance: variance,
			MA:       tensor.New(tensor.Of(x.Dtype()), tensor.WithShape(x.Shape()[1])),
		}
		err := op.Init(x, false)
		if err != nil {
			return nil, err
		}

		var outputY *gorgonia.Node
		if outputY, err = gorgonia.ApplyOp(op, x); err != nil {
			return nil, err
		}
		scale, err := gorgonia.Reshape(input[1], []int{1, input[1].Shape()[0], 1, 1})
		if err != nil {
			return nil, err
		}
		bias, err := gorgonia.Reshape(input[2], []int{1, input[2].Shape()[0], 1, 1})
		if err != nil {
			return nil, err
		}
		if outputY, err = gorgonia.HadamardProd(scale, outputY, gorgonia.NewBroadcastPattern([]byte{0, 2, 3}, nil)); err != nil {
			return nil, err
		}
		if outputY, err = gorgonia.Add(outputY, bias, gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3})); err != nil {
			return nil, err
		}

	*/
	return []*gorgonia.Node{
		outputY,
		input[3],
		input[4],
		nil,
		nil,
	}, nil

}
