package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
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
	return nil, &onnx.ErrNotImplemented{
		Operator: o.name,
		Message:  "Temporarly disabled",
	}
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
	// Reshape the scale and bias
	var err error

	var outputY, outputMean, outputVar, outputSavedMean, outputSavedVar *gorgonia.Node
	outputY, outputMean, outputVar, _, err = gorgonia.BatchNorm(input[0], input[1], input[2], o.Momentum, o.Epsilon)
	/*
		outputY, _, _, _, err = gorgonia.BatchNormONNX(input[0], nil, nil, o.Momentum, o.Epsilon)
		if err != nil {
			return nil, err
		}
		log.Println(input[1].Value())
		log.Println(input[2].Value())
		outputMean, err = gorgonia.BatchNormMulScale(outputY, input[1])
		if err != nil {
			return nil, err
		}
		outputVar, err = gorgonia.BatchNormAddBias(outputMean, input[2])
		if err != nil {
			return nil, err
		}
	*/
	return []*gorgonia.Node{
		outputY,
		outputMean,
		outputVar,
		outputSavedMean,
		outputSavedVar,
	}, err

}
