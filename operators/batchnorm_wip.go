package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
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
	// Reshape the scale and bias
	var err error
	var scaleN, biasN *gorgonia.Node
	dimN := input[0].Shape()[0]
	dimC := input[0].Shape()[1]
	dimH := input[0].Shape()[2]
	dimW := input[0].Shape()[3]
	dtype := input[0].Dtype()
	switch dtype {
	case tensor.Float32:
		backingScale := make([]float32, dimN*dimC*dimH*dimW)
		for i := 0; i < len(backingScale); i++ {
			backingScale[i] = 1
		}
		copy(backingScale[dimN:], input[1].Value().Data().([]float32))
		scaleT := tensor.New(tensor.WithBacking(backingScale), tensor.WithShape(input[0].Shape()...))
		scaleN = gorgonia.NewTensor(input[0].Graph(), dtype, 4, gorgonia.WithShape(input[0].Shape()...), gorgonia.WithValue(scaleT))

		backingBias := make([]float32, dimN*dimC*dimH*dimW)
		copy(backingBias[dimN:], input[2].Value().Data().([]float32))
		biasT := tensor.New(tensor.WithBacking(backingBias), tensor.WithShape(input[0].Shape()...))
		biasN = gorgonia.NewTensor(input[0].Graph(), dtype, 4, gorgonia.WithShape(input[0].Shape()...), gorgonia.WithValue(biasT))
	case tensor.Float64:
		backingScale := make([]float64, dimN*dimC*dimH*dimW)
		for i := 0; i < len(backingScale); i++ {
			backingScale[i] = 1
		}
		copy(backingScale[dimN:], input[1].Value().Data().([]float64))
		backingBias := make([]float64, dimN*dimC*dimH*dimW)
		copy(backingBias[dimN:], input[2].Value().Data().([]float64))
		scaleT := tensor.New(tensor.WithBacking(backingScale), tensor.WithShape(input[0].Shape()...))
		biasT := tensor.New(tensor.WithBacking(backingBias), tensor.WithShape(input[0].Shape()...))
		scaleN = gorgonia.NewTensor(input[0].Graph(), dtype, 4, gorgonia.WithShape(input[0].Shape()...), gorgonia.WithValue(scaleT))
		biasN = gorgonia.NewTensor(input[0].Graph(), dtype, 4, gorgonia.WithShape(input[0].Shape()...), gorgonia.WithValue(biasT))
	default:
		return nil, &onnx.ErrNotImplemented{
			Operator: o.name,
			Message:  "Unsupported type",
		}

	}

	var outputY, outputMean, outputVar, outputSavedMean, outputSavedVar *gorgonia.Node
	outputY, outputMean, outputVar, _, err = nnops.BatchNorm(input[0], scaleN, biasN, o.Momentum, o.Epsilon)
	if err != nil {
		return nil, err
	}
	return []*gorgonia.Node{
		outputY,
		outputMean,
		outputVar,
		outputSavedMean,
		outputSavedVar,
	}, err

}
