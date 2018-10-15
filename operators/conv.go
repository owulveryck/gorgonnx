package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
)

// Conv operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
type Conv struct {
	name        string
	Pads        []int
	Dilations   []int
	Group       int
	KernelShape tensor.Shape
	Strides     []int
}

// Init the convolution operator
func (c *Conv) Init(attrs []*onnx.AttributeProto) error {
	c.name = "conv"
	type attributes struct {
		AutoPad     string  `attributeName:"auto_pad"`
		Dilations   []int64 `attributeName:"dilations"`
		Group       int64   `attributeName:"group"`
		KernelShape []int64 `attributeName:"kernel_shape"`
		Pads        []int64 `attributeName:"pads"`
		Strides     []int64 `attributeName:"strides"`
	}
	// Set the default values
	attr := attributes{
		AutoPad:   "NOTSET",
		Group:     1,
		Strides:   []int64{1, 1},
		Pads:      []int64{0, 0},
		Dilations: []int64{1, 1},
	}
	err := onnx.UnmarshalAttributes(attrs, &attr)
	if err != nil {
		return err
	}
	// Set the obvious values
	c.KernelShape = int64ToInt(attr.KernelShape)
	c.Group = int(attr.Group)
	c.Strides = int64ToInt(attr.Strides)
	c.Dilations = int64ToInt(attr.Dilations)

	if c.Group != 1 {
		return &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "group",
			AttributeValue: c.Group,
			Message:        "attribute not implemented for a value != 1",
		}
	}
	switch attr.AutoPad {
	case "NOTSET":
	case "VALID":
		c.Pads = []int{0, 0}

	case "SAME_UPPER":
		return &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "auto_pad",
			AttributeValue: attr.AutoPad,
			Message:        "Not implemented",
		}

		/*
			//BUG(owulveryck): We need the input shape for automatic padding
				outputHeight := int(
					math.Ceil(
						float64(attr.KernelShape[2]) / float64(attr.Strides[0])))
				outputWidth := int(
					math.Ceil(
						float64(attr.KernelShape[3]) / float64(attr.Strides[1])))
				c.Pads[0] = int(
					math.Max(
						float64((outputHeight-1)*attr.Strides[0]+kernelShape[0]-input.Shape()[2]),
						float64(0)),
				) / 2
				c.Pads[1] = int(
					math.Max(
						float64((outputWidth-1)*attr.Strides[1]+kernelShape[1]-input.Shape()[3]),
						float64(0)),
				) / 2
		*/
	case "SAME_LOWER":
		return &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "auto_pad",
			AttributeValue: attr.AutoPad,
			Message:        "Not implemented",
		}

	default:
		return &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "auto_pad",
			AttributeValue: attr.AutoPad,
			Message:        "Invalide value",
		}

	}

	if attr.Pads[0] != attr.Pads[1] || attr.Pads[2] != attr.Pads[3] {
		return &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "pads",
			AttributeValue: attr.Pads,
			Message:        "Asymetric padding",
		}
	}
	c.Pads = make([]int, len(attr.Pads)/2)
	for i := 0; i < len(attr.Pads)/2; i++ {
		//pad[i] = int(attr.Ints[2*i] + attr.Ints[2*i+1])
		c.Pads[i] = int(attr.Pads[2*i])
	}
	return nil
}

// Apply ...
func (c *Conv) Apply(input []*gorgonia.Node, output []*gorgonia.Node) error {
	var err error
	if len(input) != 2 {
		return &ErrBadArity{
			Operator:      "Conv",
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	if len(output) != 1 {
		return &ErrBadArity{
			Operator:       "Conv",
			ExpectedOutput: 1,
			ActualOutput:   len(output),
		}
	}
	output[0], err = nnops.Conv2d(input[0], input[1], c.KernelShape, c.Pads, c.Strides, c.Dilations)
	return err
}
