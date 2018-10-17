package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
)

// Maxpool operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Maxpool
type Maxpool struct {
	name         string
	Pads         []int
	StorageOrder int
	KernelShape  tensor.Shape
	Strides      []int
}

// Init ...
func (o *Maxpool) Init(attrs []*onnx.AttributeProto) error {
	o.name = "Maxpool"
	type attributes struct {
		AutoPad      string  `attributeName:"auto_pad"`
		StorageOrder int64   `attributeName:"storage_order"`
		KernelShape  []int64 `attributeName:"kernel_shape" required:"true"`
		Pads         []int64 `attributeName:"pads"`
		Strides      []int64 `attributeName:"strides"`
	}
	// Set the default values
	attr := attributes{
		AutoPad:      "NOTSET",
		StorageOrder: 0,
		Strides:      []int64{1, 1},
		Pads:         []int64{0, 0},
	}
	err := onnx.UnmarshalAttributes(attrs, &attr)
	if err != nil {
		return err
	}
	// Set the obvious values
	o.KernelShape = int64ToInt(attr.KernelShape)
	o.StorageOrder = int(attr.StorageOrder)
	o.Strides = int64ToInt(attr.Strides)

	if o.StorageOrder != 0 {
		return &onnx.ErrNotImplemented{
			Operator:       o.name,
			AttributeName:  "storage_order",
			AttributeValue: o.StorageOrder,
			Message:        "attribute not implemented for a value != 0",
		}
	}
	switch attr.AutoPad {
	case "NOTSET":
	case "VALID":
		o.Pads = []int{0, 0}

	case "SAME_UPPER":
		return &onnx.ErrNotImplemented{
			Operator:       o.name,
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
			Operator:       o.name,
			AttributeName:  "auto_pad",
			AttributeValue: attr.AutoPad,
			Message:        "Not implemented",
		}

	default:
		return &onnx.ErrNotImplemented{
			Operator:       o.name,
			AttributeName:  "auto_pad",
			AttributeValue: attr.AutoPad,
			Message:        "Invalide value",
		}

	}

	if len(attr.Pads) == 4 && (attr.Pads[0] != attr.Pads[1] || attr.Pads[2] != attr.Pads[3]) {
		return &onnx.ErrNotImplemented{
			Operator:       o.name,
			AttributeName:  "pads",
			AttributeValue: attr.Pads,
			Message:        "Asymetric padding",
		}
	}
	o.Pads = make([]int, len(attr.Pads)/2)
	for i := 0; i < len(attr.Pads)/2; i++ {
		//pad[i] = int(attr.Ints[2*i] + attr.Ints[2*i+1])
		o.Pads[i] = int(attr.Pads[2*i])
	}

	return nil
}

// Apply ...
func (o *Maxpool) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 1 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 1,
			ActualInput:   len(input),
		}
	}
	if len(input[0].Shape()) != 2 {
		return nil, &onnx.ErrNotImplemented{
			Operator: o.name,
			Message:  "Not implemented for dimension != 2",
		}

	}
	n, err := nnops.MaxPool2D(input[0], o.KernelShape, o.Pads, o.Strides)
	return []*gorgonia.Node{n}, err

}
