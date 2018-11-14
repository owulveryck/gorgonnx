package operators

import (
	"math"

	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
)

// Maxpool operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
type Maxpool struct {
	name         string
	Pads         []int
	AutoPad      string
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
		o.AutoPad = attr.AutoPad
		return &onnx.ErrNotImplemented{
			Operator:       o.name,
			AttributeName:  "auto_pad",
			AttributeValue: attr.AutoPad,
			Message:        "Padding is buggy",
		}
	case "SAME_LOWER":
		o.AutoPad = attr.AutoPad
		return &onnx.ErrNotImplemented{
			Operator:       o.name,
			AttributeName:  "auto_pad",
			AttributeValue: attr.AutoPad,
			Message:        "Padding is buggy",
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
	o.Pads = make([]int, 2)
	if len(attr.Pads) == 4 {
		for i := 0; i < 2; i++ {
			o.Pads[i] = int(attr.Pads[2*i])
		}
	}
	if o.Pads[0] != 0 || o.Pads[1] != 0 {
		return &onnx.ErrNotImplemented{
			Operator:       o.name,
			AttributeName:  "pads",
			AttributeValue: attr.Pads,
			Message:        "Padding is buggy",
		}

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
	if len(o.KernelShape) != 2 {
		return nil, &onnx.ErrNotImplemented{
			Operator: o.name,
			Message:  "Not implemented for dimension != 2",
		}

	}
	switch o.AutoPad {
	case "SAME_UPPER":
		//outputHeight := int(math.Ceil(float64(input[0].Shape()[2]) / float64(o.Strides[0])))
		//outputWidth := int(math.Ceil(float64(input[0].Shape()[3]) / float64(o.Strides[1])))
		outputHeight := int(math.Ceil(float64(input[0].Shape()[2])))
		outputWidth := int(math.Ceil(float64(input[0].Shape()[3])))
		o.Pads[0] = int(math.Max(float64((outputHeight-1)*o.Strides[0]+o.KernelShape[0]-input[0].Shape()[2]), float64(0))) / 2
		o.Pads[1] = int(math.Max(float64((outputWidth-1)*o.Strides[1]+o.KernelShape[1]-input[0].Shape()[3]), float64(0))) / 2
	case "SAME_LOWER":
		return nil, &onnx.ErrNotImplemented{
			Operator:       o.name,
			AttributeName:  "auto_pad",
			AttributeValue: o.AutoPad,
			Message:        "not supported",
		}
	default:
	}
	n, err := nnops.MaxPool2D(input[0], o.KernelShape, o.Pads, o.Strides)
	return []*gorgonia.Node{n}, err

}
