package gorgonnx

import (
	"errors"
	"fmt"
	"log"
	"math"

	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/tensonnx"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant
func (cg *computationGraph) constantOp(nx *onnx.NodeProto) error {
	var t tensor.Tensor
	for _, attr := range nx.Attribute {
		switch *attr.Name {
		case "value":
			var err error
			t, err = tensonnx.NewTensor(attr.T)
			if err != nil {
				return err
			}
		default:
			return fmt.Errorf("Unknown attribute: %v for convolution operator", attr.Name)
		}
	}
	if t == nil {
		return fmt.Errorf("Value cannot be null")
	}
	cg.db.Store(nx.Output[0], cg.g.AddNode(gorgonia.NewConstant(t, gorgonia.WithName(nx.Output[0]))))

	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout
func (cg *computationGraph) dropoutOp(nx *onnx.NodeProto) error {
	if len(nx.Output) != 1 {
		return ErrToBeImplemented{
			"Dropout",
			"Output",
			"More than one",
		}
	}
	input, ok := cg.db.Load(nx.Input[0])
	if !ok {
		return errors.New("Node not found in the graph")
	}
	//kernelShape := kernel.Shape()
	var ratio float64
	for _, attr := range nx.Attribute {
		switch *attr.Name {
		case "ratio":
			ratio = float64(attr.GetF())
		default:
			return fmt.Errorf("Unknown attribute: %v for convolution operator", attr.Name)
		}
	}
	// For testing, reshape the kernel...
	n, err := gorgonia.Dropout(input.(*gorgonia.Node), ratio)
	if err != nil {
		return fmt.Errorf("Cannot apply Dropout operator: %v", err)
	}
	cg.db.Store(nx.Output[0], n)
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat
func (cg *computationGraph) concatOp(nx *onnx.NodeProto) error {

	inputs := make([]*gorgonia.Node, len(nx.Input))
	for i := 0; i < len(nx.Input); i++ {
		input, ok := cg.db.Load(nx.Input[i])
		inputs[i] = input.(*gorgonia.Node)
		if !ok {
			return errors.New("Node not found")
		}
	}
	//kernelShape := kernel.Shape()
	var axis int
	for _, attr := range nx.Attribute {
		switch *attr.Name {
		case "axis":
			axis = int(attr.GetI())
		default:
			return fmt.Errorf("Unknown attribute: %v for convolution operator", attr.Name)
		}
	}
	// For testing, reshape the kernel...
	n, err := gorgonia.Concat(axis, inputs...)
	if err != nil {
		return fmt.Errorf("Cannot apply Conccat operator: %v", err)
	}
	cg.db.Store(nx.Output[0], n)
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
func (cg *computationGraph) convOp(nx *onnx.NodeProto) error {
	inputDB, ok := cg.db.Load(nx.Input[0])
	if !ok {
		return errors.New("Node not found")
	}
	kernelDB, ok := cg.db.Load(nx.Input[1])
	if !ok {
		return errors.New("Node not found")
	}
	input := inputDB.(*gorgonia.Node)
	kernel := kernelDB.(*gorgonia.Node)
	var kernelShape tensor.Shape
	//kernelShape := kernel.Shape()
	pad := []int{0, 0}
	stride := []int{1, 1}
	dilations := []int{1, 1}
	haveStride := false
	for _, attr := range nx.Attribute {
		switch *attr.Name {
		case "kernel_shape":
			shape := make([]int, len(attr.Ints))
			for i, v := range attr.Ints {
				shape[i] = int(v)
			}
			kernelShape = tensor.Shape(shape)
		case "strides":
			haveStride = true
			stride = make([]int, len(attr.Ints))
			for i, v := range attr.Ints {
				stride[i] = int(v)
			}
		case "auto_pad":
			if !haveStride {
				log.Println("Warning, processing padding without stride")
			}
			// Evaluating the padding
			if attr.S == nil {
				return fmt.Errorf("auto_pad specified without value")
			}
			switch string(attr.S) {
			case "NOTSET":
			case "SAME_UPPER":
				outputHeight := int(math.Ceil(float64(input.Shape()[2]) / float64(stride[0])))
				outputWidth := int(math.Ceil(float64(input.Shape()[3]) / float64(stride[1])))
				pad[0] = int(math.Max(float64((outputHeight-1)*stride[0]+kernelShape[0]-input.Shape()[2]), float64(0))) / 2
				pad[1] = int(math.Max(float64((outputWidth-1)*stride[1]+kernelShape[1]-input.Shape()[3]), float64(0))) / 2
			case "SAME_LOWER":
				return ErrToBeImplemented{
					"ConvOp",
					"auto_pad",
					"SAME_LOWER",
				}
			case "VALID":
				pad = []int{0, 0}
			default:
				return fmt.Errorf("Invalid auto_pad value: %v", string(attr.S))

			}
		case "pads":
			if attr.Ints[0] != attr.Ints[1] || attr.Ints[2] != attr.Ints[3] {
				return ErrToBeImplemented{
					"ConvOp",
					"pads",
					"Asymetric padding",
				}
			}
			pad = make([]int, len(attr.Ints)/2)
			for i := 0; i < len(attr.Ints)/2; i++ {
				//pad[i] = int(attr.Ints[2*i] + attr.Ints[2*i+1])
				pad[i] = int(attr.Ints[2*i])
			}
		case "group":
			if *attr.I == int64(1) {
				continue
			}
			return ErrToBeImplemented{
				"ConvOp",
				"group",
				*attr.I,
			}
			// BUG(owulveryck): `group` attribute not implemented
		case "dilations":
			for i, v := range attr.Ints {
				dilations[i] = int(v)
			}
		default:
			return fmt.Errorf("Unknown attribute: %v for convolution operator", attr.Name)
		}
	}
	// For testing, reshape the kernel...
	n, err := nnops.Conv2d(input, kernel, kernelShape, pad, stride, dilations)
	if err != nil {
		return fmt.Errorf("Cannot apply Convolution operator: %v", err)
	}
	cg.db.Store(nx.Output[0], n)
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
func (cg *computationGraph) reshapeOp(nx *onnx.NodeProto) error {
	if len(nx.Input) != 2 {
		return fmt.Errorf("Not enough input parameters for reshape")
	}
	var data []int
	shapeDB, ok := cg.db.Load(nx.Input[1])
	if !ok {
		return errors.New("node not found")
	}
	shape := shapeDB.(*gorgonia.Node)
	inputDB, ok := cg.db.Load(nx.Input[0])
	if !ok {
		return errors.New("node not found")
	}
	input := inputDB.(*gorgonia.Node)

	d, ok := shape.Value().Data().([]int64)
	if ok {
		data = toIntSlice(d)
	} else {
		data = []int{int(shape.Value().Data().(int64))}
	}

	n, err := gorgonia.Reshape(input, data)
	if err != nil {
		return fmt.Errorf("Cannot reshape from %v to %v: %v", nx.Input[0], data, err)
	}
	cg.db.Store(nx.Output[0], n)
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
// Warning this operation is broadcastable
// See https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
//
// BUG(owulveryck): the broadcasting has to be implemented correctly in Gorgonia. see https://github.com/gorgonia/gorgonia/issues/223
func (cg *computationGraph) addOp(nx *onnx.NodeProto) error {
	b, err := cg.getNode(nx.Input[1])
	if err != nil {
		return err
	}
	a, err := cg.getNode(nx.Input[0])
	if err != nil {
		return err
	}
	var n *gorgonia.Node
	n, err = gorgonia.AddBroadcast(a, b)
	if err != nil {
		return fmt.Errorf("Cannot Add %v and %v: %v", nx.Input[0], nx.Input[1], err)
	}
	cg.db.Store(nx.Output[0], n)

	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
func (cg *computationGraph) reluOp(nx *onnx.NodeProto) error {
	input, err := cg.getNode(nx.Input[0])
	if err != nil {
		return err
	}
	n, err := nnops.Rectify(input)
	if err != nil {
		return err
	}
	cg.db.Store(nx.Output[0], n)
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
func (cg *computationGraph) averagePoolOp(nx *onnx.NodeProto) error {
	return ErrToBeImplemented{
		"averagePoolOp",
		"",
		"",
	}
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
func (cg *computationGraph) maxPoolOp(nx *onnx.NodeProto) error {

	var kernelShape tensor.Shape
	input, err := cg.getNode(nx.Input[0])
	if err != nil {
		return err
	}
	pad := []int{0, 0}
	stride := []int{1, 1}
	for _, attr := range nx.Attribute {
		switch *attr.Name {
		case "kernel_shape":
			shape := make([]int, len(attr.Ints))
			for i, v := range attr.Ints {
				shape[i] = int(v)
			}
			kernelShape = tensor.Shape(shape)
		case "strides":
			if len(attr.Ints) == 2 {
				for i, v := range attr.Ints {
					stride[i] = int(v)
				}
			}
		case "auto_pad":
			switch string(attr.S) {
			case "NOTSET":
			case "SAME_UPPER":
				return ErrToBeImplemented{
					"MaxpoolOp",
					"auto_pad",
					"SAME_UPPER",
				}
			case "SAME_LOWER":
				return ErrToBeImplemented{
					"MaxpoolOp",
					"auto_pad",
					"SAME_LOWER",
				}
			case "VALID":
				pad = []int{0, 0}
			default:
				return fmt.Errorf("Invalid auto_pad value: %v", string(attr.S))

			}
		case "pads":
			pads := attr.Ints
			if len(pads) == 4 && pads[2] != 0 && pads[3] != 0 {
				return ErrToBeImplemented{
					"MaxpoolOp",
					"pads",
					"End padding",
				}
			}
			for i := 0; i < 2; i++ {
				pad[i] = int(attr.Ints[i])
			}
		case "storage_order":
			return ErrToBeImplemented{
				"MaxpoolOp",
				"storage_order",
				"",
			}
		default:
			return fmt.Errorf("Unknown attribute: %v for maxpool operator", attr.Name)
		}
	}
	n, err := nnops.MaxPool2D(input, kernelShape, pad, stride)
	if err != nil {
		return fmt.Errorf("Cannot apply Maxpool operator: %v", err)
	}
	cg.db.Store(nx.Output[0], n)
	return nil

}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
func (cg *computationGraph) divOp(nx *onnx.NodeProto) error {
	input1, err := cg.getNode(nx.Input[0])
	if err != nil {
		return err
	}
	input2, err := cg.getNode(nx.Input[1])
	if err != nil {
		return err
	}
	n, err := gorgonia.HadamardDiv(input1, input2)
	if err != nil {
		return fmt.Errorf("Cannot Divide: %v", err)
	}
	cg.db.Store(nx.Output[0], n)

	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
//
// BUG(owulveryck): The Mul operator should be broadcastable too
func (cg *computationGraph) matMulOp(nx *onnx.NodeProto) error {
	input1, err := cg.getNode(nx.Input[0])
	if err != nil {
		return err
	}
	input2, err := cg.getNode(nx.Input[1])
	if err != nil {
		return err
	}
	n, err := gorgonia.Mul(input1, input2)
	if err != nil {
		return fmt.Errorf("Cannot Multiply: %v", err)
	}
	cg.db.Store(nx.Output[0], n)

	return nil
}
