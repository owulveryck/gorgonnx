package gorgonnx

import (
	"fmt"
	"log"
	"math"

	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
func (cg *computationGraph) convOp(nx *onnx.NodeProto) error {
	input := cg.db[nx.Input[0]]
	kernel := cg.db[nx.Input[1]]
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
	cg.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
func (cg *computationGraph) reshapeOp(nx *onnx.NodeProto) error {
	if len(nx.Input) != 2 {
		return fmt.Errorf("Not enough input parameters for reshape")
	}
	var data []int
	d, ok := cg.db[nx.Input[1]].Value().Data().([]int64)
	if ok {
		data = toIntSlice(d)
	} else {
		data = []int{int(cg.db[nx.Input[1]].Value().Data().(int64))}
	}

	n, err := gorgonia.Reshape(cg.db[nx.Input[0]], data)
	if err != nil {
		return fmt.Errorf("Cannot reshape from %v to %v: %v", nx.Input[0], data, err)
	}
	//cg.g.AddNode(n)
	cg.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
// Warning this operation is broadcastable
// See https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
//
// BUG(owulveryck): the broadcasting has to be implemented correctly in Gorgonia. see https://github.com/gorgonia/gorgonia/issues/223
func (cg *computationGraph) addOp(nx *onnx.NodeProto) error {
	b := cg.db[nx.Input[1]]
	a := cg.db[nx.Input[0]]
	var n *gorgonia.Node
	var err error
	normalAdd := true
	if len(a.Shape()) == len(b.Shape()) {
		for i := range a.Shape() {
			if a.Shape()[i] != b.Shape()[i] {
				normalAdd = false
			}
		}
	} else {
		normalAdd = false
	}
	if normalAdd {
		n, err = gorgonia.Add(a, b)
		if err != nil {
			return fmt.Errorf("Cannot Add %v and %v: %v", nx.Input[0], nx.Input[1], err)
		}
		cg.db[nx.Output[0]] = n
	} else {
		n, err = gorgonia.AddBroadcast(a, b)
		if err != nil {
			return fmt.Errorf("Cannot Add %v and %v: %v", nx.Input[0], nx.Input[1], err)
		}
		cg.db[nx.Output[0]] = n
	}

	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
func (cg *computationGraph) reluOp(nx *onnx.NodeProto) error {
	n, err := nnops.Rectify(cg.db[nx.Input[0]])
	if err != nil {
		return err
	}
	cg.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
func (cg *computationGraph) maxPoolOp(nx *onnx.NodeProto) error {

	var kernelShape tensor.Shape
	input := cg.db[nx.Input[0]]
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
		return fmt.Errorf("Cannot apply Convolution operator: %v", err)
	}
	cg.db[nx.Output[0]] = n
	return nil

}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
//
// BUG(owulveryck): The Mul operator should be broadcastable too
func (cg *computationGraph) matMulOp(nx *onnx.NodeProto) error {
	n, err := gorgonia.Mul(cg.db[nx.Input[0]], cg.db[nx.Input[1]])
	if err != nil {
		return fmt.Errorf("Cannot Multiply: %v", err)
	}
	cg.db[nx.Output[0]] = n

	return nil
}
