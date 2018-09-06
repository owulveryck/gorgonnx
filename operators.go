package gorgonnx

import (
	"fmt"
	"log"
	"math"

	onnx "github.com/owulveryck/onnx/go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
func (d *graph) convOp(nx *onnx.NodeProto) error {
	input := d.db[nx.Input[0]]
	kernel := d.db[nx.Input[1]]
	kernelShape := kernel.Shape()
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
				log.Println(pad)
			case "SAME_LOWER":
				return fmt.Errorf("Warning: lower padding not implemented")
			case "VALID":
				pad = []int{0, 0}
			default:
				return fmt.Errorf("Invalid auto_pad value: %v", string(attr.S))

			}
		case "pads":
			pad = make([]int, len(attr.Ints)/2)
			for i := 0; i < len(attr.Ints)/2; i += 1 {
				//pad[i] = int(attr.Ints[2*i] + attr.Ints[2*i+1])
				pad[i] = int(attr.Ints[2*i])
			}
			log.Println(pad)
			//return fmt.Errorf("Pad not implemented")
			// BUG(owulveryck): `pad` attribute not implemented
		case "group":
			if *attr.I == int64(1) {
				continue
			}
			return fmt.Errorf("group not implemented")
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
	d.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
func (d *graph) reshapeOp(nx *onnx.NodeProto) error {
	if len(nx.Input) != 2 {
		return fmt.Errorf("Not enough input parameters for reshape")
	}
	data := toIntSlice(d.db[nx.Input[1]].Value().Data().([]int64))

	n, err := gorgonia.Reshape(d.db[nx.Input[0]], data)
	if err != nil {
		return fmt.Errorf("Cannot reshape from %v to %v: %v", nx.Input[0], data, err)
	}
	//d.g.AddNode(n)
	d.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
// Warning this operation is broadcastable
// See https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
//
// BUG(owulveryck): the broadcasting has to be implemented correctly in Gorgonia. see https://github.com/gorgonia/gorgonia/issues/223
func (d *graph) addOp(nx *onnx.NodeProto) error {
	b := d.db[nx.Input[1]]
	a := d.db[nx.Input[0]]
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
		d.db[nx.Output[0]] = n
	} else {
		n, err = gorgonia.AddBroadcast(a, b)
		if err != nil {
			return fmt.Errorf("Cannot Add %v and %v: %v", nx.Input[0], nx.Input[1], err)
		}
		d.db[nx.Output[0]] = n
	}

	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
func (d *graph) reluOp(nx *onnx.NodeProto) error {
	n, err := nnops.Rectify(d.db[nx.Input[0]])
	if err != nil {
		return err
	}
	d.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
func (d *graph) maxPoolOp(nx *onnx.NodeProto) error {

	var kernelShape tensor.Shape
	input := d.db[nx.Input[0]]
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
				return fmt.Errorf("auto_pad %v not implemented", string(attr.S))
			case "SAME_LOWER":
				return fmt.Errorf("auto_pad %v not implemented", string(attr.S))
			case "VALID":
				pad = []int{0, 0}
			default:
				return fmt.Errorf("Invalid auto_pad value: %v", string(attr.S))

			}
		case "pads":
			pads := attr.Ints
			if len(pads) == 4 && pads[2] != 0 && pads[3] != 0 {
				return fmt.Errorf("Padding at the end not implemented")

			}
			for i := 0; i < 2; i++ {
				pad[i] = int(attr.Ints[i])
			}
		case "storage_order":
			return fmt.Errorf("Attribute: %v not implemented yet for maxpool operator", attr.Name)
		default:
			return fmt.Errorf("Unknown attribute: %v for maxpool operator", attr.Name)
		}
	}
	//log.Printf("MaxPool2D %v (%v) with %v pad:%v stride:%v", input.Name(), input.Shape(), kernelShape, pad, stride)
	n, err := nnops.MaxPool2D(input, kernelShape, pad, stride)
	if err != nil {
		return fmt.Errorf("Cannot apply Convolution operator: %v", err)
	}
	d.db[nx.Output[0]] = n
	return nil

}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
//
// BUG(owulveryck): The Mul operator should be broadcastable too
func (d *graph) matMulOp(nx *onnx.NodeProto) error {
	n, err := gorgonia.Mul(d.db[nx.Input[0]], d.db[nx.Input[1]])
	if err != nil {
		return fmt.Errorf("Cannot Multiply: %v", err)
	}
	d.db[nx.Output[0]] = n

	return nil

}
