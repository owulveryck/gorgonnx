package gorgonnx

import (
	"fmt"
	"log"

	"github.com/owulveryck/gorgonnx/onnx"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
//
// TODO(owulveryck): Check if kernel_shape corresponds to what the Conv2D operator expects
//
// TODO(owulveryck): Check if the strides are ok
func (d *Decoder) convOp(nx *onnx.NodeProto) error {
	input := d.db[nx.Input[0]]
	kernel := d.db[nx.Input[1]]
	var kernelShape tensor.Shape
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
			// http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
			if attr.S == nil {
				return fmt.Errorf("auto_pad specified without value")
			}
			switch string(attr.S) {
			case "NOTSET":
			case "SAME_UPPER":
				pad[0] = ((input.Shape()[2]-1)*stride[0] - input.Shape()[2] + kernel.Shape()[2]) / 2
				pad[1] = ((input.Shape()[3]-1)*stride[1] - input.Shape()[3] + kernel.Shape()[3]) / 2
			case "SAME_LOWER":
				pad[0] = ((input.Shape()[2]-1)*stride[0] - input.Shape()[2] + kernel.Shape()[2]) / 2
				pad[1] = ((input.Shape()[3]-1)*stride[1] - input.Shape()[3] + kernel.Shape()[3]) / 2
			case "VALID":
				pad = []int{0, 0}
			default:
				return fmt.Errorf("Invalid auto_pad value: %v", string(attr.S))

			}
		case "pads":
			// BUG(owulveryck): `pad` attribute not implemented and silently ignored
		case "group":
			// BUG(owulveryck): `group` attribute not implemented and silently ignored in the 'conv' operator
		case "dilations":
			for i, v := range attr.Ints {
				dilations[i] = int(v)
			}
		default:
			return fmt.Errorf("Unknown attribute: %v for convolution operator", attr.Name)
		}
	}
	n, err := gorgonia.Conv2d(input, kernel, kernelShape, pad, stride, dilations)
	if err != nil {
		return fmt.Errorf("Cannot apply Convolution operator: %v", err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
// BUG(owulveryck): Reshape's second parameter is a "shape tensor"
func (d *Decoder) reshapeOp(nx *onnx.NodeProto) error {
	if len(nx.Input) != 2 {
		return fmt.Errorf("Not enough input parameters for reshape")
	}
	data := d.db[nx.Input[1]]
	log.Println(data)

	n, err := gorgonia.Reshape(d.db[nx.Input[0]], d.db[nx.Input[1]].Shape())
	if err != nil {
		return fmt.Errorf("Cannot reshape: %v", err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
// Warning this operation is broadcastable
// See https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
//
// BUG(owulveryck): the broadcasting has to be implemented correctly in Gorgonia. see https://github.com/gorgonia/gorgonia/issues/223
func (d *Decoder) addOp(nx *onnx.NodeProto) error {
	b := d.db[nx.Input[1]]
	a := d.db[nx.Input[0]]
	/*
		bb, err := gorgonia.Reshape(b, a.Shape())
		if err != nil {
			return fmt.Errorf("Cannot Add %v and %v: %v", nx.Input[0], nx.Input[1], err)
		}
	*/
	n, err := gorgonia.AddBcast(a, b)
	if err != nil {
		return fmt.Errorf("Cannot Add %v and %v: %v", nx.Input[0], nx.Input[1], err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n

	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
func (d *Decoder) reluOp(nx *onnx.NodeProto) error {
	n, err := gorgonia.Rectify(d.db[nx.Input[0]])
	if err != nil {
		return err
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
func (d *Decoder) maxPoolOp(nx *onnx.NodeProto) error {

	var kernelShape tensor.Shape
	input := d.db[nx.Input[0]]
	var pad, stride []int
	for _, attr := range nx.Attribute {
		switch *attr.Name {
		case "kernel_shape":
			shape := make([]int, len(attr.Ints))
			for i, v := range attr.Ints {
				shape[i] = int(v)
			}
			kernelShape = tensor.Shape(shape)
		case "strides":
			stride = make([]int, len(attr.Ints))
			for i, v := range attr.Ints {
				stride[i] = int(v)
			}
		case "auto_pad":
			switch string(attr.S) {
			case "NOTSET":
			case "SAME_UPPER":
				return fmt.Errorf("auto_pad %v not implemented", string(attr.S))
			case "SAME_LOWER":
				return fmt.Errorf("auto_pad %v not implemented", string(attr.S))
			case "VALID":
				return fmt.Errorf("auto_pad %v not implemented", string(attr.S))
			default:
				return fmt.Errorf("Invalid auto_pad value: %v", string(attr.S))

			}
		case "pads":
			pad = make([]int, 2)
			for i := 0; i < 2; i++ {
				pad[i] = int(attr.Ints[i])
			}
		case "group":
		case "dilations":
		default:
			return fmt.Errorf("Unknown attribute: %v for convolution operator", attr.Name)
		}
	}
	n, err := gorgonia.MaxPool2D(input, kernelShape, pad, stride)
	if err != nil {
		return fmt.Errorf("Cannot apply Convolution operator: %v", err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n
	return nil

}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
//
// BUG(owulveryck): The Mul operator should be broadcastable too
func (d *Decoder) matMulOp(nx *onnx.NodeProto) error {
	n, err := gorgonia.Mul(d.db[nx.Input[0]], d.db[nx.Input[1]])
	if err != nil {
		return fmt.Errorf("Cannot Multiply: %v", err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n

	return nil

}
