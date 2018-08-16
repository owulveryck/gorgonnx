package gorgonnx

import (
	"github.com/onnx/onnx"
	"gorgonia.org/gorgonia"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
func (d *Decoder) conv(n *onnx.NodeProto) (*[]gorgonia.Node, error) {
	return nil, nil
}
