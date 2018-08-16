package gorgonnx

import (
	"fmt"

	"github.com/onnx/onnx"
)

func (d *Decoder) processNode(n *onnx.NodeProto) error {
	op, ok := operators[*n.OpType]
	if !ok {
		return fmt.Errorf("Operation %v not yet implemented", *n.OpType)
	}
	return op(n)
}
