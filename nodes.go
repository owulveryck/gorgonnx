package gorgonnx

import (
	"fmt"

	onnx "github.com/owulveryck/onnx-go"
)

func (gi *graph) processNode(nx *onnx.NodeProto) error {
	switch nType := *nx.OpType; nType {
	case "Add":
		return gi.addOp(nx)
	case "Conv":
		return gi.convOp(nx)
	case "Reshape":
		return gi.reshapeOp(nx)
	case "Relu":
		return gi.reluOp(nx)
	case "MaxPool":
		return gi.maxPoolOp(nx)
	case "MatMul":
		return gi.matMulOp(nx)
	default:
		return fmt.Errorf("Operation %v not yet implemented", nType)
	}
}
