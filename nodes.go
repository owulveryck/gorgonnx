package gorgonnx

import (
	"errors"

	onnx "github.com/owulveryck/onnx-go"
)

// ErrOpNotImplemented ...
var ErrOpNotImplemented = errors.New("Operation not implemented")

func (cg *computationGraph) processNode(nx *onnx.NodeProto) error {
	switch nType := *nx.OpType; nType {
	case "Add":
		return cg.addOp(nx)
	case "Conv":
		return cg.convOp(nx)
	case "Reshape":
		return cg.reshapeOp(nx)
	case "Relu":
		return cg.reluOp(nx)
	case "MaxPool":
		return cg.maxPoolOp(nx)
	case "MatMul":
		return cg.matMulOp(nx)
	default:
		return ErrOpNotImplemented
	}
}
