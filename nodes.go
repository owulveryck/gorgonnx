package gorgonnx

import (
	onnx "github.com/owulveryck/onnx-go"
)

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
	case "AveragePool":
		return cg.maxPoolOp(nx)
	case "MaxPool":
		return cg.maxPoolOp(nx)
	case "MatMul":
		return cg.matMulOp(nx)
	case "Concat":
		return cg.concatOp(nx)
	case "Dropout":
		return cg.dropoutOp(nx)
	default:
		return ErrToBeImplemented{
			nType,
			"",
			nil,
		}
	}
}
