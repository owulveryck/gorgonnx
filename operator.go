package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/gorgonnx/operators"
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

func (cg *computationGraph) processNode(nx *onnx.NodeProto) error {
	op, ok := AvailableOperators[*nx.OpType]
	if !ok {
		return ErrNotImplemented{
			Operator: *nx.OpType,
		}
	}
	err := op.Init(nx.Attribute)
	if err != nil {
		return err
	}
	inputs := make([]*gorgonia.Node, len(nx.Input))
	for i := 0; i < len(nx.Input); i++ {
		input, err := cg.loadNode(nx.Input[i])
		inputs[i] = input
		if err != nil {
			return err
		}
	}
	outputs := make([]*gorgonia.Node, len(nx.Output))
	for i := 0; i < len(nx.Output); i++ {
		err := cg.storeNode(nx.Output[i], outputs[i])
		if err != nil {
			return err
		}
	}
	err = op.Apply(inputs, outputs)
	if err != nil {
		return err
	}
	for _, o := range outputs {
		if o == nil {
			return fmt.Errorf("gorgonnx: Wrong number of outputs for operator %v (expected %v)", *nx.OpType, len(outputs))
		}
	}
	return nil
}

// AvailableOperators is the list of the onnx operators available linked to their implementation
var AvailableOperators = map[string]Operator{
	"conv": &operators.Conv{},
}

// Operator can be added to the computation graph
type Operator interface {
	Init([]*onnx.AttributeProto) error
	Apply(input []*gorgonia.Node, output []*gorgonia.Node) error
}
