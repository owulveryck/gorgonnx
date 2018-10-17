package gorgonnx

import (
	"errors"
	"fmt"

	"github.com/owulveryck/gorgonnx/operators"
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// AvailableOperators is the list of the onnx operators available linked to their implementation
var AvailableOperators = map[string]Operator{
	"Conv":     &operators.Conv{},
	"Mul":      &operators.Mul{},
	"Matmul":   &operators.Matmul{},
	"Div":      &operators.Div{},
	"Add":      &operators.Add{},
	"Relu":     &operators.Relu{},
	"Maxpool":  &operators.Maxpool{},
	"Concat":   &operators.Concat{},
	"Constant": &operators.Constant{},
	"Reshape":  &operators.Reshape{},
}

// Operator can be added to the computation graph
type Operator interface {
	Init([]*onnx.AttributeProto) error
	Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error)
}

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
		outputs[i] = &gorgonia.Node{}
		err := cg.storeNode(nx.Output[i], outputs[i])
		if err != nil {
			return err
		}
	}
	o, err := op.Apply(inputs...)
	if err != nil {
		return err
	}
	if len(o) != len(nx.Output) {
		return errors.New("Bad number of output")
	}
	for i := range o {
		outputs[i] = o[i]
	}
	for _, o := range outputs {
		if o == nil {
			return fmt.Errorf("gorgonnx: Wrong number of outputs for operator %v (expected %v)", *nx.OpType, len(outputs))
		}
	}
	return nil
}
