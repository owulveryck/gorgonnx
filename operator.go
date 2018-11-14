package gorgonnx

import (
	"github.com/davecgh/go-spew/spew"
	"github.com/owulveryck/gorgonnx/operators"
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// AvailableOperators is the list of the onnx operators available linked to their implementation
var AvailableOperators = map[string]Operator{
	"Conv":               &operators.Conv{},
	"Mul":                &operators.Mul{},
	"MatMul":             &operators.Matmul{},
	"Div":                &operators.Div{},
	"Add":                &operators.Add{},
	"Relu":               &operators.Relu{},
	"MaxPool":            &operators.Maxpool{},
	"Concat":             &operators.Concat{},
	"Constant":           &operators.Constant{},
	"Reshape":            &operators.Reshape{},
	"BatchNormalization": &operators.Batchnorm{},
	"Unsqueeze":          &operators.Unsqueeze{},
}

// Operator can be added to the computation graph
type Operator interface {
	Init([]*onnx.AttributeProto) error
	Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error)
}

func (cg *computationGraph) processNode(nx *onnx.NodeProto) error {
	spew.Dump(nx)
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
		if err != nil {
			return err
		}
		inputs[i] = input
	}

	o, err := op.Apply(inputs...)
	if err != nil {
		return err
	}
	/*
		if len(o) != len(nx.Output) {
			return fmt.Errorf("Bad number of output for op %v, expected %v, got %v", op, len(nx.Output), len(o))
		}
	*/
	for i := 0; i < len(nx.Output); i++ {
		err := cg.storeNode(nx.Output[i], o[i])
		if err != nil {
			return err
		}
	}
	return nil
}
