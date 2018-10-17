#!/bin/ksh

export OP=$1
cat <<EOF
package operators

import (
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// $OP operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#$OP
type $OP struct {
	name string
}

// Init ...
func (o *$OP) Init(attrs []*onnx.AttributeProto) error {
	o.name = "$OP"
	return nil
}

// Apply ...
func (o *$OP) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      o.name,
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	return nil, nil
}
EOF
