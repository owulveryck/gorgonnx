package gorgonnx

import (
	"fmt"

	onnx "github.com/owulveryck/onnx-go"
)

// ErrNotImplemented is a special error type used in the tests
type ErrNotImplemented struct {
	Operator  string
	Attribute *onnx.AttributeProto
	Err       error
}

func (e ErrNotImplemented) Error() string {
	if e.Attribute != nil {
		return fmt.Sprintf("gorgonnx: operator %v not fully implemented. Error while processing attribute %v - %v", e.Operator, e.Attribute, e.Err)
	}
	return "gorgonnx: operator " + e.Operator + " not implemented"
}
