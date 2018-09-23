package gorgonnx

import "fmt"

// ErrToBeImplemented is a special error type used in the tests
type ErrToBeImplemented struct {
	Operator       string
	AttributeName  string
	AttributeValue interface{}
}

func (e ErrToBeImplemented) Error() string {
	return fmt.Sprintf("Attribute %v with value %v not yet implemented for operator %v", e.AttributeName, e.AttributeValue, e.Operator)
}
