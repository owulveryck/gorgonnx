package operators

import "fmt"

// ErrBadArity is raised when an operator do not have the correct amount
// of input or output
type ErrBadArity struct {
	Operator       string
	ExpectedInput  int
	ActualInput    int
	ExpectedOutput int
	ActualOutput   int
}

func (e *ErrBadArity) Error() string {
	var msg string
	msg = "err arity: operator: " + e.Operator
	if e.ExpectedInput != 0 {
		msg = fmt.Sprintf("%v. Expected %v input nodes and got %v ", msg, e.ExpectedInput, e.ActualInput)
	}
	if e.ExpectedOutput != 0 {
		msg = fmt.Sprintf("%v. Expected %v output nodes and got %v ", msg, e.ExpectedOutput, e.ActualOutput)
	}
	return msg
}
