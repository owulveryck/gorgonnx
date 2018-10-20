package main

type io struct {
	Name  string
	Shape string
	Data  string
}

type attribute struct {
	VarName        string
	Name           string
	Type           string
	AssignableType string
	Value          string
	IsPointer      bool
}

type unitTest struct {
	Filename   string
	TestName   string
	Operator   string
	Attributes []attribute
	Inputs     []io
	Outputs    []io
}
