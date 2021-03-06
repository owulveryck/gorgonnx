package main

const headerTmpl = `
package operators

import (
	"os"
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)
`

const testTmpl = `
// Test{{ .TestName }} is autogenerated from {{ .Filename }}
func Test{{ .TestName }}(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &{{ .Operator }}{}

	{{  range $key, $value := .Attributes }}
	attribute{{ $key }}Name := "{{ $value.Name }}"
	attribute{{ $key }}Type := onnx.AttributeProto_AttributeType({{ $value.Type }})
	{{ if $value.IsPointer }}attribute{{ $key }}Value := {{ $value.Value }}{{end}}
	attribute{{ $key }} := &onnx.AttributeProto{
		Name: &attribute{{ $key }}Name,
		Type: &attribute{{ $key }}Type,
		{{ $value.AssignableType }}: {{ if $value.IsPointer }}&attribute{{ $key }}Value{{ else }}{{ $value.Value }}{{ end }},
	}
	{{ end }}

	attributes := []*onnx.AttributeProto{
		{{  range $key, $value := .Attributes }}attribute{{ $key }},
		{{ end }}
	}

	if len(attributes) != 0 {
		err := op.Init(attributes)
		t.Logf("Info: operator %#v", op)
		if err != nil {
			_, ok := err.(*onnx.ErrNotImplemented)
			if ok && skip {
				t.Skip(err)
			}

			t.Fatal(err)
		}
	}
	{{ range .Inputs }}
	{{ .Name}} := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape{{ .Shape }},
			tensor.WithBacking({{ .Data }})),
			gorgonia.WithName("{{ .Name }}"))
	{{ end }}
	{{ range .Outputs }}
	{{ .Name }}T := tensor.New(
		tensor.WithShape{{ .Shape }},
		tensor.WithBacking({{ .Data }}))
	{{ .Name}} := new(gorgonia.Node)
	{{ end}} 
	o, err := op.Apply(
		{{ range .Inputs }}{{ .Name}},{{end}}
	)
	if err != nil {
		_, ok := err.(*onnx.ErrNotImplemented)
		if ok && skip {
			t.Skip(err)
		}
		_, ok = err.(*gorgonia.ErrNotImplemented)
		if ok && skip {
			t.Skip(err)
		}

		t.Fatal(err)
	}
	{{ range $key, $value := .Outputs }}
	{{ $value.Name}} = o[{{ $key }}]
	{{end}}

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	{{ range .Outputs }}
	assert.Equal({{ .Name }}T.Shape(), {{ .Name }}.Shape(), "Tensors should be the same")
	assert.InDeltaSlice({{ .Name }}T.Data(), {{ .Name }}.Value().Data(), 1e-5,"Tensors should be the same")
	{{end}}
}
`
