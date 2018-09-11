package gorgonnx

import (
	"fmt"
	"io/ioutil"
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"gonum.org/v1/gonum/graph/encoding/dot"
)

func TestGraph(t *testing.T) {
	b, err := ioutil.ReadFile("./mnist/model.onnx")
	if err != nil {
		t.Fatal(err)
	}
	model := new(onnx.ModelProto)
	err = model.Unmarshal(b)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGraph(model.GetGraph())
	if err != nil {
		t.Fatal("Cannot decode ", err)
	}
	b, err = dot.Marshal(g, "MNIST", "", "    ", false)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(string(b))
}
