package main

import (
	"io/ioutil"
	"log"

	"github.com/owulveryck/gorgonnx"
	onnx "github.com/owulveryck/onnx/go"
	"github.com/y0ssar1an/q"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor/tensonnx"
)

func main() {
	b, err := ioutil.ReadFile("../mnist/model.onnx")
	if err != nil {
		log.Fatal(err)
	}
	model := new(onnx.ModelProto)
	err = model.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	g, err := gorgonnx.NewGraph(model.GetGraph())
	if err != nil {
		log.Fatal("Cannot decode ", err)
	}

	// Open the tensorproto sample file

	b, err = ioutil.ReadFile("../mnist/test_data_set_1/input_0.pb")
	if err != nil {
		log.Fatal(err)
	}
	sampleTestData := new(onnx.TensorProto)
	err = sampleTestData.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	t, err := tensonnx.NewTensor(sampleTestData)
	if err != nil {
		log.Fatal(err)
	}
	gorgonia.Let(g.ByName("Input3")[0], t)
	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	output := gorgonnx.GetOutputGraphNodes(g)
	for _, n := range output {
		log.Printf("%v: %v", n.Name(), n.Value())
	}
	for _, n := range g.AllNodes() {
		q.Q(n.Name(), n.Value())
	}
}
