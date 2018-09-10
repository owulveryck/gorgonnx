package main

import (
	"io/ioutil"
	"log"
	"os"

	"github.com/owulveryck/gorgonnx"
	onnx "github.com/owulveryck/onnx-go"
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

	b, err = ioutil.ReadFile("../mnist/test_data_set_0/input_0.pb")
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
	logger := log.New(os.Stdout, "", 0)
	machine := gorgonia.NewTapeMachine(g, gorgonia.WithLogger(logger), gorgonia.WithWatchlist())
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	output := gorgonnx.GetOutputGraphNodes(g)
	for _, n := range output {
		log.Printf("%v: %v", n.Name(), n.Value())
	}
}
