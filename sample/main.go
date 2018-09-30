package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/owulveryck/gorgonnx"
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor/tensonnx"
)

func main() {
	modelFile := flag.String("model", "", "Path to the model path")
	inputFile := flag.String("input", "", "Path to the input file")
	flag.Parse()
	if *modelFile == "" || *inputFile == "" {
		flag.Usage()
		os.Exit(0)
	}
	b, err := ioutil.ReadFile(*modelFile)
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

	b, err = ioutil.ReadFile(*inputFile)
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
	gorgonia.Let(g.Inputs()[0], t)
	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	output := gorgonnx.GetOutputGraphNodes(g)
	for _, n := range output {
		fmt.Printf("%v: %v", n.Name(), n.Value())
	}
}
