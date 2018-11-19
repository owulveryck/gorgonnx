package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"sync"

	"github.com/owulveryck/gorgonnx"
	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/tracer"
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
	t, err := sampleTestData.Tensor()
	if err != nil {
		log.Fatal(err)
	}
	gorgonia.Let(g.Inputs()[0], t)
	//machine := gorgonia.NewLispMachine(g, gorgonia.ExecuteFwdOnly())
	// start the tracer
	var wg sync.WaitGroup
	go func() {
		wg.Add(1)
		defer wg.Done()
		log.Fatal(tracer.StartDebugger(g, ":8080"))
	}()
	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	output := gorgonnx.GetOutputGraphNodes(g)
	for _, n := range output {
		fmt.Printf("%v: %v", n.Name(), n.Value())
	}
	wg.Wait()
}
