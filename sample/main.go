package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/owulveryck/gorgonnx"
	"github.com/owulveryck/gorgonnx/onnx"
	"gorgonia.org/gorgonia"
)

func main() {
	b, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	model := new(onnx.ModelProto)
	err = model.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	gx := model.GetGraph()
	dec := gorgonnx.NewDecoder()
	g, err := dec.Decode(gx)
	if err != nil {
		log.Println(g)
		log.Fatal("Cannot decode ", err)
	}
	// Open the tensorproto sample file

	b, err = ioutil.ReadFile(os.Args[2])
	if err != nil {
		log.Fatal(err)
	}
	sampleTestData := new(onnx.TensorProto)
	err = sampleTestData.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	t, err := gorgonnx.Tensorize(sampleTestData)
	if err != nil {
		log.Fatal(err)
	}
	for _, n := range g.Inputs() {
		if n.Name() == "Input3" {
			gorgonia.Let(n, t)
		}
	}
	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	fmt.Println(t)

}
