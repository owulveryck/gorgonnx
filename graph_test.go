package gorgonnx

import (
	"fmt"
	"io/ioutil"
	"log"

	onnx "github.com/owulveryck/onnx-go"
)

func ExampleDecoder() {
	b, err := ioutil.ReadFile("mnist/model.onnx")
	if err != nil {
		log.Fatal(err)
	}
	model := new(onnx.ModelProto)
	err = model.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	g, err := NewGraph(model.GetGraph())
	if err != nil {
		log.Fatal("Cannot decode ", err)
	}
	// Do something with g...
	fmt.Println(g.ToDot())
}
