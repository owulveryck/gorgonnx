package gorgonnx

import (
	"fmt"
	"io/ioutil"
	"log"

	"github.com/owulveryck/gorgonnx/onnx"
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
	gx := model.GetGraph()
	dec := NewDecoder()
	g, err := dec.Decode(gx)
	if err != nil {
		log.Fatal("Cannot decode ", err)
	}
	// Do something with g...
	fmt.Println(g.ToDot())
}
