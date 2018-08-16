package main

import (
	"fmt"
	"io/ioutil"
	"log"

	"github.com/onnx/onnx"
	"github.com/owulveryck/gorgonnx"
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
	gx := model.GetGraph()
	dec := gorgonnx.NewDecoder()
	g, err := dec.Decode(gx)
	if err != nil {
		log.Println(g)
		log.Fatal("Cannot decode ", err)
	}
	// Do something with g...
	fmt.Println(g)
}
