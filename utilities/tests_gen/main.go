package main

import (
	"fmt"
	"io/ioutil"
	"log"

	onnx "github.com/owulveryck/onnx-go"
)

func main() {
	b, err := ioutil.ReadFile("model.onnx")
	if err != nil {
		log.Fatal(err)
	}
	model := new(onnx.ModelProto)
	err = model.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	if len(model.GetGraph().GetNode()) > 1 {
		log.Fatal("Not supported")
	}
	fmt.Println(`
	package operators

	// TestConv ...
	func TestConv(t *testing.T) {
     	assert := assert.New(t)

	  g := gorgonia.NewGraph() 
	  var op gorgonnx.Operator
	  `)

	node := model.GetGraph().GetNode()[0]
	inputNames := node.GetInput()
	for i, attr := range node.GetAttribute() {
		var value string
		fmt.Printf("attribute%vName :=%#v\n", i, attr.GetName())
		fmt.Printf("attribute%vType :=onnx.AttributeProto_AttributeType(%#v)\n", i, attr.GetType())
		switch attr.GetType() {
		case onnx.AttributeProto_UNDEFINED:
		case onnx.AttributeProto_FLOAT:
			fmt.Printf("attribute%vValue :=%v\n", i, attr.GetF())
			value = fmt.Sprintf("F: &attribute%vValue,\n", i)
		case onnx.AttributeProto_INT:
			fmt.Printf("attribute%vValue :=%v\n", i, attr.GetI())
			value = fmt.Sprintf("I: &attribute%vValue,\n", i)
		case onnx.AttributeProto_STRING:
			fmt.Printf("attribute%vValue :=%#v\n", i, attr.GetS())
			value = fmt.Sprintf("S: &attribute%vValue,\n", i)
		case onnx.AttributeProto_TENSOR:
		case onnx.AttributeProto_GRAPH:
		case onnx.AttributeProto_FLOATS:
			value = fmt.Sprintf("Floats: %#v,\n", attr.GetFloats())
		case onnx.AttributeProto_INTS:
			value = fmt.Sprintf("Ints: %#v,\n", attr.GetInts())
		case onnx.AttributeProto_STRINGS:
		case onnx.AttributeProto_TENSORS:
		case onnx.AttributeProto_GRAPHS:
		}
		fmt.Printf("attribute%v := &onnx.AttributeProto{\n", i)
		fmt.Printf("Name: &attribute%vName,\n", i)
		fmt.Printf("Type: &attribute%vType,\n", i)
		fmt.Println(value)
		fmt.Printf("}\n")
	}
	fmt.Printf("attributes := []*onnx.AttributeProto{\n")
	for i := range node.GetAttribute() {
		fmt.Printf("attribute%v,\n", i)
	}
	fmt.Println("}")
	opApply := "err := op.Apply(\n"
	inputApply := "[]*gorgonia.Node{\n"
	for i, inputName := range inputNames {
		// Open the tensorproto sample file
		filename := fmt.Sprintf("test_data_set_0/input_%v.pb", i)
		b, err = ioutil.ReadFile(filename)
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
		fmt.Printf("%v := gorgonia.NodeFromAny(g,\n tensor.New(\ntensor.WithShape%s,\ntensor.WithBacking(%#v)))\n", inputName, t.Shape(), t.Data())
		inputApply = fmt.Sprintf("%v%v,\n", inputApply, inputName)
	}
	opApply = fmt.Sprintf("%v%v},\n", opApply, inputApply)
	outputApply := "[]*gorgonia.Node{\n"
	for i, outputName := range node.GetOutput() {
		// Open the tensorproto sample file
		filename := fmt.Sprintf("test_data_set_0/output_%v.pb", i)
		b, err = ioutil.ReadFile(filename)
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
		fmt.Printf("%vT := tensor.New(\ntensor.WithShape%s,\ntensor.WithBacking(%#v))\n", outputName, t.Shape(), t.Data())
		fmt.Printf("%v := new(gorgonia.Node)\n", outputName)
		outputApply = fmt.Sprintf("%v%v,\n", outputApply, outputName)
	}
	opApply = fmt.Sprintf("%v%v},\n", opApply, outputApply)
	fmt.Printf("%v)\n", opApply)
	fmt.Printf(`
	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		f.Fatal(err)
	}
			assert.Equal(yT.Shape(), y.Shape(), "Tensors should be the same")
			assert.Equal(yT.Data(), y.Value().Data(), "Tensors should be the same")

	`)
	fmt.Println("}")

}
