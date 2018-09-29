package gorgonnx

import (
	"fmt"
	"io/ioutil"
	"log"

	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor/tensonnx"
)

func Example_mnist() {
	b, err := ioutil.ReadFile("./mnist/model.onnx")
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

	// Open the tensorproto sample file
	b, err = ioutil.ReadFile("./mnist/test_data_set_1/input_0.pb")
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
	fmt.Printf("%v", GetOutputGraphNodes(g)[0].Value().Data())

	// __Output: [5041.8887 -3568.878 -187.82423 -1685.797 -1183.3232 -614.42926 892.6643 -373.65845 -290.2623 -111.176216]

}
