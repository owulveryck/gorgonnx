// This file is auto-generated
package gorgonnx_test

import (
	"io/ioutil"
	"regexp"
	"testing"

	"github.com/owulveryck/gorgonnx"
	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor/tensonnx"
)

func TestSimpleRnnDefaults(t *testing.T) {
	t.Skip()
	assert := assert.New(t)

	onnxTest := "./test_data/test_simple_rnn_defaults/"
	b, err := ioutil.ReadFile(onnxTest + "model.onnx")
	if err != nil {
		t.Fatal(err)
	}
	model := new(onnx.ModelProto)
	err = model.Unmarshal(b)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gorgonnx.NewGraph(model.GetGraph())
	if err != nil {
		t.Fatal("Cannot decode ", err)
	}

	// Open the tensorproto sample file
	files, err := ioutil.ReadDir(onnxTest)
	if err != nil {
		t.Fatal(err)
	}
	outputs := make([]*onnx.TensorProto, 0)
	for _, dir := range files {
		if dir.IsDir() {
			files, err := ioutil.ReadDir(onnxTest + dir.Name())
			if err != nil {
				t.Fatal(err)
			}
			for _, file := range files {
				matched, err := regexp.MatchString("input.*pb", file.Name())
				if err != nil {
					t.Fatal(err)
				}
				if matched {
					b, err = ioutil.ReadFile(onnxTest + dir.Name() + "/" + file.Name())
					if err != nil {
						t.Fatal(err)
					}
					sampleTestData := new(onnx.TensorProto)
					err = sampleTestData.Unmarshal(b)
					if err != nil {
						t.Fatal(err)
					}
					tens, err := tensonnx.NewTensor(sampleTestData)
					if err != nil {
						t.Fatal(err)
					}
					gorgonia.Let(g.ByName(*sampleTestData.Name)[0], tens)

				}
				matched, err = regexp.MatchString("output.*pb", file.Name())
				if err != nil {
					t.Fatal(err)
				}
				if matched {
					b, err = ioutil.ReadFile(onnxTest + dir.Name() + "/" + file.Name())
					if err != nil {
						t.Fatal(err)
					}
					sampleTestData := new(onnx.TensorProto)
					err = sampleTestData.Unmarshal(b)
					if err != nil {
						t.Fatal(err)
					}
					outputs = append(outputs, sampleTestData)
				}
			}

		}
	}
	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	if len(gorgonnx.GetOutputGraphNodes(g)) != len(outputs) {
		t.Fatal("Bad number of output")
	}
	if len(gorgonnx.GetOutputGraphNodes(g)) == 1 {
		computedOutput := gorgonnx.GetOutputGraphNodes(g)[0]
		//expectedOutput := outputs[0]
		expectedOutput, err := tensonnx.NewTensor(outputs[0])
		if err != nil {
			t.Fatal(err)
		}
		if len(computedOutput.Shape()) != len(expectedOutput.Shape()) {
			t.Fatalf("Different shape: expected %v, got %v", expectedOutput.Shape(), computedOutput.Shape())
		}
		for i := range expectedOutput.Shape() {
			if expectedOutput.Shape()[i] != computedOutput.Shape()[i] {
				t.Fatalf("Different shape: expected %v, got %v", expectedOutput.Shape(), computedOutput.Shape())
			}
		}
		assert.Equal(expectedOutput.Data(), computedOutput.Value().Data(), "Tensors should be the same")
	} else {
		t.Fail()
	}
}
