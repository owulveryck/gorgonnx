package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"strings"

	"github.com/alecthomas/template"
	onnx "github.com/owulveryck/onnx-go"
)

func main() {
	testdir := flag.String("testpath", ".", "path to the onnx test directory")
	op := flag.String("op", "", "the operator who needs tests")
	flag.Parse()
	if *op == "" {
		flag.Usage()
		os.Exit(0)
	}
	// locate all the directories with the pattern test_op_...
	files, err := ioutil.ReadDir(*testdir)
	if err != nil {
		log.Fatal(err)
	}
	re := regexp.MustCompile("^test_" + *op + "(_*)(.*)")
	for _, file := range files {
		if !file.IsDir() {
			continue
		}
		elements := re.FindAllStringSubmatch(file.Name(), -1)
		if len(elements) == 0 {
			continue
		}
		b, err := ioutil.ReadFile(*testdir + file.Name() + "/model.onnx")
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
		node := model.GetGraph().GetNode()[0]
		attributes := make([]attribute, len(node.GetAttribute()))
		for i, attr := range node.GetAttribute() {
			attributes[i].Name = attr.GetName()
			attributes[i].Type = fmt.Sprintf("%#v", attr.GetType())
			switch attr.GetType() {
			case onnx.AttributeProto_UNDEFINED:
			case onnx.AttributeProto_FLOAT:
				attributes[i].Value = fmt.Sprintf("%v", attr.GetF())
				attributes[i].AssignableType = "F"
				attributes[i].IsPointer = true
			case onnx.AttributeProto_INT:
				attributes[i].Value = fmt.Sprintf("%v", attr.GetI())
				attributes[i].AssignableType = "I"
				attributes[i].IsPointer = true
			case onnx.AttributeProto_STRING:
				attributes[i].Value = fmt.Sprintf("%v", attr.GetS())
				attributes[i].AssignableType = "S"
				attributes[i].IsPointer = true
			case onnx.AttributeProto_TENSOR:
			case onnx.AttributeProto_GRAPH:
			case onnx.AttributeProto_FLOATS:
				attributes[i].Value = fmt.Sprintf("%#v", attr.GetFloats())
				attributes[i].AssignableType = "Floats"
				attributes[i].IsPointer = false
			case onnx.AttributeProto_INTS:
				attributes[i].Value = fmt.Sprintf("%#v", attr.GetInts())
				attributes[i].AssignableType = "Ints"
				attributes[i].IsPointer = false
			case onnx.AttributeProto_STRINGS:
			case onnx.AttributeProto_TENSORS:
			case onnx.AttributeProto_GRAPHS:
			}
		}
		inputs := make([]io, len(node.GetInput()))
		for i, inputName := range node.GetInput() {
			// Open the tensorproto sample file
			filename := fmt.Sprintf("%v%v/test_data_set_0/input_%v.pb", *testdir, file.Name(), i)
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
			inputs[i].Name = inputName
			inputs[i].Shape = fmt.Sprintf("%v", t.Shape())
			inputs[i].Data = fmt.Sprintf("%#v", t.Data())
		}
		outputs := make([]io, len(node.GetOutput()))
		for i, outputName := range node.GetOutput() {
			// Open the tensorproto sample file
			filename := fmt.Sprintf("%v%v/test_data_set_0/output_%v.pb", *testdir, file.Name(), i)
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
			outputs[i].Name = outputName
			outputs[i].Shape = fmt.Sprintf("%v", t.Shape())
			outputs[i].Data = fmt.Sprintf("%#v", t.Data())
		}
		testCase := unitTest{
			Filename:   file.Name(),
			TestName:   strings.Title(*op),
			Operator:   strings.Title(*op),
			Attributes: attributes,
			Inputs:     inputs,
			Outputs:    outputs,
		}
		// Create a new template and parse the letter into it.
		t := template.Must(template.New("unitTest").Parse(testTmpl))
		err = t.Execute(os.Stdout, testCase)
		if err != nil {
			log.Println("executing template:", err)
		}
	}
}
