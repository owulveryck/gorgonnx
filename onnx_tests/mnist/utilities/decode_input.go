package main

import (
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"os"

	onnx "github.com/owulveryck/onnx-go"
)

// This little utility reads an encoded tensor and draw a picture
func main() {
	// Read the input
	b, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	sampleTestData := new(onnx.TensorProto)
	err = sampleTestData.Unmarshal(b)
	if err != nil {
		panic(err)
	}
	t, err := sampleTestData.Tensor()
	if err != nil {
		panic(err)
	}
	width := t.Shape()[2]
	height := t.Shape()[3]
	im := image.NewGray(image.Rectangle{Max: image.Point{X: width, Y: height}})
	for w := 0; w < width; w++ {
		for h := 0; h < height; h++ {
			v, err := t.At(0, 0, w, h)
			if err != nil {
				panic(err)
			}
			im.Set(w, h, color.Gray{uint8(v.(float32))})
		}
	}
	enc := png.Encoder{}
	err = enc.Encode(os.Stdout, im)
	if err != nil {
		panic(err)
	}
	// Read the output...
	b, err = ioutil.ReadFile(os.Args[2])
	if err != nil {
		panic(err)
	}
	sampleTestData = new(onnx.TensorProto)
	err = sampleTestData.Unmarshal(b)
	if err != nil {
		panic(err)
	}
	t, err = sampleTestData.Tensor()
	if err != nil {
		panic(err)
	}

	log.Println("Expected outout:", t)

}
