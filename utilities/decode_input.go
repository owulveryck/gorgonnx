package main

import (
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"os"

	"github.com/owulveryck/gorgonnx"
	"github.com/owulveryck/gorgonnx/onnx"
)

// This little utility reads an encoded tensor and draw a picture
func main() {
	b, err := ioutil.ReadFile(os.Args[1])
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
	width := t.Shape()[2]
	height := t.Shape()[3]
	im := image.NewGray(image.Rectangle{Max: image.Point{X: width, Y: height}})
	for w := 0; w < width; w++ {
		for h := 0; h < height; h++ {
			v, err := t.At(0, 0, w, h)
			if err != nil {
				log.Fatal(err)
			}
			im.Set(w, h, color.Gray{uint8(v.(float32))})
		}
	}
	enc := png.Encoder{}
	err = enc.Encode(os.Stdout, im)
	if err != nil {
		log.Fatal(err)
	}

}
