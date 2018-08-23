package gorgonnx

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"

	"github.com/owulveryck/gorgonnx/onnx"
	"gorgonia.org/tensor"
)

// Tensorize a TensorProto into a tensor.Tensor
func Tensorize(tx *onnx.TensorProto) (tensor.Tensor, error) {
	var t tensor.Tensor
	// Get the datatype
	dt, err := toDtype(tx.DataType)
	if err != nil {
		return nil, err
	}
	switch {
	case tx.RawData != nil:
		switch dt {
		case tensor.Float32:
			var backing []float32
			buf := bytes.NewReader(tx.RawData)
			element := make([]byte, 4)
			var err error
			for {
				var n int
				n, err = buf.Read(element)
				if err != nil || n != 4 {
					break
				}
				uintElement := binary.LittleEndian.Uint32(element)
				backing = append(backing, math.Float32frombits(uintElement))
			}
			if err != io.EOF {
				return nil, err
			}
			t = tensor.New(tensor.WithShape(toIntSlice(tx.Dims)...), tensor.WithBacking(backing))
		default:
			return nil, errors.New("type not yet implemented for rawdata")
		}
	default:
		return nil, errors.New("No data found or not implemented")
	}
	return t, nil
}
