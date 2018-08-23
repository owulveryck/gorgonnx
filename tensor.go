package gorgonnx

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/owulveryck/gorgonnx/onnx"
	"gorgonia.org/tensor"
)

// NewTensor returns a Gorgonia compatible tensor from a onnx.TensorProto structure
func NewTensor(tx *onnx.TensorProto) (tensor.Tensor, error) {
	// Get the data type
	dt, err := toDtype(tx.DataType)
	if err != nil {
		return nil, err
	}
	var size = make([]int, len(tx.Dims))
	for i := range tx.Dims {
		size[i] = int(tx.Dims[i])
	}
	opts := []tensor.ConsOpt{tensor.WithShape(size...), tensor.Of(dt)}
	switch dt {
	case tensor.Float32:
		if len(tx.FloatData) == 0 {
			buf := bytes.NewReader(tx.RawData)
			element := make([]byte, 4)
			var err error
			var backing []float32
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
			opts = append(opts, tensor.WithBacking(backing))
		} else {
			opts = append(opts, tensor.WithBacking(tx.FloatData))
		}
	case tensor.Float64:
		if len(tx.DoubleData) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		opts = append(opts, tensor.WithBacking(tx.DoubleData))
	case tensor.Int64:
		if len(tx.Int64Data) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
		opts = append(opts, tensor.WithBacking(tx.Int64Data))
	case tensor.Int32:
		if len(tx.Int32Data) == 0 {
			return nil, fmt.Errorf("No data found. Maybe a raw data but not yet implemented")
		}
	default:
		return nil, fmt.Errorf("Backend not yet implemented")

	}

	return tensor.New(opts...), nil
}
