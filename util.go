package gorgonnx

func toIntSlice(d []int64) []int {
	output := make([]int, len(d))
	for i, v := range d {
		output[i] = int(v)
	}
	return output
}
