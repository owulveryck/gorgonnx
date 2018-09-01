package gorgonia

// AddBroadcast to see how it works
func AddBroadcast(a, b *Node) (*Node, error) {

	switch {
	// Check if any of the tensor needs reshaping
	case len(a.Shape()) == 3 && len(b.Shape()) == 4:
		// Add a new dimension to the second operator
		x, err := Reshape(a, append([]int{1}, a.Shape()...))
		if err != nil {
			return nil, err
		}
		// Now broadcast
		//return Broadcast(addOpType, y, b, NewBroadcastPattern([]byte{1}, []byte{0, 0, 0, 0}))
		return Broadcast(addOpType, x, b, NewBroadcastPattern([]byte{0, 2, 3}, nil))
	case len(a.Shape()) == 4 && len(b.Shape()) == 3:
		// Add a new dimension to the second operator
		newShape := make([]int, len(a.Shape()))
		newShape[0] = 1
		newShape[1] = b.Shape()[0]
		newShape[2] = b.Shape()[1]
		newShape[3] = b.Shape()[2]
		y, err := Reshape(b, newShape)
		if err != nil {
			return nil, err
		}
		// Now broadcast
		return Broadcast(addOpType, a, y, NewBroadcastPattern(nil, []byte{0, 2, 3}))
	default:
		return Add(a, b)
	}
}
