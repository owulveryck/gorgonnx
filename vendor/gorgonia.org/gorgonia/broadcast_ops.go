package gorgonia

// AddBroadcast to see how it works
func AddBroadcast(a, b *Node) (*Node, error) {

	switch {
	case len(a.Shape()) < len(b.Shape()):
		dimsToAdd := len(b.Shape()) - len(a.Shape())
		dims := make([]int, dimsToAdd)
		for i := range dims {
			dims[i] = 1
		}
		x, err := Reshape(a, append(dims, a.Shape()...))
		if err != nil {
			return nil, err
		}
		// Now broadcast
		var left, right []byte
		for i := range a.Shape() {
			if a.Shape()[i] == 1 && x.Shape()[i] != 1 {
				left = append(left, byte(i))
			}
			if a.Shape()[i] != 1 && x.Shape()[i] == 1 {
				right = append(right, byte(i))
			}

		}
		// Now broadcast
		return Broadcast(addOpType, x, b, NewBroadcastPattern(left, right))
	case len(a.Shape()) > len(b.Shape()):
		dimsToAdd := len(a.Shape()) - len(b.Shape())
		dims := make([]int, dimsToAdd)
		for i := range dims {
			dims[i] = 1
		}
		x, err := Reshape(b, append(dims, b.Shape()...))
		if err != nil {
			return nil, err
		}
		var left, right []byte
		for i := range a.Shape() {
			if a.Shape()[i] == 1 && x.Shape()[i] != 1 {
				left = append(left, byte(i))
			}
			if a.Shape()[i] != 1 && x.Shape()[i] == 1 {
				right = append(right, byte(i))
			}

		}
		// Now broadcast
		return Broadcast(addOpType, a, x, NewBroadcastPattern(left, right))
	default:
		return Add(a, b)
	}
}

// HadamardProdBroadcast to see how it works
func HadamardProdBroadcast(a, b *Node) (*Node, error) {

	switch {
	case len(a.Shape()) < len(b.Shape()):
		dimsToAdd := len(b.Shape()) - len(a.Shape())
		dims := make([]int, dimsToAdd)
		for i := range dims {
			dims[i] = 1
		}
		x, err := Reshape(a, append(dims, a.Shape()...))
		if err != nil {
			return nil, err
		}
		// Now broadcast
		var left, right []byte
		for i := range a.Shape() {
			if a.Shape()[i] == 1 && x.Shape()[i] != 1 {
				left = append(left, byte(i))
			}
			if a.Shape()[i] != 1 && x.Shape()[i] == 1 {
				right = append(right, byte(i))
			}

		}
		// Now broadcast
		return Broadcast(mulOpType, x, b, NewBroadcastPattern(left, right))
	case len(a.Shape()) > len(b.Shape()):
		dimsToAdd := len(a.Shape()) - len(b.Shape())
		dims := make([]int, dimsToAdd)
		for i := range dims {
			dims[i] = 1
		}
		x, err := Reshape(b, append(dims, b.Shape()...))
		if err != nil {
			return nil, err
		}
		var left, right []byte
		for i := range a.Shape() {
			if a.Shape()[i] == 1 && x.Shape()[i] != 1 {
				left = append(left, byte(i))
			}
			if a.Shape()[i] != 1 && x.Shape()[i] == 1 {
				right = append(right, byte(i))
			}

		}
		// Now broadcast
		return Broadcast(mulOpType, a, x, NewBroadcastPattern(left, right))
	default:
		return HadamardProdBroadcast(a, b)
	}
}
