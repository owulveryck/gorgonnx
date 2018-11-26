package value

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func dtypeOf(t hm.Type) (retVal tensor.Dtype, err error) {
	switch p := t.(type) {
	case tensor.Dtype:
		retVal = p
	case TensorType:
		return dtypeOf(p.Of)
	case hm.TypeVariable:
		err = errors.Errorf("instance %v does not have a dtype", p)
	default:
		err = errors.Errorf(nyiFail, "dtypeOf", p)
		return
	}

	return
}
