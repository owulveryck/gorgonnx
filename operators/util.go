package operators

func int64ToInt(i64 []int64) []int {
	o := make([]int, len(i64))
	for i := 0; i < len(i64); i++ {
		o[i] = int(i64[i])
	}
	return o
}
