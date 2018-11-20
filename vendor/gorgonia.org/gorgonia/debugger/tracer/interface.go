package tracer

import "gonum.org/v1/gonum/graph/encoding/dot"

// DotSubgrapher is any type that can represent itself as a subgraph
type DotSubgrapher interface {
	dot.Graph
	dot.Attributers
}
