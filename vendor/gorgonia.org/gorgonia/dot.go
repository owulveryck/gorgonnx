package gorgonia

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"

	"gonum.org/v1/gonum/graph/encoding"
)

// DOTID is used for the graphviz output. It fulfils the gonum encoding interface
func (n *Node) DOTID() string {
	//	return strconv.Itoa(int(n.ID()))
	return fmt.Sprintf("Node_%p", n)
}

// Attributes is for graphviz output. It specifies the "label" of the node (a table)
func (n *Node) Attributes() []encoding.Attribute {
	return []encoding.Attribute{
		encoding.Attribute{
			Key:   "href",
			Value: fmt.Sprintf(`"/nodes/%p"`, n),
		},
		encoding.Attribute{
			Key:   "label",
			Value: n.dotLabel(),
		},
	}
}

type attributer []encoding.Attribute

func (a attributer) Attributes() []encoding.Attribute { return a }

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g *ExprGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "nodesep",
			Value: "1",
		},
		encoding.Attribute{
			Key:   "rankdir",
			Value: "TB",
		},
		encoding.Attribute{
			Key:   "ranksep",
			Value: `"1.5 equally"`,
		},
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "fontname",
			Value: "monospace",
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "none",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}

// ServeHTTP to get the value of the node via http
func (n *Node) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "%v", n.Value())
}

// StartDebugger runs a http webserver
func (g *ExprGraph) StartDebugger(addr string) {
	svg, err := generateSVG(g)
	if err != nil {
		panic(err)
	}
	handler := http.NewServeMux()
	for _, n := range g.AllNodes() {
		handler.Handle(fmt.Sprintf("/nodes/%p", n), n)
	}
	handler.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/svg+xml; charset=UTF-8")
		io.WriteString(w, string(svg))
	})

	err = http.ListenAndServe(addr, handler)

	if err != nil {
		log.Fatalf("Could not start server: %s\n", err.Error())
	}
}

func generateSVG(g *ExprGraph) ([]byte, error) {

	dotProcess := exec.Command("dot", "-Tsvg")

	// Set the stdin stdout and stderr of the dot subprocess
	stdinOfDotProcess, err := dotProcess.StdinPipe()
	if err != nil {
		return nil, err
	}
	defer stdinOfDotProcess.Close() // the doc says subProcess.Wait will close it, but I'm not sure, so I kept this line
	readCloser, err := dotProcess.StdoutPipe()
	if err != nil {
		return nil, err //replace with logger, or anything you want
	}
	dotProcess.Stderr = os.Stderr

	// Actually run the dot subprocess
	if err = dotProcess.Start(); err != nil { //Use start, not run
		return nil, err //replace with logger, or anything you want
	}
	/*
		b, err := dot.Marshal(g, "", "", "\t")
		if err != nil {
			return nil, err
		}
	*/
	fmt.Fprintf(stdinOfDotProcess, "%v", g.ToDot())
	stdinOfDotProcess.Close()
	// Read from stdout and store it in the correct structure
	var buf bytes.Buffer
	buf.ReadFrom(readCloser)

	dotProcess.Wait()
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// dotLabel a graphviz compatible label
func (n *Node) dotLabel() string {
	var buf bytes.Buffer
	if err := exprNodeTempl.ExecuteTemplate(&buf, "node", n); err != nil {
		panic(err)
	}

	return buf.String()
}
