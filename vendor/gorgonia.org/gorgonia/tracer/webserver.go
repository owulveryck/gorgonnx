package tracer

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"

	"github.com/rakyll/statik/fs"
	"gonum.org/v1/gonum/graph/encoding/dot"

	"gorgonia.org/gorgonia"
	_ "gorgonia.org/gorgonia/tracer/htdocs/statik" // Initialize the FS for static files
)

// StartDebugger runs a http webserver
func StartDebugger(g *gorgonia.ExprGraph, listenAddress string) error {
	statikFS, err := fs.New()
	if err != nil {
		return err
	}

	dg := generateDotGraph(g)
	b, err := dot.Marshal(dg, "", "    ", "     ")
	if err != nil {
		return err
	}
	svg, err := generateSVG(b)
	if err != nil {
		return err
	}
	handler := http.NewServeMux()
	/*
		for _, n := range g.AllNodes() {
			handler.Handle(fmt.Sprintf("/nodes/%p", n), n)
		}
	*/
	handler.HandleFunc("/graph.dot", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=UTF-8")
		io.Copy(w, bytes.NewReader(b))
	})
	handler.HandleFunc("/graph.svg", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/svg+xml; charset=UTF-8")
		io.WriteString(w, string(svg))
	})
	http.Handle("/", http.FileServer(statikFS))

	return http.ListenAndServe(listenAddress, handler)
}

func generateSVG(b []byte) ([]byte, error) {

	dotProcess := exec.Command("dot", "-Tsvg")

	// Set the stdin stdout and stderr of the dot subprocess
	stdinOfDotProcess, err := dotProcess.StdinPipe()
	if err != nil {
		return nil, err
	}
	defer stdinOfDotProcess.Close() // the doc says subProcess.Wait will close it, but I'm not sure, so I kept this line
	readCloser, err := dotProcess.StdoutPipe()
	if err != nil {
		return nil, err
	}
	dotProcess.Stderr = os.Stderr

	// Actually run the dot subprocess
	if err = dotProcess.Start(); err != nil { //Use start, not run
		return nil, err //replace with logger, or anything you want
	}
	fmt.Fprintf(stdinOfDotProcess, "%v", string(b))
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
