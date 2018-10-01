package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"
)

type GeneratedFile struct {
	TestName string
	FileName string
}

func main() {
	basedir := "test_data/"
	files, err := filepath.Glob(basedir + "/*")
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range files {
		filename := strings.Replace(file, basedir, "", -1)

		filename = strings.Replace(filename, "test_", "", -1)

		testname := ""

		if strings.Index(filename, "_") < 0 {
			testname = "Test" + strings.Title(filename) + "Op"
		} else {
			newtestname := strings.Split(filename, "_")
			testname = "Test" + strings.Title(newtestname[0]) + "Op"
			for i, _ := range newtestname {
				if i > 0 {
					testname = testname + "_" + newtestname[i]
				}
			}
		}

		filename = filename + "_test.go"

		genfile := GeneratedFile{testname, file + "/"}

		f, err := os.Create(filename)

		if err != nil {
			log.Fatal("Cannot create file", err)
		}

		defer f.Close()

		tmpl, err := template.New("template_test.go.tmpl").ParseFiles("template_test.go.tmpl")
		if err != nil {
			panic(err)
		}
		err = tmpl.Execute(f, genfile)
		f.Sync()
		if err != nil {
			panic(err)
		}

		fmt.Println(filename)

		cmd := exec.Command("gofmt", "-w", filename)
		cmd.Run()

	}
}
