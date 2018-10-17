PKGS := $(shell go list ./... | grep -v /vendor)

.PHONY: test
test: 
	go test $(PKGS)

BIN_DIR := $(GOPATH)/bin

BINARY := mytool
VERSION ?= vlatest
PLATFORMS := windows linux darwin
os = $(word 1, $@)

.PHONY: $(PLATFORMS)
$(PLATFORMS): test
	mkdir -p release
	GOOS=$(os) GOARCH=amd64 go build -o release/$(BINARY)-$(VERSION)-$(os)-amd64 cmd/*.go

.PHONY: release
release: windows linux darwin
