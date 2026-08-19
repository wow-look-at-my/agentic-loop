// Command cai talks to a model from a shell.
//
// It exposes no XML at all. That is the point: the format is for programs, and
// a shell is the worst place to build one -- a prompt with a quote in it, a
// path with an ampersand, an image someone pasted, and a hand-assembled
// document is malformed in a way nothing catches until an upstream rejects it.
// So the CLI takes flags and files, builds the document itself, and prints the
// answer as text.
package main

import (
	"fmt"
	"os"

	"github.com/wow-look-at-my/agentic-loop/cli"
)

func main() {
	if err := cli.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "cai:", err)
		os.Exit(1)
	}
}
