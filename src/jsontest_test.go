package agentic

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/agentic-loop/src/internal/jsontest"
)

// Test fixtures are built from Go VALUES, never spliced into JSON text.
//
// Writing fmt.Sprintf(`{"sha":%q}`, sha) or `{"sha":"` + sha + `"}` makes a
// fixture's correctness depend on what the value happens to contain: a quote,
// a backslash or a newline in it produces a malformed body, and the test then
// fails somewhere far from the splice -- or worse, passes for the wrong reason
// because the code under test rejected a body the fixture never meant to send.
// Marshaling a Go value cannot do that; the encoder escapes.
//
// Static JSON with nothing interpolated into it is not this problem and stays
// as it is: no value can corrupt a constant.

// jsonMust / jsonObj / jsonArr are the names this package's tests use.
// The implementation lives in internal/jsontest.
var jsonMust = jsontest.Must

type jsonObj = jsontest.Obj
type jsonArr = jsontest.Arr

// sprintfJSON matches fmt.Sprintf over a template that opens as JSON.
var sprintfJSON = regexp.MustCompile("fmt\\.Sprintf\\(`\\s*[\\[{]")

// concatJSON matches a backtick string that both looks like JSON and is glued
// to an expression: `{"a":"` + v, or v + `"}`. A trailing `+ "\n"` on a whole
// document is not a splice, so a plain quoted string on the far side is
// allowed.
var concatJSON = regexp.MustCompile("`[^`]*[\\[{][^`]*`\\s*\\+\\s*[^\"\\s]|\\+\\s*`\\s*[,:}\\]]")

// The rule is a build failure rather than a convention, because this is a
// defect the source application already paid for.
func TestNoTestSplicesAValueIntoJSONText(t *testing.T) {
	var offenders []string
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == "build" {
				return fs.SkipDir
			}
			return nil
		}
		// This file quotes both shapes in order to describe them.
		if !strings.HasSuffix(path, "_test.go") || filepath.Base(path) == "jsontest_test.go" {
			return nil
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for i, line := range strings.Split(string(src), "\n") {
			if sprintfJSON.MatchString(line) || concatJSON.MatchString(line) {
				offenders = append(offenders, "  "+path+":"+strconv.Itoa(i+1)+": "+strings.TrimSpace(line))
			}
		}
		return nil
	})
	require.NoError(t, err)

	assert.Empty(t, offenders,
		"these build JSON by splicing a value into text, which breaks on a quote, a backslash or a newline in the value.\n"+
			"Build the fixture from a Go value instead:\n"+
			"\tjsontest.Must(jsontest.Obj{\"sha\": sha})\n"+
			"A JSON fragment that must stay raw (a deliberately wrong-typed value) is json.RawMessage(frag).\n"+
			"Offenders:\n%s", strings.Join(offenders, "\n"))
}
