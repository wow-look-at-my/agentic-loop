package loop

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

	"github.com/wow-look-at-my/agentic-loop/internal/jsontest"
)

// Test fixtures are built from Go VALUES, never spliced into JSON text; the encoder escapes.

// jsonMust / jsonObj / jsonArr are the names this package's tests use.
var jsonMust = jsontest.Must

type jsonObj = jsontest.Obj
type jsonArr = jsontest.Arr

// sprintfJSON matches fmt.Sprintf over a template that opens as JSON.
var sprintfJSON = regexp.MustCompile("fmt\\.Sprintf\\(`\\s*[\\[{]")

// concatJSON matches a backtick string glued to an expression (a JSON splice).
var concatJSON = regexp.MustCompile("`[^`]*[\\[{][^`]*`\\s*\\+\\s*[^\"\\s]|\\+\\s*`\\s*[,:}\\]]")

// The rule is a build failure rather than a convention, because this is a
// defect the source application already paid for.
func TestNoTestSplicesAValueIntoJSONText(t *testing.T) {
	var offenders []string
	err := filepath.WalkDir(repoRoot(t), func(path string, d fs.DirEntry, err error) error {
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
