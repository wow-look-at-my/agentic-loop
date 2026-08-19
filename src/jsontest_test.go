package agentic

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

// jsonMust marshals v to compact JSON. It panics rather than returning an
// error: the input is a literal written in a test, so a failure is a mistake
// in the test itself, and a fixture that silently became "" would make the
// assertion that reads it meaningless.
func jsonMust(v any) string {
	raw, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("jsonMust: marshaling %T: %v", v, err))
	}
	return string(raw)
}

// jsonObj is a JSON object literal, shorter than map[string]any at every call
// site: jsonMust(jsonObj{"sha": sha, "type": "file"}).
type jsonObj = map[string]any

// jsonArr is a JSON array literal.
type jsonArr = []any

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
	entries, err := os.ReadDir(".")
	require.NoError(t, err)

	var offenders []string
	for _, entry := range entries {
		name := entry.Name()
		// This file quotes both shapes in order to describe them.
		if entry.IsDir() || !strings.HasSuffix(name, "_test.go") || name == "jsontest_test.go" {
			continue
		}
		src, err := os.ReadFile(name)
		require.NoError(t, err)
		for i, line := range strings.Split(string(src), "\n") {
			if sprintfJSON.MatchString(line) || concatJSON.MatchString(line) {
				offenders = append(offenders, "  "+name+":"+strconv.Itoa(i+1)+": "+strings.TrimSpace(line))
			}
		}
	}

	assert.Empty(t, offenders,
		"these build JSON by splicing a value into text, which breaks on a quote, a backslash or a newline in the value.\n"+
			"Build the fixture from a Go value instead:\n"+
			"\tjsonMust(jsonObj{\"sha\": sha})\n"+
			"A JSON fragment that must stay raw (a deliberately wrong-typed value) is json.RawMessage(frag).\n"+
			"Offenders:\n%s", strings.Join(offenders, "\n"))
}
