package agentic

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The loop package must not import optional tool families. Hosts compose them.
var optionalImportSuffixes = []string{
	"/vfs",
	"/repo",
	"/subagent",
	"/webfetch",
	"/todo",
	"/resources",
}

func TestLoopPackageDoesNotImportOptionalFamilies(t *testing.T) {
	entries, err := os.ReadDir(".")
	require.NoError(t, err)
	fset := token.NewFileSet()
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		af, err := parser.ParseFile(fset, e.Name(), nil, parser.ImportsOnly)
		require.NoError(t, err)
		for _, spec := range af.Imports {
			path := strings.Trim(spec.Path.Value, `"`)
			for _, suf := range optionalImportSuffixes {
				assert.NotContains(t, path, suf, "%s must not import %s", e.Name(), path)
			}
		}
	}
}

func TestLoopImportPathIsTheModuleRoot(t *testing.T) {
	data, err := os.ReadFile("go.mod")
	require.NoError(t, err)
	first := strings.SplitN(string(data), "\n", 2)[0]
	assert.Equal(t, "module github.com/wow-look-at-my/agentic-loop", first,
		"package agentic must occupy the module-root import path")
}

func TestOptionalFamiliesLiveInTheirOwnPackages(t *testing.T) {
	for _, dir := range []string{"vfs", "repo", "subagent", "webfetch", "todo", "resources"} {
		info, err := os.Stat(dir)
		require.NoError(t, err, dir)
		assert.True(t, info.IsDir(), dir)
		matches, err := filepath.Glob(filepath.Join(dir, "*.go"))
		require.NoError(t, err)
		assert.NotEmpty(t, matches, "%s must contain Go sources", dir)
	}
}
