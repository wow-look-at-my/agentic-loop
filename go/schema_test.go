package agentic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type schemaFixture struct {
	What    string   `json:"what" jsonschema:"Which read."`
	Name    string   `json:"name,omitempty" jsonschema:"Optional name."`
	Count   int      `json:"count,omitempty"`
	Ratio   float64  `json:"ratio,omitempty"`
	Deep    bool     `json:"deep,omitempty" jsonschema:"Go deeper."`
	Tags    []string `json:"tags,omitempty" jsonschema:"Zero or more tags."`
	private string   //nolint:unused // an unexported field is not an argument
	Skipped string   `json:"-"`
}

// The advertised schema IS the struct: field order, names, types, prose and
// required-ness all come from one declaration.
func TestInferSchemaMirrorsTheStruct(t *testing.T) {
	got := string(EnumSchema[schemaFixture](map[string][]string{"what": {"a", "b"}}))
	want := `{"type":"object","properties":{` +
		`"what":{"type":"string","description":"Which read.","enum":["a","b"]},` +
		`"name":{"type":"string","description":"Optional name."},` +
		`"count":{"type":"integer"},` +
		`"ratio":{"type":"number"},` +
		`"deep":{"type":"boolean","description":"Go deeper."},` +
		`"tags":{"type":"array","description":"Zero or more tags.","items":{"type":"string"}}` +
		`},"required":["what"],"additionalProperties":false}`
	assert.Equal(t, want, got)
	assert.True(t, json.Valid([]byte(got)))
}

// A struct with nothing required still advertises an empty list rather than
// omitting the key, so a consumer never has to tell "none" from "unstated".
func TestInferSchemaWithNothingRequired(t *testing.T) {
	type optional struct {
		X string `json:"x,omitempty"`
	}
	assert.Contains(t, string(InferSchema[optional]()), `"required":[]`)
}

// Every one of these is a wiring mistake in the package that declares the
// tool, so it fails at construction rather than on the one call that reaches
// it -- a tool advertised with a contract its handler does not honour is worse
// than a build that stops.
func TestInferSchemaPanicsOnAMisdeclaredArgument(t *testing.T) {
	type noTag struct{ Path string }
	type emptyName struct {
		Path string `json:","`
	}
	type unsupported struct {
		When map[string]string `json:"when"`
	}
	type listOfStructs struct {
		Items []noTag `json:"items"`
	}
	assert.PanicsWithValue(t, "agentic: noTag.Path has no json tag; every argument field must name itself",
		func() { InferSchema[noTag]() })
	assert.PanicsWithValue(t, "agentic: emptyName.Path has an empty json name",
		func() { InferSchema[emptyName]() })
	assert.PanicsWithValue(t, "agentic: unsupported.When is map[string]string, which has no tool-argument schema",
		func() { InferSchema[unsupported]() })
	assert.PanicsWithValue(t, "agentic: listOfStructs.Items is []agentic.noTag; a tool argument list must hold scalars",
		func() { InferSchema[listOfStructs]() })
	assert.PanicsWithValue(t, `agentic: cannot constrain unknown property "nope" on agentic.schemaFixture`,
		func() { EnumSchema[schemaFixture](map[string][]string{"nope": {"x"}}) })
	assert.PanicsWithValue(t, "agentic: a tool's arguments must be a struct, got string",
		func() { InferSchema[string]() })
}

// The file tools' schemas come off their argument structs, so this is the one
// check that matters for them: the tool decodes exactly what it advertises.
func TestFileToolSchemasAreInferred(t *testing.T) {
	reg, _ := fileRig()
	tool, ok := reg.Find(GrepToolName)
	require.True(t, ok)
	assert.JSONEq(t, string(InferSchema[grepArgs]()), string(tool.Decl().InputSchema))
}
