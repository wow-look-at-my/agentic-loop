package commonai

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// The vocabulary is the library's, so a host's UI does not carry a copy
// that goes stale when a dialect is added.
//
// Reading a dialect off an endpoint is FetchModelList's job -- see
// modellist_test.go, which covers both envelopes, both bare-item forms, and
// what a document of neither shape answers.
func TestDialectVocabulary(t *testing.T) {
	assert.Equal(t, []Dialect{DialectAuto, DialectOpenAI, DialectAnthropic, DialectResponses}, Dialects())
	for _, d := range Dialects() {
		assert.True(t, d.Valid())
		assert.NotEmpty(t, d.Label())
	}
	assert.False(t, Dialect("cohere").Valid(), "a dialect this library cannot speak is not valid")
	assert.Equal(t, "detect", DialectAuto.Label())
}
