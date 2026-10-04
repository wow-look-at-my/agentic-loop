package modelinfo

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sonnet is a record shaped like modelinfo's own: a canonical undated id, rates in USD per token as decimal strings,
// and limits.
const sonnet = `{"id":"anthropic/claude-sonnet-4-5",
  "aliases":["claude-sonnet-4-5","claude-sonnet-4-5-20250929"],
  "pricing":{"prompt":"0.000003","completion":"0.000015","input_cache_read":"0.0000003","input_cache_write":"0.00000375"},
  "max_input_tokens":1000000,"max_output_tokens":64000}`

var catalogue = map[string]string{
	"anthropic/claude-sonnet-4-5":          sonnet,
	"claude-sonnet-4-5":                    sonnet,
	"anthropic/claude-sonnet-4-5-20250929": sonnet,
	"claude-sonnet-4-5-20250929":           sonnet,
	"some/unpriced":                        `{"id":"some/unpriced","context_length":8192}`,
	"some/garbage":                         `{"id":"some/garbage","pricing":{"prompt":"nonsense","completion":""}}`,
	"some/free":                            `{"id":"some/free","pricing":{"prompt":"0","completion":"0"}}`,
	"a/dupe":                               `{"id":"a/dupe","pricing":{"prompt":"0.000001"}}`,
	"b/dupe":                               `{"id":"b/dupe","pricing":{"prompt":"0.000002"}}`,
}

func serve(t *testing.T, records map[string]string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		name, ok := strings.CutPrefix(r.URL.Path, "/v1/models/")
		require.True(t, ok, "unexpected path %q", r.URL.Path)
		rec, ok := records[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(rec))
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func TestALookupResolvesEveryNameAModelAnswersTo(t *testing.T) {
	srv, _ := serve(t, catalogue)
	c := New(srv.URL, srv.Client())

	for _, name := range []string{
		"anthropic/claude-sonnet-4-5",
		"claude-sonnet-4-5",
		"anthropic/claude-sonnet-4-5-20250929",
		"claude-sonnet-4-5-20250929",
		"CLAUDE-SONNET-4-5-20250929",
	} {
		m, ok, err := c.Lookup(context.Background(), name)
		require.NoError(t, err)
		require.True(t, ok, name)
		assert.Equal(t, "anthropic/claude-sonnet-4-5", m.ID)
		require.True(t, m.Priced)
		assert.InDelta(t, 0.000003, m.Rates.Prompt, 1e-12)
		assert.InDelta(t, 0.000015, m.Rates.Completion, 1e-12)
		assert.InDelta(t, 0.0000003, m.Rates.CacheRead, 1e-12)
		assert.InDelta(t, 0.00000375, m.Rates.CacheWrite, 1e-12)
		assert.Equal(t, 1000000, m.Limits.ContextWindow)
		assert.Equal(t, 64000, m.Limits.MaxOutput)
	}

	_, ok, err := c.Lookup(context.Background(), "claude-opus-4-1")
	require.NoError(t, err)
	assert.False(t, ok, "a model the catalogue does not carry")
}

func TestUnpricedIsDistinctFromFree(t *testing.T) {
	srv, _ := serve(t, catalogue)
	c := New(srv.URL, srv.Client())

	m, ok, err := c.Lookup(context.Background(), "some/unpriced")
	require.NoError(t, err)
	require.True(t, ok, "a model the catalogue knows, priced or not")
	assert.False(t, m.Priced)
	assert.Equal(t, 8192, m.Limits.ContextWindow)

	m, _, _ = c.Lookup(context.Background(), "some/garbage")
	assert.False(t, m.Priced, "a block with no usable number is not a price")

	m, _, _ = c.Lookup(context.Background(), "some/free")
	assert.True(t, m.Priced, "zero is a real price")
	assert.Zero(t, m.Rates.Prompt)
}

func TestANameTwoModelsClaimAnswersForNeither(t *testing.T) {
	srv, _ := serve(t, catalogue)
	c := New(srv.URL, srv.Client())

	_, ok, err := c.Lookup(context.Background(), "dupe")
	require.NoError(t, err)
	assert.False(t, ok)
	_, ok, _ = c.Lookup(context.Background(), "a/dupe")
	assert.True(t, ok)
	_, ok, _ = c.Lookup(context.Background(), "b/dupe")
	assert.True(t, ok)
}

func TestEachNameIsAskedOnce(t *testing.T) {
	srv, hits := serve(t, catalogue)
	c := New(srv.URL, srv.Client())
	for range 3 {
		_, ok, err := c.Lookup(context.Background(), "a/dupe")
		require.NoError(t, err)
		require.True(t, ok)
		_, ok, err = c.Lookup(context.Background(), "nobody/knows")
		require.NoError(t, err)
		require.False(t, ok)
	}
	assert.Equal(t, int32(2), hits.Load(), "a known model and an unknown one, each asked once")
}

func TestAFailedReadIsAnsweredUntilRetryAfter(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if hits.Add(1) == 1 {
			http.Error(w, "down", http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(catalogue["a/dupe"]))
	}))
	defer srv.Close()
	c := New(srv.URL, srv.Client())
	clock := time.Unix(0, 0)
	c.now = func() time.Time { return clock }

	_, _, err := c.Lookup(context.Background(), "a/dupe")
	require.ErrorContains(t, err, "500")
	_, _, err = c.Lookup(context.Background(), "a/dupe")
	require.ErrorContains(t, err, "500", "the same failure, without a second request")
	assert.Equal(t, int32(1), hits.Load())

	clock = clock.Add(RetryAfter)
	_, ok, err := c.Lookup(context.Background(), "a/dupe")
	require.NoError(t, err)
	assert.True(t, ok)
	assert.Equal(t, int32(2), hits.Load())
}

func TestAnUnreadableRecordIsAnError(t *testing.T) {
	srv, _ := serve(t, map[string]string{"m": `<html>not a record</html>`, "n": `{"name":"no id"}`})
	_, _, err := New(srv.URL, srv.Client()).Lookup(context.Background(), "m")
	require.ErrorContains(t, err, "not a model record")
	_, _, err = New(srv.URL, srv.Client()).Lookup(context.Background(), "n")
	require.ErrorContains(t, err, "names no id")
}

func TestALookupNeedsAURLAndAName(t *testing.T) {
	_, _, err := New("", nil).Lookup(context.Background(), "m")
	require.ErrorContains(t, err, "no URL")
	_, _, err = New(DefaultURL, nil).Lookup(context.Background(), " ")
	require.ErrorContains(t, err, "no id")
}

func TestAnOversizedCatalogueIsRefused(t *testing.T) {
	big := make([]byte, maxBytes+1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(big) }))
	defer srv.Close()
	_, _, err := New(srv.URL, srv.Client()).Lookup(context.Background(), "m")
	require.ErrorContains(t, err, "more than")
}
