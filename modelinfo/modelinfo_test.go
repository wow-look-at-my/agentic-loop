package modelinfo

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// catalogue is shaped like modelinfo's own answer: canonical undated ids, dated snapshots as aliases, rates in USD per
// token as decimal strings, and limits.
const catalogue = `{"object":"list","data":[
  {"id":"anthropic/claude-sonnet-4-5",
   "aliases":["claude-sonnet-4-5","anthropic/claude-sonnet-4-5-20250929","claude-sonnet-4-5-20250929"],
   "pricing":{"prompt":"0.000003","completion":"0.000015","input_cache_read":"0.0000003","input_cache_write":"0.00000375"},
   "context_length":1000000,"max_output_length":64000},
  {"id":"some/unpriced","context_length":8192},
  {"id":"some/garbage","pricing":{"prompt":"nonsense","completion":""}},
  {"id":"some/free","pricing":{"prompt":"0","completion":"0"}},
  {"id":"a/dupe","pricing":{"prompt":"0.000001"}},
  {"id":"b/dupe","pricing":{"prompt":"0.000002"}}
]}`

func serve(t *testing.T, body string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		assert.Equal(t, "/v1/models", r.URL.Path)
		assert.Equal(t, "all", r.URL.Query().Get("mode"), "the default view drops every non-chat model")
		_, _ = w.Write([]byte(body))
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

func TestOneReadServesEveryLookup(t *testing.T) {
	srv, hits := serve(t, catalogue)
	c := New(srv.URL, srv.Client())
	for range 3 {
		_, ok, err := c.Lookup(context.Background(), "a/dupe")
		require.NoError(t, err)
		require.True(t, ok)
	}
	assert.Equal(t, int32(1), hits.Load())
}

func TestAFailedReadIsAnsweredUntilRetryAfter(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if hits.Add(1) == 1 {
			http.Error(w, "down", http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(catalogue))
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

func TestAnUnreadableCatalogueIsAnError(t *testing.T) {
	srv, _ := serve(t, `<html>not a catalogue</html>`)
	_, _, err := New(srv.URL, srv.Client()).Lookup(context.Background(), "m")
	require.Error(t, err)
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
