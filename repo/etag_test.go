package repo

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestARepeatReadIsConditionalAndReplaysOn304(t *testing.T) {
	var conditional []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conditional = append(conditional, r.Header.Get("If-None-Match"))
		if r.Header.Get("If-None-Match") == `"v1"` {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		w.Header().Set("ETag", `"v1"`)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"state":"success"}`)
	}))
	defer srv.Close()
	g := NewGitHub(GitHubConfig{APIBaseURL: srv.URL, Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}, NoAnonymous: true})

	first, err := g.FetchURL(t.Context(), "", srv.URL+"/repos/o/r/commits/abc/status", "application/vnd.github+json")
	require.NoError(t, err)
	second, err := g.FetchURL(t.Context(), "", srv.URL+"/repos/o/r/commits/abc/status", "application/vnd.github+json")
	require.NoError(t, err)

	assert.Equal(t, []string{"", `"v1"`}, conditional, "the second read is conditional")
	assert.Equal(t, http.StatusOK, second.Status(), "a 304 is replayed as the stored 200")
	assert.Equal(t, first.Body(), second.Body())
}

// Each credential keeps its own entry: one token's cached answer is never served for another.
func TestTheReplayCacheIsPerCredential(t *testing.T) {
	c := newETagCache()
	res := GHResponse{status: http.StatusOK, header: http.Header{"Etag": {`"v1"`}}, body: []byte("a")}
	c.store(etagKey{target: "u", token: "a"}, res)
	_, _, ok := c.lookup(etagKey{target: "u", token: "b"})
	assert.False(t, ok)
	_, _, ok = c.lookup(etagKey{target: "u", token: "a"})
	assert.True(t, ok)
}

// The cache holds a bounded number of entries and drops the least recently used first.
func TestTheReplayCacheDropsTheOldestEntry(t *testing.T) {
	c := newETagCache()
	res := GHResponse{status: http.StatusOK, header: http.Header{"Etag": {`"v"`}}}
	for i := range etagCacheEntries + 1 {
		c.store(etagKey{target: string(rune(i + 1))}, res)
	}
	_, _, ok := c.lookup(etagKey{target: string(rune(1))})
	assert.False(t, ok, "the first entry is gone")
	_, _, ok = c.lookup(etagKey{target: string(rune(etagCacheEntries + 1))})
	assert.True(t, ok)
}
