package repo

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// NoAnonymous is the host's policy knob: a server holding at least one PAT
// must never let a read fall through to an unauthenticated request, because
// GitHub buckets the anonymous request by the server's own IP and answers it
// with a different (worse) verdict than the token's real failure. These tests
// pin that the client-level flag drops the anonymous attempt from the read
// paths (FetchURLOpts, Credentials, OwnerRepos) while leaving the default
// (zero tokens) anonymous-friendly for public repositories.

func TestNoAnonymousDropsTheUnauthenticatedAttemptFromReads(t *testing.T) {
	var auths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auths = append(auths, trimBearer(r.Header.Get("Authorization")))
		if r.Header.Get("Authorization") == "" {
			// Anonymous would succeed; it must never be tried.
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[]`))
			return
		}
		// The token's real verdict: refused.
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"message":"API rate limit exceeded"}`))
	}))
	t.Cleanup(srv.Close)

	e := NewGitHub(GitHubConfig{
		HTTPClient:  srv.Client(),
		APIBaseURL:  srv.URL,
		Tokens:      []GitHubToken{{ID: "t1", Token: "tok"}},
		NoAnonymous: true,
	})
	res, err := e.FetchURLOpts(t.Context(), "", srv.URL+"/repos/o/r/commits", "application/vnd.github+json", FetchOptions{})
	require.NoError(t, err)
	assert.Equal(t, http.StatusForbidden, res.Status())
	assert.Equal(t, []string{"tok"}, auths, "only the one configured token is tried; the anonymous attempt is dropped")
}

func TestCredentialsNoAnonymousContainNoAnonymousAttempt(t *testing.T) {
	e := NewGitHub(GitHubConfig{
		Tokens:      []GitHubToken{{ID: "t1", Token: "tok"}},
		NoAnonymous: true,
	})
	assert.Equal(t, []Credential{{ID: "t1", Token: "tok"}}, e.Credentials("o/r", false),
		"per-call NoAnonymous=false must not re-add the anonymous attempt when the client forbids it")

	e2 := NewGitHub(GitHubConfig{})
	assert.Equal(t, []Credential{{ID: "", Name: "", Token: ""}}, e2.Credentials("o/r", false),
		"a host with no credentials keeps the anonymous attempt for public repositories")
}

// A cache that remembers the anonymous attempt as a repository's winner -- a
// store seeded before the user added any token -- must not put it ahead of the
// tokens, and under NoAnonymous must not put it anywhere.
func TestCachedAnonymousWinnerNeverOutranksATokenOrDefeatsNoAnonymous(t *testing.T) {
	var auths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auths = append(auths, trimBearer(r.Header.Get("Authorization")))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`[]`))
	}))
	t.Cleanup(srv.Close)

	cache := newMemCache()
	cache.Put("o/r", "")
	e := NewGitHub(GitHubConfig{
		HTTPClient:  srv.Client(),
		APIBaseURL:  srv.URL,
		Tokens:      []GitHubToken{{ID: "t1", Token: "tok"}},
		NoAnonymous: true,
		Cache:       cache,
	})
	assert.Equal(t, []Credential{{ID: "t1", Token: "tok"}}, e.Credentials("o/r", false),
		"the remembered anonymous winner is not an attempt when the client forbids anonymous reads")
	res, err := e.FetchURLOpts(t.Context(), "o/r", srv.URL+"/repos/o/r/commits", "application/vnd.github+json", FetchOptions{})
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, res.Status())
	assert.Equal(t, []string{"tok"}, auths)

	permissive := NewGitHub(GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}, Cache: cache})
	assert.Equal(t, []Credential{{ID: "t1", Token: "tok"}, {}}, permissive.Credentials("o/r", false),
		"with anonymous allowed it still runs last, never promoted by the cache")
}

// An anonymous win is not remembered: the cache holds tokens only, so a token
// configured later is tried first rather than shadowed by a stored preference
// for no credential.
func TestAnAnonymousWinIsNotRemembered(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`[{"name":"hello","private":false}]`))
	}))
	t.Cleanup(srv.Close)

	cache := newMemCache()
	e := NewGitHub(GitHubConfig{HTTPClient: srv.Client(), APIBaseURL: srv.URL, Cache: cache})
	res, err := e.FetchURLOpts(t.Context(), "o/r", srv.URL+"/repos/o/r/commits", "application/vnd.github+json", FetchOptions{})
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, res.Status())
	_, ok := cache.Get("o/r")
	assert.False(t, ok, "the anonymous winner is not stored")

	_, _, _, err = e.OwnerRepos(t.Context(), "octo")
	require.NoError(t, err)
	_, ok = cache.Get("octo")
	assert.False(t, ok, "an anonymous owner listing is not stored either")

	e.Remember("o/r", "")
	_, ok = cache.Get("o/r")
	assert.False(t, ok, "Remember drops an anonymous win too")
}

func TestNoAnonymousOwnerReposDropsTheUnauthenticatedAttempt(t *testing.T) {
	var auths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auths = append(auths, trimBearer(r.Header.Get("Authorization")))
		if r.Header.Get("Authorization") == "" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[]`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`[{"name":"hello","private":false}]`))
	}))
	t.Cleanup(srv.Close)

	e := NewGitHub(GitHubConfig{
		HTTPClient:  srv.Client(),
		APIBaseURL:  srv.URL,
		Tokens:      []GitHubToken{{ID: "t1", Token: "tok"}},
		NoAnonymous: true,
	})
	repos, _, last, err := e.OwnerRepos(t.Context(), "octo")
	require.NoError(t, err)
	require.NotNil(t, repos)
	assert.Equal(t, 1, len(repos))
	assert.Equal(t, 200, last.Status())
	for _, a := range auths {
		assert.NotEmpty(t, a, "every owner-listing request must carry a credential")
	}
}
