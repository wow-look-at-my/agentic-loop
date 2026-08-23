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
