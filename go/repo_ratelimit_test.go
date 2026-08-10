package agentic

import (
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Content search no longer touches a rate-limited endpoint at all, but the
// metadata reads (commits, PRs, issues) still use the API, and the failure that
// broke the original session can still happen there: a token's real 403 being
// overwritten by the anonymous attempt's answer, so a wait measured in seconds
// reads as a permanent authentication problem.
func TestRateLimitIsReportedAsTransient(t *testing.T) {
	reset := time.Now().Add(45 * time.Second)
	g, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}},
		func(c ghCall) (int, string) {
			if c.Auth == "" {
				return http.StatusUnauthorized, `{"message":"Requires authentication"}`
			}
			return http.StatusForbidden, `{"message":"API rate limit exceeded for user ID 1."}`
		})
	g.headers = func(c ghCall) http.Header {
		if c.Auth == "" {
			return nil
		}
		h := http.Header{}
		h.Set("X-RateLimit-Remaining", "0")
		h.Set("X-RateLimit-Reset", strconv.FormatInt(reset.Unix(), 10))
		h.Set("X-RateLimit-Resource", "core")
		return h
	}

	r := execRepoTool(t, ex, RepoReadToolName, map[string]any{"what": "commits", "org": "o", "repo": "r"})
	require.True(t, r.IsError)
	assert.Contains(t, r.Content, "TRANSIENT")
	assert.Contains(t, r.Content, "rate limit")
	assert.Regexp(t, `clears in \d+s`, r.Content, "a transient failure has to say how long it lasts")
	assert.NotContains(t, r.Content, "Requires authentication",
		"the anonymous attempt's answer must never be what a rate limit is reported as")
	assert.Contains(t, g.Auths(), "tok")
}

// The same masking broke every other read: a private repo answered 404 to the
// anonymous attempt, and that 404 overwrote the token's real answer — telling
// the model a repository it had just read "may not exist".
func TestPrivateRepoReportsTheTokenFailureNotTheAnonymous404(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}},
		func(c ghCall) (int, string) {
			if c.Auth == "" {
				return http.StatusNotFound, `{"message":"Not Found"}`
			}
			return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
		})
	r := execRepoTool(t, ex, RepoReadToolName, map[string]any{"what": "prs", "org": "o", "repo": "r"})
	require.True(t, r.IsError)
	assert.Contains(t, r.Content, "access denied")
	assert.NotContains(t, r.Content, "not found (404)",
		"an existing repository must never be reported as possibly nonexistent because the anonymous attempt could not see it")
}

// Three differently-worded queries produced one byte-identical repo-wide commit
// list, because what=commits never reads "query" — which then tripped the
// output deduper into telling the model it had repeated itself.
func TestRepoReadRejectsArgumentsTheReadIgnores(t *testing.T) {
	g, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}},
		func(ghCall) (int, string) { return http.StatusOK, `[]` })

	r := execRepoTool(t, ex, RepoReadToolName, map[string]any{
		"what": "commits", "org": "o", "repo": "r", "query": "SSR stochastic noise",
	})
	require.True(t, r.IsError)
	assert.Empty(t, g.Calls(), "the call must not run: its result would look like an answer")
	assert.Contains(t, r.Content, `"query"`)
	assert.Contains(t, r.Content, "newest commits, unfiltered")
	assert.Contains(t, r.Content, GrepToolName, "and must name the tool that does search contents")
	assert.Contains(t, r.Content, "org, repo, path, ref, per_page")
}

func TestRepoReadAcceptsTheArgumentsItsReadUses(t *testing.T) {
	g, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}},
		func(ghCall) (int, string) { return http.StatusOK, `[]` })
	r := execRepoTool(t, ex, RepoReadToolName, map[string]any{
		"what": "commits", "org": "o", "repo": "r", "path": "Engine/Shaders", "ref": "main", "per_page": 5,
	})
	require.False(t, r.IsError, r.Content)
	require.NotEmpty(t, g.Calls())
	assert.Equal(t, "Engine/Shaders", g.Calls()[0].Query.Get("path"), "the path filter still reaches GitHub")
}

func TestRepoReadRejectionNamesEveryIgnoredField(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(ghCall) (int, string) { return http.StatusOK, `[]` })
	r := execRepoTool(t, ex, RepoReadToolName, map[string]any{
		"what": "issue", "org": "o", "repo": "r", "number": 1, "query": "x", "sha": "deadbeef",
	})
	require.True(t, r.IsError)
	assert.True(t, strings.Contains(r.Content, `"query"`) && strings.Contains(r.Content, `"sha"`), r.Content)
}
