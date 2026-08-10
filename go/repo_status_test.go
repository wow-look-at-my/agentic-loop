package agentic

import (
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRepoStatusReportsCombinedStatusAndCheckRuns(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Cache: newMemCache()}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"failure","sha":"0123456789abcdef","statuses":[
				{"context":"all-builds","state":"failure","description":"1/2 builds failed","target_url":"https://example.com/checks"}
			]}`
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusOK, `{"check_runs":[
				{"name":"build","status":"completed","conclusion":"failure","html_url":"https://example.com/run/1"},
				{"name":"lint","status":"in_progress","conclusion":"","html_url":"https://example.com/run/2"}
			]}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "CI status for /repos/octo/hello @ main (commit 0123456789)")
	assert.Contains(t, res.Content, "Commit status: FAILURE")
	assert.Contains(t, res.Content, "all-builds: failure -- 1/2 builds failed (https://example.com/checks)")
	assert.Contains(t, res.Content, "build: failure (https://example.com/run/1)")
	assert.Contains(t, res.Content, "lint: in_progress (https://example.com/run/2)")
}

func TestRepoStatusWithNoRefUsesTheDefaultBranchHead(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"trunk"}`
		case "/repos/octo/hello/commits/trunk/status":
			return http.StatusOK, `{"state":"success","sha":"abc","statuses":[]}`
		case "/repos/octo/hello/commits/trunk/check-runs":
			return http.StatusOK, `{"check_runs":[]}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "CI status for /repos/octo/hello @ trunk")
	assert.Contains(t, res.Content, "Commit status: SUCCESS")
	assert.Contains(t, res.Content, "(no statuses posted)")
	assert.Contains(t, res.Content, "Check runs: (none)")
}

// A check-runs failure (a token without Checks API access is common) is
// noted, not fatal — the legacy status is still a real, useful answer.
func TestRepoStatusCheckRunsFailureIsNotedNotFatal(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"pending","sha":"abc","statuses":[]}`
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "Commit status: PENDING")
	assert.Contains(t, res.Content, "Check runs: unavailable")
	assert.Contains(t, res.Content, "lack the required permission")
}

func TestRepoStatusCombinedStatusFailureIsFatal(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		return http.StatusNotFound, `{"message":"Not Found"}`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "ghost", Ref: "main"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "could not read the CI status of /repos/octo/ghost@main")
}

// A near-expiry token gets the SAME advisory on a successful what=status
// call as it does on an explained failure — the header is real either way.
func TestRepoStatusAppendsANearExpiryAdvisory(t *testing.T) {
	expiry := time.Now().Add(2 * 24 * time.Hour)
	g, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"success","sha":"abc","statuses":[]}`
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusOK, `{"check_runs":[]}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	g.headers = func(c ghCall) http.Header {
		h := http.Header{}
		h.Set("GitHub-Authentication-Token-Expiration", expiry.UTC().Format(githubTokenExpirationLayout))
		return h
	}
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "expires on")
	assert.Contains(t, res.Content, "rotate it")
}

func TestRepoStatusRequiresOrgRepo(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `requires both "org" and "repo"`)
}

// A branch name containing "/" (a common convention) must survive in the
// URL literally, not as %2F — EscapeSegments is what the rest of this
// package already relies on for the same reason.
func TestRepoStatusRefWithSlashIsNotPercentEncoded(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		assert.NotContains(t, c.Path, "%2F", "a literal slash in the ref must stay a path separator")
		switch c.Path {
		case "/repos/octo/hello/commits/feature/foo/status":
			return http.StatusOK, `{"state":"success","sha":"abc","statuses":[]}`
		case "/repos/octo/hello/commits/feature/foo/check-runs":
			return http.StatusOK, `{"check_runs":[]}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "feature/foo"})
	require.False(t, res.IsError, res.Content)
}
