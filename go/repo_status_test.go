package agentic

import (
	"net/http"
	"strings"
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

// A check-runs failure (a token granted `actions` and not `checks` is the
// common case) is noted, not fatal — the legacy status is still a real answer.
func TestRepoStatusCheckRunsFailureIsNotedNotFatal(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"pending","sha":"abc","statuses":[]}`
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
		case "/repos/octo/hello/actions/runs":
			return http.StatusOK, `{"total_count":0,"workflow_runs":[]}`
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

// The bug: a token without `checks` got "CI is red" and no reason, which is
// the whole question. The same runs are readable through the Actions API on
// the `actions` permission, so a failed job and the step that failed inside it
// must still reach the reader.
func TestRepoStatusFallsBackToActionsWhenCheckRunsAreUnreadable(t *testing.T) {
	g, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"failure","sha":"0123456789abcdef","statuses":[
				{"context":"all-builds","state":"failure","description":"1/2 builds failed"}
			]}`
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
		case "/repos/octo/hello/actions/runs":
			assert.Equal(t, "0123456789abcdef", c.Query.Get("head_sha"), "the Actions read must be scoped to the same commit")
			return http.StatusOK, `{"total_count":1,"workflow_runs":[
				{"id":77,"name":"CI","status":"completed","conclusion":"failure","html_url":"https://example.com/run/77"}
			]}`
		case "/repos/octo/hello/actions/runs/77/jobs":
			return http.StatusOK, `{"total_count":2,"jobs":[
				{"name":"build","status":"completed","conclusion":"failure","html_url":"https://example.com/job/1","steps":[
					{"name":"checkout","status":"completed","conclusion":"success","number":1},
					{"name":"go-toolchain","status":"completed","conclusion":"failure","number":2}
				]},
				{"name":"smoke","status":"completed","conclusion":"success","html_url":"https://example.com/job/2","steps":[]}
			]}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	// The header GitHub really sends on a Checks API 403.
	g.headers = func(c ghCall) http.Header {
		h := http.Header{}
		if strings.HasSuffix(c.Path, "/check-runs") {
			h.Set("X-Accepted-GitHub-Permissions", "checks=read")
		}
		return h
	}
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "Workflow runs (the Actions API, since the check runs could not be read):")
	assert.Contains(t, res.Content, "CI: failure (https://example.com/run/77)")
	// GitHub's 403 names a permission a PAT cannot be granted; the report must
	// not leave that standing as an instruction.
	assert.Contains(t, res.Content, "cannot be granted \"Checks\" at all")
	assert.Contains(t, res.Content, "build: failure (https://example.com/job/1)")
	assert.Contains(t, res.Content, "step 2 failed: go-toolchain (failure)")
	// A passing job names no steps: the reader is after what broke.
	assert.Contains(t, res.Content, "smoke: success")
	assert.NotContains(t, res.Content, "step 1 failed")
}

// The fallback's own failure is stated. Falling through to silence would leave
// the check-runs note reading as the last word, which is the defect again.
func TestRepoStatusActionsFallbackFailureIsStated(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"failure","sha":"abc","statuses":[]}`
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
		case "/repos/octo/hello/actions/runs":
			return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "Workflow runs (the Actions API, since the check runs could not be read): unavailable")
	assert.Contains(t, res.Content, "could not read the workflow runs of /repos/octo/hello@abc")
}

// A jobs read that fails says so under its run, rather than rendering a run
// with no jobs — which reads as a run that did nothing.
func TestRepoStatusActionsJobsFailureIsStatedUnderItsRun(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"failure","sha":"abc","statuses":[]}`
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusForbidden, `{"message":"no"}`
		case "/repos/octo/hello/actions/runs":
			return http.StatusOK, `{"total_count":1,"workflow_runs":[
				{"id":9,"name":"CI","status":"completed","conclusion":"failure","html_url":"https://example.com/run/9"}
			]}`
		case "/repos/octo/hello/actions/runs/9/jobs":
			return http.StatusInternalServerError, `{"message":"boom"}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "CI: failure")
	assert.Contains(t, res.Content, "jobs unavailable --")
}

// Readable check runs are the whole answer: the Actions API is not consulted,
// so a working Checks permission costs no extra requests.
func TestRepoStatusDoesNotCallActionsWhenCheckRunsAreReadable(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"failure","sha":"abc","statuses":[]}`
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusOK, `{"check_runs":[{"name":"build","status":"completed","conclusion":"success"}]}`
		default:
			t.Fatalf("unexpected path %q -- the Actions fallback must not run when the check runs are readable", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)
	assert.NotContains(t, res.Content, "Workflow runs")
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
