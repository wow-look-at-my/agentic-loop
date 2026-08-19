package agentic

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// "CI failed" is the whole of what a user usually says, so a report that
// answers it with "build: failure" only renames the question. These pin that
// what=status carries the REASON, that anything it could not explain is named
// rather than dropped, and that what=check_run is the drill-down.

// checkRun builds one check-run fixture.
func checkRun(id int, name, conclusion string, extra jsonObj) jsonObj {
	o := jsonObj{
		"id": id, "name": name, "status": "completed", "conclusion": conclusion,
		"html_url": fmt.Sprintf("https://example.com/run/%d", id),
	}
	for k, v := range extra {
		o[k] = v
	}
	return o
}

func TestRepoStatusExplainsEveryFailingCheck(t *testing.T) {
	g, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, jsonMust(jsonObj{"state": "failure", "sha": "0123456789abcdef", "statuses": jsonArr{}})
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusOK, jsonMust(jsonObj{"check_runs": jsonArr{
				checkRun(41, "build", "failure", nil),
				checkRun(42, "lint", "success", nil),
			}})
		case "/repos/octo/hello/check-runs/41":
			return http.StatusOK, jsonMust(checkRun(41, "build", "failure", jsonObj{
				"output": jsonObj{
					"title":             "Process completed with exit code 1",
					"summary":           "go.mod was rewritten during the run\nworking tree is dirty in CI",
					"annotations_count": 2,
				},
			}))
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)

	// The verdict, the reason, and the id that reaches the rest of it.
	assert.Contains(t, res.Content, "build: failure [id 41]")
	assert.Contains(t, res.Content, "Process completed with exit code 1")
	assert.Contains(t, res.Content, "working tree is dirty in CI")
	assert.Contains(t, res.Content, "2 error annotation(s)")
	assert.Contains(t, res.Content, `"what":"check_run"`)

	// Only the FAILING check is followed up: a green run has nothing to explain
	// and its detail read would be pure latency.
	for _, c := range g.Calls() {
		assert.NotEqual(t, "/repos/octo/hello/check-runs/42", c.Path)
	}
}

// A detail read that fails must leave the check NAMED as unexplained. A report
// that quietly drops it reads as a complete account of the failure.
func TestRepoStatusNamesTheChecksItCouldNotExplain(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/commits/main/status":
			return http.StatusOK, jsonMust(jsonObj{"state": "failure", "sha": "abc", "statuses": jsonArr{}})
		case "/repos/octo/hello/commits/main/check-runs":
			return http.StatusOK, jsonMust(jsonObj{"check_runs": jsonArr{checkRun(41, "build", "failure", nil)}})
		case "/repos/octo/hello/check-runs/41":
			return http.StatusForbidden, jsonMust(jsonObj{"message": "Resource not accessible by personal access token"})
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "build: failure")
	assert.Contains(t, res.Content, "Not explained here: build")
}

// The follow-up count is bounded, and the bound says what it left out.
func TestRepoStatusBoundsTheDetailReadsAndSaysSo(t *testing.T) {
	failing := jsonArr{}
	for i := 1; i <= statusFailureDetailLimit+2; i++ {
		failing = append(failing, checkRun(i, fmt.Sprintf("job-%d", i), "failure", nil))
	}
	g, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch {
		case c.Path == "/repos/octo/hello/commits/main/status":
			return http.StatusOK, jsonMust(jsonObj{"state": "failure", "sha": "abc", "statuses": jsonArr{}})
		case c.Path == "/repos/octo/hello/commits/main/check-runs":
			return http.StatusOK, jsonMust(jsonObj{"check_runs": failing})
		case strings.HasPrefix(c.Path, "/repos/octo/hello/check-runs/"):
			return http.StatusOK, jsonMust(jsonObj{"id": 1, "name": "job", "status": "completed", "conclusion": "failure"})
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, res.IsError, res.Content)

	detailReads := 0
	for _, c := range g.Calls() {
		if strings.HasPrefix(c.Path, "/repos/octo/hello/check-runs/") {
			detailReads++
		}
	}
	assert.Equal(t, statusFailureDetailLimit, detailReads, "the follow-up reads must be capped")
	assert.Contains(t, res.Content, "Not explained here:")
	assert.Contains(t, res.Content, fmt.Sprintf("job-%d", statusFailureDetailLimit+2))
}

func TestRepoCheckRunReportsOutputAndAnnotations(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/check-runs/41":
			return http.StatusOK, jsonMust(checkRun(41, "build", "failure", jsonObj{
				"output": jsonObj{
					"title": "Process completed with exit code 1", "summary": "one step failed",
					"text": "go-toolchain: working tree is dirty in CI", "annotations_count": 1,
				},
			}))
		case "/repos/octo/hello/check-runs/41/annotations":
			return http.StatusOK, jsonMust(jsonArr{jsonObj{
				"path": "go.mod", "start_line": 14, "end_line": 14,
				"annotation_level": "failure", "message": "unexpected rewrite",
			}})
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "check_run", Org: "octo", Repo: "hello", ID: 41})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "Check run 41 of /repos/octo/hello: build -- failure")
	assert.Contains(t, res.Content, "go-toolchain: working tree is dirty in CI")
	assert.Contains(t, res.Content, "failure go.mod:14 -- unexpected rewrite")
}

// A check that reported nothing must SAY it reported nothing. Rendering an
// empty section instead reads as "this check flagged no errors", which is a
// different claim entirely.
func TestRepoCheckRunSaysWhenTheCheckReportedNothing(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/check-runs/41":
			return http.StatusOK, jsonMust(checkRun(41, "build", "failure", nil))
		case "/repos/octo/hello/check-runs/41/annotations":
			return http.StatusOK, jsonMust(jsonArr{})
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "check_run", Org: "octo", Repo: "hello", ID: 41})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "This check reported no output and no annotations")
	assert.Contains(t, res.Content, "https://example.com/run/41")
}

// An annotations read the token cannot make is a NOTE, not an empty list: the
// check's own output is still worth having, and "(none)" would be a lie.
func TestRepoCheckRunNotesAnUnreadableAnnotationList(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Name: "one", Token: "tok"}}}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/check-runs/41":
			return http.StatusOK, jsonMust(checkRun(41, "build", "failure", jsonObj{
				"output": jsonObj{"title": "boom"},
			}))
		case "/repos/octo/hello/check-runs/41/annotations":
			return http.StatusForbidden, jsonMust(jsonObj{"message": "Resource not accessible by personal access token"})
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "check_run", Org: "octo", Repo: "hello", ID: 41})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "boom")
	assert.Contains(t, res.Content, "Annotations: unavailable")
	assert.NotContains(t, res.Content, "Annotations: (none)")
}

// A capped annotation listing must say it was capped, or it reads as the full
// set of everything that went wrong.
func TestRepoCheckRunSaysWhenTheAnnotationListingIsCapped(t *testing.T) {
	full := jsonArr{}
	for i := 0; i < checkRunAnnotationLimit; i++ {
		full = append(full, jsonObj{"path": "x.go", "start_line": i + 1, "annotation_level": "failure", "message": "bad"})
	}
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/check-runs/41":
			return http.StatusOK, jsonMust(checkRun(41, "build", "failure", nil))
		case "/repos/octo/hello/check-runs/41/annotations":
			return http.StatusOK, jsonMust(full)
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "check_run", Org: "octo", Repo: "hello", ID: 41})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, fmt.Sprintf("listing capped at %d", checkRunAnnotationLimit))
}

func TestRepoCheckRunWithoutAnIDTeachesWhereIDsComeFrom(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "check_run", Org: "octo", Repo: "hello"})
	require.True(t, res.IsError)
	assert.Contains(t, res.Content, `requires "id"`)
	assert.Contains(t, res.Content, `"what":"status"`)
}

// what=check_run ignores every argument but org/repo/id, and an ignored one is
// rejected rather than dropped (see repo_args.go).
func TestRepoCheckRunRejectsArgumentsItIgnores(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoReadToolName, map[string]any{
		"what": "check_run", "org": "octo", "repo": "hello", "id": 41, "ref": "main",
	})
	require.True(t, res.IsError)
	assert.Contains(t, res.Content, `repo_read what=check_run ignores "ref"`)
}

func TestRepoCheckRunRequiresOrgRepo(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "check_run", Org: "octo", ID: 41})
	require.True(t, res.IsError)
	assert.Contains(t, res.Content, `requires both "org" and "repo"`)
}

func TestRepoCheckRunUnreadableCheckIsAnError(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(ghCall) (int, string) {
		return http.StatusNotFound, jsonMust(jsonObj{"message": "Not Found"})
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "check_run", Org: "octo", Repo: "hello", ID: 7})
	require.True(t, res.IsError)
	assert.Contains(t, res.Content, "check run 7 of /repos/octo/hello")
}

// what=commits names the ref it listed even when the caller named none.
// Without that, a caller who needs a SPECIFIC branch cannot tell whether the
// answer is that branch's, and re-checks every repository against the GitHub
// API by hand.
func TestRepoCommitsNamesTheRefEvenWhenUnspecified(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		return http.StatusOK, jsonMust(jsonArr{jsonObj{
			"sha":    "abcdef1234567890",
			"commit": jsonObj{"message": "hi", "author": jsonObj{"name": "a", "date": "2026-01-01T00:00:00Z"}},
		}})
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commits", Org: "octo", Repo: "hello"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "the repository's default branch")
	assert.Contains(t, res.Content, `pass "ref" to list another`)

	res = execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commits", Org: "octo", Repo: "hello", Ref: "master"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "ref master")
	assert.NotContains(t, res.Content, "default branch")
}
