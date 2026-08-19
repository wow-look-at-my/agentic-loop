package repo

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRepoIssueListFiltersOutPullRequests(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		assert.Equal(t, "/repos/octo/hello/issues", c.Path)
		assert.Equal(t, "open", c.Query.Get("state"))
		assert.Equal(t, "bug,ui", c.Query.Get("labels"))
		return http.StatusOK, `[
			{"number":11,"title":"Crash on load","state":"open","updated_at":"2026-07-01T10:00:00Z",
			 "user":{"login":"alice"},"labels":[{"name":"bug"},{"name":"ui"}]},
			{"number":12,"title":"A pull request","state":"open","updated_at":"2026-07-01T11:00:00Z",
			 "user":{"login":"bob"},"pull_request":{"url":"https://api.github.com/repos/octo/hello/pulls/12"}},
			{"number":13,"title":"Feature request","state":"open","updated_at":"2026-06-29T08:00:00Z",
			 "user":{"login":"carol"},"labels":[]}
		]`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "issues", Org: "octo", Repo: "hello", Labels: "bug,ui"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "issues of /repos/octo/hello (state open, labels bug,ui)")
	assert.Contains(t, res.Content, "#11  Crash on load  [bug, ui]  (alice, open, updated 2026-07-01T10:00:00Z)")
	assert.Contains(t, res.Content, "#13  Feature request  (carol, open, updated 2026-06-29T08:00:00Z)")
	// The pull request entry is filtered out.
	assert.NotContains(t, res.Content, "#12")
	assert.NotContains(t, res.Content, "A pull request")
}

func TestRepoIssueListEmptyAndStateValidation(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		assert.Equal(t, "closed", c.Query.Get("state"))
		return http.StatusOK, `[]`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "issues", Org: "octo", Repo: "hello", State: "closed"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "(no issues found)")

	bad := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "issues", Org: "octo", Repo: "hello", State: "wontfix"})
	assert.True(t, bad.IsError)
	assert.Contains(t, bad.Content, `"open", "closed", "all"`)
}

func TestRepoIssueReadRendersBodyAndComments(t *testing.T) {
	longComment := strings.Repeat("c", repoCommentMaxRunes+10)
	_, ex := newFakeGitHub(t, GitHubConfig{Cache: newMemCache()}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/issues/11":
			return http.StatusOK, `{"number":11,"title":"Crash on load","state":"open","comments":40,
				"body":"It crashes.","html_url":"https://github.com/octo/hello/issues/11",
				"updated_at":"2026-07-01T10:00:00Z","user":{"login":"alice"},"labels":[{"name":"bug"}]}`
		case "/repos/octo/hello/issues/11/comments":
			assert.Equal(t, fmt.Sprint(repoIssueMaxComments), c.Query.Get("per_page"))
			return http.StatusOK, jsonMust(jsonArr{
				jsonObj{
					"body":       "Repro steps here",
					"created_at": "2026-07-01T11:00:00Z",
					"user": jsonObj{
						"login": "bob",
					},
				},
				jsonObj{
					"body":       longComment,
					"created_at": "2026-07-01T12:00:00Z",
					"user": jsonObj{
						"login": "carol",
					},
				},
			})
		}
		t.Fatalf("unexpected path %s", c.Path)
		return 0, ""
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "issue", Org: "octo", Repo: "hello", Number: 11})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "issue #11 of /repos/octo/hello: Crash on load")
	assert.Contains(t, res.Content, "open, by alice, updated 2026-07-01T10:00:00Z, labels [bug]")
	assert.Contains(t, res.Content, "It crashes.")
	// 40 total, only the fetched page shown, with an explicit note.
	assert.Contains(t, res.Content, "comments (40, showing first 2):")
	assert.Contains(t, res.Content, "--- bob (2026-07-01T11:00:00Z)\nRepro steps here")
	assert.Contains(t, res.Content, fmt.Sprintf("(truncated to %d characters)", repoCommentMaxRunes))
}

func TestRepoIssueReadNoComments(t *testing.T) {
	gh, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		return http.StatusOK, `{"number":2,"title":"Quiet","state":"closed","comments":0,
			"user":{"login":"alice"},"updated_at":"2026-07-01T10:00:00Z"}`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "issue", Org: "octo", Repo: "hello", Number: 2})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "(no description)")
	assert.Contains(t, res.Content, "(no comments)")
	// No comments request was made.
	for _, c := range gh.Calls() {
		assert.NotContains(t, c.Path, "/comments")
	}
}

func TestRepoIssueReadRedirectsPullRequest(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		return http.StatusOK, `{"number":12,"title":"A PR","state":"open",
			"pull_request":{"url":"https://api.github.com/repos/octo/hello/pulls/12"}}`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "issue", Org: "octo", Repo: "hello", Number: 12})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "is a pull request, not an issue")
	assert.Contains(t, res.Content, "repo_read what=pr")
}

func TestRepoIssueReadArgErrors(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "issue", Org: "octo", Repo: "hello"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `positive "number"`)

	res = execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "issue", Repo: "hello", Number: 1})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `requires both "org" and "repo"`)
}
