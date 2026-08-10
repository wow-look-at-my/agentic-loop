package agentic

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRepoPRListRendersRows(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		assert.Equal(t, "/repos/octo/hello/pulls", c.Path)
		assert.Equal(t, "open", c.Query.Get("state")) // default state
		assert.Equal(t, "10", c.Query.Get("per_page"))
		return http.StatusOK, `[
			{"number":7,"title":"Add feature","state":"open","draft":true,"updated_at":"2026-07-01T10:00:00Z",
			 "user":{"login":"alice"},"head":{"ref":"feature-x"},"base":{"ref":"main"}},
			{"number":5,"title":"Fix bug","state":"open","draft":false,"updated_at":"2026-06-30T09:00:00Z",
			 "user":{"login":"bob"},"head":{"ref":"bugfix"},"base":{"ref":"main"}}
		]`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "prs", Org: "octo", Repo: "hello"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "pull requests of /repos/octo/hello (state open)")
	assert.Contains(t, res.Content, "#7  Add feature  (alice, feature-x -> main, open, draft, updated 2026-07-01T10:00:00Z)")
	assert.Contains(t, res.Content, "#5  Fix bug  (bob, bugfix -> main, open, updated 2026-06-30T09:00:00Z)")
}

func TestRepoPRListStateAllAndEmpty(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		assert.Equal(t, "all", c.Query.Get("state"))
		return http.StatusOK, `[]`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "prs", Org: "octo", Repo: "hello", State: "ALL"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "(no pull requests found)")
}

func TestRepoPRListInvalidState(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "prs", Org: "octo", Repo: "hello", State: "merged"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `"open", "closed", "all"`)
}

func TestRepoPRReadRendersDetailsAndFiles(t *testing.T) {
	longBody := strings.Repeat("b", repoBodyMaxRunes+10)
	_, ex := newFakeGitHub(t, GitHubConfig{Cache: newMemCache()}, func(c ghCall) (int, string) {
		switch {
		case c.Path == "/repos/octo/hello/pulls/7" && strings.Contains(c.Accept, "vnd.github+json"):
			return http.StatusOK, jsonMust(jsonObj{
				"number":        7,
				"title":         "Add feature",
				"state":         "open",
				"draft":         true,
				"body":          longBody,
				"html_url":      "https://github.com/octo/hello/pull/7",
				"updated_at":    "2026-07-01T10:00:00Z",
				"changed_files": 2,
				"user": jsonObj{
					"login": "alice",
				},
				"head": jsonObj{
					"ref": "feature-x",
				},
				"base": jsonObj{
					"ref": "main",
				},
			})
		case c.Path == "/repos/octo/hello/pulls/7/files":
			assert.Equal(t, fmt.Sprint(repoPRMaxFiles), c.Query.Get("per_page"))
			return http.StatusOK, `[
				{"filename":"a.go","status":"modified","additions":10,"deletions":2},
				{"filename":"b.go","status":"added","additions":30,"deletions":0}
			]`
		}
		t.Fatalf("unexpected request %s %s (accept %s)", c.Method, c.Path, c.Accept)
		return 0, ""
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "pr", Org: "octo", Repo: "hello", Number: 7})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "pull request #7 of /repos/octo/hello: Add feature")
	assert.Contains(t, res.Content, "open, draft, by alice, feature-x -> main, updated 2026-07-01T10:00:00Z")
	assert.Contains(t, res.Content, "https://github.com/octo/hello/pull/7")
	assert.Contains(t, res.Content, fmt.Sprintf("(truncated to %d characters)", repoBodyMaxRunes))
	assert.Contains(t, res.Content, "changed files (2):")
	assert.Contains(t, res.Content, "  a.go  +10 -2")
	assert.Contains(t, res.Content, "  b.go  +30 -0  (added)")
	assert.NotContains(t, res.Content, "diff:") // include_diff not requested
}

func TestRepoPRReadIncludeDiff(t *testing.T) {
	var diffFetched bool
	_, ex := newFakeGitHub(t, GitHubConfig{Cache: newMemCache()}, func(c ghCall) (int, string) {
		switch {
		case c.Path == "/repos/octo/hello/pulls/7" && strings.Contains(c.Accept, "diff"):
			diffFetched = true
			return http.StatusOK, "diff --git a/a.go b/a.go\n+added line"
		case c.Path == "/repos/octo/hello/pulls/7":
			return http.StatusOK, `{"number":7,"title":"T","state":"closed","merged":true,
				"user":{"login":"alice"},"head":{"ref":"h"},"base":{"ref":"m"},"changed_files":1}`
		case c.Path == "/repos/octo/hello/pulls/7/files":
			return http.StatusOK, `[{"filename":"a.go","additions":1,"deletions":0,"status":"modified"}]`
		}
		t.Fatalf("unexpected request %s %s", c.Method, c.Path)
		return 0, ""
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "pr", Org: "octo", Repo: "hello", Number: 7, IncludeDiff: true})
	require.False(t, res.IsError, res.Content)
	assert.True(t, diffFetched)
	assert.Contains(t, res.Content, "merged, by alice")
	assert.Contains(t, res.Content, "(no description)")
	assert.Contains(t, res.Content, "diff:\ndiff --git a/a.go b/a.go")
}

func TestRepoPRReadFilesTruncationNote(t *testing.T) {
	var fileRows []string
	for i := 0; i < repoPRMaxFiles; i++ {
		fileRows = append(fileRows, jsonMust(jsonObj{
			"filename":  fmt.Sprintf("f%03d.go", i),
			"additions": 1,
			"deletions": 1,
			"status":    "modified",
		}))
	}
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/pulls/9":
			return http.StatusOK, `{"number":9,"title":"Big","state":"open","changed_files":150,
				"user":{"login":"alice"},"head":{"ref":"h"},"base":{"ref":"m"}}`
		case "/repos/octo/hello/pulls/9/files":
			return http.StatusOK, "[" + strings.Join(fileRows, ",") + "]"
		}
		t.Fatalf("unexpected path %s", c.Path)
		return 0, ""
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "pr", Org: "octo", Repo: "hello", Number: 9})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, fmt.Sprintf("changed files (150, showing first %d):", repoPRMaxFiles))
}

func TestRepoPRReadNotFoundAndArgErrors(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		return http.StatusNotFound, `{"message":"Not Found"}`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "pr", Org: "octo", Repo: "hello", Number: 404})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "could not read pull request #404 of /repos/octo/hello")

	res = execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "pr", Org: "octo", Repo: "hello"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `positive "number"`)
}
