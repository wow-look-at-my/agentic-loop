package agentic

import (
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRepoCommitsListsAndScopes(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Cache: newMemCache()}, func(c ghCall) (int, string) {
		assert.Equal(t, "/repos/octo/hello/commits", c.Path)
		assert.Equal(t, "main", c.Query.Get("sha"))
		assert.Equal(t, "docs/intro.md", c.Query.Get("path"))
		assert.Equal(t, "10", c.Query.Get("per_page"))
		return http.StatusOK, `[
			{"sha":"0123456789abcdef","commit":{"message":"Add intro\n\nLonger body","author":{"name":"Alice","date":"2026-07-01T10:00:00Z"}}},
			{"sha":"fedcba9876543210","commit":{"message":"Fix typo","author":{"name":"","date":"2026-06-30T09:00:00Z"}},"author":{"login":"bob"}}
		]`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commits", Org: "octo", Repo: "hello", Ref: "main", Path: "docs/intro.md"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "commits of /repos/octo/hello (ref main, path docs/intro.md)")
	// Short SHA, ISO date, author, and the message subject only.
	assert.Contains(t, res.Content, "0123456789  2026-07-01T10:00:00Z  Alice  Add intro")
	assert.NotContains(t, res.Content, "Longer body")
	// git-author name empty: falls back to the GitHub login.
	assert.Contains(t, res.Content, "fedcba9876  2026-06-30T09:00:00Z  bob  Fix typo")
}

func TestRepoCommitsPerPageClampAndEmpty(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		assert.Equal(t, "30", c.Query.Get("per_page"))
		assert.Empty(t, c.Query.Get("sha"))
		return http.StatusOK, `[]`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commits", Org: "octo", Repo: "hello", PerPage: 500})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "(no commits found)")
}

func TestRepoCommitsNotFound(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		return http.StatusNotFound, `{"message":"Not Found"}`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commits", Org: "octo", Repo: "ghost"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "could not list commits of /repos/octo/ghost")
	assert.Contains(t, res.Content, "not found (404)")
}

func TestRepoCommitsArgErrors(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commits", Org: "octo"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `requires both "org" and "repo"`)
}

func TestRepoCommitDiffReturnsPatch(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{Cache: newMemCache()}, func(c ghCall) (int, string) {
		assert.Equal(t, "/repos/octo/hello/commits/abc1234", c.Path)
		assert.Contains(t, c.Accept, "application/vnd.github.diff")
		return http.StatusOK, "diff --git a/f.go b/f.go\n-old\n+new"
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commit", Org: "octo", Repo: "hello", SHA: "abc1234"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "commit abc1234 of /repos/octo/hello")
	assert.Contains(t, res.Content, "diff --git a/f.go b/f.go")
	assert.NotContains(t, res.Content, "truncated")
}

func TestRepoCommitDiffTruncates(t *testing.T) {
	huge := strings.Repeat("a", repoDiffMaxRunes+100)
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		return http.StatusOK, huge
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commit", Org: "octo", Repo: "hello", SHA: "abc"})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "(diff truncated to")
	assert.Less(t, len(res.Content), repoDiffMaxRunes+200)
}

func TestRepoCommitDiffRequiresSHA(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commit", Org: "octo", Repo: "hello"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `requires "sha"`)
}
