package repo

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// repo_pr_create, and the properties both write tools share: they are not
// Readonly (so every call goes to the Approver and a run without one refuses
// them), they never reach a subagent's readonly view, and they can be turned
// off like any other tool.

func TestRepoPRCreateDefaultsToDraftAndDefaultBranch(t *testing.T) {
	cache := newMemCache()
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{Cache: cache}, GitHubToken{ID: "t1", Token: "writer"}), func(c ghCall) (int, string) {
		switch {
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Method == http.MethodPost && c.Path == "/repos/octo/hello/pulls":
			return http.StatusCreated, `{"number":8,"html_url":"https://github.com/octo/hello/pull/8","draft":true}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoPRCreateToolName, repoPRCreateArgs{
		Org: "octo", Repo: "hello", Title: "Add feature", Head: "feature-x",
	})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "created draft pull request #8 in octo/hello: Add feature")
	assert.Contains(t, res.Content, "feature-x -> main")
	assert.Contains(t, res.Content, "https://github.com/octo/hello/pull/8")

	var post ghCall
	for _, c := range gh.Calls() {
		if c.Method == http.MethodPost {
			post = c
		}
	}
	require.Equal(t, http.MethodPost, post.Method)
	// base defaults to the repo's default branch; draft defaults to true; no
	// body key when the description is empty.
	assert.JSONEq(t, `{"title":"Add feature","head":"feature-x","base":"main","draft":true}`, post.Body)

	id, _ := cache.Get("octo/hello")
	assert.Equal(t, "t1", id)
}

func TestRepoPRCreateExplicitBaseBodyAndNonDraft(t *testing.T) {
	draftFalse := false
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "writer"}), func(c ghCall) (int, string) {
		switch {
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Method == http.MethodPost && c.Path == "/repos/octo/hello/pulls":
			return http.StatusCreated, `{"number":9,"html_url":"https://github.com/octo/hello/pull/9","draft":false}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoPRCreateToolName, repoPRCreateArgs{
		Org: "octo", Repo: "hello", Title: "T", Head: "h", Base: "release", Body: "Details", Draft: &draftFalse,
	})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "created pull request #9")
	assert.NotContains(t, res.Content, "draft pull request")
	assert.Contains(t, res.Content, "h -> release")

	for _, c := range gh.Calls() {
		if c.Method == http.MethodPost {
			assert.JSONEq(t, `{"title":"T","head":"h","base":"release","body":"Details","draft":false}`, c.Body)
		}
	}
}

func TestRepoPRCreateValidationErrorIsFatal(t *testing.T) {
	postCalls := 0
	_, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "w1"}, GitHubToken{ID: "t2", Token: "w2"}), func(c ghCall) (int, string) {
		switch {
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Method == http.MethodPost:
			postCalls++
			return http.StatusUnprocessableEntity, `{"message":"Validation Failed","errors":[{"message":"A pull request already exists for octo:feature-x."}]}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoPRCreateToolName, repoPRCreateArgs{
		Org: "octo", Repo: "hello", Title: "T", Head: "feature-x",
	})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "A pull request already exists for octo:feature-x.")
	assert.Equal(t, 1, postCalls, "422 must not be retried with other tokens")
}

func TestRepoPRCreateFallsThroughTokens(t *testing.T) {
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "bad"}, GitHubToken{ID: "t2", Token: "writer"}), func(c ghCall) (int, string) {
		if c.Auth == "bad" {
			return http.StatusNotFound, `{"message":"Not Found"}`
		}
		switch {
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Method == http.MethodPost:
			return http.StatusCreated, `{"number":1,"html_url":"u","draft":true}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoPRCreateToolName, repoPRCreateArgs{
		Org: "octo", Repo: "hello", Title: "T", Head: "h",
	})
	require.False(t, res.IsError, res.Content)
	assert.Equal(t, []string{"bad", "writer", "writer"}, gh.Auths())
}

func TestRepoPRCreateArgErrors(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoPRCreateToolName, repoPRCreateArgs{Org: "octo", Repo: "hello", Title: "T"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `"head"`)
}
