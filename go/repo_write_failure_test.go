package agentic

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

// What an exhausted credential rotation reports. A 404 covers a repository no
// token can see, one none of them may write to, and an object that is not
// there -- so which of those it was has to be established, never assumed.

func TestRepoFileWriteAllTokensLackWriteAccess(t *testing.T) {
	_, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "a"}, GitHubToken{ID: "t2", Token: "b"}), func(c ghCall) (int, string) {
		return http.StatusNotFound, `{"message":"Not Found"}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "f.txt",
		Content: "x", Message: "m",
	})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "failed with every configured GitHub token")
	// Every call 404s, the repository probe included, so no credential can see
	// it at all -- the one case where blaming the credentials is established.
	assert.Contains(t, res.Content, "none of them can read octo/hello at all")
}

// The repository reads fine and only the mutation is refused: the tokens are
// the problem, but "cannot see it" would be wrong, so the message says which.
func TestRepoFileWriteReadableButNotWritable(t *testing.T) {
	_, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "a"}), func(c ghCall) (int, string) {
		switch {
		case c.Method == http.MethodPut:
			return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
		case c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Path == "/repos/octo/hello/git/ref/heads/docs":
			return http.StatusOK, `{"object":{"sha":"refsha"}}`
		case c.Path == "/repos/octo/hello/contents/f.txt":
			return http.StatusNotFound, `{"message":"Not Found"}`
		}
		return http.StatusNotFound, `{"message":"Not Found"}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "f.txt",
		Content: "x", Message: "m",
	})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "can read octo/hello but none of them could write to it")
	assert.Contains(t, res.Content, "Contents: read & write")
}
