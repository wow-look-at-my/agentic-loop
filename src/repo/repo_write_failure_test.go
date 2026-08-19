package repo

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

// authTestBaseTip is a commit sha long enough that clipping it would show.
const authTestBaseTip = "1111111111111111111111111111111111111111"

func TestAuthFailureRankPrefersWhatExplains(t *testing.T) {
	missing := GitHubAuthError{status: 404, what: "read the base commit " + authTestBaseTip, object: "commit " + authTestBaseTip}
	denied := GitHubAuthError{status: 403, what: "create the tree"}
	hidden := GitHubAuthError{status: 404, what: "create the tree"}
	bad := GitHubAuthError{status: 401, what: "create the tree"}

	// A named missing object is the only one of these that identifies a cause
	// rather than a credential, so nothing displaces it.
	for _, weaker := range []GitHubAuthError{denied, hidden, bad} {
		assert.Equal(t, missing, MoreInformativeAuthFailure(missing, weaker), "%d must not displace a missing object", weaker.status)
		assert.Equal(t, missing, MoreInformativeAuthFailure(weaker, missing), "a missing object must win from either side")
	}
	// A 401 only says the credential was rejected; a 403 says it was
	// recognized and denied, which is a real signal about access.
	assert.Equal(t, denied, MoreInformativeAuthFailure(denied, bad))
	assert.Equal(t, denied, MoreInformativeAuthFailure(bad, denied))
	// Nothing to compare against yet: the first failure is the best so far.
	assert.Equal(t, bad, MoreInformativeAuthFailure(GitHubAuthError{}, bad))
	// Equal rank keeps the later attempt, which is what the loop did before.
	assert.Equal(t, hidden, MoreInformativeAuthFailure(GitHubAuthError{status: 404, what: "write x"}, hidden))
}

func TestCommitReadNamesTheWholeSHA(t *testing.T) {
	// A 404 on the branch point makes that sha the thing the reader has to go
	// and look up, so it is reported whole rather than clipped to ten
	// characters by the message that reports it.
	err := ClassifyObjectRead("read the base commit "+authTestBaseTip, "commit "+authTestBaseTip,
		GHResponse{status: 404, body: []byte(`{"message":"Not Found"}`)})
	var auth GitHubAuthError
	require.ErrorAs(t, err, &auth)
	assert.Equal(t, "commit "+authTestBaseTip, auth.object)
	assert.Contains(t, auth.Error(), authTestBaseTip)
	assert.NotEqual(t, shortSHA(authTestBaseTip), authTestBaseTip, "the sha is long enough for clipping to matter")
}
