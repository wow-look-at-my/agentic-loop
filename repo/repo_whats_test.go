package repo

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Every advertised repo_read "what" against the arguments it declares. The
// enum, the dispatch table and the per-what argument allowlist are three views
// of one list, and a read present in one and missing from another is either a
// tool the model is never told about or one that answers "unknown what".

// repoReadArgValues is a plausible value for each argument repo_read
// advertises, so a call can be assembled from any read's field list.
var repoReadArgValues = map[string]any{
	"org": "octo", "repo": "hello", "path": "src", "ref": "main",
	"sha": "abc123def456", "number": 7, "id": 42,
	"state": "all", "labels": "bug", "per_page": 5, "include_diff": true,
}

// respondAnyRepoRead answers every endpoint the reads touch with a
// well-formed empty body of the right JSON shape.
func respondAnyRepoRead(c ghCall) (int, string) {
	for _, listEndpoint := range []string{"/commits", "/pulls", "/issues", "/files", "/comments", "/annotations"} {
		if strings.HasSuffix(c.Path, listEndpoint) {
			return 200, `[]`
		}
	}
	return 200, `{}`
}

func TestEveryRepoReadWhatAnswersACallBuiltFromItsOwnArguments(t *testing.T) {
	for _, what := range repoReadWhatOrder {
		t.Run(what, func(t *testing.T) {
			fields, ok := repoReadFields[what]
			require.True(t, ok, "%q is advertised with no argument list, so nothing validates its calls", what)

			args := map[string]any{"what": what}
			for _, f := range fields {
				v, known := repoReadArgValues[f]
				require.True(t, known, "%q lists an argument %q that repo_read does not advertise", what, f)
				args[f] = v
			}

			g, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "tok"}}}, respondAnyRepoRead)
			res := execRepoTool(t, ex, RepoReadToolName, args)
			require.False(t, res.IsError, "what=%s refused the arguments its own schema documents: %s", what, res.Content)
			assert.NotEmpty(t, g.Calls(), "what=%s answered without reading anything from GitHub", what)
		})
	}
}

// The mirror image: an argument belonging to a different read is refused by
// name, never quietly dropped. A dropped argument returns the read's default
// answer, which looks exactly like an answer to the question that was asked.
func TestEveryRepoReadWhatRefusesAnArgumentItDoesNotUse(t *testing.T) {
	for _, what := range repoReadWhatOrder {
		t.Run(what, func(t *testing.T) {
			var foreign string
			for f := range repoReadArgValues {
				if !fieldAllowed(repoReadFields[what], f) && f != "org" && f != "repo" {
					if foreign == "" || f < foreign {
						foreign = f
					}
				}
			}
			require.NotEmpty(t, foreign, "every advertised argument is used by what=%s", what)

			args := map[string]any{"what": what, "org": "octo", "repo": "hello", foreign: repoReadArgValues[foreign]}
			for _, f := range repoReadFields[what] {
				args[f] = repoReadArgValues[f]
			}

			_, ex := newFakeGitHub(t, GitHubConfig{}, respondAnyRepoRead)
			res := execRepoTool(t, ex, RepoReadToolName, args)
			require.True(t, res.IsError, "what=%s accepted %q, which it ignores", what, foreign)
			assert.Contains(t, res.Content, `"`+foreign+`"`, "the refusal has to name the argument that was wrong")
			assert.Contains(t, res.Content, "was NOT run")
		})
	}
}

// repo_read is history and metadata only: there is no content-search "what"
// left for a caller to reach for.
func TestRepoReadHasNoContentSearchWhat(t *testing.T) {
	for _, gone := range []string{"grep", "search", "code"} {
		_, ok := repoReadWhats[gone]
		assert.False(t, ok, "what=%s must not exist: content search is grep", gone)
	}
	assert.NotContains(t, string(repoReadSchema), `"grep"`)
	assert.NotContains(t, repoReadDescription, "code search")
}

// The advertised enum and the dispatch table are two views of one list; a read
// present in either and missing from the other is a tool the model calls and
// gets "unknown what" from, or one it is never told about.
func TestEveryAdvertisedWhatHasAHandler(t *testing.T) {
	assert.Len(t, repoReadWhats, len(repoReadWhatOrder))
	for _, what := range repoReadWhatOrder {
		assert.Contains(t, repoReadWhats, what)
		assert.Contains(t, repoReadFields, what, "%q needs an argument allowlist too, or its call is never validated", what)
	}
}
