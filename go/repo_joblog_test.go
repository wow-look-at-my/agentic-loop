package agentic

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// numberedLog builds a log whose every line names its own number, so an
// assertion about a window is an assertion about which lines came back.
func numberedLog(n int) string {
	lines := make([]string, n)
	for i := range lines {
		lines[i] = "line " + strconv.Itoa(i+1)
	}
	return strings.Join(lines, "\n") + "\n"
}

func TestRepoJobLogReturnsTheWholeLogWhenItFits(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		require.Equal(t, "/repos/octo/hello/actions/jobs/77/logs", c.Path)
		return http.StatusOK, "compiling\nFAIL: TestThing\nexit 1\n"
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "job_log", Org: "octo", Repo: "hello", JobID: 77})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "Log of job 77 of /repos/octo/hello -- lines 1-3 of 3")
	assert.Contains(t, res.Content, "FAIL: TestThing")
}

func TestRepoJobLogReturnsTheTailOfALongLogAndSaysSo(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(ghCall) (int, string) {
		return http.StatusOK, numberedLog(jobLogTailLines + 50)
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "job_log", Org: "octo", Repo: "hello", JobID: 5})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, fmt.Sprintf("lines 51-%d of %d", jobLogTailLines+50, jobLogTailLines+50))
	assert.Contains(t, res.Content, "the tail")
	// The window is what it claims: the first 50 lines are absent, and the
	// last one is present.
	assert.NotContains(t, res.Content, "\nline 50\n")
	assert.Contains(t, res.Content, "line "+strconv.Itoa(jobLogTailLines+50))
}

func TestRepoJobLogWindowIsAddressableSoTheWholeLogIsReachable(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(ghCall) (int, string) {
		return http.StatusOK, numberedLog(1000)
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{
		What: "job_log", Org: "octo", Repo: "hello", JobID: 5, Offset: 1, Limit: 10,
	})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "lines 1-10 of 1000")
	assert.Contains(t, res.Content, `{"offset":11}`)
	assert.Contains(t, res.Content, "line 1\n")
	assert.NotContains(t, res.Content, "line 11\n")
}

// GitHub does not serve the log from the API host: it answers 302 with a
// short-lived signed URL on storage. Following that is the whole read, and the
// Authorization header must NOT cross to the other host -- the signature is the
// credential there, and forwarding a PAT to storage would hand it over.
func TestRepoJobLogFollowsTheRedirectToStorageWithoutForwardingTheToken(t *testing.T) {
	var storageAuth string
	var sawStorage bool
	storage := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawStorage = true
		storageAuth = r.Header.Get("Authorization")
		fmt.Fprint(w, "step failed\npanic: boom\n")
	}))
	t.Cleanup(storage.Close)

	g, ex := newFakeGitHub(t, GitHubConfig{Tokens: []GitHubToken{{ID: "t1", Token: "secret-pat"}}},
		func(c ghCall) (int, string) {
			require.Equal(t, "/repos/octo/hello/actions/jobs/8/logs", c.Path)
			return http.StatusFound, ""
		})
	g.headers = func(ghCall) http.Header {
		return http.Header{"Location": []string{storage.URL + "/signed"}}
	}
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "job_log", Org: "octo", Repo: "hello", JobID: 8})
	require.False(t, res.IsError, res.Content)
	require.True(t, sawStorage, "the redirect was never followed, so no log was read")
	assert.Empty(t, storageAuth, "the PAT was forwarded to the storage host")
	assert.Contains(t, res.Content, "panic: boom")
}

func TestRepoJobLogNeedsAJobID(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(ghCall) (int, string) {
		t.Fatal("no request should be made without a job id")
		return 0, ""
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "job_log", Org: "octo", Repo: "hello"})
	require.True(t, res.IsError)
	assert.Contains(t, res.Content, "job_id")
}

func TestRepoJobLogFailureExplainsItself(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(ghCall) (int, string) {
		return http.StatusNotFound, `{"message":"Not Found"}`
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "job_log", Org: "octo", Repo: "hello", JobID: 9})
	require.True(t, res.IsError)
	assert.Contains(t, res.Content, "job 9 of /repos/octo/hello")
	assert.Contains(t, res.Content, "will not fix itself")
}

// GitHub 404s the logs endpoint for a job that has not finished -- the log is
// not archived to storage yet -- and that looks identical on the wire to a job
// no token can see. The tool must tell them apart rather than call a job that
// is simply still running "gone".
func TestRepoJobLogStillRunningNamesTheJobsRealState(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/actions/jobs/9/logs":
			return http.StatusNotFound, `{"message":"Not Found"}`
		case "/repos/octo/hello/actions/jobs/9":
			return http.StatusOK, `{"id":9,"name":"build","status":"in_progress","conclusion":""}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "job_log", Org: "octo", Repo: "hello", JobID: 9})
	require.True(t, res.IsError)
	assert.Contains(t, res.Content, "still in_progress")
	assert.NotContains(t, res.Content, "will not fix itself")
	assert.NotContains(t, res.Content, "does not exist")
}

// A job GitHub reports as genuinely completed, whose log 404s anyway, keeps
// the ordinary explanation -- the job-status re-read has nothing to add over
// what explainFailure already says.
func TestRepoJobLogCompletedJobKeepsTheOrdinaryFailure(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch c.Path {
		case "/repos/octo/hello/actions/jobs/9/logs":
			return http.StatusNotFound, `{"message":"Not Found"}`
		case "/repos/octo/hello/actions/jobs/9":
			return http.StatusOK, `{"id":9,"name":"build","status":"completed","conclusion":"failure"}`
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "job_log", Org: "octo", Repo: "hello", JobID: 9})
	require.True(t, res.IsError)
	assert.Contains(t, res.Content, "will not fix itself")
}

// The whole chain a PAT-only host has: the Checks API is refused, the Actions
// fallback names the failing job AND the id needed to read its log, and that
// read answers with the log. A step name without a reachable log is where this
// chain used to end.
func TestRepoStatusPointsAtTheFailingJobsLog(t *testing.T) {
	_, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		switch {
		case c.Path == "/repos/octo/hello/commits/main/status":
			return http.StatusOK, `{"state":"failure","sha":"0123456789abcdef","statuses":[]}`
		case c.Path == "/repos/octo/hello/commits/main/check-runs":
			return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
		case strings.HasPrefix(c.Path, "/repos/octo/hello/actions/runs") && strings.HasSuffix(c.Path, "/jobs"):
			return http.StatusOK, `{"total_count":1,"jobs":[{"id":4242,"name":"build","status":"completed","conclusion":"failure","html_url":"https://example.com/j/4242","steps":[{"name":"go-toolchain","status":"completed","conclusion":"failure","number":6}]}]}`
		case strings.HasPrefix(c.Path, "/repos/octo/hello/actions/runs"):
			return http.StatusOK, `{"total_count":1,"workflow_runs":[{"id":31,"name":"CI","status":"completed","conclusion":"failure","html_url":"https://example.com/r/31"}]}`
		case c.Path == "/repos/octo/hello/actions/jobs/4242/logs":
			return http.StatusOK, "go-toolchain: FAIL github.com/octo/hello/internal/x\n"
		default:
			t.Fatalf("unexpected path %q", c.Path)
			return 0, ""
		}
	})

	status := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "status", Org: "octo", Repo: "hello", Ref: "main"})
	require.False(t, status.IsError, status.Content)
	assert.Contains(t, status.Content, "step 6 failed: go-toolchain")
	assert.Contains(t, status.Content, "[job 4242]")
	assert.Contains(t, status.Content, `{"what":"job_log","org":"octo","repo":"hello","job_id":4242}`)

	log := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "job_log", Org: "octo", Repo: "hello", JobID: 4242})
	require.False(t, log.IsError, log.Content)
	assert.Contains(t, log.Content, "FAIL github.com/octo/hello/internal/x")
}
