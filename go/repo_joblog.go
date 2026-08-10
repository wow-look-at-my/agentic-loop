package agentic

import (
	"context"
	"fmt"
	"strings"
)

// what=job_log reads one Actions job's own log.
//
// It is the end of the only chain a token without the Checks API has. The
// check-run detail read (what=check_run) carries a failing step's error
// annotations, and a fine-grained PAT cannot reach it at all -- "Checks" is not
// a repository permission a PAT can be granted. The Actions fallback
// (repo_actions.go) gets as far as naming the step that failed, which renames
// the question rather than answering it. The log is where the compiler error,
// the failing assertion and the stack trace actually are, and
// /actions/jobs/{id}/logs is served under `actions`, which a PAT CAN hold.
//
// GitHub answers that endpoint with a 302 to a short-lived signed storage URL.
// The redirect is followed by the http.Client, which drops the Authorization
// header crossing hosts -- correct here, since the signature is the credential
// and forwarding a PAT to storage would leak it.

const (
	// jobLogMaxBytes caps one log read. A job that installs a toolchain and
	// runs a test suite produces a few MB; this holds those whole.
	jobLogMaxBytes = 24 << 20
	// jobLogTailLines is how much of an over-long log is returned when the
	// caller asked for no particular window. The tail is where a failure is:
	// the step that died is the last thing that ran.
	jobLogTailLines = 400
	// jobLogMaxLines caps an explicit window, so one call cannot return a
	// hundred thousand lines into the model's context.
	jobLogMaxLines = 2_000
)

func (e *repoTools) jobLogRead(ctx context.Context, in repoReadArgs) ToolResult {
	if err := requireOrgRepo("repo_read what=job_log", in.Org, in.Repo); err != nil {
		return ToolResult{Content: err.Error(), IsError: true}
	}
	if in.JobID <= 0 {
		return ToolResult{Content: `repo_read what=job_log needs "job_id": the job id what=status prints beside each job as [job N].`, IsError: true}
	}

	resource := fmt.Sprintf("job %d of %s", in.JobID, RepoPath(in.Org, in.Repo, ""))
	target := fmt.Sprintf("%s/actions/jobs/%d/logs", e.gh.RepoURL(in.Org, in.Repo), in.JobID)
	// application/json, though the body that arrives is plain text: the
	// endpoint 415s "Must accept 'application/json'" on the honest Accept,
	// because what it serves under that header is the 302 to storage.
	res, err := e.gh.FetchURLOpts(ctx, RepoCacheKey(in.Org, in.Repo), target, "application/json", FetchOptions{MaxBytes: jobLogMaxBytes})
	if err != nil {
		return ToolResult{Content: fmt.Sprintf("reading the log of %s failed: %s", resource, err.Error()), IsError: true}
	}
	if res.status < 200 || res.status >= 300 {
		return ToolResult{Content: DescribeResourceFailure("read the log of", resource, res, len(e.gh.tokens)), IsError: true}
	}

	log := strings.ReplaceAll(string(res.body), "\r\n", "\n")
	lines := strings.Split(strings.TrimRight(log, "\n"), "\n")
	if len(lines) == 1 && lines[0] == "" {
		return ToolResult{Content: fmt.Sprintf("The log of %s is empty.", resource)}
	}
	return ToolResult{Content: renderJobLog(in, resource, lines, res.truncated)}
}

// renderJobLog picks the window and states, always, which lines these are out
// of how many. A log slice that does not say where it came from reads as the
// whole log, and "the error is not in here" is then a conclusion drawn from
// text the reader never saw.
func renderJobLog(in repoReadArgs, resource string, lines []string, truncated bool) string {
	total := len(lines)
	start, end, tailed := jobLogWindow(in.Offset, in.Limit, total)

	var b strings.Builder
	fmt.Fprintf(&b, "Log of %s -- lines %d-%d of %d", resource, start+1, end, total)
	switch {
	case tailed:
		fmt.Fprintf(&b, " (the tail, where a failing step's error is; earlier lines with {\"offset\":1,\"limit\":%d})", jobLogTailLines)
	case end < total:
		fmt.Fprintf(&b, " (continue with {\"offset\":%d})", end+1)
	}
	b.WriteString("\n")
	if truncated {
		fmt.Fprintf(&b, "GitHub's log exceeded this tool's %d MiB cap and was cut at that point; what follows is the part that was read.\n", jobLogMaxBytes>>20)
	}
	b.WriteString("\n")
	b.WriteString(strings.Join(lines[start:end], "\n"))
	return b.String()
}

// jobLogWindow resolves the requested line window. offset is 1-based to match
// what the header prints, so a reader can page by quoting the numbers back.
func jobLogWindow(offset, limit, total int) (start, end int, tailed bool) {
	if offset <= 0 && limit <= 0 {
		if total <= jobLogTailLines {
			return 0, total, false
		}
		return total - jobLogTailLines, total, true
	}
	start = offset - 1
	if start < 0 {
		start = 0
	}
	if start > total {
		start = total
	}
	if limit <= 0 || limit > jobLogMaxLines {
		limit = jobLogMaxLines
	}
	end = start + limit
	if end > total {
		end = total
	}
	return start, end, false
}
