package repo

import (
	"context"
	"encoding/json"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"strings"
)

// what=job_log reads Actions job's own log, where the failing assertion lives.

const (
	// jobLogMaxBytes caps log read.
	jobLogMaxBytes = 24 << 20
	// jobLogTailLines is how much of an over-long log is returned when no window is asked.
	jobLogTailLines = 400
	// jobLogMaxLines caps an explicit window.
	jobLogMaxLines = 2_000
)

func (e *repoTools) jobLogRead(ctx context.Context, in repoReadArgs) agentic.ToolResult {
	if err := requireOrgRepo("repo_read what=job_log", in.Org, in.Repo); err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}
	}
	if in.JobID <= 0 {
		return agentic.ToolResult{Content: `repo_read what=job_log needs "job_id": the job id what=status prints beside each job as [job N].`, IsError: true}
	}

	resource := fmt.Sprintf("job %d of %s", in.JobID, RepoPath(in.Org, in.Repo, ""))
	target := fmt.Sprintf("%s/actions/jobs/%d/logs", e.gh.RepoURL(in.Org, in.Repo), in.JobID)
	// application/json though the body is plain text; the endpoint 415s otherwise.
	res, err := e.gh.FetchURLOpts(ctx, RepoCacheKey(in.Org, in.Repo), target,
		"application/json", FetchOptions{MaxBytes: jobLogMaxBytes, NoRedirect: true})
	if err != nil {
		return agentic.ToolResult{Content: fmt.Sprintf("reading the log of %s failed: %s", resource, err.Error()), IsError: true}
	}
	if loc := res.header.Get("Location"); res.status >= 300 && res.status < 400 && loc != "" {
		res, err = e.gh.FetchRedirectTarget(ctx, loc, jobLogMaxBytes)
		if err != nil {
			return agentic.ToolResult{Content: fmt.Sprintf("reading the log of %s failed at its storage URL: %s", resource, err.Error()), IsError: true}
		}
	}
	if res.status < 200 || res.status >= 300 {
		if job, ok := e.fetchJobForLogFailure(ctx, in.Org, in.Repo, in.JobID); ok {
			switch {
			case job.Status != "completed":
				state := job.Status
				if state == "" {
					state = "queued"
				}
				return agentic.ToolResult{Content: fmt.Sprintf(
					"The log of %s is not available yet: the job is still %s, and GitHub does not serve a log until it finishes. This is not a permission or existence problem -- try again once it completes.",
					resource, state), IsError: true}
			case job.Conclusion == "skipped":
				return agentic.ToolResult{Content: fmt.Sprintf(
					"%s was skipped and never ran a step, so GitHub never produced a log for it. This is not a permission problem.",
					strings.ToUpper(resource[:1])+resource[1:]), IsError: true}
			default:
				return agentic.ToolResult{Content: fmt.Sprintf(
					"The log of %s could not be read (404), but the job itself is confirmed to exist and be readable with these tokens (status: completed, conclusion: %s) -- so this is not a token or existence problem with the job. The LOG specifically is what is missing, most likely because it aged out under this repository's or organization's Actions log retention setting (Settings -> Actions -> General -> Artifact and log retention); it will not come back on retry.",
					resource, job.Conclusion), IsError: true}
			}
		}
		return agentic.ToolResult{Content: DescribeResourceFailure("read the log of", resource, res, len(e.gh.tokens)), IsError: true}
	}

	log := strings.ReplaceAll(string(res.body), "\r\n", "\n")
	lines := strings.Split(strings.TrimRight(log, "\n"), "\n")
	if len(lines) == 1 && lines[0] == "" {
		return agentic.ToolResult{Content: fmt.Sprintf("The log of %s is empty.", resource)}
	}
	return agentic.ToolResult{Content: renderJobLog(in, resource, lines, res.truncated)}
}

// fetchJobForLogFailure re-reads GET /actions/jobs/{id} after a failed log
// fetch, so the log failure can be explained by the job's own confirmed
// state instead of assumed to be a permission or existence problem with the
// job. ok is false only when this re-read itself failed -- a job GitHub
// cannot be reached to confirm adds nothing over the log fetch's own error,
// which is the explanation then.
func (e *repoTools) fetchJobForLogFailure(ctx context.Context, org, repo string, jobID int64) (job ghJob, ok bool) {
	target := fmt.Sprintf("%s/actions/jobs/%d", e.gh.RepoURL(org, repo), jobID)
	res, err := e.gh.FetchURL(ctx, RepoCacheKey(org, repo), target, "application/vnd.github+json")
	if err != nil || res.status < 200 || res.status >= 300 {
		return ghJob{}, false
	}
	if json.Unmarshal(res.body, &job) != nil {
		return ghJob{}, false
	}
	return job, true
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

// jobLogWindow resolves the requested line window. offset is -based to match
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
