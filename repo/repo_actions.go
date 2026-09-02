package repo

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// The Actions-API account of a commit's CI, used when the Checks API cannot be read.

const (
	// actionsRunLimit bounds how many workflow runs for commit are reported.
	actionsRunLimit = 5
	// actionsJobLimit bounds the jobs listed per run.
	actionsJobLimit = 30
	// actionsStepLimit bounds the failed steps named per job.
	actionsStepLimit = 10
)

// ghWorkflowRun decodes entry of /actions/runs.
type ghWorkflowRun struct {
	ID         int64  `json:"id"`
	Name       string `json:"name"`
	Status     string `json:"status"`
	Conclusion string `json:"conclusion"`
	HTMLURL    string `json:"html_url"`
	Event      string `json:"event"`
}

// ghWorkflowRuns decodes the /actions/runs listing.
type ghWorkflowRuns struct {
	TotalCount   int             `json:"total_count"`
	WorkflowRuns []ghWorkflowRun `json:"workflow_runs"`
}

// ghJob decodes entry of /actions/runs/{id}/jobs.
type ghJob struct {
	ID         int64  `json:"id"`
	Name       string `json:"name"`
	Status     string `json:"status"`
	Conclusion string `json:"conclusion"`
	HTMLURL    string `json:"html_url"`
	Steps      []struct {
		Name       string `json:"name"`
		Status     string `json:"status"`
		Conclusion string `json:"conclusion"`
		Number     int    `json:"number"`
	} `json:"steps"`
}

// ghJobs decodes the jobs listing.
type ghJobs struct {
	TotalCount int     `json:"total_count"`
	Jobs       []ghJob `json:"jobs"`
}

// actionsReport renders the workflow runs for commit, each failed job and
// the steps that failed inside it. The return is a note explaining why
// nothing could be rendered; exactly of the is ever non-empty, so a
// failed read can never pass for "no runs".
func (e *repoTools) actionsReport(ctx context.Context, org, repo, sha string) (string, string) {
	resource := RepoPath(org, repo, "") + "@" + sha
	target := fmt.Sprintf("%s/actions/runs?head_sha=%s&per_page=%d", e.gh.RepoURL(org, repo), EscapeSegments(sha), actionsRunLimit)
	res, err := e.gh.FetchURL(ctx, RepoCacheKey(org, repo), target, "application/vnd.github+json")
	switch {
	case err != nil:
		return "", "the request failed: " + err.Error()
	case res.status < 200 || res.status >= 300:
		return "", DescribeResourceFailure("read the workflow runs of", resource, res, len(e.gh.tokens))
	}
	var runs ghWorkflowRuns
	if uerr := json.Unmarshal(res.body, &runs); uerr != nil {
		return "", "could not parse GitHub's workflow-runs response: " + uerr.Error()
	}
	if len(runs.WorkflowRuns) == 0 {
		return "", "GitHub reports no workflow runs for this commit"
	}

	var b strings.Builder
	for _, run := range runs.WorkflowRuns {
		state := run.Status
		if run.Status == "completed" && run.Conclusion != "" {
			state = run.Conclusion
		}
		fmt.Fprintf(&b, "  %s: %s", nameOrUnnamed(run.Name), state)
		if run.HTMLURL != "" {
			fmt.Fprintf(&b, " (%s)", run.HTMLURL)
		}
		b.WriteString("\n")
		b.WriteString(e.jobsReport(ctx, org, repo, run))
	}
	// The listing is capped, and a cap that does not say so reads as the whole
	// set of runs this commit produced.
	if runs.TotalCount > len(runs.WorkflowRuns) {
		fmt.Fprintf(&b, "  (%d of %d runs shown)\n", len(runs.WorkflowRuns), runs.TotalCount)
	}
	return strings.TrimRight(b.String(), "\n"), ""
}

// jobsReport renders run's jobs, indented under it: every job's verdict,
// and for a failed the steps that failed inside it — the line a reader is
// actually after.
func (e *repoTools) jobsReport(ctx context.Context, org, repo string, run ghWorkflowRun) string {
	target := fmt.Sprintf("%s/actions/runs/%d/jobs?per_page=%d", e.gh.RepoURL(org, repo), run.ID, actionsJobLimit)
	res, err := e.gh.FetchURL(ctx, RepoCacheKey(org, repo), target, "application/vnd.github+json")
	switch {
	case err != nil:
		return fmt.Sprintf("    jobs unavailable -- the request failed: %s\n", err.Error())
	case res.status < 200 || res.status >= 300:
		return fmt.Sprintf("    jobs unavailable -- %s\n",
			DescribeResourceFailure("read the jobs of", fmt.Sprintf("run %d of %s", run.ID, RepoPath(org, repo, "")), res, len(e.gh.tokens)))
	}
	var jobs ghJobs
	if uerr := json.Unmarshal(res.body, &jobs); uerr != nil {
		return "    jobs unavailable -- could not parse GitHub's jobs response: " + uerr.Error() + "\n"
	}
	if len(jobs.Jobs) == 0 {
		return "    (no jobs)\n"
	}

	var b strings.Builder
	for _, job := range jobs.Jobs {
		state := job.Status
		if job.Status == "completed" && job.Conclusion != "" {
			state = job.Conclusion
		}
		fmt.Fprintf(&b, "    %s: %s", nameOrUnnamed(job.Name), state)
		if job.ID != 0 {
			fmt.Fprintf(&b, " [job %d]", job.ID)
		}
		if job.HTMLURL != "" {
			fmt.Fprintf(&b, " (%s)", job.HTMLURL)
		}
		b.WriteString("\n")
		if !failedConclusions.Contains(strings.ToLower(job.Conclusion)) {
			continue
		}
		shown := 0
		for _, step := range job.Steps {
			if !failedConclusions.Contains(strings.ToLower(step.Conclusion)) {
				continue
			}
			if shown >= actionsStepLimit {
				fmt.Fprintf(&b, "      (more failed steps not listed; %s has them)\n", job.HTMLURL)
				break
			}
			fmt.Fprintf(&b, "      step %d failed: %s (%s)\n", step.Number, nameOrUnnamed(step.Name), step.Conclusion)
			shown++
		}
		// Naming the step that died renames the question; the log has the real error.
		if job.ID != 0 {
			fmt.Fprintf(&b, "      %s {\"what\":\"job_log\",\"org\":%q,\"repo\":%q,\"job_id\":%d} gives this job's full log.\n",
				RepoReadToolName, org, repo, job.ID)
		}
	}
	if jobs.TotalCount > len(jobs.Jobs) {
		fmt.Fprintf(&b, "    (%d of %d jobs shown)\n", len(jobs.Jobs), jobs.TotalCount)
	}
	return b.String()
}

// nameOrUnnamed keeps an empty name from rendering as a blank column.
func nameOrUnnamed(s string) string {
	if t := strings.TrimSpace(s); t != "" {
		return t
	}
	return "(unnamed)"
}
