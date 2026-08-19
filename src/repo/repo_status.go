package repo

import (
	"context"
	"encoding/json"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop/src"
	"strconv"
	"strings"
	"time"
)

// what=status reads a commit's CI state through GitHub's two separate,
// never-merged mechanisms: the legacy commit-status API (posted by
// "context" — this org's own all-builds gate is one) and the native GitHub
// Actions checks API (check runs). Both are reported, because GitHub itself
// has no single verdict combining them over REST.
//
// "ref" (a branch, tag, or commit SHA) resolves the same way both endpoints
// already resolve it; omitted, it resolves to the repository's default
// branch (defaultBranch, repo_search.go), so "what's the status of this
// repo" means its default branch's HEAD without a round trip to ask what
// that branch is first.
//
// A red check is reported WITH ITS REASON. "CI failed" is the whole of what a
// user usually says, and a report that answers it with "Build and test:
// failure" only renames the question — so every failing check run is followed
// up with its own detail read (title, summary, annotations) and rendered
// inline. see docs/tools/repo-tools.md

const (
	// statusFailureDetailLimit bounds how many failing check runs are followed
	// up with a detail read, so one catastrophic commit cannot turn a single
	// tool call into fifty. Anything past it is NAMED as unexplained rather
	// than dropped: a truncated report that looks complete is worse than none.
	statusFailureDetailLimit = 5
	// statusSummaryMaxRunes caps each inlined failure summary. The drill-down
	// (what=check_run) carries the full text.
	statusSummaryMaxRunes = 4_000
	// checkRunTextMaxRunes caps what=check_run's own output text.
	checkRunTextMaxRunes = 60_000
	// checkRunAnnotationLimit bounds the annotations listed for one check run.
	checkRunAnnotationLimit = 50
)

// ghCombinedStatus decodes a /commits/{ref}/status response.
type ghCombinedStatus struct {
	State    string `json:"state"`
	SHA      string `json:"sha"`
	Statuses []struct {
		Context     string `json:"context"`
		State       string `json:"state"`
		Description string `json:"description"`
		TargetURL   string `json:"target_url"`
	} `json:"statuses"`
}

// ghCheckRun decodes one check run, from either the per-commit listing or the
// single-check-run read.
type ghCheckRun struct {
	ID         int64  `json:"id"`
	Name       string `json:"name"`
	Status     string `json:"status"`
	Conclusion string `json:"conclusion"`
	HTMLURL    string `json:"html_url"`
	DetailsURL string `json:"details_url"`
	Output     struct {
		Title            string `json:"title"`
		Summary          string `json:"summary"`
		Text             string `json:"text"`
		AnnotationsCount int    `json:"annotations_count"`
	} `json:"output"`
}

// ghCheckRunsResponse decodes a /commits/{ref}/check-runs response.
type ghCheckRunsResponse struct {
	CheckRuns []ghCheckRun `json:"check_runs"`
}

// ghAnnotation decodes one check-run annotation: where GitHub Actions records
// the actual error lines a failing step produced.
type ghAnnotation struct {
	Path            string `json:"path"`
	StartLine       int    `json:"start_line"`
	EndLine         int    `json:"end_line"`
	AnnotationLevel string `json:"annotation_level"`
	Title           string `json:"title"`
	Message         string `json:"message"`
}

// failedConclusions are the check-run conclusions that mean a human has to go
// look. "cancelled" is among them: a cancelled run explains nothing by itself,
// and the reason a run was cancelled is exactly what the reader is after.
var failedConclusions = map[string]bool{
	"failure": true, "timed_out": true, "action_required": true, "cancelled": true, "stale": true,
}

// checkRunFailed reports whether a completed check run needs explaining.
func checkRunFailed(c ghCheckRun) bool {
	return c.Status == "completed" && failedConclusions[strings.ToLower(c.Conclusion)]
}

func (e *repoTools) statusRead(ctx context.Context, in repoReadArgs) agentic.ToolResult {
	if err := requireOrgRepo("repo_read what=status", in.Org, in.Repo); err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}
	}
	ref := strings.TrimSpace(in.Ref)
	if ref == "" {
		branch, err := e.gh.DefaultBranch(ctx, in.Org, in.Repo)
		if err != nil {
			return agentic.ToolResult{Content: "repo_read what=status: could not resolve the default branch: " + err.Error(), IsError: true}
		}
		ref = branch
	}
	text, res, err := e.ciStatusReport(ctx, in.Org, in.Repo, ref)
	if err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}
	}
	return agentic.ToolResult{Content: text + tokenExpiryDetail(res, time.Now())}
}

// ciStatusReport renders one commit's full CI report — both mechanisms, with
// every failing check run explained. It is shared with workspace_read
// what=checks, which asks the same question about the attached PR's head, so
// the two can never drift into reporting CI differently.
//
// The returned GHResponse is the combined-status response, carried back only so
// the caller can append its token-expiry note.
// CIStatusReport renders a commit's CI state -- legacy commit statuses and
// Actions check runs together, every failing check explained. A host reporting
// a working copy's own CI calls this, so what it shows and what
// repo_read what=status shows are one rendering rather than two.
func (e *GitHub) CIStatusReport(ctx context.Context, org, repo, ref string) (string, GHResponse, error) {
	return (&repoTools{gh: e}).ciStatusReport(ctx, org, repo, ref)
}

func (e *repoTools) ciStatusReport(ctx context.Context, org, repo, ref string) (string, GHResponse, error) {
	resource := RepoPath(org, repo, "") + "@" + ref

	statusTarget := fmt.Sprintf("%s/commits/%s/status", e.gh.RepoURL(org, repo), EscapeSegments(ref))
	statusRes, err := e.gh.FetchURL(ctx, RepoCacheKey(org, repo), statusTarget, "application/vnd.github+json")
	if err != nil {
		return "", GHResponse{}, fmt.Errorf("reading the CI status of %s failed: %w", resource, err)
	}
	if statusRes.status < 200 || statusRes.status >= 300 {
		return "", GHResponse{}, errStr(DescribeResourceFailure("read the CI status of", resource, statusRes, len(e.gh.tokens)))
	}
	var combined ghCombinedStatus
	if uerr := json.Unmarshal(statusRes.body, &combined); uerr != nil {
		return "", GHResponse{}, fmt.Errorf("could not parse GitHub's status response for %s: %w", resource, uerr)
	}

	checksTarget := fmt.Sprintf("%s/commits/%s/check-runs", e.gh.RepoURL(org, repo), EscapeSegments(ref))
	checksRes, err := e.gh.FetchURL(ctx, RepoCacheKey(org, repo), checksTarget, "application/vnd.github+json")
	var checks ghCheckRunsResponse
	var checksNote string
	switch {
	case err != nil:
		checksNote = "the request failed: " + err.Error()
	case checksRes.status < 200 || checksRes.status >= 300:
		checksNote = DescribeResourceFailure("read the check runs of", resource, checksRes, len(e.gh.tokens))
	default:
		if uerr := json.Unmarshal(checksRes.body, &checks); uerr != nil {
			checksNote = "could not parse GitHub's check-runs response: " + uerr.Error()
		}
	}

	details, undetailed := e.explainFailures(ctx, org, repo, checks.CheckRuns)
	return formatStatus(org, repo, ref, combined, checks, checksNote, details, undetailed), statusRes, nil
}

// errStr is an error whose text is exactly the message given: the failure
// describers already produce the sentence the model should read, and wrapping
// it would only prefix it with the plumbing's own words.
type errStr string

func (e errStr) Error() string { return string(e) }

// explainFailures fetches the detail of each failing check run, up to
// statusFailureDetailLimit. It returns the fetched details by check-run id and
// the names it did NOT explain, so the report can say so rather than let a
// bounded read pass for a complete one.
func (e *repoTools) explainFailures(ctx context.Context, org, repo string, runs []ghCheckRun) (map[int64]ghCheckRun, []string) {
	details := map[int64]ghCheckRun{}
	var undetailed []string
	for _, c := range runs {
		if !checkRunFailed(c) || c.ID == 0 {
			continue
		}
		if len(details) >= statusFailureDetailLimit {
			undetailed = append(undetailed, c.Name)
			continue
		}
		full, err := e.fetchCheckRun(ctx, org, repo, c.ID)
		if err != nil {
			undetailed = append(undetailed, c.Name)
			continue
		}
		details[c.ID] = full
	}
	return details, undetailed
}

// fetchCheckRun reads one check run by id.
func (e *repoTools) fetchCheckRun(ctx context.Context, org, repo string, id int64) (ghCheckRun, error) {
	target := fmt.Sprintf("%s/check-runs/%d", e.gh.RepoURL(org, repo), id)
	res, err := e.gh.FetchURL(ctx, RepoCacheKey(org, repo), target, "application/vnd.github+json")
	if err != nil {
		return ghCheckRun{}, err
	}
	if res.status < 200 || res.status >= 300 {
		return ghCheckRun{}, errStr(DescribeResourceFailure("read", fmt.Sprintf("check run %d of %s", id, RepoPath(org, repo, "")), res, len(e.gh.tokens)))
	}
	var c ghCheckRun
	if err := json.Unmarshal(res.body, &c); err != nil {
		return ghCheckRun{}, fmt.Errorf("could not parse GitHub's check-run response: %w", err)
	}
	return c, nil
}

// formatStatus renders both CI mechanisms as one report. A check-runs failure
// is noted, not fatal — a token can read the legacy status and lack Checks API
// access (or vice versa), and a partial answer beats none.
func formatStatus(org, repo, ref string, combined ghCombinedStatus, checks ghCheckRunsResponse, checksNote string, details map[int64]ghCheckRun, undetailed []string) string {
	sha := combined.SHA
	if sha == "" {
		sha = ref
	}
	var b strings.Builder
	fmt.Fprintf(&b, "CI status for %s @ %s (commit %s)\n", RepoPath(org, repo, ""), ref, shortSHA(sha))

	state := combined.State
	if state == "" {
		state = "none"
	}
	fmt.Fprintf(&b, "\nCommit status: %s", strings.ToUpper(state))
	if len(combined.Statuses) == 0 {
		b.WriteString(" (no statuses posted)\n")
	} else {
		b.WriteString("\n")
		for _, s := range combined.Statuses {
			fmt.Fprintf(&b, "  %s: %s", s.Context, s.State)
			if s.Description != "" {
				fmt.Fprintf(&b, " -- %s", s.Description)
			}
			if s.TargetURL != "" {
				fmt.Fprintf(&b, " (%s)", s.TargetURL)
			}
			b.WriteString("\n")
		}
	}

	b.WriteString("\nCheck runs")
	switch {
	case checksNote != "":
		fmt.Fprintf(&b, ": unavailable -- %s\n", checksNote)
	case len(checks.CheckRuns) == 0:
		b.WriteString(": (none)\n")
	default:
		b.WriteString(":\n")
		for _, c := range checks.CheckRuns {
			runState := c.Status
			if c.Status == "completed" && c.Conclusion != "" {
				runState = c.Conclusion
			}
			fmt.Fprintf(&b, "  %s: %s", c.Name, runState)
			if c.ID != 0 {
				fmt.Fprintf(&b, " [id %d]", c.ID)
			}
			if c.HTMLURL != "" {
				fmt.Fprintf(&b, " (%s)", c.HTMLURL)
			}
			b.WriteString("\n")
			b.WriteString(formatFailureDetail(details[c.ID]))
		}
	}
	if len(undetailed) > 0 {
		fmt.Fprintf(&b, "\nNot explained here: %s. Read each with %s {\"what\":\"check_run\",\"org\":%q,\"repo\":%q,\"id\":<id>}.\n",
			strings.Join(undetailed, ", "), RepoReadToolName, org, repo)
	}
	if anyFailed(checks.CheckRuns) {
		fmt.Fprintf(&b, "\n%s {\"what\":\"check_run\",\"org\":%q,\"repo\":%q,\"id\":<id from above>} gives one check's full output and its error annotations.\n",
			RepoReadToolName, org, repo)
	}
	return strings.TrimRight(b.String(), "\n")
}

// anyFailed reports whether any check run needs explaining.
func anyFailed(runs []ghCheckRun) bool {
	for _, c := range runs {
		if checkRunFailed(c) {
			return true
		}
	}
	return false
}

// formatFailureDetail renders a failing check's own account of itself, indented
// under its line. A zero-value detail (not fetched, or nothing reported)
// renders nothing.
func formatFailureDetail(c ghCheckRun) string {
	var b strings.Builder
	if t := strings.TrimSpace(c.Output.Title); t != "" {
		fmt.Fprintf(&b, "    %s\n", t)
	}
	if s := strings.TrimSpace(c.Output.Summary); s != "" {
		summary, truncated := agentic.TruncateRunes(s, statusSummaryMaxRunes)
		for _, line := range strings.Split(summary, "\n") {
			fmt.Fprintf(&b, "    %s\n", line)
		}
		if truncated {
			fmt.Fprintf(&b, "    (summary truncated to %d characters; what=check_run has the rest)\n", statusSummaryMaxRunes)
		}
	}
	if c.Output.AnnotationsCount > 0 {
		fmt.Fprintf(&b, "    %d error annotation(s) -- what=check_run id=%d lists them with file and line\n", c.Output.AnnotationsCount, c.ID)
	}
	return b.String()
}

// checkRunRead is what=check_run: one check run's full output and its
// annotations. what=status names the ids; this is the drill-down for when its
// inlined summary is not enough.
func (e *repoTools) checkRunRead(ctx context.Context, in repoReadArgs) agentic.ToolResult {
	if err := requireOrgRepo("repo_read what=check_run", in.Org, in.Repo); err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}
	}
	if in.ID <= 0 {
		return agentic.ToolResult{Content: fmt.Sprintf(
			`repo_read what=check_run requires "id": a check run id, which %s {"what":"status","org":%q,"repo":%q} lists as [id N] beside each check.`,
			RepoReadToolName, in.Org, in.Repo), IsError: true}
	}
	c, err := e.fetchCheckRun(ctx, in.Org, in.Repo, in.ID)
	if err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}
	}
	annotations, annNote := e.fetchAnnotations(ctx, in.Org, in.Repo, in.ID)
	return agentic.ToolResult{Content: formatCheckRun(in.Org, in.Repo, c, annotations, annNote)}
}

// fetchAnnotations reads a check run's annotations. A failure is reported as a
// note rather than as an error: the check's own output is still worth having,
// and a silently empty annotation list would read as "no errors were flagged".
func (e *repoTools) fetchAnnotations(ctx context.Context, org, repo string, id int64) ([]ghAnnotation, string) {
	target := fmt.Sprintf("%s/check-runs/%d/annotations?per_page=%d", e.gh.RepoURL(org, repo), id, checkRunAnnotationLimit)
	res, err := e.gh.FetchURL(ctx, RepoCacheKey(org, repo), target, "application/vnd.github+json")
	if err != nil {
		return nil, "the request failed: " + err.Error()
	}
	if res.status < 200 || res.status >= 300 {
		return nil, DescribeResourceFailure("read the annotations of", fmt.Sprintf("check run %d of %s", id, RepoPath(org, repo, "")), res, len(e.gh.tokens))
	}
	var out []ghAnnotation
	if err := json.Unmarshal(res.body, &out); err != nil {
		return nil, "could not parse GitHub's annotations response: " + err.Error()
	}
	return out, ""
}

// formatCheckRun renders one check run in full.
func formatCheckRun(org, repo string, c ghCheckRun, annotations []ghAnnotation, annNote string) string {
	var b strings.Builder
	runState := c.Status
	if c.Status == "completed" && c.Conclusion != "" {
		runState = c.Conclusion
	}
	fmt.Fprintf(&b, "Check run %d of %s: %s -- %s\n", c.ID, RepoPath(org, repo, ""), c.Name, runState)
	if c.HTMLURL != "" {
		fmt.Fprintf(&b, "%s\n", c.HTMLURL)
	}
	if t := strings.TrimSpace(c.Output.Title); t != "" {
		fmt.Fprintf(&b, "\n%s\n", t)
	}
	if s := strings.TrimSpace(c.Output.Summary); s != "" {
		fmt.Fprintf(&b, "\n%s\n", s)
	}
	if txt := strings.TrimSpace(c.Output.Text); txt != "" {
		body, truncated := agentic.TruncateRunes(txt, checkRunTextMaxRunes)
		fmt.Fprintf(&b, "\n%s\n", body)
		if truncated {
			fmt.Fprintf(&b, "(output truncated to %d characters)\n", checkRunTextMaxRunes)
		}
	}

	b.WriteString("\nAnnotations")
	switch {
	case annNote != "":
		fmt.Fprintf(&b, ": unavailable -- %s\n", annNote)
	case len(annotations) == 0:
		b.WriteString(": (none)\n")
	default:
		b.WriteString(":\n")
		for _, a := range annotations {
			fmt.Fprintf(&b, "  %s\n", formatAnnotation(a))
		}
		// The listing is capped, and a cap that does not say so reads as a
		// complete list of everything that went wrong.
		if len(annotations) >= checkRunAnnotationLimit {
			fmt.Fprintf(&b, "  (listing capped at %d; the check may carry more -- %s has the full set)\n", checkRunAnnotationLimit, c.HTMLURL)
		}
	}
	if strings.TrimSpace(c.Output.Title+c.Output.Summary+c.Output.Text) == "" && len(annotations) == 0 && annNote == "" {
		fmt.Fprintf(&b, "\nThis check reported no output and no annotations. Whatever it logged is only in its own run page: %s\n", c.HTMLURL)
	}
	return strings.TrimRight(b.String(), "\n")
}

// formatAnnotation renders one annotation as "level path:line -- message".
func formatAnnotation(a ghAnnotation) string {
	var b strings.Builder
	level := a.AnnotationLevel
	if level == "" {
		level = "notice"
	}
	b.WriteString(level)
	if a.Path != "" {
		b.WriteString(" " + a.Path)
		if a.StartLine > 0 {
			b.WriteString(":" + strconv.Itoa(a.StartLine))
			if a.EndLine > a.StartLine {
				b.WriteString("-" + strconv.Itoa(a.EndLine))
			}
		}
	}
	msg := strings.TrimSpace(a.Title)
	if m := strings.TrimSpace(a.Message); m != "" {
		if msg != "" {
			msg += ": "
		}
		msg += m
	}
	if msg != "" {
		b.WriteString(" -- " + strings.ReplaceAll(msg, "\n", " "))
	}
	return b.String()
}
