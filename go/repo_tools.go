package agentic

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// The repo tools give the model real GitHub capability. A required "what"
// selector picks the read, and every one of them is history or metadata the
// REST API serves: commits, one commit's diff, pull requests, issues, CI. A
// repository's FILES are not here -- they are a filesystem the host mounts at
// /repos, listed, read, found and grepped with the file tools (files.go),
// which share this client's credentials and caching. Two approval-gated write
// tools (a single-file commit and PR creation) stay separate: they reach
// GitHub, they use ONLY the write credential list, and they never fall through
// to unauthenticated.
const (
	RepoReadToolName      = "repo_read"
	RepoFileWriteToolName = "repo_file_write"
	RepoPRCreateToolName  = "repo_pr_create"

	// RepoFileMaxRunes caps the text of one file fed back to the model.
	RepoFileMaxRunes = 200_000

	repoListDefaultPerPage = 10      // default page size for the list reads
	repoListMaxPerPage     = 30      // hard cap on a list read's per_page
	RepoDiffMaxRunes       = 200_000 // cap on a commit/PR diff fed back to the model
	repoBodyMaxRunes       = 20_000  // cap on one PR/issue body
	repoCommentMaxRunes    = 5_000   // cap on one issue comment body

	repoReadDescription = "Reads a GitHub repository's HISTORY and METADATA — the parts of a repository that are not files — selected by the required \"what\": " +
		"commits (commit list), commit (one commit with its diff), prs (pull request list), pr (one pull request with changed files), " +
		"issues (issue list), issue (one issue with comments), status (a commit's CI state: legacy commit statuses and GitHub Actions check runs). " +
		"The repository's FILES are not here: they are a filesystem under /repos/<org>/<repo>/<path>, so list them with " + ListDirToolName +
		", read them with " + ReadFileToolName + ", find them by name with " + FindFilesToolName + ", and search what is INSIDE them with " +
		GrepToolName + ". Results are capped, and a capped result always says so."
)

var repoReadSchema = EnumSchema[repoReadArgs](map[string][]string{
	"what":  repoReadWhatOrder,
	"state": {"open", "closed", "all"},
})

// RepoToolsConfig configures NewRepoTools.
type RepoToolsConfig struct {
	// GitHub is the client every read and write runs on. Nil yields no tools:
	// a run with no GitHub access is never offered a tool that can only fail.
	GitHub *GitHub
	// Blocked vetoes a WRITE whose repository the host has open some other way
	// -- a working copy whose staged state a direct commit would bypass. It
	// returns the model-facing refusal, naming what to use instead, or nil to
	// allow. Reads are deliberately never asked: history, pull requests and CI
	// are not things a working copy holds a version of, and gating them left a
	// checked-out repository's own CI unreachable by any route.
	Blocked func(org, repo string) *ToolResult
}

// NewRepoTools returns repo_read plus the two approval-gated writes. The
// writes declare NeedsApproval: they reach GitHub, and no undo exists.
func NewRepoTools(cfg RepoToolsConfig) Tools {
	if cfg.GitHub == nil {
		return nil
	}
	e := &repoTools{gh: cfg.GitHub, blocked: cfg.Blocked}
	return Tools{
		NewTool(ToolDecl{
			Name: RepoReadToolName, Description: repoReadDescription,
			InputSchema: repoReadSchema, Readonly: true,
		}, wrapRepoTool(e.repoRead)),
		approvalTool{NewTool(ToolDecl{
			Name: RepoFileWriteToolName, Description: repoFileWriteDescription,
			InputSchema: repoFileWriteSchema,
		}, wrapRepoTool(e.fileWrite))},
		approvalTool{NewTool(ToolDecl{
			Name: RepoPRCreateToolName, Description: repoPRCreateDescription,
			InputSchema: repoPRCreateSchema,
		}, wrapRepoTool(e.prCreate))},
	}
}

// repoTools is the shared state behind the three tools.
type repoTools struct {
	gh      *GitHub
	blocked func(org, repo string) *ToolResult
}

// wrapRepoTool adapts a handler that cannot fail: every repo failure is a
// recoverable tool result the model can react to, never a failed turn.
func wrapRepoTool(run func(context.Context, json.RawMessage) ToolResult) func(context.Context, json.RawMessage) (ToolResult, error) {
	return func(ctx context.Context, args json.RawMessage) (ToolResult, error) { return run(ctx, args), nil }
}

// approvalTool is a tool whose every call is gated. A host may still override
// the answer -- the preference is the tool's, the decision is the user's.
type approvalTool struct{ Tool }

func (t approvalTool) NeedsApproval() bool { return true }

// block asks the host whether this repository is off limits for a write.
func (e *repoTools) block(org, repo string) *ToolResult {
	if e.blocked == nil {
		return nil
	}
	return e.blocked(org, repo)
}

// repoReadArgs is repo_read's argument union: "what" selects the read, the
// remaining fields apply per-what (see repoReadSchema's field descriptions).
type repoReadArgs struct {
	What        string `json:"what" jsonschema:"Which read to perform: commits (commit list), commit (one commit with diff), prs (PR list), pr (one PR), issues (issue list), issue (one issue with comments), status (a commit's CI state, with every failing check explained), check_run (one check run's full output and error annotations), job_log (a failing Actions job's own log — the compiler error, the failing assertion, the stack trace). To list, read, find or SEARCH files use the filesystem tools on /repos/<org>/<repo>/<path> instead."`
	Org         string `json:"org,omitempty" jsonschema:"Repository owner (org or user)."`
	Repo        string `json:"repo,omitempty" jsonschema:"Repository name."`
	Path        string `json:"path,omitempty" jsonschema:"For what=commits: only list commits touching this file or directory."`
	Ref         string `json:"ref,omitempty" jsonschema:"Optional branch, tag, or commit SHA. For what=commits: the starting point to list from. For what=status: the commit whose CI status to report — pass a branch to get that branch's HEAD. Defaults to the repository's default branch."`
	ID          int64  `json:"id,omitempty" jsonschema:"For what=check_run: the check run id, as what=status lists it beside each check ([id N])."`
	SHA         string `json:"sha,omitempty" jsonschema:"For what=commit: the commit SHA (or any ref resolving to one) whose details and diff to fetch, e.g. from what=commits."`
	Number      int    `json:"number,omitempty" jsonschema:"For what=pr / what=issue: the pull request or issue number, e.g. from what=prs / what=issues."`
	State       string `json:"state,omitempty" jsonschema:"For what=prs / what=issues: which to list. Defaults to open."`
	Labels      string `json:"labels,omitempty" jsonschema:"For what=issues: optional comma-separated label names; only issues carrying all of them are listed."`
	PerPage     int    `json:"per_page,omitempty" jsonschema:"Result count for the list reads (what=commits/prs/issues): 1-30, default 10."`
	IncludeDiff bool   `json:"include_diff,omitempty" jsonschema:"For what=pr: also fetch the PR's full diff (capped). Defaults to false."`
	JobID       int64  `json:"job_id,omitempty" jsonschema:"For what=job_log: the Actions job id, as what=status prints it beside each job ([job N])."`
	Offset      int    `json:"offset,omitempty" jsonschema:"For what=job_log: 1-based first line to return. Omitted, the log's tail is returned (where a failing step's error is)."`
	Limit       int    `json:"limit,omitempty" jsonschema:"For what=job_log: how many lines to return from offset."`
}

// repoReadWhatOrder is the one declaration of which reads exist and in what
// order. The handler table, the schema's enum and the "must be one of" error
// all derive from it, so a read cannot be added to one and missing from
// another.
var repoReadWhatOrder = []string{"commits", "commit", "prs", "pr", "issues", "issue", "status", "check_run", "job_log"}

// repoReadWhats maps each valid "what" to its implementation.
var repoReadWhats = map[string]func(*repoTools, context.Context, repoReadArgs) ToolResult{
	"commits":   (*repoTools).commitList,
	"commit":    (*repoTools).commitRead,
	"prs":       (*repoTools).prList,
	"pr":        (*repoTools).prRead,
	"issues":    (*repoTools).issueList,
	"issue":     (*repoTools).issueRead,
	"status":    (*repoTools).statusRead,
	"check_run": (*repoTools).checkRunRead,
	"job_log":   (*repoTools).jobLogRead,
}

var repoReadWhatList = strings.Join(repoReadWhatOrder, ", ")

// repoReadMovedWhats names the reads that became filesystem operations, so a
// model still calling them by the old name is redirected rather than told the
// what is merely unknown.
var repoReadMovedWhats = map[string]string{
	"tree":      ListDirToolName + ` on the repository path, e.g. {"path": "/repos/<org>/<repo>/<dir>"}`,
	"file":      ReadFileToolName + ` on the repository path, e.g. {"path": "/repos/<org>/<repo>/<path/to/file>"}`,
	"filenames": FindFilesToolName + ` on the repository path, e.g. {"path": "/repos/<org>/<repo>", "pattern": "*.go"}`,
}

// repoRead is the repo_read tool: it parses the argument union, validates
// "what", and dispatches to the per-read implementation. Every failure is a
// recoverable tool error whose text teaches the correct call shape.
func (e *repoTools) repoRead(ctx context.Context, args json.RawMessage) ToolResult {
	var in repoReadArgs
	if err := json.Unmarshal(args, &in); err != nil {
		return ToolResult{Content: "invalid repo_read arguments: " + err.Error(), IsError: true}
	}
	what := strings.ToLower(strings.TrimSpace(in.What))
	if what == "" {
		return ToolResult{Content: `repo_read requires "what": one of ` + repoReadWhatList, IsError: true}
	}
	handler, ok := repoReadWhats[what]
	if !ok {
		if moved, gone := repoReadMovedWhats[what]; gone {
			return ToolResult{Content: fmt.Sprintf(
				"repo_read no longer does what=%s: repository files are a filesystem now. Use %s.", what, moved), IsError: true}
		}
		return ToolResult{Content: fmt.Sprintf("repo_read: unknown what %q; must be one of %s", in.What, repoReadWhatList), IsError: true}
	}
	if err := validateRepoReadArgs(what, args); err != nil {
		return ToolResult{Content: err.Error(), IsError: true}
	}
	// Deliberately NOT gated by workspace mode. Every read here is history or
	// metadata — commits, pull requests, issues, CI — and the workspace holds
	// no version of any of it, so there is nothing a direct read could show
	// staler than the workspace does. Gating them left the workspace
	// repository's CI unreachable by any route at all; the files, which the
	// workspace DOES hold, are gated in fs.go instead. see workspace_mode.go
	return handler(e, ctx, in)
}
