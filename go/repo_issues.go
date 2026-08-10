package agentic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

// what=issues and what=issue read a repository's issues. GitHub's /issues
// endpoints return pull requests too (a PR is an issue underneath); these
// reads filter them out (list) or redirect to what=pr (read) so the model
// gets true issues only.
const repoIssueMaxComments = 30 // comments shown per issue read

// ghIssue decodes the slice of an issue the reads surface.
type ghIssue struct {
	Number    int       `json:"number"`
	Title     string    `json:"title"`
	State     string    `json:"state"`
	Body      string    `json:"body"`
	HTMLURL   string    `json:"html_url"`
	UpdatedAt string    `json:"updated_at"`
	Comments  int       `json:"comments"`
	Labels    []ghLabel `json:"labels"`
	User      struct {
		Login string `json:"login"`
	} `json:"user"`
	// PullRequest is present when the "issue" is actually a pull request (the
	// issues API returns both kinds).
	PullRequest json.RawMessage `json:"pull_request"`
}

// isPR reports whether this issues-API entry is really a pull request.
func (i ghIssue) isPR() bool {
	s := strings.TrimSpace(string(i.PullRequest))
	return s != "" && s != "null"
}

// ghIssueComment is one issue comment.
type ghIssueComment struct {
	Body      string `json:"body"`
	CreatedAt string `json:"created_at"`
	User      struct {
		Login string `json:"login"`
	} `json:"user"`
}

func (e *repoTools) issueList(ctx context.Context, in repoReadArgs) ToolResult {
	if err := requireOrgRepo("repo_read what=issues", in.Org, in.Repo); err != nil {
		return ToolResult{Content: err.Error(), IsError: true}
	}
	state, err := parseListState(in.State)
	if err != nil {
		return ToolResult{Content: "repo_read what=issues: " + err.Error(), IsError: true}
	}
	q := url.Values{}
	q.Set("state", state)
	q.Set("per_page", strconv.Itoa(clampPerPage(in.PerPage)))
	labels := strings.TrimSpace(in.Labels)
	if labels != "" {
		q.Set("labels", labels)
	}
	target := fmt.Sprintf("%s/repos/%s/%s/issues?%s", e.gh.base, url.PathEscape(in.Org), url.PathEscape(in.Repo), q.Encode())
	res, ferr := e.gh.FetchURL(ctx, RepoCacheKey(in.Org, in.Repo), target, "application/vnd.github+json")
	if ferr != nil {
		return ToolResult{Content: "repo_read what=issues request failed: " + ferr.Error(), IsError: true}
	}
	if res.status < 200 || res.status >= 300 {
		return ToolResult{Content: DescribeResourceFailure("list issues of", RepoPath(in.Org, in.Repo, ""), res, len(e.gh.tokens)), IsError: true}
	}
	var issues []ghIssue
	if uerr := json.Unmarshal(res.body, &issues); uerr != nil {
		return ToolResult{Content: "repo_read what=issues: could not parse GitHub's response: " + uerr.Error(), IsError: true}
	}
	// GitHub's issues list includes pull requests; keep true issues only.
	kept := issues[:0:0]
	for _, is := range issues {
		if !is.isPR() {
			kept = append(kept, is)
		}
	}
	return ToolResult{Content: formatIssueList(in.Org, in.Repo, state, labels, kept)}
}

// formatIssueList renders one issue per line.
func formatIssueList(org, repo, state, labels string, issues []ghIssue) string {
	header := fmt.Sprintf("issues of %s (state %s", RepoPath(org, repo, ""), state)
	if labels != "" {
		header += ", labels " + labels
	}
	header += ")"
	if len(issues) == 0 {
		return header + "\n(no issues found)"
	}
	var b strings.Builder
	b.WriteString(header + "\n")
	for _, is := range issues {
		fmt.Fprintf(&b, "#%d  %s", is.Number, firstLine(is.Title))
		if names := labelNames(is.Labels); names != "" {
			fmt.Fprintf(&b, "  [%s]", names)
		}
		fmt.Fprintf(&b, "  (%s, %s, updated %s)\n", is.User.Login, is.State, is.UpdatedAt)
	}
	return strings.TrimRight(b.String(), "\n")
}

func (e *repoTools) issueRead(ctx context.Context, in repoReadArgs) ToolResult {
	if err := requireOrgRepo("repo_read what=issue", in.Org, in.Repo); err != nil {
		return ToolResult{Content: err.Error(), IsError: true}
	}
	if in.Number <= 0 {
		return ToolResult{Content: `repo_read what=issue requires a positive "number" (an issue number, e.g. from what=issues)`, IsError: true}
	}
	key := RepoCacheKey(in.Org, in.Repo)
	issueURL := fmt.Sprintf("%s/repos/%s/%s/issues/%d", e.gh.base, url.PathEscape(in.Org), url.PathEscape(in.Repo), in.Number)
	what := fmt.Sprintf("issue #%d of %s", in.Number, RepoPath(in.Org, in.Repo, ""))

	res, err := e.gh.FetchURL(ctx, key, issueURL, "application/vnd.github+json")
	if err != nil {
		return ToolResult{Content: "repo_read what=issue request failed: " + err.Error(), IsError: true}
	}
	if res.status < 200 || res.status >= 300 {
		return ToolResult{Content: DescribeResourceFailure("read", what, res, len(e.gh.tokens)), IsError: true}
	}
	var is ghIssue
	if uerr := json.Unmarshal(res.body, &is); uerr != nil {
		return ToolResult{Content: "repo_read what=issue: could not parse GitHub's response: " + uerr.Error(), IsError: true}
	}
	if is.isPR() {
		return ToolResult{Content: fmt.Sprintf("#%d in %s/%s is a pull request, not an issue — use repo_read what=pr to read it.", in.Number, in.Org, in.Repo), IsError: true}
	}

	// Comments are best-effort: a failure becomes a note, not a failed read.
	var comments []ghIssueComment
	commentsErr := ""
	if is.Comments > 0 {
		curl := fmt.Sprintf("%s/comments?per_page=%d", issueURL, repoIssueMaxComments)
		cres, cerr := e.gh.FetchURL(ctx, key, curl, "application/vnd.github+json")
		switch {
		case cerr != nil:
			commentsErr = cerr.Error()
		case cres.status < 200 || cres.status >= 300:
			commentsErr = fmt.Sprintf("status %d", cres.status)
		default:
			if uerr := json.Unmarshal(cres.body, &comments); uerr != nil {
				commentsErr = uerr.Error()
			}
		}
	}
	return ToolResult{Content: formatIssue(in.Org, in.Repo, is, comments, commentsErr)}
}

// formatIssue renders one issue: header, body (capped), and comments (each
// capped, count-capped with an explicit note).
func formatIssue(org, repo string, is ghIssue, comments []ghIssueComment, commentsErr string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "issue #%d of %s: %s\n", is.Number, RepoPath(org, repo, ""), firstLine(is.Title))
	line := fmt.Sprintf("%s, by %s, updated %s", is.State, is.User.Login, is.UpdatedAt)
	if names := labelNames(is.Labels); names != "" {
		line += ", labels [" + names + "]"
	}
	b.WriteString(line + "\n")
	if is.HTMLURL != "" {
		b.WriteString(is.HTMLURL + "\n")
	}
	b.WriteString("\n")
	if body := strings.TrimSpace(is.Body); body != "" {
		b.WriteString(CappedText(body, repoBodyMaxRunes) + "\n")
	} else {
		b.WriteString("(no description)\n")
	}
	switch {
	case commentsErr != "":
		b.WriteString("\n(could not fetch comments: " + commentsErr + ")")
	case is.Comments == 0:
		b.WriteString("\n(no comments)")
	default:
		shown := comments
		if len(shown) > repoIssueMaxComments {
			shown = shown[:repoIssueMaxComments]
		}
		fmt.Fprintf(&b, "\ncomments (%d", is.Comments)
		if is.Comments > len(shown) {
			fmt.Fprintf(&b, ", showing first %d", len(shown))
		}
		b.WriteString("):\n")
		for _, c := range shown {
			fmt.Fprintf(&b, "\n--- %s (%s)\n%s\n", c.User.Login, c.CreatedAt, CappedText(strings.TrimSpace(c.Body), repoCommentMaxRunes))
		}
	}
	return strings.TrimRight(b.String(), "\n")
}
