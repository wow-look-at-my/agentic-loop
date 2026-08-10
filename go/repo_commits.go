package agentic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

// what=commits and what=commit read a repository's history: a capped commit
// listing (optionally scoped to a ref and/or path) and one commit's unified
// diff via the GitHub diff media type.

// ghCommitEntry decodes one entry of a /repos/{o}/{r}/commits response.
type ghCommitEntry struct {
	SHA    string `json:"sha"`
	Commit struct {
		Message string `json:"message"`
		Author  struct {
			Name string `json:"name"`
			Date string `json:"date"`
		} `json:"author"`
	} `json:"commit"`
	Author *struct {
		Login string `json:"login"`
	} `json:"author"`
}

func (e *repoTools) commitList(ctx context.Context, in repoReadArgs) ToolResult {
	if err := requireOrgRepo("repo_read what=commits", in.Org, in.Repo); err != nil {
		return ToolResult{Content: err.Error(), IsError: true}
	}
	q := url.Values{}
	q.Set("per_page", strconv.Itoa(clampPerPage(in.PerPage)))
	if ref := strings.TrimSpace(in.Ref); ref != "" {
		q.Set("sha", ref) // the commits API calls the starting ref "sha"
	}
	if p := strings.Trim(strings.TrimSpace(in.Path), "/"); p != "" {
		q.Set("path", p)
	}
	target := fmt.Sprintf("%s/repos/%s/%s/commits?%s", e.gh.base, url.PathEscape(in.Org), url.PathEscape(in.Repo), q.Encode())
	res, err := e.gh.FetchURL(ctx, RepoCacheKey(in.Org, in.Repo), target, "application/vnd.github+json")
	if err != nil {
		return ToolResult{Content: "repo_read what=commits request failed: " + err.Error(), IsError: true}
	}
	if res.status < 200 || res.status >= 300 {
		return ToolResult{Content: DescribeResourceFailure("list commits of", RepoPath(in.Org, in.Repo, ""), res, len(e.gh.tokens)), IsError: true}
	}
	var commits []ghCommitEntry
	if uerr := json.Unmarshal(res.body, &commits); uerr != nil {
		return ToolResult{Content: "repo_read what=commits: could not parse GitHub's response: " + uerr.Error(), IsError: true}
	}
	return ToolResult{Content: formatCommits(in.Org, in.Repo, in.Ref, in.Path, commits)}
}

// formatCommits renders one commit per line: short SHA, ISO date, author, and
// the message subject.
func formatCommits(org, repo, ref, path string, commits []ghCommitEntry) string {
	header := "commits of " + RepoPath(org, repo, "")
	var scope []string
	// Name the ref even when the caller passed none. Saying only "commits of
	// <repo>" leaves the reader unable to tell WHICH branch answered, so one
	// that needs a NAMED branch cannot use the result and re-reads every
	// repository through the raw API instead: six such reads cost ~10k prompt
	// tokens to establish six branch heads. A caller who wants a named branch
	// passes "ref"; this says plainly when they did not, which is what makes
	// the second call targeted instead of a hand audit.
	if r := strings.TrimSpace(ref); r != "" {
		scope = append(scope, "ref "+r)
	} else {
		scope = append(scope, "the repository's default branch -- pass \"ref\" to list another")
	}
	if p := strings.TrimSpace(path); p != "" {
		scope = append(scope, "path "+p)
	}
	if len(scope) > 0 {
		header += " (" + strings.Join(scope, ", ") + ")"
	}
	if len(commits) == 0 {
		return header + "\n(no commits found)"
	}
	var b strings.Builder
	b.WriteString(header + "\n")
	for _, c := range commits {
		author := strings.TrimSpace(c.Commit.Author.Name)
		if author == "" && c.Author != nil {
			author = c.Author.Login
		}
		if author == "" {
			author = "(unknown)"
		}
		fmt.Fprintf(&b, "%s  %s  %s  %s\n", shortSHA(c.SHA), c.Commit.Author.Date, author, firstLine(c.Commit.Message))
	}
	return strings.TrimRight(b.String(), "\n")
}

// commitRead (what=commit) fetches one commit's unified diff by SHA.
func (e *repoTools) commitRead(ctx context.Context, in repoReadArgs) ToolResult {
	if err := requireOrgRepo("repo_read what=commit", in.Org, in.Repo); err != nil {
		return ToolResult{Content: err.Error(), IsError: true}
	}
	sha := strings.TrimSpace(in.SHA)
	if sha == "" {
		return ToolResult{Content: `repo_read what=commit requires "sha" (a commit SHA, e.g. from what=commits)`, IsError: true}
	}
	target := fmt.Sprintf("%s/repos/%s/%s/commits/%s", e.gh.base, url.PathEscape(in.Org), url.PathEscape(in.Repo), url.PathEscape(sha))
	res, err := e.gh.FetchURL(ctx, RepoCacheKey(in.Org, in.Repo), target, "application/vnd.github.diff")
	if err != nil {
		return ToolResult{Content: "repo_read what=commit request failed: " + err.Error(), IsError: true}
	}
	if res.status < 200 || res.status >= 300 {
		return ToolResult{Content: DescribeResourceFailure("read", "commit "+sha+" of "+RepoPath(in.Org, in.Repo, ""), res, len(e.gh.tokens)), IsError: true}
	}
	diff, truncated := TruncateRunes(string(res.body), RepoDiffMaxRunes)
	header := fmt.Sprintf("commit %s of %s", sha, RepoPath(in.Org, in.Repo, ""))
	if truncated || res.truncated {
		header += fmt.Sprintf("\n(diff truncated to %d characters)", RepoDiffMaxRunes)
	}
	return ToolResult{Content: header + "\n\n" + diff}
}
