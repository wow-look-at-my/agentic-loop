package repo

import (
	"context"
	"encoding/json"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"net/url"
	"strings"
)

// what=prs and what=pr read a repository's pull requests: a capped listing,
// and one PR's metadata + body + changed files (optionally with the full diff
// appended).
const repoPRMaxFiles = 100 // changed files listed per PR read

// ghPull decodes the slice of a pull request the reads surface.
type ghPull struct {
	Number    int    `json:"number"`
	Title     string `json:"title"`
	State     string `json:"state"`
	Draft     bool   `json:"draft"`
	Merged    bool   `json:"merged"`
	Body      string `json:"body"`
	HTMLURL   string `json:"html_url"`
	UpdatedAt string `json:"updated_at"`
	// ChangedFiles is only present on the single-PR endpoint (not the list).
	ChangedFiles int `json:"changed_files"`
	User         struct {
		Login string `json:"login"`
	} `json:"user"`
	Head struct {
		Ref string `json:"ref"`
	} `json:"head"`
	Base struct {
		Ref string `json:"ref"`
	} `json:"base"`
}

// ghPRFile is one changed file of a PR.
type ghPRFile struct {
	Filename  string `json:"filename"`
	Status    string `json:"status"`
	Additions int    `json:"additions"`
	Deletions int    `json:"deletions"`
}

func (e *repoTools) prList(ctx context.Context, in repoReadArgs) agentic.ToolResult {
	if err := requireOrgRepo("repo_read what=prs", in.Org, in.Repo); err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}
	}
	state, err := parseListState(in.State)
	if err != nil {
		return agentic.ToolResult{Content: "repo_read what=prs: " + err.Error(), IsError: true}
	}
	target := fmt.Sprintf("%s/repos/%s/%s/pulls?state=%s&per_page=%d",
		e.gh.base, url.PathEscape(in.Org), url.PathEscape(in.Repo), state, clampPerPage(in.PerPage))
	res, ferr := e.gh.FetchURL(ctx, RepoCacheKey(in.Org, in.Repo), target, "application/vnd.github+json")
	if ferr != nil {
		return agentic.ToolResult{Content: "repo_read what=prs request failed: " + ferr.Error(), IsError: true}
	}
	if res.status < 200 || res.status >= 300 {
		return agentic.ToolResult{Content: DescribeResourceFailure("list pull requests of", RepoPath(in.Org, in.Repo, ""), res, len(e.gh.tokens)), IsError: true}
	}
	var pulls []ghPull
	if uerr := json.Unmarshal(res.body, &pulls); uerr != nil {
		return agentic.ToolResult{Content: "repo_read what=prs: could not parse GitHub's response: " + uerr.Error(), IsError: true}
	}
	return agentic.ToolResult{Content: formatPRList(in.Org, in.Repo, state, pulls)}
}

// formatPRList renders one PR per line.
func formatPRList(org, repo, state string, pulls []ghPull) string {
	header := fmt.Sprintf("pull requests of %s (state %s)", RepoPath(org, repo, ""), state)
	if len(pulls) == 0 {
		return header + "\n(no pull requests found)"
	}
	var b strings.Builder
	b.WriteString(header + "\n")
	for _, p := range pulls {
		flags := p.State
		if p.Draft {
			flags += ", draft"
		}
		fmt.Fprintf(&b, "#%d  %s  (%s, %s -> %s, %s, updated %s)\n",
			p.Number, firstLine(p.Title), p.User.Login, p.Head.Ref, p.Base.Ref, flags, p.UpdatedAt)
	}
	return strings.TrimRight(b.String(), "\n")
}

func (e *repoTools) prRead(ctx context.Context, in repoReadArgs) agentic.ToolResult {
	if err := requireOrgRepo("repo_read what=pr", in.Org, in.Repo); err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}
	}
	if in.Number <= 0 {
		return agentic.ToolResult{Content: `repo_read what=pr requires a positive "number" (a pull request number, e.g. from what=prs)`, IsError: true}
	}
	key := RepoCacheKey(in.Org, in.Repo)
	prURL := fmt.Sprintf("%s/repos/%s/%s/pulls/%d", e.gh.base, url.PathEscape(in.Org), url.PathEscape(in.Repo), in.Number)
	what := fmt.Sprintf("pull request #%d of %s", in.Number, RepoPath(in.Org, in.Repo, ""))

	res, err := e.gh.FetchURL(ctx, key, prURL, "application/vnd.github+json")
	if err != nil {
		return agentic.ToolResult{Content: "repo_read what=pr request failed: " + err.Error(), IsError: true}
	}
	if res.status < 200 || res.status >= 300 {
		return agentic.ToolResult{Content: DescribeResourceFailure("read", what, res, len(e.gh.tokens)), IsError: true}
	}
	var pr ghPull
	if uerr := json.Unmarshal(res.body, &pr); uerr != nil {
		return agentic.ToolResult{Content: "repo_read what=pr: could not parse GitHub's response: " + uerr.Error(), IsError: true}
	}

	// Changed files and the optional diff are best-effort: a failure becomes a
	// note in the output rather than failing the whole read.
	var files []ghPRFile
	filesErr := ""
	if fres, ferr := e.gh.FetchURL(ctx, key, fmt.Sprintf("%s/files?per_page=%d", prURL, repoPRMaxFiles), "application/vnd.github+json"); ferr != nil {
		filesErr = ferr.Error()
	} else if fres.status < 200 || fres.status >= 300 {
		filesErr = fmt.Sprintf("status %d", fres.status)
	} else if uerr := json.Unmarshal(fres.body, &files); uerr != nil {
		filesErr = uerr.Error()
	}

	diff, diffNote := "", ""
	if in.IncludeDiff {
		dres, derr := e.gh.FetchURL(ctx, key, prURL, "application/vnd.github.diff")
		switch {
		case derr != nil:
			diffNote = "(could not fetch diff: " + derr.Error() + ")"
		case dres.status < 200 || dres.status >= 300:
			diffNote = fmt.Sprintf("(could not fetch diff: status %d)", dres.status)
		default:
			var truncated bool
			diff, truncated = agentic.TruncateRunes(string(dres.body), RepoDiffMaxRunes)
			if truncated || dres.truncated {
				diffNote = fmt.Sprintf("(diff truncated to %d characters)", RepoDiffMaxRunes)
			}
		}
	}
	return agentic.ToolResult{Content: formatPR(in.Org, in.Repo, pr, files, filesErr, diff, diffNote)}
}

// formatPR renders one pull request: header, body (capped), changed files
// (capped with an explicit note), and the optional diff.
func formatPR(org, repo string, pr ghPull, files []ghPRFile, filesErr, diff, diffNote string) string {
	flags := pr.State
	if pr.Merged {
		flags = "merged"
	}
	if pr.Draft {
		flags += ", draft"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "pull request #%d of %s: %s\n", pr.Number, RepoPath(org, repo, ""), firstLine(pr.Title))
	fmt.Fprintf(&b, "%s, by %s, %s -> %s, updated %s\n", flags, pr.User.Login, pr.Head.Ref, pr.Base.Ref, pr.UpdatedAt)
	if pr.HTMLURL != "" {
		b.WriteString(pr.HTMLURL + "\n")
	}
	b.WriteString("\n")
	if body := strings.TrimSpace(pr.Body); body != "" {
		b.WriteString(CappedText(body, repoBodyMaxRunes) + "\n")
	} else {
		b.WriteString("(no description)\n")
	}
	b.WriteString("\n")
	switch {
	case filesErr != "":
		b.WriteString("(could not list changed files: " + filesErr + ")\n")
	case len(files) == 0:
		b.WriteString("(no changed files listed)\n")
	default:
		total := pr.ChangedFiles
		if total < len(files) {
			total = len(files)
		}
		shown := files
		if len(shown) > repoPRMaxFiles {
			shown = shown[:repoPRMaxFiles]
		}
		fmt.Fprintf(&b, "changed files (%d", total)
		if total > len(shown) {
			fmt.Fprintf(&b, ", showing first %d", len(shown))
		}
		b.WriteString("):\n")
		for _, f := range shown {
			fmt.Fprintf(&b, "  %s  +%d -%d", f.Filename, f.Additions, f.Deletions)
			if f.Status != "" && f.Status != "modified" {
				fmt.Fprintf(&b, "  (%s)", f.Status)
			}
			b.WriteString("\n")
		}
	}
	if diffNote != "" {
		b.WriteString("\n" + diffNote + "\n")
	}
	if diff != "" {
		b.WriteString("\ndiff:\n" + diff)
	}
	return strings.TrimRight(b.String(), "\n")
}
