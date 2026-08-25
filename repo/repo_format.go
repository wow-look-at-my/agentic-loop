package repo

import (
	"bytes"
	"encoding/json"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"strings"
	"time"
)

// The shapes GitHub's REST responses decode into, and the shared formatting.

// GHRepo is one repository in an org/user repos response. Only the fields the
// listing surfaces are decoded.
type GHRepo struct {
	Name        string `json:"name"`
	Private     bool   `json:"private"`
	Archived    bool   `json:"archived"`
	Fork        bool   `json:"fork"`
	Description string `json:"description"`
}

// parseRepoArray decodes a /orgs|users/<owner>/repos JSON array.
func parseRepoArray(body []byte) ([]GHRepo, error) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return nil, fmt.Errorf("expected a JSON array of repositories")
	}
	var repos []GHRepo
	if err := json.Unmarshal(trimmed, &repos); err != nil {
		return nil, err
	}
	return repos, nil
}

// IsDirectoryListing reports whether a raw-media file read returned a directory listing.
func IsDirectoryListing(body []byte, ctype string) bool {
	if !strings.Contains(ctype, "json") {
		return false
	}
	t := bytes.TrimSpace(body)
	return len(t) > 0 && t[0] == '['
}

// DescribeGitHubFailure turns a non-2xx contents response into a model-facing message.
func DescribeGitHubFailure(op, org, repo, inner string, res GHResponse, numTokens int) string {
	return DescribeResourceFailure(op, RepoPath(org, repo, inner), res, numTokens)
}

// DescribeResourceFailure is DescribeGitHubFailure for an arbitrary named resource
// (a PR, an issue, a commit, ...) instead of a contents path.
func DescribeResourceFailure(op, what string, res GHResponse, numTokens int) string {
	return explainFailure(op, what, res, numTokens, time.Now())
}

// DescribeOwnerFailure explains why an owner-level listing (/repos/<owner>)
// could not be produced.
func DescribeOwnerFailure(owner string, res GHResponse, numTokens int) string {
	return explainFailure("list repositories under", "/repos/"+owner, res, numTokens, time.Now())
}

func GitHubErrorMessage(body []byte) string {
	var e struct {
		Message string `json:"message"`
	}
	if json.Unmarshal(body, &e) == nil {
		return e.Message
	}
	return ""
}

// GitHubErrorDetail extracts GitHub's error message plus the first detailed
// sub-error: 422 validation responses put the useful text ("A pull request
// already exists...", "Reference already exists", ...) in errors[], as either
// objects with a message or plain strings.
func GitHubErrorDetail(body []byte) string {
	var e struct {
		Message string            `json:"message"`
		Errors  []json.RawMessage `json:"errors"`
	}
	if json.Unmarshal(body, &e) != nil {
		return ""
	}
	msg := e.Message
	for _, raw := range e.Errors {
		var detail struct {
			Message string `json:"message"`
		}
		if json.Unmarshal(raw, &detail) == nil && detail.Message != "" {
			return msg + " (" + detail.Message + ")"
		}
		var s string
		if json.Unmarshal(raw, &s) == nil && s != "" {
			return msg + " (" + s + ")"
		}
	}
	return msg
}

// requireOrgRepo validates the org/repo pair the API-object tools need,
// teaching the model the correct shape when either is missing.
func requireOrgRepo(tool, org, repo string) error {
	if strings.TrimSpace(org) == "" || strings.TrimSpace(repo) == "" {
		return fmt.Errorf(`%s requires both "org" and "repo", e.g. {"org":"octocat","repo":"hello-world", ...}`, tool)
	}
	return nil
}

// clampPerPage normalizes a list tool's per_page: the default when unset or
// non-positive, silently capped at repoListMaxPerPage.
func clampPerPage(n int) int {
	if n <= 0 {
		return repoListDefaultPerPage
	}
	if n > repoListMaxPerPage {
		return repoListMaxPerPage
	}
	return n
}

// parseListState validates a PR/issue list state argument (default open).
func parseListState(raw string) (string, error) {
	s := strings.ToLower(strings.TrimSpace(raw))
	if s == "" {
		return "open", nil
	}
	switch s {
	case "open", "closed", "all":
		return s, nil
	}
	return "", fmt.Errorf(`invalid state %q: must be one of "open", "closed", "all" (default open)`, raw)
}

// firstLine returns the trimmed first line of s (the subject of a commit
// message or title).
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

// shortSHA abbreviates a commit SHA for listings.
func shortSHA(sha string) string {
	if len(sha) > 10 {
		return sha[:10]
	}
	return sha
}

// CappedText rune-caps s, appending an explicit truncation note when trimmed.
func CappedText(s string, max int) string {
	out, truncated := agentic.TruncateRunes(s, max)
	if truncated {
		out += fmt.Sprintf("\n(truncated to %d characters)", max)
	}
	return out
}

// ghLabel is one issue/PR label; only the name is surfaced.
type ghLabel struct {
	Name string `json:"name"`
}

// labelNames joins label names for display ("bug, help wanted").
func labelNames(labels []ghLabel) string {
	var names []string
	for _, l := range labels {
		if l.Name != "" {
			names = append(names, l.Name)
		}
	}
	return strings.Join(names, ", ")
}
