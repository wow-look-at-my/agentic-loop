package repo

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/wow-look-at-my/agentic-loop/vfs"
	"sort"
	"strings"
)

// Per-"what" argument validation for repo_read.
//
// repo_read is tool over an argument union, so its schema has to accept
// every field for every read. That made a field the chosen read ignores
// silently droppable — see the table below for what that cost.

// repoReadFields lists the arguments each repo_read "what" actually reads.
var repoReadFields = map[string][]string{
	"commits":   {"org", "repo", "path", "ref", "per_page"},
	"commit":    {"org", "repo", "sha"},
	"prs":       {"org", "repo", "state", "per_page"},
	"pr":        {"org", "repo", "number", "include_diff"},
	"issues":    {"org", "repo", "state", "labels", "per_page"},
	"issue":     {"org", "repo", "number"},
	"status":    {"org", "repo", "ref"},
	"check_run": {"org", "repo", "id"},
	"job_log":   {"org", "repo", "job_id", "offset", "limit"},
}

// repoReadFieldRedirect points each ignorable field at the read that does use
// it, so the rejection teaches the correct call rather than only refusing.
var repoReadFieldRedirect = map[string]string{
	"query":        `repo_read has no "query": it reads history and metadata, never file contents. To search inside files use grep on /repos/<org>/<repo>/<path>.`,
	"glob":         `repo_read has no "glob": it reads history and metadata, never files. To match filenames use find_files, or to filter a content search use grep's own "glob".`,
	"path":         `"path" filters what=commits to one file or directory. To list files use find_files on /repos/<org>/<repo>/<path>.`,
	"sha":          `"sha" names the commit what=commit fetches.`,
	"number":       `"number" names the pull request or issue what=pr / what=issue fetches.`,
	"state":        `"state" filters what=prs / what=issues.`,
	"labels":       `"labels" filters what=issues.`,
	"ref":          `"ref" picks the starting ref for what=commits, or the commit for what=status.`,
	"include_diff": `"include_diff" applies to what=pr.`,
	"id":           `"id" names the check run what=check_run reads; what=status lists the ids.`,
	"job_id":       `"job_id" names the Actions job what=job_log reads; what=status prints it beside each job as [job N].`,
	"offset":       `"offset" and "limit" pick a line window of what=job_log. To read part of a FILE use ` + vfs.ReadFileToolName + `, which takes the same two.`,
	"limit":        `"limit" bounds what=job_log's line window; the list reads take "per_page" instead.`,
}

// validateRepoReadArgs rejects a call carrying arguments the chosen read
// ignores. It decodes the raw object a time because what matters is
// which fields the caller actually supplied — a field carrying its value
// states no intent (a client that marshals the whole argument struct sends
// every key), so only a field with a value in it counts.
func validateRepoReadArgs(what string, raw json.RawMessage) error {
	allowed, known := repoReadFields[what]
	if !known {
		return nil
	}
	var present map[string]json.RawMessage
	if err := json.Unmarshal(raw, &present); err != nil {
		return nil // the caller's own decode already reported this
	}
	var ignored []string
	for k, v := range present {
		if k == "what" || fieldAllowed(allowed, k) || isZeroJSON(v) {
			continue
		}
		ignored = append(ignored, k)
	}
	if len(ignored) == 0 {
		return nil
	}
	sort.Strings(ignored)
	var b strings.Builder
	fmt.Fprintf(&b, "repo_read what=%s ignores %s, so this call was NOT run: acting on it would return a result that looks like an answer and is not one",
		what, quotedList(ignored))
	if what == "commits" || what == "prs" || what == "issues" {
		fmt.Fprintf(&b, " (what=%s would have returned the newest %s, unfiltered)", what, whatNoun(what))
	}
	fmt.Fprintf(&b, ". what=%s reads: %s.", what, strings.Join(allowed, ", "))
	for _, k := range ignored {
		if hint := repoReadFieldRedirect[k]; hint != "" {
			b.WriteString(" " + hint)
		}
	}
	return errors.New(b.String())
}

// isZeroJSON reports whether a raw argument value is a caller could not
// have meant anything by: absent-in-spirit rather than absent in fact.
func isZeroJSON(raw json.RawMessage) bool {
	switch strings.TrimSpace(string(raw)) {
	case "", "null", `""`, "0", "false", "[]", "{}":
		return true
	}
	return false
}

func fieldAllowed(allowed []string, k string) bool {
	for _, a := range allowed {
		if a == k {
			return true
		}
	}
	return false
}

func whatNoun(what string) string {
	switch what {
	case "commits":
		return "commits"
	case "prs":
		return "pull requests"
	default:
		return "issues"
	}
}

// quotedList renders names as "a", "b" and "c".
func quotedList(names []string) string {
	quoted := make([]string, len(names))
	for i, n := range names {
		quoted[i] = `"` + n + `"`
	}
	if len(quoted) == 1 {
		return quoted[0]
	}
	return strings.Join(quoted[:len(quoted)-1], ", ") + " and " + quoted[len(quoted)-1]
}
