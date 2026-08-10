package agentic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// repo_file_write and repo_pr_create are the two mutating repo tools. They
// differ from the read tools in four deliberate ways:
//   - they always require the user's approval (alwaysAsk in the tool table, so
//     NeedsApproval returns true regardless of the ask_tools setting);
//   - they never fall through to an unauthenticated attempt (writeTokenOrder);
//   - they draw from the client's separate WRITE credential list only —
//     for these model-initiated tools that is the allow_model_writes-flagged
//     subset of the user's PATs, so a token the user has not opted in to model
//     writes is structurally unreachable from here (the read list is never
//     consulted);
//   - a failure with the cached winner falls through to the other tokens, and
//     only a token that completed the write is cached — so a read-only winner
//     discovered by the read tools can never poison the cache.
const (
	repoFileWriteDescription = "Creates a single NEW file in a GitHub repository as a new commit on the given branch, " +
		"optionally creating the branch first (create_branch=true, from create_branch_from or the default branch). " +
		"CREATE-ONLY: it refuses to overwrite a file that already exists — to change an existing file, ask the user to pull the pull request (or open one for the branch) into a conversation workspace from the files pane, then edit it with workspace_edit's replace. " +
		"This MUTATES the repository — the user is asked to approve every call — and requires a GitHub PAT with write access."
	repoPRCreateDescription = "Opens a GitHub pull request from an existing head branch into base (the default branch unless set), as a draft unless draft=false. " +
		"This MUTATES the repository — the user is asked to approve every call — and requires a GitHub PAT with write access. " +
		"Commit the changes first (e.g. with repo_file_write), then open the PR."

	// noModelWriteTokensMsg is the recoverable teaching error for a
	// model-initiated write when the user has PATs configured but has flagged
	// none of them for model writes. User-initiated writes (the files-pane
	// workspace push) are unaffected — they use every enabled token.
	noModelWriteTokensMsg = `No GitHub credential permits model-initiated writes. Ask the user to enable "model can write" on a GitHub token in Settings -> github.`
)

var repoFileWriteSchema = InferSchema[repoFileWriteArgs]()

var repoPRCreateSchema = InferSchema[repoPRCreateArgs]()

// GitHubAuthError marks a write-flow step that failed in a way that may be
// credential-specific — 401, 403, or 404 (GitHub reports resources a token
// cannot access, or cannot write, as 404) — so the whole flow is retried with
// the next token.
type GitHubAuthError struct {
	status int
	what   string
	// object names the one git object this step READ, when the step was a
	// lookup of a specific commit or ref rather than a mutation. A 404 on such
	// a step means EITHER that the credential cannot see the repository OR
	// that the object is not in it, and nothing about the response
	// distinguishes them -- so the rotation records which it was and the two
	// are told apart after every credential has been tried, never during.
	object string
}

func (a GitHubAuthError) Error() string {
	return fmt.Sprintf("could not %s: status %d", a.what, a.status)
}

// classifyObjectRead is ClassifyWriteStatus for a step that reads ONE named
// git object, recording that object so an exhausted rotation can report a
// missing commit as a missing commit.
func classifyObjectRead(what, object string, res GHResponse) error {
	err := ClassifyWriteStatus(what, res)
	var auth GitHubAuthError
	if errors.As(err, &auth) {
		auth.object = object
		return auth
	}
	return err
}

// GitHubFatalError marks a write-flow failure no other credential can fix (bad
// arguments, a missing base branch, validation errors, conflicts); its message
// goes back to the model as a recoverable tool error.
type GitHubFatalError struct{ msg string }

func (f GitHubFatalError) Error() string { return f.msg }

// ClassifyWriteStatus turns a non-2xx write-flow response into a GitHubAuthError
// (retry with the next credential) or a GitHubFatalError carrying GitHub's own message.
func ClassifyWriteStatus(what string, res GHResponse) error {
	switch res.status {
	case http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound:
		return GitHubAuthError{status: res.status, what: what}
	}
	if msg := ghErrorDetail(res.body); msg != "" {
		return GitHubFatalError{msg: fmt.Sprintf("could not %s: GitHub returned %d: %s", what, res.status, msg)}
	}
	return GitHubFatalError{msg: fmt.Sprintf("could not %s: GitHub returned status %d", what, res.status)}
}

// writeTokenOrder is the credential order for mutating calls, drawn ONLY from
// the client's write list (e.gh.writeTokens — never the read list): the cached
// winner for cacheKey first — when it is a real token PRESENT IN the write
// list; the "" (public/no-auth) cache sentinel and any winner outside the
// write list (e.g. a user-only token recorded by a read or a user push) are
// skipped, and the write list rotates normally — then every write-capable
// token. Unlike reads, writes never append an unauthenticated attempt.
func (e *GitHub) writeTokenOrder(cacheKey string) []tokenAttempt {
	var order []tokenAttempt
	seen := map[string]bool{}
	add := func(id, name, token string) {
		if id == "" || seen[id] {
			return
		}
		seen[id] = true
		order = append(order, tokenAttempt{id: id, name: name, token: token})
	}
	if e.cache != nil && cacheKey != "" {
		if id, ok := e.cache.Get(cacheKey); ok && id != "" {
			if t, found := findToken(e.writeTokens, id); found {
				add(id, t.Name, t.Token)
			}
		}
	}
	for _, t := range e.writeTokens {
		add(t.ID, t.Name, t.Token)
	}
	return order
}

// runWrite drives a write flow through writeTokenOrder: attempt runs the whole
// flow with one token; a GitHubAuthError falls through to the next credential (the
// cached winner may have been discovered by a READ and lack write access), a
// GitHubFatalError or transport error stops immediately, and the first success caches
// the winning token — so only a credential that completed a write is recorded.
// An empty write list is a recoverable teaching error: these are the
// model-initiated write tools, so the fix is the user flagging (or adding) a
// "model can write" token — the loop continues and the model can relay that.
func (e *repoTools) runWrite(ctx context.Context, toolName, cacheKey string, attempt func(token string) (string, error)) ToolResult {
	order := e.gh.writeTokenOrder(cacheKey)
	if len(order) == 0 {
		if len(e.gh.tokens) > 0 {
			// Tokens exist, but none is flagged for model-initiated writes.
			return ToolResult{Content: noModelWriteTokensMsg, IsError: true}
		}
		return ToolResult{Content: toolName + ` mutates GitHub and is never attempted unauthenticated — add a GitHub personal access token with write access in Settings -> github and enable "model can write" on it first.`, IsError: true}
	}
	var bestAuth GitHubAuthError
	for _, att := range order {
		text, err := attempt(att.token)
		if err == nil {
			if e.gh.cache != nil && cacheKey != "" {
				e.gh.cache.Put(cacheKey, att.id)
			}
			return ToolResult{Content: text}
		}
		var auth GitHubAuthError
		if errors.As(err, &auth) {
			bestAuth = moreInformativeAuth(bestAuth, auth)
			continue // this credential may simply lack access; try the next
		}
		var fatal GitHubFatalError
		if errors.As(err, &fatal) {
			return ToolResult{Content: fatal.msg, IsError: true}
		}
		return ToolResult{Content: toolName + " request failed: " + err.Error(), IsError: true}
	}
	return ToolResult{Content: e.explainExhaustedWrite(ctx, toolName, cacheKey, order, bestAuth), IsError: true}
}

// ghRepoMeta is the slice of GET /repos/{org}/{repo} the write flows need.
type ghRepoMeta struct {
	DefaultBranch string `json:"default_branch"`
}

// repoMeta probes a repository with one credential and returns its metadata.
// 401/403/404 come back as GitHubAuthError so the caller tries the next credential
// (GitHub hides an inaccessible private repo behind 404).
func (e *repoTools) repoMeta(ctx context.Context, token, org, repo string) (ghRepoMeta, error) {
	res, err := e.gh.doGet(ctx, e.repoURL(org, repo), token, "application/vnd.github+json")
	if err != nil {
		return ghRepoMeta{}, err
	}
	if res.status < 200 || res.status >= 300 {
		return ghRepoMeta{}, ClassifyWriteStatus("access repository "+org+"/"+repo, res)
	}
	var meta ghRepoMeta
	if uerr := json.Unmarshal(res.body, &meta); uerr != nil {
		return ghRepoMeta{}, uerr
	}
	return meta, nil
}

func (e *repoTools) repoURL(org, repo string) string { return e.gh.RepoURL(org, repo) }

// RepoURL is the API root of one repository, which a host composing its own
// endpoint (a ref update, a blob write) builds on.
func (e *GitHub) RepoURL(org, repo string) string {
	return e.base + "/repos/" + url.PathEscape(org) + "/" + url.PathEscape(repo)
}

// resolveRefSHA resolves any ref (branch, tag, or SHA) to a commit SHA via the
// commits endpoint. The caller has already proven the repo visible with this
// token, so a 404 means the ref itself is missing — a user-correctable error,
// not a credential problem.
func (e *repoTools) resolveRefSHA(ctx context.Context, token, org, repo, ref string) (string, error) {
	res, err := e.gh.doGet(ctx, e.repoURL(org, repo)+"/commits/"+url.PathEscape(ref), token, "application/vnd.github+json")
	if err != nil {
		return "", err
	}
	if res.status == http.StatusNotFound {
		return "", GitHubFatalError{msg: fmt.Sprintf("base ref %q does not exist in %s/%s — pass an existing branch, tag, or SHA as create_branch_from (or omit it to use the default branch)", ref, org, repo)}
	}
	if res.status < 200 || res.status >= 300 {
		return "", ClassifyWriteStatus("resolve base ref "+ref, res)
	}
	var out struct {
		SHA string `json:"sha"`
	}
	if uerr := json.Unmarshal(res.body, &out); uerr != nil {
		return "", uerr
	}
	if out.SHA == "" {
		return "", GitHubFatalError{msg: fmt.Sprintf("could not resolve base ref %q to a commit SHA", ref)}
	}
	return out.SHA, nil
}

type repoFileWriteArgs struct {
	Org              string `json:"org" jsonschema:"Repository owner (org or user)."`
	Repo             string `json:"repo" jsonschema:"Repository name."`
	Branch           string `json:"branch" jsonschema:"Branch to commit to."`
	Path             string `json:"path" jsonschema:"Path of the NEW file inside the repository, e.g. docs/notes.md. Must not exist yet on the target branch (or, with create_branch=true, on the branch it is created from)."`
	Content          string `json:"content" jsonschema:"The new file's full content (plain text)."`
	Message          string `json:"message" jsonschema:"Commit message."`
	CreateBranch     bool   `json:"create_branch,omitempty" jsonschema:"Create the branch if it does not exist yet. Defaults to false."`
	CreateBranchFrom string `json:"create_branch_from,omitempty" jsonschema:"Ref the new branch starts from when create_branch=true. Defaults to the repository's default branch."`
}

func (e *repoTools) fileWrite(ctx context.Context, args json.RawMessage) ToolResult {
	var in repoFileWriteArgs
	if err := json.Unmarshal(args, &in); err != nil {
		return ToolResult{Content: "invalid repo_file_write arguments: " + err.Error(), IsError: true}
	}
	if err := requireOrgRepo(RepoFileWriteToolName, in.Org, in.Repo); err != nil {
		return ToolResult{Content: err.Error(), IsError: true}
	}
	// Workspace mode: writes to the conversation's workspace repository go
	// through workspace_edit (staged locally, pushed only by the user).
	if r := e.block(in.Org, in.Repo); r != nil {
		return *r
	}
	in.Branch = strings.TrimSpace(in.Branch)
	in.Path = strings.Trim(strings.TrimSpace(in.Path), "/")
	in.Message = strings.TrimSpace(in.Message)
	if in.Branch == "" || in.Path == "" || in.Message == "" {
		return ToolResult{Content: `repo_file_write requires non-empty "branch", "path", and "message" (plus "content"), e.g. {"org":"octocat","repo":"hello-world","branch":"docs","path":"README.md","content":"...","message":"update readme"}`, IsError: true}
	}
	// branchCreated survives across credential attempts: if one token created
	// the branch but could not commit, the token that finally succeeds still
	// reports the creation.
	branchCreated := false
	createdFrom := ""
	return e.runWrite(ctx, RepoFileWriteToolName, RepoCacheKey(in.Org, in.Repo), func(token string) (string, error) {
		return e.tryFileWrite(ctx, token, in, &branchCreated, &createdFrom)
	})
}

// tryFileWrite runs the whole create-file flow with a single credential: probe
// the repo (learning the default branch), find the branch (noting when it must
// be created), verify the path does NOT already exist, then create the branch
// (when asked) and PUT the new contents without a blob SHA.
//
// The tool is CREATE-ONLY: a path that already exists is refused with a
// teaching error — existing files are edited through a PR workspace's
// workspace_edit replace, never rewritten wholesale by a direct commit. The
// existence check runs against the ref the content would actually come from —
// the target branch, or, when the branch is still to be created, the ref it
// would be created from — so a fresh branch cannot be used to dodge the check
// (and a refused call creates nothing, branch included).
func (e *repoTools) tryFileWrite(ctx context.Context, token string, in repoFileWriteArgs, branchCreated *bool, createdFrom *string) (string, error) {
	meta, err := e.repoMeta(ctx, token, in.Org, in.Repo)
	if err != nil {
		return "", err
	}
	repoURL := e.repoURL(in.Org, in.Repo)

	// Does the branch exist? The repo probe above succeeded with this token, so
	// a 404 here means the branch really is missing (not an access problem).
	res, err := e.gh.doGet(ctx, repoURL+"/git/ref/"+EscapeSegments("heads/"+in.Branch), token, "application/vnd.github+json")
	if err != nil {
		return "", err
	}
	// checkRef is where the new file's absence is verified: the branch itself,
	// or — when the branch is yet to be created — the ref it will start from
	// (whose tree the new branch inherits). needBranch defers the branch
	// creation until after the check, so a refusal mutates nothing.
	checkRef := in.Branch
	needBranch := false
	branchBaseSHA := ""
	switch {
	case res.status == http.StatusNotFound:
		if !in.CreateBranch {
			return "", GitHubFatalError{msg: fmt.Sprintf("branch %q does not exist in %s/%s. Pass create_branch=true to create it (from create_branch_from, default: the repository's default branch %q), or commit to an existing branch.", in.Branch, in.Org, in.Repo, meta.DefaultBranch)}
		}
		base := strings.TrimSpace(in.CreateBranchFrom)
		if base == "" {
			base = meta.DefaultBranch
		}
		if base == "" {
			return "", GitHubFatalError{msg: "could not determine a base branch for the new branch — pass create_branch_from explicitly"}
		}
		sha, serr := e.resolveRefSHA(ctx, token, in.Org, in.Repo, base)
		if serr != nil {
			return "", serr
		}
		checkRef = base
		needBranch = true
		branchBaseSHA = sha
		*createdFrom = base
	case res.status < 200 || res.status >= 300:
		return "", ClassifyWriteStatus("look up branch "+in.Branch, res)
	}

	// CREATE-ONLY gate: the path must not exist on the ref the content would
	// come from (2xx = it does; 404 = genuinely new).
	ContentsURL := repoURL + "/contents/" + EscapeSegments(in.Path)
	fres, err := e.gh.doGet(ctx, ContentsURL+"?ref="+url.QueryEscape(checkRef), token, "application/vnd.github+json")
	if err != nil {
		return "", err
	}
	switch {
	case fres.status >= 200 && fres.status < 300:
		if t := bytes.TrimSpace(fres.body); len(t) > 0 && t[0] == '[' {
			return "", GitHubFatalError{msg: fmt.Sprintf("%s is a directory in %s/%s — repo_file_write writes a single file", in.Path, in.Org, in.Repo)}
		}
		where := fmt.Sprintf("on branch %q", checkRef)
		if needBranch {
			where = fmt.Sprintf("on %q (the ref branch %q would be created from)", checkRef, in.Branch)
		}
		return "", GitHubFatalError{msg: fmt.Sprintf("%s already exists in %s/%s %s — repo_file_write only creates new files, it never overwrites. To change an existing file, ask the user to pull the pull request (or open one for the branch) into a conversation workspace from the files pane, then edit it with workspace_edit's replace.", in.Path, in.Org, in.Repo, where)}
	case fres.status == http.StatusNotFound:
		// Genuinely new — proceed.
	default:
		return "", ClassifyWriteStatus("check the existing file "+in.Path, fres)
	}

	if needBranch {
		body, _ := json.Marshal(map[string]string{"ref": "refs/heads/" + in.Branch, "sha": branchBaseSHA})
		cres, cerr := e.gh.doRequest(ctx, http.MethodPost, repoURL+"/git/refs", token, "application/vnd.github+json", body)
		if cerr != nil {
			return "", cerr
		}
		if cres.status < 200 || cres.status >= 300 {
			return "", ClassifyWriteStatus("create branch "+in.Branch, cres)
		}
		*branchCreated = true
	}

	put := map[string]string{
		"message": in.Message,
		"content": base64.StdEncoding.EncodeToString([]byte(in.Content)),
		"branch":  in.Branch,
	}
	body, _ := json.Marshal(put)
	pres, err := e.gh.doRequest(ctx, http.MethodPut, ContentsURL, token, "application/vnd.github+json", body)
	if err != nil {
		return "", err
	}
	if pres.status < 200 || pres.status >= 300 {
		return "", ClassifyWriteStatus("write "+in.Path, pres)
	}
	var out struct {
		Commit struct {
			SHA     string `json:"sha"`
			HTMLURL string `json:"html_url"`
		} `json:"commit"`
	}
	_ = json.Unmarshal(pres.body, &out)
	return formatFileWrite(in, *branchCreated, *createdFrom, out.Commit.SHA, out.Commit.HTMLURL), nil
}

// formatFileWrite renders a successful single-file creation commit.
func formatFileWrite(in repoFileWriteArgs, branchCreated bool, createdFrom, commitSHA, htmlURL string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "created %s on branch %s of %s/%s", in.Path, in.Branch, in.Org, in.Repo)
	if branchCreated {
		fmt.Fprintf(&b, "\nbranch %s created from %s", in.Branch, createdFrom)
	}
	if commitSHA != "" {
		fmt.Fprintf(&b, "\ncommit %s", commitSHA)
	}
	if htmlURL != "" {
		fmt.Fprintf(&b, "\n%s", htmlURL)
	}
	return b.String()
}

type repoPRCreateArgs struct {
	Org   string `json:"org" jsonschema:"Repository owner (org or user)."`
	Repo  string `json:"repo" jsonschema:"Repository name."`
	Title string `json:"title" jsonschema:"Pull request title."`
	Head  string `json:"head" jsonschema:"Branch carrying the changes (must already exist)."`
	Base  string `json:"base,omitempty" jsonschema:"Branch to merge into. Defaults to the repository's default branch."`
	Body  string `json:"body,omitempty" jsonschema:"Optional pull request description (markdown)."`
	// Draft is a pointer so an omitted argument keeps the default (true) rather
	// than reading as an explicit false.
	Draft *bool `json:"draft,omitempty" jsonschema:"Open as a draft pull request. Defaults to true."`
}

func (e *repoTools) prCreate(ctx context.Context, args json.RawMessage) ToolResult {
	var in repoPRCreateArgs
	if err := json.Unmarshal(args, &in); err != nil {
		return ToolResult{Content: "invalid repo_pr_create arguments: " + err.Error(), IsError: true}
	}
	if err := requireOrgRepo(RepoPRCreateToolName, in.Org, in.Repo); err != nil {
		return ToolResult{Content: err.Error(), IsError: true}
	}
	// Workspace mode: the workspace repository already has its pull request —
	// the one attached to this conversation.
	if r := e.block(in.Org, in.Repo); r != nil {
		return *r
	}
	in.Title = strings.TrimSpace(in.Title)
	in.Head = strings.TrimSpace(in.Head)
	if in.Title == "" || in.Head == "" {
		return ToolResult{Content: `repo_pr_create requires non-empty "title" and "head" (the branch carrying the changes), e.g. {"org":"octocat","repo":"hello-world","title":"Fix typo","head":"fix-typo"}`, IsError: true}
	}
	draft := true
	if in.Draft != nil {
		draft = *in.Draft
	}
	return e.runWrite(ctx, RepoPRCreateToolName, RepoCacheKey(in.Org, in.Repo), func(token string) (string, error) {
		return e.tryPRCreate(ctx, token, in, draft)
	})
}

// tryPRCreate runs the create-PR flow with a single credential: probe the repo
// (learning the default branch for the base fallback), then POST /pulls.
func (e *repoTools) tryPRCreate(ctx context.Context, token string, in repoPRCreateArgs, draft bool) (string, error) {
	meta, err := e.repoMeta(ctx, token, in.Org, in.Repo)
	if err != nil {
		return "", err
	}
	base := strings.TrimSpace(in.Base)
	if base == "" {
		base = meta.DefaultBranch
	}
	payload := map[string]any{"title": in.Title, "head": in.Head, "base": base, "draft": draft}
	if strings.TrimSpace(in.Body) != "" {
		payload["body"] = in.Body
	}
	body, _ := json.Marshal(payload)
	res, err := e.gh.doRequest(ctx, http.MethodPost, e.repoURL(in.Org, in.Repo)+"/pulls", token, "application/vnd.github+json", body)
	if err != nil {
		return "", err
	}
	if res.status < 200 || res.status >= 300 {
		return "", ClassifyWriteStatus("create pull request", res)
	}
	var pr struct {
		Number  int    `json:"number"`
		HTMLURL string `json:"html_url"`
		Draft   bool   `json:"draft"`
	}
	_ = json.Unmarshal(res.body, &pr)
	kind := "pull request"
	if pr.Draft {
		kind = "draft pull request"
	}
	return fmt.Sprintf("created %s #%d in %s/%s: %s\n%s -> %s\n%s", kind, pr.Number, in.Org, in.Repo, in.Title, in.Head, base, pr.HTMLURL), nil
}

// NewGitHubFatalError is a write failure that must NOT be retried with another
// credential: the request was understood and refused on its merits, so trying
// the next token only repeats it.
func NewGitHubFatalError(msg string) error { return GitHubFatalError{msg: msg} }
