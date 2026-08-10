package agentic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// This file holds the shared GitHub REST-API plumbing behind the repo tools:
// the token-trying fetch, path parsing, argument helpers, and the rendering of
// directory listings and error responses. The tool surface (schemas, Tools,
// Execute) lives in repo.go; the per-tool handlers live in their sibling files
// (repo_search.go, repo_commits.go, repo_prs.go, repo_issues.go, repo_write.go).

// GHResponse is one GitHub API response.
type GHResponse struct {
	status    int
	body      []byte
	ctype     string
	header    http.Header
	truncated bool
	// authed records whether the credential that produced this response was a
	// token. A failure from a token explains far more than the anonymous
	// attempt's, which is what failureRank uses it for.
	authed bool
}

// Status is the HTTP status GitHub answered with.
func (r GHResponse) Status() int { return r.status }

// Body is the response body, already capped at the read limit.
func (r GHResponse) Body() []byte { return r.body }

// ContentType is the response's declared media type, which is how a contents
// read tells a directory listing from a file.
func (r GHResponse) ContentType() string { return r.ctype }

// Truncated reports that the body hit the read cap, so what is here is a
// PREFIX -- never treat it as the whole document.
func (r GHResponse) Truncated() bool { return r.truncated }

// FetchOptions tunes one repo fetch.
type FetchOptions struct {
	// NoAnonymous drops the unauthenticated attempt that makes public
	// resources readable without a PAT. Endpoints github.com never serves
	// anonymously (code search) set it, so a token's real failure — a rate
	// limit, say — is what the model is told about, instead of the anonymous
	// attempt's inevitable "Requires authentication".
	NoAnonymous bool
	// MaxBytes overrides the per-response read cap (default GitHubMaxResponseBytes).
	MaxBytes int64
}

// Fetch performs a GET against the contents endpoint for org/repo/inner. It is
// exported because a host mounting /repos as a folder reads files through the
// same credential rotation the tools do.
func (e *GitHub) Fetch(ctx context.Context, org, repo, inner, ref, accept string) (GHResponse, error) {
	return e.FetchURL(ctx, RepoCacheKey(org, repo), e.contentsURL(org, repo, inner, ref), accept)
}

// FetchURL performs a GET against an arbitrary API URL with the default
// options: every credential, then an unauthenticated attempt.
func (e *GitHub) FetchURL(ctx context.Context, cacheKey, target, accept string) (GHResponse, error) {
	return e.FetchURLOpts(ctx, cacheKey, target, accept, FetchOptions{})
}

// FetchURLOpts performs a GET against an arbitrary API URL, trying the cached
// token for cacheKey first (if any), then every configured token, then an
// unauthenticated request (so public resources work without a PAT). On the
// first 2xx it records the winning token id in the cache (an empty cacheKey
// disables caching) and returns the response.
//
// When every attempt fails it returns the MOST INFORMATIVE failure, not the
// last one: the last attempt is the anonymous one, whose 401 says only that no
// credential was sent. Returning that hid the actual reason a configured token
// was refused — a spent code-search rate limit reads as "401 Requires
// authentication", so a transient wait looks like a permanent auth problem.
func (e *GitHub) FetchURLOpts(ctx context.Context, cacheKey, target, accept string, opt FetchOptions) (GHResponse, error) {
	var best GHResponse
	bestRank := -1
	var lastErr error
	for _, att := range e.tokenOrder(cacheKey, opt.NoAnonymous) {
		res, err := e.doGetOpts(ctx, target, att.token, accept, opt)
		if err != nil {
			lastErr = err
			continue
		}
		if res.status >= 200 && res.status < 300 {
			if e.cache != nil && cacheKey != "" {
				e.cache.Put(cacheKey, att.id)
			}
			return res, nil
		}
		res.authed = att.token != ""
		if r := failureRank(res); r > bestRank {
			bestRank, best = r, res
		}
	}
	if best.status == 0 && lastErr != nil {
		return GHResponse{}, lastErr
	}
	return best, nil
}

type tokenAttempt struct {
	id string
	// name is the label the user gave this token in Settings. It exists only
	// so a failure can name the credential in terms the reader can find; the
	// anonymous attempt has none.
	name  string
	token string
}

// tokenOrder is the credential order every repo read tries: the cached winner
// for cacheKey first (if any; an empty cacheKey skips the cache), then every
// configured token, then an unauthenticated attempt (so public resources work
// without a PAT) unless NoAnonymous drops it. Duplicates are dropped so each
// distinct credential is tried once. Writes use writeTokenOrder
// (repo_write.go) instead, which never falls through to unauthenticated.
func (e *GitHub) tokenOrder(cacheKey string, NoAnonymous bool) []tokenAttempt {
	var order []tokenAttempt
	seen := map[string]bool{}
	add := func(id, name, token string) {
		if seen[id] {
			return
		}
		seen[id] = true
		order = append(order, tokenAttempt{id: id, name: name, token: token})
	}
	if e.cache != nil && cacheKey != "" {
		if id, ok := e.cache.Get(cacheKey); ok {
			if id == "" {
				add("", "", "")
			} else if t, found := e.tokenByID(id); found {
				add(id, t.Name, t.Token)
			}
		}
	}
	for _, t := range e.tokens {
		add(t.ID, t.Name, t.Token)
	}
	if !NoAnonymous {
		add("", "", "") // unauthenticated fallback (public repositories)
	}
	return order
}

func (e *GitHub) tokenByID(id string) (GitHubToken, bool) {
	return findToken(e.tokens, id)
}

// findToken looks a token id up in a credential list. writeTokenOrder resolves
// the cached winner against the WRITE list with this, so a winner recorded by
// a read (or by a user-initiated push) that is not write-capable for this
// client is skipped rather than handed to a write flow.
func findToken(list []GitHubToken, id string) (GitHubToken, bool) {
	for _, t := range list {
		if t.ID == id {
			return t, true
		}
	}
	return GitHubToken{}, false
}

func (e *GitHub) contentsURL(org, repo, inner, ref string) string {
	u := e.base + "/repos/" + url.PathEscape(org) + "/" + url.PathEscape(repo) + "/contents"
	if inner != "" {
		u += "/" + EscapeSegments(inner)
	}
	if ref = strings.TrimSpace(ref); ref != "" {
		u += "?ref=" + url.QueryEscape(ref)
	}
	return u
}

// OwnerRepos enumerates the repositories owned by a GitHub org or user.
// Listing /repos/<owner> (no repo segment) means "show every repository under
// this owner". GitHub splits this across two endpoints — /orgs/<owner>/repos for
// organizations and /users/<owner>/repos for personal accounts — so each
// candidate credential is tried against the org endpoint first, then the user
// endpoint, until one returns 2xx. The winning credential is cached under the
// lowercased owner name (no slash, so it never collides with an "org/repo" key).
func (e *GitHub) OwnerRepos(ctx context.Context, owner string) ([]GHRepo, bool, GHResponse, error) {
	cacheKey := strings.ToLower(owner)
	var best GHResponse
	bestRank := -1
	var lastErr error
	for _, att := range e.tokenOrder(cacheKey, false) {
		for _, isUser := range []bool{false, true} {
			res, err := e.doGet(ctx, e.ownerReposURL(owner, isUser, 1), att.token, "application/vnd.github+json")
			if err != nil {
				lastErr = err
				continue
			}
			if res.status >= 200 && res.status < 300 {
				if e.cache != nil {
					e.cache.Put(cacheKey, att.id)
				}
				repos, truncated, perr := e.collectOwnerRepos(ctx, owner, isUser, att.token, res.body)
				return repos, truncated, GHResponse{status: res.status}, perr
			}
			// Same most-informative-failure rule as FetchURLOpts: the
			// anonymous attempt runs last and its answer explains least.
			res.authed = att.token != ""
			if r := failureRank(res); r > bestRank {
				bestRank, best = r, res
			}
		}
	}
	if best.status == 0 && lastErr != nil {
		return nil, false, GHResponse{}, lastErr
	}
	return nil, false, best, ErrOwnerListing
}

// ErrOwnerListing signals that no credential yielded a 2xx owner listing; the
// returned GHResponse carries the last status so the caller can explain it.
var ErrOwnerListing = fmt.Errorf("owner listing returned no successful response")

// collectOwnerRepos parses the first page of an owner's repositories and follows
// pagination (up to ownerRepoMaxPages full pages) using the credential that
// already worked. It reports truncated=true when more pages remain past the cap.
func (e *GitHub) collectOwnerRepos(ctx context.Context, owner string, isUser bool, token string, firstBody []byte) (repos []GHRepo, truncated bool, err error) {
	repos, err = parseRepoArray(firstBody)
	if err != nil {
		return nil, false, err
	}
	lastLen := len(repos)
	for page := 2; lastLen == ownerRepoPerPage; page++ {
		if page > ownerRepoMaxPages {
			truncated = true
			break
		}
		res, derr := e.doGet(ctx, e.ownerReposURL(owner, isUser, page), token, "application/vnd.github+json")
		if derr != nil || res.status < 200 || res.status >= 300 {
			break
		}
		more, perr := parseRepoArray(res.body)
		if perr != nil || len(more) == 0 {
			break
		}
		repos = append(repos, more...)
		lastLen = len(more)
	}
	return repos, truncated, nil
}

func (e *GitHub) ownerReposURL(owner string, isUser bool, page int) string {
	kind := "orgs"
	if isUser {
		kind = "users"
	}
	return fmt.Sprintf("%s/%s/%s/repos?per_page=%d&page=%d&sort=full_name",
		e.base, kind, url.PathEscape(owner), ownerRepoPerPage, page)
}

func (e *GitHub) doGet(ctx context.Context, target, token, accept string) (GHResponse, error) {
	return e.doRequest(ctx, http.MethodGet, target, token, accept, nil)
}

// doGetOpts is doGet honouring a per-call response cap.
func (e *GitHub) doGetOpts(ctx context.Context, target, token, accept string, opt FetchOptions) (GHResponse, error) {
	if opt.MaxBytes <= 0 {
		return e.doGet(ctx, target, token, accept)
	}
	return e.doRequestCapped(ctx, http.MethodGet, target, token, accept, nil, opt.MaxBytes)
}

// doRequest performs one GitHub API call with a single credential. A non-nil
// body is sent as JSON.
func (e *GitHub) doRequest(ctx context.Context, method, target, token, accept string, body []byte) (GHResponse, error) {
	return e.doRequestCapped(ctx, method, target, token, accept, body, GitHubMaxResponseBytes)
}

// doRequestCapped is doRequest reading at most MaxBytes of the response body.
// The git-tree reads raise the cap: the default file cap severs a large
// repository's tree mid-JSON, which surfaced as an unexplained decode error
// rather than as the size problem it is.
func (e *GitHub) doRequestCapped(ctx context.Context, method, target, token, accept string, body []byte, MaxBytes int64) (GHResponse, error) {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, target, rd)
	if err != nil {
		return GHResponse{}, err
	}
	req.Header.Set("Accept", accept)
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := e.hc.Do(req)
	if err != nil {
		return GHResponse{}, err
	}
	defer resp.Body.Close()
	data, truncated, err := readCapped(resp.Body, MaxBytes)
	if err != nil {
		return GHResponse{}, err
	}
	return GHResponse{
		status:    resp.StatusCode,
		body:      data,
		ctype:     resp.Header.Get("Content-Type"),
		header:    resp.Header,
		truncated: truncated,
	}, nil
}

func EscapeSegments(p string) string {
	parts := strings.Split(p, "/")
	for i, s := range parts {
		parts[i] = url.PathEscape(s)
	}
	return strings.Join(parts, "/")
}

// DefaultBranch resolves a repository's default branch via the repo metadata
// endpoint, falling back to "main" when GitHub reports none.
func (e *GitHub) DefaultBranch(ctx context.Context, org, repo string) (string, error) {
	target := e.base + "/repos/" + url.PathEscape(org) + "/" + url.PathEscape(repo)
	res, err := e.FetchURL(ctx, RepoCacheKey(org, repo), target, "application/vnd.github+json")
	if err != nil {
		return "", err
	}
	if res.status < 200 || res.status >= 300 {
		return "", errors.New(DescribeGitHubFailure("inspect", org, repo, "", res, len(e.tokens)))
	}
	var meta struct {
		DefaultBranch string `json:"default_branch"`
	}
	if err := json.Unmarshal(res.body, &meta); err != nil {
		return "", err
	}
	if strings.TrimSpace(meta.DefaultBranch) == "" {
		return "main", nil
	}
	return meta.DefaultBranch, nil
}
