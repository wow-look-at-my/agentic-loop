package repo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"github.com/wow-look-at-my/go-containers/set"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// GitHub plumbing: token-trying fetch, path parsing, arg helpers, listing/error rendering.

// GHResponse is GitHub API response.
type GHResponse struct {
	status    int
	body      []byte
	ctype     string
	header    http.Header
	truncated bool
	// authed records whether the credential that produced this response was a token.
	authed bool
	// credentialName is the Settings label of the token that produced this response.
	credentialName string
}

// Status is the HTTP status GitHub answered with.
func (r GHResponse) Status() int { return r.status }

// Body is the response body, already capped at the read limit.
func (r GHResponse) Body() []byte { return r.body }

// ContentType is the response's declared media type, which tells a directory listing from a file.
func (r GHResponse) ContentType() string { return r.ctype }

// Truncated reports that the body hit the read cap, so it is a PREFIX.
func (r GHResponse) Truncated() bool { return r.truncated }

// FetchOptions tunes repo fetch.
type FetchOptions struct {
	// NoAnonymous drops the unauthenticated attempt that makes public resources readable.
	NoAnonymous bool
	// MaxBytes overrides the per-response read cap (default GitHubMaxResponseBytes).
	MaxBytes int64
	// NoRedirect returns a 3xx instead of following it.
	NoRedirect bool
}

// Fetch performs a GET against the contents endpoint for org/repo/inner. It is
// exported because a host mounting /repos as a folder reads files through the
// same credential rotation the tools do.
func (e *GitHub) Fetch(ctx context.Context, org, repo, inner, ref, accept string) (GHResponse, error) {
	return e.FetchURL(ctx, RepoCacheKey(org, repo), e.ContentsURL(org, repo, inner, ref), accept)
}

// FetchURL performs a GET against an arbitrary API URL with the default
// options: every credential, then an unauthenticated attempt.
func (e *GitHub) FetchURL(ctx context.Context, cacheKey, target, accept string) (GHResponse, error) {
	return e.FetchURLOpts(ctx, cacheKey, target, accept, FetchOptions{})
}

// FetchURLOpts performs a GET against an arbitrary API URL, trying the cached
// token for cacheKey (if any), then every configured token, then an
// unauthenticated request (so public resources work without a PAT). On the
// 2xx it records the winning token id in the cache (an empty cacheKey
// disables caching) and returns the response.
//
// A registered VFS mount is consulted FIRST: a path a mount owns is answered
// by that mount and never reaches GitHub or the token rotation.
//
// When every attempt fails it returns the MOST INFORMATIVE failure, not the
// last: the last attempt is the anonymous, whose says only that no
// credential was sent. Returning that hid the actual reason a configured token
// was refused — a spent code-search rate limit reads as " Requires
// authentication", so a transient wait looks like a permanent auth problem.
func (e *GitHub) FetchURLOpts(ctx context.Context, cacheKey, target, accept string, opt FetchOptions) (GHResponse, error) {
	// A virtual filesystem owns this path: answer from it, never GitHub.
	if e.vfs != nil {
		if vfs, rest, ok := e.vfs.Resolve(target); ok {
			return vfs.Read(ctx, rest, opt)
		}
	}
	var best GHResponse
	bestRank := rankNone
	// bestAuthed is the most informative failure produced by a TOKEN attempt.
	// It is kept separate from best so an anonymous failure can never
	// supersede it: when a configured PAT was tried and produced a failure,
	// the caller must be told about THAT, never "this was anonymous." The
	// anonymous attempt runs last, so without this separation its (typically
	// less informative) 401/403 would overwrite the token's answer on the
	// last-write-wins tiebreak.
	var bestAuthed GHResponse
	bestAuthedRank := rankNone
	var lastErr error
	for _, att := range e.tokenOrder(cacheKey, opt.NoAnonymous) {
		res, err := e.doGetOpts(ctx, target, att.token, accept, opt)
		if err != nil {
			lastErr = err
			continue
		}
		// With NoRedirect a 3xx is the answer, not a failure: the resource lives at the Location.
		if res.status >= 200 && res.status < 300 || (opt.NoRedirect && res.status >= 300 && res.status < 400) {
			e.Remember(cacheKey, att.id)
			return res, nil
		}
		res.authed = att.token != ""
		res.credentialName = att.name
		r := failureRank(res)
		// A token's failure outranks any anonymous one: an anonymous result
		// says only "no credential was sent", so when a PAT was configured
		// and tried, its answer is what the caller must see.
		if att.token != "" {
			if r > bestAuthedRank {
				bestAuthedRank, bestAuthed = r, res
			}
			continue
		}
		if r > bestRank {
			bestRank, best = r, res
		}
	}
	// If any token produced a failure, that is the answer: never report an
	// anonymous outcome when a configured PAT was tried and failed.
	if bestAuthedRank != rankNone {
		return bestAuthed, nil
	}
	if best.status == 0 && lastErr != nil {
		return GHResponse{}, lastErr
	}
	return best, nil
}

type tokenAttempt struct {
	id string
	// name is the label the user gave this token in Settings; the anonymous attempt has none.
	name  string
	token string
}

// tokenOrder is the credential order every repo read tries: the cached winner
// for cacheKey (an empty cacheKey skips the cache), then every configured
// token, then an unauthenticated attempt so public resources work without a
// PAT -- unless NoAnonymous drops it, the host's policy that a server holding
// a credential never lets a read fall through anonymously. Duplicates are
// dropped. Writes use writeTokenOrder, which never falls through.
func (e *GitHub) tokenOrder(cacheKey string, NoAnonymous bool) []tokenAttempt {
	var order []tokenAttempt
	seen := set.New[string]()
	add := func(id, name, token string) {
		if !seen.Add(id) {
			return
		}
		order = append(order, tokenAttempt{id: id, name: name, token: token})
	}
	if e.cache != nil && cacheKey != "" {
		if id, ok := e.cache.Get(cacheKey); ok && id != "" {
			if t, found := e.tokenByID(id); found {
				add(id, t.Name, t.Token)
			}
		}
	}
	for _, t := range e.tokens {
		add(t.ID, t.Name, t.Token)
	}
	if !NoAnonymous && !e.noAnonymous {
		add("", "", "") // unauthenticated fallback (public repositories)
	}
	return order
}

func (e *GitHub) tokenByID(id string) (GitHubToken, bool) {
	return findToken(e.tokens, id)
}

// findToken looks a token id up in a credential list.
func findToken(list []GitHubToken, id string) (GitHubToken, bool) {
	for _, t := range list {
		if t.ID == id {
			return t, true
		}
	}
	return GitHubToken{}, false
}

func (e *GitHub) ContentsURL(org, repo, inner, ref string) string {
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
// this owner". GitHub splits this across endpoints — /orgs/<owner>/repos for
// organizations and /users/<owner>/repos for personal accounts — so each
// candidate credential is tried against the org endpoint, then the user
// endpoint, until returns 2xx. The winning credential is cached under the
// lowercased owner name (no slash, so it never collides with an "org/repo" key).
func (e *GitHub) OwnerRepos(ctx context.Context, owner string) ([]GHRepo, bool, GHResponse, error) {
	cacheKey := strings.ToLower(owner)
	var best GHResponse
	bestRank := rankNone
	var lastErr error
	for _, att := range e.tokenOrder(cacheKey, false) {
		for _, isUser := range []bool{false, true} {
			res, err := e.doGet(ctx, e.ownerReposURL(owner, isUser, 1), att.token, "application/vnd.github+json")
			if err != nil {
				lastErr = err
				continue
			}
			if res.status >= 200 && res.status < 300 {
				e.Remember(cacheKey, att.id)
				repos, truncated, perr := e.collectOwnerRepos(ctx, owner, isUser, att.token, res.body)
				return repos, truncated, GHResponse{status: res.status}, perr
			}
			// Same most-informative-failure rule as FetchURLOpts: anonymous explains least.
			res.authed = att.token != ""
			res.credentialName = att.name
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

// ErrOwnerListing signals that no credential yielded a 2xx owner listing.
var ErrOwnerListing = fmt.Errorf("owner listing returned no successful response")

// collectOwnerRepos parses the page of an owner's repositories and follows
// pagination (up to OwnerReposMaxPages full pages) using the credential that
// already worked. It reports truncated=true when more pages remain past the cap.
func (e *GitHub) collectOwnerRepos(ctx context.Context, owner string, isUser bool, token string, firstBody []byte) (repos []GHRepo, truncated bool, err error) {
	repos, err = parseRepoArray(firstBody)
	if err != nil {
		return nil, false, err
	}
	lastLen := len(repos)
	for page := 2; lastLen == OwnerReposPerPage; page++ {
		if page > OwnerReposMaxPages {
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
		e.base, kind, url.PathEscape(owner), OwnerReposPerPage, page)
}

func (e *GitHub) doGet(ctx context.Context, target, token, accept string) (GHResponse, error) {
	return e.doRequest(ctx, http.MethodGet, target, token, accept, nil)
}

// doGetOpts is doGet honouring a per-call response cap.
func (e *GitHub) doGetOpts(ctx context.Context, target, token, accept string, opt FetchOptions) (GHResponse, error) {
	if opt.NoRedirect {
		return e.doRequestOn(ctx, e.noRedirectClient(), http.MethodGet, target, token, accept, nil, opt.maxBytes())
	}
	if opt.MaxBytes <= 0 {
		return e.doGet(ctx, target, token, accept)
	}
	return e.doRequestCapped(ctx, http.MethodGet, target, token, accept, nil, opt.MaxBytes)
}

func (o FetchOptions) maxBytes() int64 {
	if o.MaxBytes <= 0 {
		return GitHubMaxResponseBytes
	}
	return o.MaxBytes
}

// noRedirectClient is this client with redirect following turned off.
func (e *GitHub) noRedirectClient() *http.Client {
	c := *e.hc
	c.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return &c
}

// FetchRedirectTarget performs an UNAUTHENTICATED GET of an absolute URL a redirect sent.
func (e *GitHub) FetchRedirectTarget(ctx context.Context, target string, maxBytes int64) (GHResponse, error) {
	if maxBytes <= 0 {
		maxBytes = GitHubMaxResponseBytes
	}
	return e.doRequestOn(ctx, e.hc, http.MethodGet, target, "", "", nil, maxBytes)
}

// doRequest performs GitHub API call with a single credential. A non-nil
// body is sent as JSON.
func (e *GitHub) doRequest(ctx context.Context, method, target, token, accept string, body []byte) (GHResponse, error) {
	return e.doRequestCapped(ctx, method, target, token, accept, body, GitHubMaxResponseBytes)
}

// doRequestCapped is doRequest reading at most MaxBytes of the response body.
func (e *GitHub) doRequestCapped(ctx context.Context, method, target, token, accept string, body []byte, MaxBytes int64) (GHResponse, error) {
	return e.doRequestOn(ctx, e.hc, method, target, token, accept, body, MaxBytes)
}

// doRequestOn is doRequestCapped on a named client, which is how a caller that
// must see a 3xx rather than follow it gets.
func (e *GitHub) doRequestOn(ctx context.Context, hc *http.Client, method, target, token, accept string, body []byte, MaxBytes int64) (GHResponse, error) {
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
	resp, err := hc.Do(req)
	if err != nil {
		return GHResponse{}, err
	}
	defer resp.Body.Close()
	data, truncated, err := agentic.ReadCapped(resp.Body, MaxBytes)
	if err != nil {
		return GHResponse{}, err
	}
	// Every response states the quota, so nothing asks. see rate_limit_headers.go
	e.observeRateLimit(token, resp.Header)
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
