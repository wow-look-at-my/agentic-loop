package repo

import (
	"context"
	"net/http"
	"strings"
	"time"
)

// The GitHub client the repo tools run on, and the host seams around it.

const (
	defaultGitHubAPIBase = "https://api.github.com"
	// GitHubMaxResponseBytes is the hard cap on one API response body.
	GitHubMaxResponseBytes = 5 << 20
	githubRequestTimeout   = 30 * time.Second
	// OwnerReposPerPage and OwnerReposMaxPages bound an owner listing.
	OwnerReposPerPage  = 100
	OwnerReposMaxPages = 5
)

// GitHubToken is one configured PAT; AllowModelWrites opts it into model writes.
type GitHubToken struct {
	ID               string
	Name             string
	Token            string
	AllowModelWrites bool
}

// ModelWriteTokens filters a credential list to the tokens flagged for model writes.
func ModelWriteTokens(tokens []GitHubToken) []GitHubToken {
	var out []GitHubToken
	for _, t := range tokens {
		if t.AllowModelWrites {
			out = append(out, t)
		}
	}
	return out
}

// RepoKeyCache remembers which token works for a lowercased org/repo; must be concurrency-safe.
type RepoKeyCache interface {
	Get(repo string) (tokenID string, ok bool)
	Put(repo, tokenID string)
}

// GitHubConfig configures NewGitHub; APIBaseURL defaults to https://api.github.com.
type GitHubConfig struct {
	HTTPClient *http.Client
	APIBaseURL string
	// Tokens is the READ credential list; writes never touch it.
	Tokens []GitHubToken
	// WriteTokens is the WRITE list; nil means this client cannot write at all.
	WriteTokens []GitHubToken
	// NoAnonymous drops the unauthenticated attempt from every read order.
	NoAnonymous bool
	Cache       RepoKeyCache
	// OnRateLimit hears every response's core-quota headers, so a host tracks
	// each credential's standing without asking GitHub a second time.
	OnRateLimit func(RateLimitObservation)
}

// GitHub is the credential-rotating GitHub REST client behind the repo tools.
type GitHub struct {
	hc          *http.Client
	base        string
	tokens      []GitHubToken
	writeTokens []GitHubToken
	noAnonymous bool
	cache       RepoKeyCache
	onRateLimit func(RateLimitObservation)
}

// NewGitHub builds the client; the only constructor, so all callers share one cache.
func NewGitHub(cfg GitHubConfig) *GitHub {
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: githubRequestTimeout}
	}
	base := strings.TrimRight(strings.TrimSpace(cfg.APIBaseURL), "/")
	if base == "" {
		base = defaultGitHubAPIBase
	}
	return &GitHub{
		hc:          hc,
		base:        base,
		tokens:      cfg.Tokens,
		writeTokens: cfg.WriteTokens,
		noAnonymous: cfg.NoAnonymous,
		cache:       cfg.Cache,
		onRateLimit: cfg.OnRateLimit,
	}
}

// BaseURL is the API root every request is built on.
func (e *GitHub) BaseURL() string { return e.base }

// Credentials is the order a read tries: cached winner, then tokens, then anonymous.
func (e *GitHub) Credentials(cacheKey string, NoAnonymous bool) []Credential {
	order := e.tokenOrder(cacheKey, NoAnonymous)
	out := make([]Credential, 0, len(order))
	for _, a := range order {
		out = append(out, Credential{ID: a.id, Name: a.name, Token: a.token})
	}
	return out
}

// Credential is one attempt's identity; an empty Token is the anonymous attempt.
type Credential struct {
	ID    string
	Name  string
	Token string
}

// Remember records which credential reached a repository, so the next call starts with it.
func (e *GitHub) Remember(cacheKey, credentialID string) {
	if e.cache != nil && cacheKey != "" {
		e.cache.Put(cacheKey, credentialID)
	}
}

// RepoCacheKey is the RepoKeyCache key for an org/repo pair.
func RepoCacheKey(org, repo string) string { return strings.ToLower(org + "/" + repo) }

// RepoPath renders the canonical /repos/<org>/<repo>/<inner> form for messages.
func RepoPath(org, repo, inner string) string {
	p := "/repos/" + org + "/" + repo
	if inner != "" {
		p += "/" + inner
	}
	return p
}

// TokenCount is how many tokens this client will try; the failure reads differ by count.
func (e *GitHub) TokenCount() int { return len(e.tokens) }

// WriteCredentials: cached write-capable winner, then write tokens, never anonymous.
func (e *GitHub) WriteCredentials(cacheKey string) []Credential {
	order := e.writeTokenOrder(cacheKey)
	out := make([]Credential, 0, len(order))
	for _, a := range order {
		out = append(out, Credential{ID: a.id, Name: a.name, Token: a.token})
	}
	return out
}

// Get performs one GET with ONE credential, for a host running its own rotation.
func (e *GitHub) Get(ctx context.Context, target, token, accept string) (GHResponse, error) {
	return e.doGet(ctx, target, token, accept)
}

// Status is the HTTP status behind a credential-specific failure.
func (a GitHubAuthError) Status() int { return a.status }

// Object names the git object a failing step read, if it read one.
func (a GitHubAuthError) Object() string { return a.object }

// Do performs one request with ONE credential, for a host running its own rotation.
func (e *GitHub) Do(ctx context.Context, method, target, token, accept string, body []byte) (GHResponse, error) {
	return e.doRequest(ctx, method, target, token, accept, body)
}
