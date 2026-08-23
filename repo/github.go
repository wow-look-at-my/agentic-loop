package repo

import (
	"context"
	"net/http"
	"strings"
	"time"
)

// The GitHub client the repo tools run on, and the host seams around it.
//
// It is a separate value from the tools because a host needs it directly: the
// same credential rotation and the same winning-token cache serve the repo
// tools, a /repos filesystem folder, and a settings "test this token" button,
// and building three clients would mean three caches disagreeing about which
// credential works.
//
// Nothing here reads an environment or a config file. Credentials arrive as
// values, and where they are STORED is the host's business.

const (
	defaultGitHubAPIBase = "https://api.github.com"
	// GitHubMaxResponseBytes is the hard cap on one API response body.
	GitHubMaxResponseBytes = 5 << 20
	githubRequestTimeout   = 30 * time.Second
	// OwnerReposPerPage and OwnerReposMaxPages bound an owner listing: how many
	// repositories one page holds, and how many pages are fetched before the
	// listing is reported truncated.
	OwnerReposPerPage  = 100
	OwnerReposMaxPages = 5
)

// GitHubToken is one configured personal access token the repo tools may use.
// Token is the secret value; ID is an opaque identifier used as the cache value
// (so the cache never stores the secret). AllowModelWrites carries the user's
// per-token opt-in to MODEL-initiated writes: reads may use any enabled token,
// but a write driven by a model tool may only use flagged ones (see
// ModelWriteTokens and GitHubConfig.WriteTokens).
type GitHubToken struct {
	ID               string
	Name             string
	Token            string
	AllowModelWrites bool
}

// ModelWriteTokens filters a credential list down to the tokens the user has
// explicitly flagged for model-initiated writes. This is what a toolset built
// for MODEL tool calls passes as WriteTokens, so the write plumbing is handed
// only credentials the user consented to -- a user-only token never even
// reaches the model's write path.
func ModelWriteTokens(tokens []GitHubToken) []GitHubToken {
	var out []GitHubToken
	for _, t := range tokens {
		if t.AllowModelWrites {
			out = append(out, t)
		}
	}
	return out
}

// RepoKeyCache remembers which token works for a given lowercased "org/repo" so
// the repo tools don't re-probe every token on every call. Implementations must
// be safe for concurrent use. An empty token id (and ok=true) means the repo is
// reachable without authentication (public).
type RepoKeyCache interface {
	Get(repo string) (tokenID string, ok bool)
	Put(repo, tokenID string)
}

// GitHubConfig configures NewGitHub. Cache may be nil (no caching; every call
// re-probes). APIBaseURL defaults to https://api.github.com.
type GitHubConfig struct {
	HTTPClient *http.Client
	APIBaseURL string
	// VFS, when non-nil, is consulted by every read before GitHub is. A path
	// a registered mount owns is answered by that mount -- a virtual
	// filesystem (diagnostics, workspace state, anything) exposed through the
	// same read tools. Paths no mount owns fall through to the real GitHub
	// API. See VFSMux.
	VFS *VFSMux
	// Tokens is the READ credential list: every read rotates through it (then
	// unauthenticated, so public repositories work with no token at all).
	// Writes never touch it.
	Tokens []GitHubToken
	// WriteTokens is the WRITE credential list -- the only tokens a write will
	// ever use, and it never falls through to unauthenticated. The host
	// filters it by initiator: a toolset serving MODEL tool calls gets only the
	// allow_model_writes-flagged tokens (ModelWriteTokens); one acting for the
	// USER gets every enabled token. Nil means this client cannot write at all
	// (default-closed), so a write can never reach a credential the initiator
	// was not granted.
	WriteTokens []GitHubToken
	// NoAnonymous drops the unauthenticated attempt from EVERY read order this
	// client issues, even if the per-call FetchURLOpts/Credentials are passed a
	// NoAnonymous=false. It is the host's policy knob: a server with at least
	// one configured PAT must never let a read fall through to an anonymous
	// request just because every token was refused, since GitHub buckets the
	// anonymous request by the server's own IP and answers it with a different
	// (worse) verdict than a token's real failure. Hosts with no credentials
	// leave it false, and public-repository reads keep working anonymously as
	// before.
	NoAnonymous bool
	Cache       RepoKeyCache
}

// GitHub is the credential-rotating GitHub REST client: commits, pull
// requests, issues, CI, file contents and the gated writes, with the token
// rotation and the winning-token cache behind them. It also serves the VFS
// mounts registered in its config, so virtual filesystems ride the same read
// path as GitHub.
type GitHub struct {
	hc          *http.Client
	base        string
	tokens      []GitHubToken
	writeTokens []GitHubToken
	noAnonymous bool
	cache       RepoKeyCache
	vfs         *VFSMux
}

// NewGitHub builds the client. It is deliberately the only constructor: the
// repo tools, a filesystem folder over /repos and a token-test button all take
// the same value, so they share one cache and one credential order.
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
		vfs:         cfg.VFS,
	}
}

// BaseURL is the API root every request is built on, for a host composing a
// URL this client has no method for.
func (e *GitHub) BaseURL() string { return e.base }

// Credentials is the order a read tries: the cached winner for cacheKey first
// (an empty key skips the cache), then every configured token, then an
// unauthenticated attempt unless NoAnonymous drops it. Exported because a host
// doing its own fetching -- a git clone, say -- must offer the same
// credentials in the same order, or it reaches a different set of
// repositories than the tools do.
func (e *GitHub) Credentials(cacheKey string, NoAnonymous bool) []Credential {
	order := e.tokenOrder(cacheKey, NoAnonymous)
	out := make([]Credential, 0, len(order))
	for _, a := range order {
		out = append(out, Credential{ID: a.id, Name: a.name, Token: a.token})
	}
	return out
}

// Credential is one attempt's identity: the opaque id to remember, the label
// the user gave it (so a failure names something they can find), and the
// secret. An empty Token is the anonymous attempt.
type Credential struct {
	ID    string
	Name  string
	Token string
}

// Remember records which credential reached a repository, so the next call
// starts with the one that works. A host that fetches outside this client
// calls it with what worked; nothing else keeps the two in agreement.
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

// TokenCount is how many tokens this client will try. A failure explanation
// reads differently when none is configured than when three were refused, and
// only the client knows which.
func (e *GitHub) TokenCount() int { return len(e.tokens) }

// WriteCredentials is Credentials for a WRITE: the cached winner when it is
// write-capable, then every write token, and NEVER an anonymous attempt. A
// host pushing with git runs the same rotation this returns, so a push reaches
// exactly the repositories the write tools do.
func (e *GitHub) WriteCredentials(cacheKey string) []Credential {
	order := e.writeTokenOrder(cacheKey)
	out := make([]Credential, 0, len(order))
	for _, a := range order {
		out = append(out, Credential{ID: a.id, Name: a.name, Token: a.token})
	}
	return out
}

// Get performs one GET with ONE credential, for a host running its own
// rotation. The rotating reads (Fetch, FetchURL) are what a caller wants
// otherwise.
func (e *GitHub) Get(ctx context.Context, target, token, accept string) (GHResponse, error) {
	return e.doGet(ctx, target, token, accept)
}

// Status is the HTTP status behind a credential-specific failure.
func (a GitHubAuthError) Status() int { return a.status }

// Object names the one git object the failing step read, when it read one. A
// 404 there means EITHER that the credential cannot see the repository OR that
// the object is not in it, and nothing in the response distinguishes them --
// which is why the two are told apart after every credential has been tried,
// never during.
func (a GitHubAuthError) Object() string { return a.object }

// Do performs one request with ONE credential, for a host running its own
// rotation -- a git push builds refs, blobs and trees itself and must send
// them through the same client, or the two disagree about which token works.
// A non-nil body is sent as JSON.
func (e *GitHub) Do(ctx context.Context, method, target, token, accept string, body []byte) (GHResponse, error) {
	return e.doRequest(ctx, method, target, token, accept, body)
}
