package repo

import (
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func rateLimitHeaders(remaining string, reset time.Time, resource string) http.Header {
	h := http.Header{}
	h.Set("X-RateLimit-Remaining", remaining)
	h.Set("X-RateLimit-Reset", strconv.FormatInt(reset.Unix(), 10))
	h.Set("X-RateLimit-Resource", resource)
	return h
}

func TestClassifyRateLimitPrimaryCarriesResetAndResource(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	res := GHResponse{
		status: http.StatusForbidden,
		header: rateLimitHeaders("0", now.Add(42*time.Second), "code_search"),
		body:   []byte(`{"message":"API rate limit exceeded for user ID 1."}`),
	}
	rl, limited := classifyRateLimit(res, now)
	require.True(t, limited, "403 with remaining=0 is a rate limit")
	assert.Equal(t, "code_search", rl.resource)
	assert.Equal(t, 42*time.Second, rl.retryIn)
	assert.False(t, rl.secondary, "remaining=0 is the primary quota")
}

func TestClassifyRateLimitSecondaryUsesRetryAfter(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	h := http.Header{}
	h.Set("Retry-After", "60")
	res := GHResponse{
		status: http.StatusTooManyRequests,
		header: h,
		body:   []byte(`{"message":"You have exceeded a secondary rate limit."}`),
	}
	rl, limited := classifyRateLimit(res, now)
	require.True(t, limited)
	assert.True(t, rl.secondary)
	assert.Equal(t, time.Minute, rl.retryIn)
}

func TestClassifyRateLimitIgnoresPlainDenials(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	for _, res := range []GHResponse{
		{status: http.StatusForbidden, header: http.Header{}, body: []byte(`{"message":"Resource not accessible"}`)},
		{status: http.StatusUnauthorized, header: http.Header{}, body: []byte(`{"message":"Requires authentication"}`)},
		{status: http.StatusNotFound, header: http.Header{}, body: []byte(`{"message":"Not Found"}`)},
	} {
		_, limited := classifyRateLimit(res, now)
		assert.False(t, limited, "status %d without rate-limit signals is not a rate limit", res.status)
	}
}

// The anonymous attempt is always last, so before this ranking its 401 was the
// failure every caller reported — hiding a token's real, transient 403.
func TestFailureRankPrefersTokenFailureOverAnonymous401(t *testing.T) {
	anon401 := GHResponse{status: http.StatusUnauthorized, header: http.Header{}, authed: false}
	tokenDenied := GHResponse{status: http.StatusForbidden, header: http.Header{}, authed: true}
	limited := GHResponse{
		status: http.StatusForbidden,
		header: rateLimitHeaders("0", time.Now().Add(time.Minute), "code_search"),
		authed: true,
	}
	assert.Greater(t, failureRank(tokenDenied), failureRank(anon401), "a token's failure outranks the anonymous 401")
	assert.Greater(t, failureRank(limited), failureRank(tokenDenied), "a rate limit outranks any other failure")
	anon404 := GHResponse{status: http.StatusNotFound, header: http.Header{}, authed: false}
	assert.Greater(t, failureRank(anon404), failureRank(anon401), "anonymous 401 is the least informative failure there is")
}

func TestExplainFailureRateLimitIsMarkedTransientWithItsWait(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	res := GHResponse{
		status: http.StatusForbidden,
		header: rateLimitHeaders("0", now.Add(42*time.Second), "code_search"),
		body:   []byte(`{"message":"API rate limit exceeded"}`),
		authed: true,
	}
	msg := explainFailure("read", "/repos/o/r/x.go", res, 1, now)
	assert.Contains(t, msg, "TRANSIENT")
	assert.Contains(t, msg, "42s")
	assert.Contains(t, msg, "code_search")
	assert.Contains(t, msg, "retry the same call")
	assert.Contains(t, msg, "one of your configured tokens", "authed with no recorded name still says a real token was hit")
	assert.NotContains(t, msg, "Settings -> github", "a rate limit must not send the user off to reconfigure a working token")
}

// The one fact a bare "rate limit exceeded" never used to say: whether the
// request that got rate-limited carried a real credential at all.
func TestExplainFailureRateLimitNamesTheCredentialThatHitIt(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	named := GHResponse{
		status: http.StatusForbidden,
		header: rateLimitHeaders("0", now.Add(time.Minute), "core"),
		body:   []byte(`{"message":"API rate limit exceeded"}`),
		authed: true, credentialName: "wow-look-at-my",
	}
	msg := explainFailure("read", "/repos/o/r", named, 1, now)
	assert.Contains(t, msg, `your "wow-look-at-my" token`)

	anon := GHResponse{
		status: http.StatusForbidden,
		header: rateLimitHeaders("0", now.Add(time.Minute), "core"),
		body:   []byte(`{"message":"API rate limit exceeded"}`),
		authed: false,
	}
	msg = explainFailure("read", "/repos/o/r", anon, 1, now)
	assert.Contains(t, msg, "unauthenticated (anonymous) request, not one of your configured tokens")
}

func TestExplainFailureDistinguishesTheDenialModes(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	notFound := GHResponse{status: http.StatusNotFound, header: http.Header{}}

	withTokens := explainFailure("read", "/repos/o/r", notFound, 2, now)
	assert.Contains(t, withTokens, "2 configured token(s)")
	assert.Contains(t, withTokens, "will not fix itself", "404 with tokens is permanent — say so, so it is not retried")

	noTokens := explainFailure("read", "/repos/o/r", notFound, 0, now)
	assert.Contains(t, noTokens, "no GitHub token is configured")
	assert.Contains(t, noTokens, "Settings -> github")

	denied := explainFailure("read", "/repos/o/r", GHResponse{status: http.StatusForbidden, header: http.Header{}}, 2, now)
	assert.Contains(t, denied, "lack the required permission")
	assert.NotContains(t, denied, "does not exist", "a 403 is not evidence about existence")
}

// A 401 means GitHub rejected the credential itself — the previous message
// ("the tokens are valid but lack access") said the opposite of what a 401
// means, since a 403 (not a 401) is what GitHub sends for a valid-but-scoped
// token.
func TestExplainFailure401RejectsTheCredentialNotJustAccess(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	res := GHResponse{
		status: http.StatusUnauthorized,
		header: http.Header{},
		body:   []byte(`{"message":"Bad credentials"}`),
	}
	msg := explainFailure("read", "/repos/o/r/x.go", res, 2, now)
	assert.Contains(t, msg, "rejected (401)")
	assert.Contains(t, msg, `"Bad credentials"`)
	assert.Contains(t, msg, "invalid, revoked and expired")
	assert.NotContains(t, msg, "valid but lack", "a 401 is a rejected credential, not a valid one lacking scope")
}

// GitHub does not expose any signal distinguishing an expired token from a
// revoked or simply wrong one on a 401 — the message must say so rather than
// invent a distinction the API does not make.
func TestExplainFailure401WithNoBodySaysNothingItCannotKnow(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	res := GHResponse{status: http.StatusUnauthorized, header: http.Header{}}
	msg := explainFailure("read", "/repos/o/r", res, 1, now)
	assert.Contains(t, msg, "cannot be told from here")
	assert.NotContains(t, msg, "GitHub says", "no body message to quote")
}

// X-Accepted-GitHub-Permissions is the real, documented signal GitHub sends
// on a fine-grained-PAT/GitHub-App permission failure, naming the exact
// permission needed — far more actionable than the generic denial.
func TestExplainFailure403NamesTheMissingPermission(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	h := http.Header{}
	h.Set("X-Accepted-GitHub-Permissions", "contents=read")
	res := GHResponse{
		status: http.StatusForbidden,
		header: h,
		body:   []byte(`{"message":"Resource not accessible by personal access token"}`),
	}
	msg := explainFailure("read", "/repos/o/r/x.go", res, 1, now)
	assert.Contains(t, msg, "needs the contents=read permission")
	assert.NotContains(t, msg, "Resource not accessible", "the named permission is more useful than the generic body message")
}

// Classic PATs predate X-Accepted-GitHub-Permissions, so a 403 without it
// falls back to GitHub's own message body instead of going detail-free.
func TestExplainFailure403FallsBackToBodyMessageWithoutThePermissionHeader(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	res := GHResponse{
		status: http.StatusForbidden,
		header: http.Header{},
		body:   []byte(`{"message":"Must have admin rights to Repository."}`),
	}
	msg := explainFailure("read", "/repos/o/r", res, 1, now)
	assert.Contains(t, msg, `"Must have admin rights to Repository."`)
}

// X-GitHub-SSO's "required; url=..." form names a one-hour authorization
// link — a different fix (visit the URL) than any scope/permission gap, and
// GitHub sends it independently of whether the token has the right scope.
func TestExplainFailure403SurfacesTheSSOAuthorizeURL(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	h := http.Header{}
	h.Set("X-GitHub-SSO", "required; url=https://github.com/orgs/octo-org/sso?authorization_request=abc123")
	h.Set("X-Accepted-GitHub-Permissions", "contents=read")
	res := GHResponse{status: http.StatusForbidden, header: h}
	msg := explainFailure("read", "/repos/o/r/x.go", res, 1, now)
	assert.Contains(t, msg, "https://github.com/orgs/octo-org/sso?authorization_request=abc123")
	assert.Contains(t, msg, "SAML SSO")
	assert.NotContains(t, msg, "contents=read", "SSO is a policy block, not a scope gap — it outranks the permission header")
}

// X-OAuth-Scopes/X-Accepted-OAuth-Scopes is the classic-PAT equivalent of
// X-Accepted-GitHub-Permissions: diff what the token has against what the
// endpoint needs to name the exact missing scope.
func TestExplainFailure403NamesTheMissingOAuthScope(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	h := http.Header{}
	h.Set("X-OAuth-Scopes", "public_repo, user")
	h.Set("X-Accepted-OAuth-Scopes", "repo")
	res := GHResponse{status: http.StatusForbidden, header: h}
	msg := explainFailure("read", "/repos/o/r", res, 1, now)
	assert.Contains(t, msg, "needs the repo scope")
}

// A scope already present in X-OAuth-Scopes is not missing, even if it is
// also listed as accepted — only the difference is worth reporting.
func TestExplainFailure403OAuthScopeDetailIsEmptyWhenTokenAlreadyHasIt(t *testing.T) {
	res := GHResponse{status: http.StatusForbidden, header: http.Header{}}
	assert.Equal(t, "", oauthScopeDetail(res), "no X-Accepted-OAuth-Scopes at all — fine-grained PATs and GitHub Apps never send it")

	h := http.Header{}
	h.Set("X-OAuth-Scopes", "repo, user")
	h.Set("X-Accepted-OAuth-Scopes", "repo")
	assert.Equal(t, "", oauthScopeDetail(GHResponse{status: http.StatusForbidden, header: h}), "the token already has every accepted scope")
}

func ghHeaderWithExpiry(t time.Time) http.Header {
	h := http.Header{}
	h.Set("GitHub-Authentication-Token-Expiration", t.UTC().Format(githubTokenExpirationLayout))
	return h
}

// A token with months of runway must generate no advisory at all, on every
// call — the whole point of the warn window is silence in the common case.
func TestTokenExpiryDetailIsSilentFarFromExpiry(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	res := GHResponse{header: ghHeaderWithExpiry(now.Add(90 * 24 * time.Hour))}
	assert.Equal(t, "", tokenExpiryDetail(res, now))
}

func TestTokenExpiryDetailIsSilentWhenTheHeaderIsAbsentOrUnparseable(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	assert.Equal(t, "", tokenExpiryDetail(GHResponse{header: http.Header{}}, now), "no header at all")

	bad := http.Header{}
	bad.Set("GitHub-Authentication-Token-Expiration", "not a date")
	assert.Equal(t, "", tokenExpiryDetail(GHResponse{header: bad}, now), "a value that doesn't parse must not crash or guess")
}

func TestTokenExpiryDetailWarnsInsideTheWindow(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	res := GHResponse{header: ghHeaderWithExpiry(now.Add(3 * 24 * time.Hour))}
	detail := tokenExpiryDetail(res, now)
	assert.Contains(t, detail, "expires on")
	assert.Contains(t, detail, "rotate it")
}

func TestTokenExpiryDetailReportsAnAlreadyExpiredToken(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	res := GHResponse{header: ghHeaderWithExpiry(now.Add(-24 * time.Hour))}
	assert.Contains(t, tokenExpiryDetail(res, now), "expired on")
}

// The expiry advisory rides the SAME response that explained the denial —
// GitHub identified the credential to answer a 403, so its expiration header
// is meaningful there, not just on a bare 2xx.
func TestExplainFailure403AppendsExpiryAdvisoryAlongsideThePermissionDetail(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	h := http.Header{}
	h.Set("X-Accepted-GitHub-Permissions", "contents=read")
	for k, v := range ghHeaderWithExpiry(now.Add(2 * 24 * time.Hour)) {
		h[k] = v
	}
	res := GHResponse{status: http.StatusForbidden, header: h}
	msg := explainFailure("read", "/repos/o/r/x.go", res, 1, now)
	assert.Contains(t, msg, "needs the contents=read permission")
	assert.Contains(t, msg, "rotate it")
}
