package repo

import (
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Failure classification for the repo reads; see docs/tools/repo-tools.md.

// rateLimit describes a GitHub rate-limit refusal.
type rateLimit struct {
	// resource names the exhausted bucket, empty when GitHub did not say.
	resource string
	// retryIn is how long until the bucket refills, 0 when unknown.
	retryIn time.Duration
	// secondary marks the abuse/secondary limit, which clears on its own.
	secondary bool
}

// classifyRateLimit reports whether a non-2xx response is a rate limit, and
// what kind. GitHub signals the primary limit with 403/429 plus
// x-ratelimit-remaining: 0 (x-ratelimit-reset carries the epoch second it
// refills), and the secondary limit with a retry-after header or a message
// naming it.
func classifyRateLimit(res GHResponse, now time.Time) (rateLimit, bool) {
	if res.status != http.StatusForbidden && res.status != http.StatusTooManyRequests {
		return rateLimit{}, false
	}
	rl := rateLimit{resource: strings.TrimSpace(res.header.Get("X-RateLimit-Resource"))}
	hit := false
	if strings.TrimSpace(res.header.Get("X-RateLimit-Remaining")) == "0" {
		hit = true
		if secs, err := strconv.ParseInt(strings.TrimSpace(res.header.Get("X-RateLimit-Reset")), 10, 64); err == nil {
			if d := time.Unix(secs, 0).Sub(now); d > 0 {
				rl.retryIn = d.Round(time.Second)
			}
		}
	}
	if after := strings.TrimSpace(res.header.Get("Retry-After")); after != "" {
		if secs, err := strconv.Atoi(after); err == nil && secs > 0 {
			hit, rl.secondary = true, true
			rl.retryIn = time.Duration(secs) * time.Second
		}
	}
	if msg := strings.ToLower(GitHubErrorMessage(res.body)); strings.Contains(msg, "rate limit") {
		hit = true
		if strings.Contains(msg, "secondary rate limit") {
			rl.secondary = true
		}
	}
	return rl, hit
}

// whichCredentialHit names the credential that produced a rate-limited response.
func whichCredentialHit(res GHResponse) string {
	if !res.authed {
		return "the unauthenticated (anonymous) request, not one of your configured tokens"
	}
	if res.credentialName != "" {
		return fmt.Sprintf("your %q token", res.credentialName)
	}
	return "one of your configured tokens"
}

// waitAdvice renders the wait a rate limit implies.
func (rl rateLimit) waitAdvice() string {
	if rl.retryIn <= 0 {
		return "It clears on its own shortly"
	}
	return fmt.Sprintf("It clears in %s", rl.retryIn)
}

// failureRank scores a non-2xx response by how much it explains, so
// FetchURLOpts can report the most informative attempt instead of the last
// one. The last attempt is the anonymous fallback, and its 401 ("Requires
// authentication") only restates that no credential was sent: preferring it
// over a token's own 403 turned a 42-second code-search rate limit into what
// looked like a permanent authentication failure.
//
// A rate-limited TOKEN attempt outranks a rate-limited ANONYMOUS attempt even
// though both are "a 403 saying the quota is spent": the token's 403 proves a
// credential WAS tried and also hit the wall, whereas the anonymous 403 only
// says no credential was sent. Reporting the token's version tells the caller
// "your PAT is rate-limited" instead of the misleading "this was anonymous."
func failureRank(res GHResponse) failureRankLevel {
	if _, limited := classifyRateLimit(res, time.Now()); limited {
		if res.authed {
			return rankTokenRateLimited
		}
		return rankAnonRateLimited
	}
	switch {
	case res.authed:
		return rankTokenFailure
	case res.status == http.StatusUnauthorized:
		return rankAnon401 // "you sent no credential" — true and useless
	default:
		return rankAnonFailure
	}
}

// failureRankLevel is an ORDERED ranking of how informative a failed attempt is.
type failureRankLevel int

const (
	// rankNone is the "no failure yet" floor, below every real ranking.
	rankNone failureRankLevel = iota
	// rankAnon401 is the anonymous 401: no credential was sent.
	rankAnon401
	// rankAnonFailure is any other anonymous failure (e.g. a 404 on a public resource).
	rankAnonFailure
	// rankTokenFailure is a token's non-rate-limit failure (403 denied, 404 not found).
	rankTokenFailure
	// rankAnonRateLimited is the anonymous rate limit: the unauthenticated quota was spent.
	rankAnonRateLimited
	// rankTokenRateLimited is a TOKEN's rate limit; outranks the anonymous one.
	rankTokenRateLimited
)

// githubTokenExpirationLayout matches GitHub's GitHub-Authentication-Token-Expiration header.
const githubTokenExpirationLayout = "2006-01-02 15:04:05 MST"

// tokenExpiryWarnWindow is how far ahead of expiry this starts warning.
const tokenExpiryWarnWindow = 14 * 24 * time.Hour

// tokenExpiryDetail renders GitHub's GitHub-Authentication-Token-Expiration
// header when the token behind this call is expired or expiring soon. GitHub
// sends it on requests it could identify the token for — which includes a 403
// (permission denied, but the credential itself was recognized) as well as
// any 2xx — so a caller with a real GHResponse from either case can surface it.
// Absent or outside the warn window, this says nothing: a token with months
// of runway generates no advisory, on every call, forever.
func tokenExpiryDetail(res GHResponse, now time.Time) string {
	raw := strings.TrimSpace(res.header.Get("GitHub-Authentication-Token-Expiration"))
	if raw == "" {
		return ""
	}
	expires, err := time.Parse(githubTokenExpirationLayout, raw)
	if err != nil {
		return ""
	}
	if !expires.After(now) {
		return fmt.Sprintf(" Also: this token expired on %s.", expires.Format("2006-01-02"))
	}
	if until := expires.Sub(now); until <= tokenExpiryWarnWindow {
		return fmt.Sprintf(" Also: this token expires on %s (in %s) — rotate it before then.", expires.Format("2006-01-02"), until.Round(time.Hour))
	}
	return ""
}

// authRejectionDetail surfaces GitHub's own explanation for a 401, if it sent one.
func authRejectionDetail(res GHResponse, now time.Time) string {
	var detail string
	if msg := GitHubErrorMessage(res.body); msg != "" {
		detail = fmt.Sprintf(" GitHub says: %q.", msg)
	}
	return detail + tokenExpiryDetail(res, now)
}

// ssoAuthorizeDetail renders GitHub's SAML SSO block and its one-hour authorization URL.
func ssoAuthorizeDetail(res GHResponse) string {
	sso := res.header.Get("X-GitHub-SSO")
	idx := strings.Index(sso, "url=")
	if idx < 0 {
		return ""
	}
	url := strings.TrimSpace(sso[idx+len("url="):])
	if url == "" {
		return ""
	}
	return fmt.Sprintf(" This organization requires SAML SSO and the token has not been authorized for it — authorize it (link expires in an hour): %s", url)
}

// oauthScopeDetail is the classic-PAT/OAuth-token counterpart of
// X-Accepted-GitHub-Permissions: X-OAuth-Scopes lists what the token has,
// X-Accepted-OAuth-Scopes lists what the endpoint needs, and a scope present
// in the second but absent from the first is exactly what is missing.
// Fine-grained PATs and GitHub Apps don't send either header — nothing to
// derive when X-Accepted-OAuth-Scopes is absent.
func oauthScopeDetail(res GHResponse) string {
	accepted := splitScopeHeader(res.header.Get("X-Accepted-OAuth-Scopes"))
	if len(accepted) == 0 {
		return ""
	}
	have := splitScopeHeader(res.header.Get("X-OAuth-Scopes"))
	var missing []string
	for _, scope := range accepted {
		if !slices.Contains(have, scope) {
			missing = append(missing, scope)
		}
	}
	if len(missing) == 0 {
		return ""
	}
	return fmt.Sprintf(" It needs the %s scope.", strings.Join(missing, ", "))
}

func splitScopeHeader(header string) []string {
	var scopes []string
	for _, scope := range strings.Split(header, ",") {
		if scope = strings.TrimSpace(scope); scope != "" {
			scopes = append(scopes, scope)
		}
	}
	return scopes
}

// missingPermissionDetail renders what a 403 is missing, most actionable signal first.
func missingPermissionDetail(res GHResponse, now time.Time) string {
	return primaryDenialDetail(res) + tokenExpiryDetail(res, now)
}

// primaryDenialDetail is missingPermissionDetail without the expiry
// advisory, kept separate so each candidate is only evaluated once.
func primaryDenialDetail(res GHResponse) string {
	if detail := ssoAuthorizeDetail(res); detail != "" {
		return detail
	}
	if perm := strings.TrimSpace(res.header.Get("X-Accepted-GitHub-Permissions")); perm != "" {
		return fmt.Sprintf(" It needs the %s permission.", perm)
	}
	if detail := oauthScopeDetail(res); detail != "" {
		return detail
	}
	if msg := GitHubErrorMessage(res.body); msg != "" {
		return fmt.Sprintf(" GitHub says: %q.", msg)
	}
	return ""
}

// explainFailure renders a non-2xx response as one model-facing sentence:
// what failed, which of the distinct causes it was, and what to do about it.
// op reads as a verb phrase ("read", "list commits of"), what names the
// resource.
func explainFailure(op, what string, res GHResponse, numTokens int, now time.Time) string {
	lead := fmt.Sprintf("could not %s %s", op, what)
	if rl, limited := classifyRateLimit(res, now); limited {
		kind := "rate limit"
		if rl.secondary {
			kind = "secondary rate limit"
		}
		bucket := ""
		if rl.resource != "" {
			bucket = " on the " + rl.resource + " quota"
		}
		return fmt.Sprintf("%s: GitHub %s exceeded%s (%d) — TRANSIENT, not an auth problem. This was %s. %s; retry the same call after that.",
			lead, kind, bucket, res.status, whichCredentialHit(res), rl.waitAdvice())
	}
	switch res.status {
	case http.StatusNotFound:
		if numTokens == 0 {
			return fmt.Sprintf("%s: not found (404), and no GitHub token is configured. If it is private, add a token in Settings -> github; if it is public, check the spelling.", lead)
		}
		return fmt.Sprintf("%s: not found (404) with any of your %d configured token(s). It does not exist, or none of those tokens can see it — this will not fix itself; try a different path or repository.", lead, numTokens)
	case http.StatusUnauthorized, http.StatusForbidden:
		if numTokens == 0 {
			return fmt.Sprintf("%s: authentication required (%d) and no GitHub token is configured. Add one in Settings -> github.", lead, res.status)
		}
		if res.status == http.StatusUnauthorized {
			return fmt.Sprintf("%s: every configured token was rejected (401) — GitHub does not accept the credential itself.%s That covers invalid, revoked and expired tokens alike; GitHub's API returns the same response for all three, so which one it is cannot be told from here. Create a fresh token in Settings -> github.", lead, authRejectionDetail(res, now))
		}
		return fmt.Sprintf("%s: access denied (403) with every configured token. The token(s) are accepted but lack the required permission.%s", lead, missingPermissionDetail(res, now))
	default:
		if msg := GitHubErrorMessage(res.body); msg != "" {
			return fmt.Sprintf("%s: GitHub returned %d: %s", lead, res.status, msg)
		}
		return fmt.Sprintf("%s: GitHub returned status %d", lead, res.status)
	}
}
