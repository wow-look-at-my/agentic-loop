package repo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// RateLimitStatus is the result of one GET /rate_limit probe: GitHub's own
// accounting of the "core" REST budget for the credential that made the
// call, not a figure derived from some other response's headers. A blank
// Token probes anonymously -- the reading for "this server's own IP,
// unauthenticated" -- since GitHub buckets an unauthenticated request by
// the caller's IP address the same way it buckets an authenticated one by
// token.
type RateLimitStatus struct {
	OK        bool      `json:"ok"`
	Error     string    `json:"error,omitempty"`
	Limit     int       `json:"limit,omitempty"`
	Remaining int       `json:"remaining,omitempty"`
	Used      int       `json:"used,omitempty"`
	ResetAt   time.Time `json:"reset_at,omitempty"`
}

// RateLimit issues the probe. apiBase and httpClient mirror GitHubConfig
// (empty apiBase defaults to https://api.github.com); token empty means an
// anonymous request. Unlike a repo_read call, this never rotates through
// other credentials or falls back to anything -- it reports exactly the
// bucket the given token (or no token) belongs to, so a caller comparing
// several of these side by side sees each credential's real, independent
// standing.
func RateLimit(ctx context.Context, apiBase, token string, httpClient *http.Client) RateLimitStatus {
	e := NewGitHub(GitHubConfig{HTTPClient: httpClient, APIBaseURL: apiBase})
	res, err := e.doGet(ctx, e.base+"/rate_limit", token, "application/vnd.github+json")
	if err != nil {
		return RateLimitStatus{Error: "could not reach GitHub: " + err.Error()}
	}
	if res.status < 200 || res.status >= 300 {
		where := ""
		if res.target != "" {
			where = " from " + res.target
		}
		if msg := GitHubErrorMessage(res.body); msg != "" {
			return RateLimitStatus{Error: fmt.Sprintf("GitHub returned %d%s: %s", res.status, where, msg)}
		}
		return RateLimitStatus{Error: fmt.Sprintf("GitHub returned status %d%s", res.status, where)}
	}
	var body struct {
		Resources struct {
			Core struct {
				Limit     int   `json:"limit"`
				Remaining int   `json:"remaining"`
				Used      int   `json:"used"`
				Reset     int64 `json:"reset"`
			} `json:"core"`
		} `json:"resources"`
	}
	if err := json.Unmarshal(res.body, &body); err != nil {
		return RateLimitStatus{Error: "could not parse GitHub's rate limit response: " + err.Error()}
	}
	c := body.Resources.Core
	return RateLimitStatus{
		OK: true, Limit: c.Limit, Remaining: c.Remaining, Used: c.Used,
		ResetAt: time.Unix(c.Reset, 0),
	}
}
