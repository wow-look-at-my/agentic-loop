package repo

import (
	"net/http"
	"strconv"
	"strings"
	"time"
)

// RateLimitStatus is a credential's core-quota standing. Every GitHub response
// carries it in headers, so this is READ OFF a call that was going to happen
// anyway. There is deliberately no probe: GET /rate_limit is a second request
// asking what the first one already answered.
type RateLimitStatus struct {
	OK        bool      `json:"ok"`
	Error     string    `json:"error,omitempty"`
	Limit     int       `json:"limit,omitempty"`
	Remaining int       `json:"remaining,omitempty"`
	Used      int       `json:"used,omitempty"`
	ResetAt   time.Time `json:"reset_at,omitempty"`
	// ObservedAt is when the response carrying these numbers came back. A
	// reader needs it: unlike a probe's answer, this is as old as the last
	// call made with that credential. see USAGE.md
	ObservedAt time.Time `json:"observed_at,omitempty"`
}

// RateLimitObservation is one credential's standing as one response reported
// it. Anonymous marks the unauthenticated bucket, whose CredentialID is empty.
type RateLimitObservation struct {
	CredentialID   string
	CredentialName string
	Anonymous      bool
	Status         RateLimitStatus
}

// ReadRateLimit pulls the core-quota standing out of a response's headers,
// reporting whether they were there at all. Only the CORE resource counts: a
// code-search call spends a different, much smaller budget (GitHub names it in
// x-ratelimit-resource), and folding the two together reports a quota the
// reader does not have. An absent resource header reads as core, since only
// the search and graphql routes set it to anything else.
func ReadRateLimit(h http.Header, now time.Time) (RateLimitStatus, bool) {
	if res := strings.TrimSpace(h.Get("X-RateLimit-Resource")); res != "" && res != "core" {
		return RateLimitStatus{}, false
	}
	limit, hasLimit := headerInt(h, "X-RateLimit-Limit")
	remaining, hasRemaining := headerInt(h, "X-RateLimit-Remaining")
	if !hasLimit || !hasRemaining {
		return RateLimitStatus{}, false
	}
	s := RateLimitStatus{OK: true, Limit: limit, Remaining: remaining, ObservedAt: now}
	if used, ok := headerInt(h, "X-RateLimit-Used"); ok {
		s.Used = used
	} else {
		s.Used = limit - remaining
	}
	if secs, ok := headerInt64(h, "X-RateLimit-Reset"); ok {
		s.ResetAt = time.Unix(secs, 0)
	}
	return s, true
}

func headerInt(h http.Header, name string) (int, bool) {
	v, err := strconv.Atoi(strings.TrimSpace(h.Get(name)))
	return v, err == nil
}

func headerInt64(h http.Header, name string) (int64, bool) {
	v, err := strconv.ParseInt(strings.TrimSpace(h.Get(name)), 10, 64)
	return v, err == nil
}

// observeRateLimit reports one response's rate-limit headers to the host, if
// it asked to hear about them. token names which bucket was spent.
func (e *GitHub) observeRateLimit(token string, h http.Header) {
	if e.onRateLimit == nil {
		return
	}
	s, ok := ReadRateLimit(h, time.Now())
	if !ok {
		return
	}
	obs := RateLimitObservation{Anonymous: token == "", Status: s}
	if !obs.Anonymous {
		t, found := e.tokenByValue(token)
		if !found {
			// A token this client was not configured with spends a bucket
			// nothing can name, and an empty id would read as the anonymous one.
			return
		}
		obs.CredentialID, obs.CredentialName = t.ID, t.Name
	}
	e.onRateLimit(obs)
}

// tokenByValue finds the configured credential a raw token belongs to, across
// both the read and write lists.
func (e *GitHub) tokenByValue(token string) (GitHubToken, bool) {
	for _, list := range [][]GitHubToken{e.tokens, e.writeTokens} {
		for _, t := range list {
			if t.Token == token {
				return t, true
			}
		}
	}
	return GitHubToken{}, false
}
