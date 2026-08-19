package agentic

import (
	"context"
	"errors"
	"github.com/wow-look-at-my/go-containers/set"
	"regexp"
	"strings"
	"sync"
)

// This file implements provider-agnostic recovery from a "rejected parameter"
// 400. Extra params are forwarded verbatim (faithful passthrough), but some
// upstreams reject a parameter another upstream accepts (e.g. xAI rejects
// reasoning_effort). When the upstream rejects a parameter at request time —
// a 400 returned before any token streams — the middleware parses the
// offending parameter name out of the error text (which embeds the HTTP 400
// body via APIError), strips that one key from the request's Extra, and
// retries the same request once. Params are thus still sent by default; only
// one is dropped, and only after the upstream said no.

// rejectParamPatterns matches the common OpenAI-compatible phrasings for a
// rejected/unsupported request parameter, each capturing the parameter name.
// The name capture is permissive (it stops at whitespace, a closing quote, or
// common trailing punctuation) because the surrounding quoting differs per
// provider; the captured token is cleaned by trimParamName afterwards.
var rejectParamPatterns = []*regexp.Regexp{
	// xAI: Model grok-build-0.1 does not support parameter reasoningEffort.
	regexp.MustCompile(`(?i)does not support parameter\s+["'` + "`" + `]?([^"'` + "`" + `\s,.;:)]+)`),
	// OpenAI: Unsupported parameter: 'reasoning_effort' ... / unsupported parameter reasoning_effort
	regexp.MustCompile(`(?i)unsupported parameter:?\s+["'` + "`" + `]?([^"'` + "`" + `\s,.;:)]+)`),
	// OpenAI/strict JSON: unknown field "reasoning_effort"
	regexp.MustCompile(`(?i)unknown field\s+["'` + "`" + `]?([^"'` + "`" + `\s,.;:)]+)`),
	// OpenAI: unrecognized request argument: reasoning_effort
	regexp.MustCompile(`(?i)unrecognized request argument:?\s+["'` + "`" + `]?([^"'` + "`" + `\s,.;:)]+)`),
}

// rejectedParamName extracts the name of a rejected/unsupported request
// parameter from an upstream error text, returning ("", false) when no known
// phrasing matches. It is intentionally targeted: an unparseable error means
// no retry (there is no name to strip), so the original error surfaces
// unchanged.
func rejectedParamName(errBody string) (string, bool) {
	for _, re := range rejectParamPatterns {
		if m := re.FindStringSubmatch(errBody); m != nil {
			if name := trimParamName(m[1]); name != "" {
				return name, true
			}
		}
	}
	return "", false
}

// trimParamName strips the whitespace, surrounding quotes, and trailing
// sentence punctuation an error message may wrap a parameter name in
// (e.g. `"reasoningEffort".` -> reasoningEffort).
func trimParamName(s string) string {
	return strings.Trim(s, " \t\"'`.,:;)")
}

// normalizeParamName lowercases a name and removes underscores so an
// upstream's camelCased report (reasoningEffort) matches a snake_case key
// (reasoning_effort). Both forms normalize to "reasoningeffort".
func normalizeParamName(name string) string {
	return strings.ReplaceAll(strings.ToLower(name), "_", "")
}

// paramStripper is the stateful strip-and-retry middleware.
type paramStripper struct {
	inner Provider

	mu       sync.Mutex
	stripped set.Set[string] // normalized names of params already stripped
}

// NewParamStripper wraps a Provider with rejected-parameter recovery: when a
// call fails before anything streamed and the error text names a parameter
// present in the request's Extra (matched by normalized form, so
// reasoningEffort matches reasoning_effort), that key is removed and the call
// is retried ONCE. The strip is remembered, so subsequent calls through the
// same stripper drop the key up front — mirroring the persistent in-place
// strip of the source loop, without mutating the caller's Extra map. A
// context cancellation is never treated as a parameter problem, and a call
// that already streamed (a non-nil completion) is never retried.
func NewParamStripper(p Provider) Provider {
	return &paramStripper{inner: p, stripped: set.New[string]()}
}

// Complete implements Provider.
func (s *paramStripper) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	req.Extra = s.withoutStripped(req.Extra)
	comp, err := s.inner.Complete(ctx, req, ev)
	// A non-nil completion means the call already streamed (see Provider), so
	// it is too late to strip anything and re-send.
	if err == nil || errors.Is(err, context.Canceled) || comp != nil {
		return comp, err
	}
	key, ok := matchRejectedKey(req.Extra, err.Error())
	if !ok {
		return comp, err
	}
	s.mu.Lock()
	s.stripped.Add(normalizeParamName(key))
	s.mu.Unlock()
	next := make(map[string]any, len(req.Extra))
	for k, v := range req.Extra {
		if k == key {
			continue
		}
		next[k] = v
	}
	req.Extra = next
	return s.inner.Complete(ctx, req, ev)
}

// withoutStripped returns extra with every remembered key removed, copying
// only when a removal is needed (the caller's map is never mutated).
func (s *paramStripper) withoutStripped(extra map[string]any) map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stripped.Len() == 0 || len(extra) == 0 {
		return extra
	}
	needs := false
	for k := range extra {
		if s.stripped.Contains(normalizeParamName(k)) {
			needs = true
			break
		}
	}
	if !needs {
		return extra
	}
	out := make(map[string]any, len(extra))
	for k, v := range extra {
		if s.stripped.Contains(normalizeParamName(k)) {
			continue
		}
		out[k] = v
	}
	return out
}

// matchRejectedKey parses the rejected parameter name out of errStr and, if
// it matches (by normalized form) a key actually present in extra, returns
// that key. It returns ("", false) when the error names no parameter or the
// named parameter is not in extra — in both cases the caller must NOT retry
// (there is nothing to change, so retrying would loop or be pointless).
func matchRejectedKey(extra map[string]any, errStr string) (string, bool) {
	name, ok := rejectedParamName(errStr)
	if !ok {
		return "", false
	}
	target := normalizeParamName(name)
	for k := range extra {
		if normalizeParamName(k) == target {
			return k, true
		}
	}
	return "", false
}
