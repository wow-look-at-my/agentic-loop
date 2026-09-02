package commonai

import (
	"context"
	"errors"
	"github.com/wow-look-at-my/go-containers/set"
	"regexp"
	"strings"
	"sync"
)

// This file implements provider-agnostic recovery from a "rejected parameter"
//. Extra params are forwarded verbatim (faithful passthrough), but some
// upstreams reject a parameter another upstream accepts (e.g. xAI rejects
// reasoning_effort). When the upstream rejects a parameter at request time —
// a returned before any token streams — the middleware parses the
// offending parameter name out of the error text (which embeds the HTTP
// body via APIError), strips that key from the request's Extra, and
// retries the same request. Params are thus still sent by default; only
// is dropped, and only after the upstream said no.

// rejectParamPatterns matches common OpenAI-compatible phrasings for a rejected/unsupported parameter, capturing the parameter name.
var rejectParamPatterns = []*regexp.Regexp{
	// xAI: Model grok-build- does not support parameter reasoningEffort.
	regexp.MustCompile(`(?i)does not support parameter\s+["'` + "`" + `]?([^"'` + "`" + `\s,.;:)]+)`),
	// OpenAI: Unsupported parameter: 'reasoning_effort'... / unsupported parameter reasoning_effort
	regexp.MustCompile(`(?i)unsupported parameter:?\s+["'` + "`" + `]?([^"'` + "`" + `\s,.;:)]+)`),
	// OpenAI/strict JSON: unknown field "reasoning_effort"
	regexp.MustCompile(`(?i)unknown field\s+["'` + "`" + `]?([^"'` + "`" + `\s,.;:)]+)`),
	// OpenAI: unrecognized request argument: reasoning_effort
	regexp.MustCompile(`(?i)unrecognized request argument:?\s+["'` + "`" + `]?([^"'` + "`" + `\s,.;:)]+)`),
}

// rejectedParamName extracts a rejected/unsupported parameter name from an error text; unparseable means no retry.
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

// trimParamName strips whitespace, surrounding quotes, and trailing punctuation an error may wrap a parameter name in.
func trimParamName(s string) string {
	return strings.Trim(s, " \t\"'`.,:;)")
}

// normalizeParamName lowercases and removes underscores so camelCase and snake_case forms match.
func normalizeParamName(name string) string {
	return strings.ReplaceAll(strings.ToLower(name), "_", "")
}

// paramStripper is the stateful strip-and-retry middleware.
type paramStripper struct {
	inner Provider

	mu       sync.Mutex
	stripped set.Set[string] // normalized names of params already stripped
}

// Strips a named param from Extra and retries; never on cancel or streamed calls.
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

// matchRejectedKey returns the key in extra matching the rejected name, or ("", false) when there is nothing to strip.
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
