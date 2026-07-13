package agentic

import (
	"context"
	"errors"
	"fmt"
	"regexp"
)

// APIError is a non-2xx response from a provider. Body holds up to 4 KiB of
// the response body (or the HTTP status text when the body was empty); it is
// embedded in Error() so downstream matchers — the param-strip middleware's
// regexes in particular — can see the provider's wording. ContextOverflow
// flags an HTTP 400 whose body says the prompt exceeded the model's context
// window: a permanent, never-retried condition callers should surface
// explicitly.
type APIError struct {
	Status          int
	Body            string
	ContextOverflow bool
}

// Error returns the status and the (bounded) response body.
func (e *APIError) Error() string {
	return fmt.Sprintf("api error: status %d: %s", e.Status, e.Body)
}

// contextOverflowRe matches provider wordings for a prompt that exceeded the
// model's context window ("prompt is too long", "context length", "maximum
// context", "too many tokens", ...).
var contextOverflowRe = regexp.MustCompile(`(?i)prompt (is )?too long|context (length|window)|maximum context|too many tokens|exceeds?.{0,20}(context|token)`)

// IsContextOverflow reports whether err is (or wraps) an APIError flagged as a
// context-window overflow.
func IsContextOverflow(err error) bool {
	var ae *APIError
	return errors.As(err, &ae) && ae.ContextOverflow
}

// IsTransient reports whether err is worth retrying: an APIError with status
// 408, 429, or any 5xx, or any network/transport error. Context cancellation
// and deadline expiry are never transient, and neither is any other 4xx —
// including a context-overflow 400 (retrying the same oversized prompt is
// futile).
func IsTransient(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var ae *APIError
	if errors.As(err, &ae) {
		return ae.Status == 408 || ae.Status == 429 || ae.Status >= 500
	}
	var re *requestError
	if errors.As(err, &re) {
		return false
	}
	return true
}

// requestError marks a request the library itself refused to build or send —
// deterministic misconfiguration (missing required fields, marshal failures),
// never transient.
type requestError struct{ msg string }

// Error returns the misconfiguration message.
func (e *requestError) Error() string { return e.msg }

// badRequestErr wraps a deterministic request-construction failure so
// IsTransient classifies it as permanent.
func badRequestErr(msg string) error { return &requestError{msg: msg} }
