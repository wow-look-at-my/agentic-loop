package commonai

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
// futile) — nor an error returned by one of the caller's own stream callbacks
// (the upstream call did not fail; the caller's sink did, and re-sending the
// prompt cannot fix that).
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
	var ce *callbackError
	if errors.As(err, &ce) {
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

// BadRequest builds a permanent failure for a request that was refused before
// it was sent. A layer above this one -- a loop, a transport, a CLI -- needs
// the same classification for its own refusals: without it, a caller's own
// misconfiguration reads as an unclassified error, and a retry wrapper would
// dutifully re-send something that cannot ever work.
func BadRequest(msg string) error { return badRequestErr(msg) }

// CallbackError marks an error as originating in one of the caller's own
// callbacks rather than in the upstream, which is what keeps a failed sink
// permanent instead of retried. It is transparent: the message and
// errors.Is/errors.As behavior are the wrapped error's own.
func CallbackError(err error) error { return wrapCallbackErr(err) }

// callbackError marks an error that originated in one of the caller's own
// callbacks (StreamEvents.On*, Events.OnToolCall/OnToolResult) rather than in
// the upstream call. It is transparent — Error() is the callback error's own
// text and Unwrap preserves errors.Is/errors.As against the caller's sentinel
// — and exists only so IsTransient classifies a failed sink as permanent: it
// is never an *APIError and never retried.
type callbackError struct{ err error }

// Error returns the wrapped callback error's text unchanged.
func (e *callbackError) Error() string { return e.err.Error() }

// Unwrap exposes the caller's original error to errors.Is / errors.As.
func (e *callbackError) Unwrap() error { return e.err }

// wrapCallbackErr marks err as callback-originated, idempotently: nil stays
// nil, and an error already carrying the marker is returned unchanged (the
// emit helpers can nest).
func wrapCallbackErr(err error) error {
	if err == nil {
		return nil
	}
	var ce *callbackError
	if errors.As(err, &ce) {
		return err
	}
	return &callbackError{err: err}
}
