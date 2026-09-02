package commonai

import (
	"context"
	"errors"
	"fmt"
	"regexp"
)

// APIError is a non-2xx provider response: Body holds up to KiB, ContextOverflow flags a permanent.
type APIError struct {
	Status          int
	Body            string
	ContextOverflow bool
}

// Error returns the status and the (bounded) response body.
func (e *APIError) Error() string {
	return fmt.Sprintf("api error: status %d: %s", e.Status, e.Body)
}

// contextOverflowRe matches provider wordings for a prompt that exceeded the model's context window.
var contextOverflowRe = regexp.MustCompile(`(?i)prompt (is )?too long|context (length|window)|maximum context|too many tokens|exceeds?.{0,20}(context|token)`)

// IsContextOverflow reports whether err is (or wraps) an APIError flagged as a
// context-window overflow.
func IsContextOverflow(err error) bool {
	var ae *APIError
	return errors.As(err, &ae) && ae.ContextOverflow
}

// IsTransient reports if err is retryable: //5xx or network errors; never cancellation, other 4xx, or callbacks
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

// requestError marks a request the library refused to build or send: deterministic misconfiguration, never transient.
type requestError struct{ msg string }

// Error returns the misconfiguration message.
func (e *requestError) Error() string { return e.msg }

// badRequestErr wraps a deterministic request-construction failure so IsTransient classifies it as permanent.
func badRequestErr(msg string) error { return &requestError{msg: msg} }

// BadRequest builds a permanent failure for a refused request, so upper layers can classify their own refusals.
func BadRequest(msg string) error { return badRequestErr(msg) }

// CallbackError marks an error from the caller's own callback, keeping a failed sink permanent; it is transparent.
func CallbackError(err error) error { return wrapCallbackErr(err) }

// callbackError marks an error from the caller's own callbacks, not the upstream; transparent and never retried.
type callbackError struct{ err error }

// Error returns the wrapped callback error's text unchanged.
func (e *callbackError) Error() string { return e.err.Error() }

// Unwrap exposes the caller's original error to errors.Is / errors.As.
func (e *callbackError) Unwrap() error { return e.err }

// wrapCallbackErr marks err as callback-originated, idempotently: nil stays nil and marked errors pass through.
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
