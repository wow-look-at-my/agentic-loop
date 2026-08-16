package commonai

import (
	"context"
	"errors"
)

// Error kinds, as the format names them. A reader that has to decide what to do
// next -- retry, re-prompt, give up, ask a person -- needs to know which of
// these happened, and grepping the message text for it is not a contract.
const (
	// ErrorKindAPI is a non-2xx answer from the upstream.
	ErrorKindAPI = "api"
	// ErrorKindOverflow is a prompt that exceeded the model's context window.
	// It is a permanent, never-retried condition, and it is the one failure a
	// caller can usually fix without a person.
	ErrorKindOverflow = "context-overflow"
	// ErrorKindRequest is a request this library refused to build or send:
	// deterministic misconfiguration, never transient.
	ErrorKindRequest = "request"
	// ErrorKindCanceled is the caller's own context ending the call.
	ErrorKindCanceled = "canceled"
	// ErrorKindUnsupported is a document the target dialect cannot express.
	ErrorKindUnsupported = "unsupported"
	// ErrorKindTransport is everything else: a connection that failed, a
	// stream that broke, a body that would not read.
	ErrorKindTransport = "transport"
)

// UnsupportedError is a request that cannot be expressed in the target
// dialect: an image by URL where only inline bytes work, a reasoning block
// with no replayable payload, a parameter with nowhere to go. It is an error
// rather than a silent omission because a request quietly stripped of a thing
// the caller asked for is a wrong answer that looks like a right one.
type UnsupportedError struct {
	// Dialect is the target that cannot express it.
	Dialect Dialect
	// What names the thing, in the format's own vocabulary.
	What string
	// Why explains what the dialect does instead, when there is something the
	// caller could do about it.
	Why string
}

// Error implements error.
func (e *UnsupportedError) Error() string {
	msg := string(e.Dialect) + " cannot express " + e.What
	if e.Why != "" {
		msg += ": " + e.Why
	}
	return "commonai: " + msg
}

// Unsupported builds an UnsupportedError.
func Unsupported(d Dialect, what, why string) error {
	return &UnsupportedError{Dialect: d, What: what, Why: why}
}

// IsUnsupported reports whether err is (or wraps) an UnsupportedError.
func IsUnsupported(err error) bool {
	var ue *UnsupportedError
	return errors.As(err, &ue)
}

// ErrorKind classifies a failure the way the format names it, so a caller can
// branch on what happened without matching message text.
func ErrorKind(err error) string { return errorKind(err) }

// IsBadRequest reports whether err is one the library refused to send, rather
// than one an upstream produced. A transport answering an HTTP status needs
// exactly this distinction: the caller's document was wrong, or it was not.
func IsBadRequest(err error) bool {
	var re *requestError
	return errors.As(err, &re)
}

// errorKind classifies an error for the wire.
func errorKind(err error) string {
	switch {
	case err == nil:
		return ""
	case IsContextOverflow(err):
		return ErrorKindOverflow
	case IsUnsupported(err):
		return ErrorKindUnsupported
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return ErrorKindCanceled
	}
	var ae *APIError
	if errors.As(err, &ae) {
		return ErrorKindAPI
	}
	var re *requestError
	if errors.As(err, &re) {
		return ErrorKindRequest
	}
	return ErrorKindTransport
}

// asAPIError reports whether err is (or wraps) an *APIError, filling target.
func asAPIError(err error, target **APIError) bool {
	return errors.As(err, target)
}
