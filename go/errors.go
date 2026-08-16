package agentic

import (
	"github.com/wow-look-at-my/agentic-loop/go/client"
)

// badRequestErr wraps a deterministic request-construction failure so
// IsTransient classifies it as permanent. It produces the format's own error
// type rather than a local one, because IsTransient reads the concrete type: a
// marker it does not recognize would be classified transient and retried.
func badRequestErr(msg string) error { return client.BadRequest(msg) }

// wrapCallbackErr marks err as originating in one of the caller's own
// callbacks (Events.OnToolCall/OnToolResult) rather than in a model call. It
// is transparent -- Error() is the callback error's own text and errors.Is /
// errors.As still reach the caller's sentinel -- and exists only so a failed
// sink is never mistaken for a failed upstream. nil stays nil, and an error
// already carrying the marker is returned unchanged.
func wrapCallbackErr(err error) error { return client.CallbackError(err) }
