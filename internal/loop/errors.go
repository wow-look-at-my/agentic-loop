package loop

import (
	"github.com/wow-look-at-my/agentic-loop/client"
)

// badRequestErr wraps a request-construction failure so IsTransient classifies it permanent.
func badRequestErr(msg string) error { return client.BadRequest(msg) }

// wrapCallbackErr marks err as from a caller callback, not a model call.
func wrapCallbackErr(err error) error { return client.CallbackError(err) }
