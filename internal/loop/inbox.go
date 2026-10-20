package loop

import "context"

// Inbox is a queue a Run drains so a message the model must answer is never
// dropped.
type Inbox interface {
	Receive(ctx context.Context) (Message, bool)
}

// drainInbox takes every message in the inbox and returns them. It stops on
// the first empty response or a context cancellation, whichever comes first.
func drainInbox(ctx context.Context, inbox Inbox) []Message {
	if inbox == nil {
		return nil
	}
	var out []Message
	for {
		if ctx.Err() != nil {
			return out
		}
		msg, ok := inbox.Receive(ctx)
		if !ok {
			return out
		}
		out = append(out, msg)
	}
}
