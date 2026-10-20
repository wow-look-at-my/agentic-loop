package agentic

import "context"

// Inbox is the queue a Run drains so a message the model must answer is never
// dropped.
type Inbox interface {
	Receive(ctx context.Context) (Message, bool)
}

// drainInbox appends every message in the inbox to the transcript and reports whether any were added. It stops on the first empty response or a context cancellation, whichever comes first. The host's Receive must return promptly -- false means "empty" -- since the loop
// calls it at every turn start and at the finish point.
func drainInbox(ctx context.Context, inbox Inbox, transcript []Message) ([]Message, bool) {
	if inbox == nil {
		return transcript, false
	}
	added := false
	for {
		if ctx.Err() != nil {
			return transcript, added
		}
		msg, ok := inbox.Receive(ctx)
		if !ok {
			return transcript, added
		}
		transcript = append(transcript, msg)
		added = true
	}
}
