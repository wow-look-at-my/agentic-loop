package agentic

import "context"

// Inbox is the queue a Run drains so a message the model must answer is never
// dropped. It is how a host injects a notice the model should react to -- a CI
// transition, a supervisor ping, a capability toggle -- without inventing a
// per-host queue and without a user message being silently lost mid-run.
//
// The contract is at-least-once, and it is enforced at the loop's finish
// points, not just at turn starts:
//
//   - Run drains the inbox at the top of every turn, so messages that landed
//     while the model was working are answered by the next model call.
//   - Run drains AGAIN at the point where it would otherwise end (the model
//     asked for no tools). Only an EMPTY drain lets the loop finish. A message
//     that arrives during the final turn is therefore caught: there is one
//     more drain check before the loop declares done, and one more turn
//     answers it.
//   - Receive returning (msg, false) means "empty" -- the only way the loop
//     learns there is nothing left to answer.
//
// The host owns buffering, blocking policy, and persistence across runs. A
// host that must never lose a message backs the inbox with durable storage
// and dedups by message id on re-read, so a cancelled run can only duplicate,
// never drop.
//
// Injected messages are appended as RoleUser turns. That is the role every
// upstream accepts mid-transcript, and it is what the sub-agent-delivery path
// already uses for the same reason.
type Inbox interface {
	Receive(ctx context.Context) (Message, bool)
}

// drainInbox appends every message currently in the inbox to the transcript
// and reports whether any were added. It stops on the first empty response or
// a context cancellation, whichever comes first. The host's Receive must
// return promptly -- false means "empty" -- since the loop calls it at every
// turn start and at the finish point.
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
