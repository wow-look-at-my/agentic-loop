package loop

// QueuedMessage is a message queued for delivery into a running loop. Its
// two implementations, SystemMessage and UserMessage, are what tell
// MessageQueue which kind it is holding -- so one queue carries both kinds
// instead of a caller having to wire two separate MessageQueue instances
// through by hand and keep them both in sync.
type QueuedMessage interface {
	// queuedMessage returns the wrapped Message. Unexported: QueuedMessage
	// has exactly two implementations, both in this package.
	queuedMessage() Message
	// isSystemMessage reports whether this is a SystemMessage, which is what
	// lets Drain put every system message ahead of every user message queued
	// in the same window.
	isSystemMessage() bool
}

// SystemMessage is an automated notice -- a CI transition, a workspace
// toggle, a stop-hook nudge, a sub-agent report -- queued for delivery.
type SystemMessage struct{ Message }

func (m SystemMessage) queuedMessage() Message { return m.Message }
func (m SystemMessage) isSystemMessage() bool  { return true }

// UserMessage is a message from the user, queued for delivery mid-turn.
type UserMessage struct{ Message }

func (m UserMessage) queuedMessage() Message { return m.Message }
func (m UserMessage) isSystemMessage() bool  { return false }
