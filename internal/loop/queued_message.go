package loop

// QueuedMessage is a message queued for delivery into a running loop. Its
// implementations, SystemMessage and UserMessage, tell MessageQueue which kind
// it holds, so queue carries both and no caller can wire up half of it.
type QueuedMessage interface {
	// queuedMessage returns the wrapped Message; unexported keeps the set closed.
	queuedMessage() Message
	// isSystemMessage is how Drain puts system messages ahead of user ones.
	isSystemMessage() bool
}

// SystemMessage is an automated notice queued for delivery: a CI transition,
type SystemMessage struct{ Message }

func (m SystemMessage) queuedMessage() Message { return m.Message }
func (m SystemMessage) isSystemMessage() bool  { return true }

// UserMessage is a message from the user, queued for delivery mid-turn.
type UserMessage struct{ Message }

func (m UserMessage) queuedMessage() Message { return m.Message }
func (m UserMessage) isSystemMessage() bool  { return false }
