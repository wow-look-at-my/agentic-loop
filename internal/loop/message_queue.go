package loop

import "sync"

// MessageQueue is a thread-safe FIFO of messages to inject into the loop's
// transcript. The loop holds two instances: SystemMessages (automated notices,
// stop-hook nudges) and UserMessages (user-injected mid-run messages). System
// messages are always drained before user messages so an automated nudge
// precedes anything the user queued.
//
// The zero value is ready to use.
type MessageQueue struct {
	mu       sync.Mutex
	messages []Message
}

// Queue adds a message to the end of the queue.
func (q *MessageQueue) Queue(msg Message) {
	if q == nil {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.messages = append(q.messages, msg)
}

// Drain returns and clears all pending messages.
func (q *MessageQueue) Drain() []Message {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	msgs := q.messages
	q.messages = nil
	return msgs
}

// Len returns the number of pending messages.
func (q *MessageQueue) Len() int {
	if q == nil {
		return 0
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.messages)
}

// DrainBoth drains system messages before user messages, returning a single
// ordered slice. System messages always come first.
func DrainBoth(system, user *MessageQueue) []Message {
	var out []Message
	out = append(out, system.Drain()...)
	out = append(out, user.Drain()...)
	return out
}

// Pending reports whether either queue has messages waiting.
func Pending(queues ...*MessageQueue) bool {
	for _, q := range queues {
		if q.Len() > 0 {
			return true
		}
	}
	return false
}
