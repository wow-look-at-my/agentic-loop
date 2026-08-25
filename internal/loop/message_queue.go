package loop

import "sync"

// MessageQueue is a thread-safe FIFO delivering messages into a running loop.
type MessageQueue struct {
	mu       sync.Mutex
	messages []Message
	closed   bool
}

// Queue adds a message and reports whether the queue took it; false means closed.
func (q *MessageQueue) Queue(msg Message) bool {
	if q == nil {
		return false
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return false
	}
	q.messages = append(q.messages, msg)
	return true
}

// Drain returns and clears all pending messages. A closed queue still drains
// what it holds; it only stops accepting new messages.
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

// Close marks the queue closed and returns whatever was queued but never drained.
func (q *MessageQueue) Close() []Message {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	msgs := q.messages
	q.messages = nil
	return msgs
}

// Closed reports whether the queue has stopped accepting messages.
func (q *MessageQueue) Closed() bool {
	if q == nil {
		return true
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.closed
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

// closeQueues closes both queues and returns undelivered messages, system first.
func closeQueues(system, user *MessageQueue) []Message {
	var out []Message
	out = append(out, system.Close()...)
	out = append(out, user.Close()...)
	return out
}
