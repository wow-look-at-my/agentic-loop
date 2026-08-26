package loop

import (
	"sync"

	"github.com/wow-look-at-my/go-containers/concurrentqueue"
)

// MessageQueue is a thread-safe FIFO delivering messages into a running
// loop, backed by one concurrentqueue.Queue holding both automated notices
// and the user's own messages -- a queued value's own type (SystemMessage or
// UserMessage) says which kind it is, so nothing here needs two separate
// queue instances wired through by hand. Draining stable-partitions system
// messages ahead of user messages queued in the same window.
//
// The zero value is an empty, open queue ready to use.
type MessageQueue struct {
	items  concurrentqueue.Queue[QueuedMessage]
	mu     sync.Mutex // guards closed; Queue and Close must agree on it atomically
	closed bool
}

// Queue adds msg and reports whether the queue took it; false means closed.
func (q *MessageQueue) Queue(msg QueuedMessage) bool {
	if q == nil {
		return false
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return false
	}
	q.items.Enqueue(msg)
	return true
}

// Drain returns and clears every pending message, system first. A closed
// queue still drains what it holds; it only stops accepting new messages.
func (q *MessageQueue) Drain() []Message {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.drainLocked()
}

// Close marks the queue closed and returns whatever was queued but never
// drained, system first.
func (q *MessageQueue) Close() []Message {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	return q.drainLocked()
}

// drainLocked takes every value out of the underlying queue and stable-
// partitions it: every SystemMessage ahead of every UserMessage, each
// keeping its own arrival order. Must be called with q.mu held, which is
// what stops a racing Queue call from adding a message after the take-loop
// below has already decided the queue looks empty.
func (q *MessageQueue) drainLocked() []Message {
	if q.items.IsEmpty() {
		return nil
	}
	system := make([]Message, 0, q.items.Len())
	var user []Message
	for {
		msg, ok := q.items.TryDequeue()
		if !ok {
			break
		}
		if msg.isSystemMessage() {
			system = append(system, msg.queuedMessage())
		} else {
			user = append(user, msg.queuedMessage())
		}
	}
	return append(system, user...)
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

// Pending reports whether the queue has any message waiting, of either kind.
func (q *MessageQueue) Pending() bool {
	if q == nil {
		return false
	}
	return !q.items.IsEmpty()
}

// Len returns the number of pending messages, of either kind.
func (q *MessageQueue) Len() int {
	if q == nil {
		return 0
	}
	return q.items.Len()
}
