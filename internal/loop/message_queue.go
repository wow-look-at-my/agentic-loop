package loop

import "sync"

// MessageQueue is a thread-safe FIFO of messages to deliver INTO a running
// loop. The loop holds two: Config.SystemMessages (automated notices -- a CI
// status change, a stop-hook nudge, a sub-agent report) and
// Config.UserMessages (what the user sent while the model was working).
// System messages are always delivered before user messages, so an automated
// notice precedes anything the user queued in the same gap.
//
// Run drains both at the top of every turn, and a queued message starts
// another turn when the model would otherwise finish. That is the contract: a
// message this queue ACCEPTS reaches the model.
//
// A queue belongs to ONE run. Run closes both queues as it returns, so a
// producer that races the end of a run is told the truth: Queue reports false,
// and the host delivers the message another way -- by starting a new run --
// instead of leaving it where nothing will read it. Whatever was queued and
// never delivered comes back in Result.Undelivered.
//
// The zero value is ready to use.
type MessageQueue struct {
	mu       sync.Mutex
	messages []Message
	closed   bool
}

// Queue adds a message to the end of the queue and reports whether the queue
// took it. False means the queue is closed: the run that would have shown this
// message to the model has ended, and no other run will. A host that gets
// false must deliver the message itself, by starting a new run. A nil queue
// takes nothing, for the same reason -- there is no run to deliver it.
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

// Close marks the queue closed and returns whatever was queued but never
// drained. Every later Queue reports false. Closing twice is safe and returns
// nothing the second time.
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

// Closed reports whether the queue has stopped accepting messages. The answer
// is a snapshot; Queue's own return value is what settles whether a message
// landed.
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

// closeQueues closes both of a run's queues and returns everything neither one
// delivered, system first. Run calls it as it returns: a message left here was
// produced by somebody who believes the model saw it, so it goes back to the
// host in Result.Undelivered instead of disappearing with the run.
func closeQueues(system, user *MessageQueue) []Message {
	var out []Message
	out = append(out, system.Close()...)
	out = append(out, user.Close()...)
	return out
}
