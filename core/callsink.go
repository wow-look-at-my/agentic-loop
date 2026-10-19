package commonai

import (
	"fmt"
	"sync"
)

// CallSink receives one model call as it streams. A decoder makes each node when its part begins,
// writes its stream fields through Delta, fills the node's opaque fields, then calls Done.
type CallSink interface {
	// Start adds a node the call produced, in the order the parts began.
	Start(node IMessage)
	// Delta appends to a stream field of a started node.
	Delta(node IMessage, field FieldID, delta string)
	// Done ends a node. Stream fields still open finish on what streamed.
	Done(node IMessage)
	Usage(u Usage)
	Progress(p PromptProgress)
	Timings(t Timings)
	// Retry says the call is re-sent.
	Retry(a RetryAttempt)
	// Err is non-nil once a listener stopped the call. A decoder checks it after each event.
	Err() error
}

// FieldOf is the stream field a FieldID names on node, or nil.
func FieldOf(node IMessage, f FieldID) IStreamString {
	switch f {
	case FieldText:
		switch n := node.(type) {
		case *AssistantMessage:
			return n.Text
		case *ThinkingMessage:
			return n.Text
		case *CompactionMessage:
			return n.Summary
		}
		return nil
	case FieldArguments:
		if n, ok := node.(*ToolCallMessage); ok {
			return n.Arguments
		}
		return nil
	default:
		panic(fmt.Sprintf("commonai: FieldOf %v", f))
	}
}

// StreamFields are every stream field of node.
func StreamFields(node IMessage) []IStreamString {
	switch n := node.(type) {
	case *AssistantMessage:
		return []IStreamString{n.Text}
	case *ThinkingMessage:
		return []IStreamString{n.Text}
	case *CompactionMessage:
		return []IStreamString{n.Summary}
	case *ToolCallMessage:
		return []IStreamString{n.Arguments}
	}
	return nil
}

// FinishNode ends every open stream field of node on what streamed.
func FinishNode(node IMessage) {
	for _, f := range StreamFields(node) {
		if !f.Done() {
			FinishStreamed(f)
		}
	}
}

// Collector is a CallSink that keeps one call in memory, for callers that want the whole result.
type Collector struct {
	mu       sync.Mutex
	Nodes    []IMessage
	Usages   []Usage
	TimingsL []Timings
	Retries  []RetryAttempt
	// Events, when set, receives the call as host stream events.
	Events *StreamEvents
	err    error
}

func (c *Collector) Start(node IMessage) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.Nodes = append(c.Nodes, node)
}

func (c *Collector) Delta(node IMessage, f FieldID, delta string) {
	if s := FieldOf(node, f); s != nil {
		WriterOf(s).Append(delta)
	}
	var err error
	switch node.(type) {
	case *AssistantMessage:
		err = c.Events.EmitText(delta)
	case *ThinkingMessage:
		err = c.Events.EmitReasoning(delta)
	}
	c.keep(err)
}

func (c *Collector) Done(node IMessage) {
	FinishNode(node)
	if p := partOf(node); p != nil {
		c.keep(c.Events.EmitPart(p))
	}
}

func (c *Collector) Usage(u Usage) {
	c.mu.Lock()
	c.Usages = append(c.Usages, u)
	c.mu.Unlock()
	c.keep(c.Events.EmitUsage(u))
}

func (c *Collector) Progress(p PromptProgress) { c.keep(c.Events.EmitProgress(p)) }

func (c *Collector) Timings(t Timings) {
	c.mu.Lock()
	c.TimingsL = append(c.TimingsL, t)
	c.mu.Unlock()
	c.keep(c.Events.EmitTimings(t))
}

func (c *Collector) Retry(a RetryAttempt) {
	c.mu.Lock()
	c.Retries = append(c.Retries, a)
	c.mu.Unlock()
	for _, n := range c.Nodes {
		for _, f := range StreamFields(n) {
			WriterOf(f).Restart()
		}
	}
	c.keep(c.Events.EmitRetry(a))
}

func (c *Collector) keep(err error) {
	if err == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err == nil {
		c.err = err
	}
}

func (c *Collector) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}
