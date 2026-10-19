package loop

import (
	"context"
	"sync"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/client"
	"github.com/wow-look-at-my/go-containers/event"
)

// IMessage is one node of the conversation tree.
type IMessage = commonai.IMessage

// NodeEvent is the argument of OnNode: a node was added, or one of its fields changed.
type NodeEvent struct {
	event.Args
	Node  IMessage
	Added bool
}

// session owns a conversation as a tree. Every change to the conversation is a method here, so no
// change is silent and nothing is ever cut out of the tree.
type session struct {
	mu   sync.Mutex
	tree *commonai.Tree
	leaf IMessage
	// window is the newest compaction on the path, or nil for the root. A request starts there.
	window *commonai.CompactionMessage
	// dialect places the opaque blobs of flat messages a host hands in.
	dialect commonai.Dialect
	events  *Events
	index   contentIndex
}

func newSession(events *Events) *session {
	return &session{tree: commonai.NewTree(), events: events}
}

func (s *session) Leaf() IMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.leaf
}

// add links n under the leaf and makes it the leaf.
func (s *session) add(n IMessage) {
	s.mu.Lock()
	err := s.tree.Add(n, s.leaf)
	if err == nil {
		s.leaf = n
	}
	s.mu.Unlock()
	if err != nil {
		panic(err)
	}
	s.events.emitNode(NodeEvent{Node: n, Added: true})
}

// appendFlat adds a flat message as nodes and gives the nodes.
func (s *session) appendFlat(m Message) []IMessage {
	nodes := commonai.NodesFromMessage(m, s.dialect)
	for _, n := range nodes {
		s.add(n)
	}
	return nodes
}

// path is the whole conversation, root to leaf.
func (s *session) path() []IMessage {
	return s.tree.PathTo(s.Leaf())
}

// windowPath is what a request carries: the newest compaction to the leaf.
func (s *session) windowPath() []IMessage {
	p := s.path()
	s.mu.Lock()
	w := s.window
	s.mu.Unlock()
	if w == nil {
		return p
	}
	for i, n := range p {
		if n == IMessage(w) {
			return p[i:]
		}
	}
	return p
}

// messages is the window as flat messages.
func (s *session) messages() []Message {
	return commonai.MessagesFromPath(s.windowPath())
}

// compact appends a finished summary and starts the window there. Nothing before it changes.
func (s *session) compact(summary, id string) *commonai.CompactionMessage {
	s.mu.Lock()
	var from IMessage = s.window
	s.mu.Unlock()
	c := commonai.NewCompaction(from)
	if id != "" {
		c.SetID(id)
	}
	commonai.WriterOf(c.Summary).Finish(summary)
	s.add(c)
	s.mu.Lock()
	s.window = c
	s.mu.Unlock()
	return c
}

// toolResult answers call. A non-empty omitted says why the call never ran; the content says it too.
func (s *session) toolResult(call *commonai.ToolCallMessage, content string, isError bool, omitted string) *commonai.ToolResultMessage {
	r := commonai.NewToolResult(call, isError, commonai.Text(content))
	r.Omitted = omitted
	s.add(r)
	return r
}

// toolCalls are the tool-call nodes among nodes that are still on the wire.
func toolCalls(nodes []IMessage) []*commonai.ToolCallMessage {
	var out []*commonai.ToolCallMessage
	for _, n := range nodes {
		if tc, ok := n.(*commonai.ToolCallMessage); ok && !tc.Superseded() {
			out = append(out, tc)
		}
	}
	return out
}

// omitCalls answers every call in nodes that has no result, so the path never holds a call no result
// answers. The calls stay as the model made them.
func (s *session) omitCalls(nodes []IMessage, reason string) {
	answered := map[*commonai.ToolCallMessage]bool{}
	for _, n := range s.path() {
		if r, ok := n.(*commonai.ToolResultMessage); ok && r.Call != nil {
			answered[r.Call] = true
		}
	}
	for _, tc := range toolCalls(nodes) {
		if !answered[tc] {
			s.toolResult(tc, reason, true, reason)
		}
	}
}

// replaceCalls records a host's replacement batch. The model's own calls stay in the tree, left out
// of requests; the replacement joins the same turn.
func (s *session) replaceCalls(nodes []IMessage, calls []ToolCall) []*commonai.ToolCallMessage {
	var rec *commonai.CallRecord
	for _, tc := range toolCalls(nodes) {
		rec = tc.Call()
		tc.MarkSuperseded()
	}
	out := make([]*commonai.ToolCallMessage, 0, len(calls))
	for _, c := range calls {
		tc := commonai.NewToolCallArgs(commonai.CallID(c.ID), c.Name, c.Arguments)
		commonai.SetCallOf(tc, rec)
		s.add(tc)
		out = append(out, tc)
	}
	return out
}

// addText adds loop-written text to a model turn, such as the placeholder of a turn with no answer.
func (s *session) addText(rec *commonai.CallRecord, text string) {
	a := commonai.NewAssistantText(text)
	commonai.SetCallOf(a, rec)
	s.add(a)
}

// sink is the CallSink of one model call: each node joins the tree when its
// part begins.
func (s *session) sink(rec *commonai.CallRecord, firstID string) *callSink {
	return &callSink{s: s, rec: rec, firstID: firstID}
}

type callSink struct {
	s       *session
	rec     *commonai.CallRecord
	firstID string
	mu      sync.Mutex
	nodes   []IMessage
	err     error
}

func (c *callSink) Start(n IMessage) {
	c.s.tree.SetCall(n, c.rec, commonai.Producer{Dialect: c.s.dialect, Model: c.rec.Model})
	c.mu.Lock()
	if len(c.nodes) == 0 && c.firstID != "" && c.s.tree.Node(c.firstID) == nil {
		n.Base().SetID(c.firstID)
	}
	c.nodes = append(c.nodes, n)
	c.mu.Unlock()
	c.s.add(n)
}

func (c *callSink) Delta(n IMessage, f commonai.FieldID, delta string) {
	if field := commonai.FieldOf(n, f); field != nil {
		commonai.WriterOf(field).Append(delta)
	}
	var err error
	switch n.(type) {
	case *commonai.AssistantMessage:
		err = c.s.events.EmitText(delta)
	case *commonai.ThinkingMessage:
		err = c.s.events.EmitReasoning(delta)
	}
	c.keep(err)
	c.s.events.emitNode(NodeEvent{Node: n})
}

func (c *callSink) Done(n IMessage) {
	commonai.FinishNode(n)
	if p := partOfNode(n); p != nil {
		c.keep(c.s.events.EmitPart(p))
	}
	c.s.events.emitNode(NodeEvent{Node: n})
}

func (c *callSink) Usage(u commonai.Usage) {
	c.rec.Usages = append(c.rec.Usages, u)
	c.keep(c.s.events.EmitUsage(u))
}

func (c *callSink) Progress(p commonai.PromptProgress) { c.keep(c.s.events.EmitProgress(p)) }

func (c *callSink) Timings(t commonai.Timings) {
	c.rec.Timings = append(c.rec.Timings, t)
	c.keep(c.s.events.EmitTimings(t))
}

// Retry keeps what each node received as a retry revision, so the next try writes new revisions.
func (c *callSink) Retry(a commonai.RetryAttempt) {
	c.mu.Lock()
	nodes := append([]IMessage(nil), c.nodes...)
	c.mu.Unlock()
	for _, n := range nodes {
		for _, f := range commonai.StreamFields(n) {
			commonai.WriterOf(f).Restart()
		}
	}
	c.keep(c.s.events.EmitRetry(a))
}

func (c *callSink) keep(err error) {
	if err == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err == nil {
		c.err = err
	}
}

func (c *callSink) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

// produced is every node the call added, in the order the parts began.
func (c *callSink) produced() []IMessage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]IMessage(nil), c.nodes...)
}

// partOfNode is the flat part of one model-output node, for OnPart.
func partOfNode(n IMessage) Part {
	m := commonai.MessageFromNodes([]IMessage{n})
	if ps := m.EffectiveParts(); len(ps) == 1 {
		return ps[0]
	}
	return nil
}

// streamCall runs one model call into the tree and gives its result and the nodes it produced.
func (s *session) streamCall(ctx context.Context, p Provider, req Request, turn int, firstID string) (*Completion, []IMessage, *commonai.CallRecord, error) {
	rec := &commonai.CallRecord{Turn: uint(turn), Status: commonai.CallStreaming, Started: time.Now()}
	sink := s.sink(rec, firstID)
	comp, err := client.Stream(ctx, p, req, sink)
	rec.Ended = time.Now()
	nodes := sink.produced()
	for _, n := range nodes {
		commonai.FinishNode(n)
	}
	switch {
	case err == nil:
		rec.Status = commonai.CallDone
	case ctx.Err() != nil:
		rec.Status, rec.Err = commonai.CallCancelled, err.Error()
	default:
		rec.Status, rec.Err = commonai.CallFailed, err.Error()
	}
	if comp != nil {
		rec.StopReason, rec.Model, rec.Streamed = comp.StopReason, comp.Model, comp.Streamed
	}
	if err == nil {
		err = sink.Err()
	}
	return comp, nodes, rec, err
}
