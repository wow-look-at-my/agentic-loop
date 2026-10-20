package loop

import (
	"context"
	"errors"
	"fmt"
	"sync"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/go-containers/event"
)

// IFuture is the async primitive: every operation that can wait returns one,
// and nothing in the session API blocks.
type IFuture[T any] interface {
	// Done is closed on every exit path, a panic and a cancel included.
	Done() <-chan struct{}
	Wait(ctx context.Context) (T, error)
	// Result polls without waiting; ok is false while the operation runs.
	Result() (v T, err error, ok bool)
	Cancel()
}

type future[T any] struct {
	done   chan struct{}
	cancel context.CancelFunc
	once   sync.Once
	v      T
	err    error
}

func newFuture[T any](cancel context.CancelFunc) *future[T] {
	return &future[T]{done: make(chan struct{}), cancel: cancel}
}

func (f *future[T]) settle(v T, err error) {
	f.once.Do(func() {
		f.v, f.err = v, err
		close(f.done)
	})
}

func (f *future[T]) Done() <-chan struct{} { return f.done }

func (f *future[T]) Wait(ctx context.Context) (T, error) {
	select {
	case <-f.done:
		return f.v, f.err
	case <-ctx.Done():
		var zero T
		return zero, ctx.Err()
	}
}

func (f *future[T]) Result() (T, error, bool) {
	select {
	case <-f.done:
		return f.v, f.err, true
	default:
		var zero T
		return zero, nil, false
	}
}

func (f *future[T]) Cancel() {
	if f.cancel != nil {
		f.cancel()
	}
}

// settled is a future that is already done.
func settled[T any](v T, err error) *future[T] {
	f := newFuture[T](nil)
	f.settle(v, err)
	return f
}

// ITurn is one Send: an awaitable handle on the running loop.
type ITurn interface {
	Done() <-chan struct{}
	Wait(ctx context.Context) error
	// Err is the turn's error once it is done, else nil.
	Err() error
	Cancel()
	// Result is the run's outcome once the turn is done, else nil.
	Result() *Result
}

type turnHandle struct {
	f *future[*Result]
	// panicked holds a panic out of the turn; Wait raises it again on the caller's goroutine.
	panicked *any
}

func (t turnHandle) Done() <-chan struct{} { return t.f.Done() }
func (t turnHandle) Wait(ctx context.Context) error {
	_, err := t.f.Wait(ctx)
	// The turn is done only when Result says so; only then is panicked safe to read.
	if _, _, done := t.f.Result(); done && t.panicked != nil && *t.panicked != nil {
		panic(*t.panicked)
	}
	return err
}
func (t turnHandle) Err() error {
	_, err, _ := t.f.Result()
	return err
}
func (t turnHandle) Cancel() { t.f.Cancel() }
func (t turnHandle) Result() *Result {
	r, _, _ := t.f.Result()
	return r
}

// SessionConfig is everything on the requesting side: the loop's wiring, and
// the request template every call starts from.
type SessionConfig struct {
	Loop    Config
	Request Request
}

// SessionState is what the server reported and what the session counted, across every turn.
type SessionState struct {
	// Usages are every call's usage, in order.
	Usages      []Usage
	Turns       int
	Compactions int
	StateLosses []StateLossEvent
	// Undelivered are queued messages a turn that ended never delivered.
	Undelivered []Message
}

// ISession owns one conversation as a tree. Every change goes through its methods.
type ISession interface {
	// Send appends msgs under the leaf and runs the loop until it answers. It returns at once.
	Send(ctx context.Context, msgs ...Message) ITurn
	// Queue delivers msg at the next turn boundary: the running turn's, or the next Send's.
	Queue(msg QueuedMessage) bool
	// Append adds a node under parent, such as an edit or a regenerate. It does not run the loop.
	Append(msg, parent IMessage) error
	SetLeaf(n IMessage) error
	Leaf() IMessage
	// Path is the conversation root to leaf; Window is the part a request carries.
	Path() []IMessage
	Window() []IMessage
	Messages() []Message
	Node(id string) IMessage
	Nodes() []IMessage
	State() SessionState
	Config() SessionConfig
	OnNode() *event.Event[NodeEvent]
	// Close waits for a running turn and refuses further work.
	Close(ctx context.Context) IFuture[struct{}]
}

// ErrSessionBusy is the error of a Send while a turn runs. A message for a running turn goes through Queue.
var ErrSessionBusy = errors.New("agentic: a turn is already running in this session")

// ErrSessionClosed is the error of a Send after Close.
var ErrSessionClosed = errors.New("agentic: the session is closed")

type publicSession struct {
	s   *session
	cfg SessionConfig

	mu      sync.Mutex
	running *future[*Result]
	queue   *MessageQueue
	pending []QueuedMessage
	closed  bool
	state   SessionState
}

// NewSession is an empty conversation. The window configuration is checked here.
func NewSession(cfg SessionConfig) (ISession, error) {
	if cfg.Loop.Provider == nil {
		return nil, badRequestErr("agentic: SessionConfig.Loop.Provider is required")
	}
	if err := checkWindowConfig(cfg.Loop); err != nil {
		return nil, err
	}
	if cfg.Loop.Events == nil {
		cfg.Loop.Events = &Events{}
	}
	return &publicSession{s: newSession(cfg.Loop.Events), cfg: cfg}, nil
}

func (p *publicSession) Send(ctx context.Context, msgs ...Message) ITurn {
	p.mu.Lock()
	switch {
	case p.closed:
		p.mu.Unlock()
		return turnHandle{f: settled[*Result](nil, ErrSessionClosed)}
	case p.running != nil:
		p.mu.Unlock()
		return turnHandle{f: settled[*Result](nil, ErrSessionBusy)}
	}
	runCtx, cancel := context.WithCancel(ctx)
	f := newFuture[*Result](cancel)
	// A host queue serves one turn: the turn closes it as it returns.
	q := p.cfg.Loop.Messages
	if q == nil {
		q = &MessageQueue{}
	}
	for _, m := range p.pending {
		q.Queue(m)
	}
	p.pending = nil
	p.running, p.queue = f, q
	p.mu.Unlock()

	for _, m := range msgs {
		p.s.appendFlat(m)
	}
	cfg := p.cfg.Loop
	cfg.Messages = q
	var panicked any
	go func() {
		var res *Result
		var err error
		defer func() {
			if r := recover(); r != nil {
				panicked = r
				err = errors.Join(err, fmt.Errorf("agentic: the turn panicked: %v", r))
			}
			p.finish(res)
			f.settle(res, err)
			cancel()
		}()
		res, err = runOn(runCtx, cfg, p.cfg.Request, p.s)
	}()
	return turnHandle{f: f, panicked: &panicked}
}

func (p *publicSession) finish(res *Result) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.running, p.queue = nil, nil
	if res == nil {
		return
	}
	p.state.Usages = append(p.state.Usages, res.Usages...)
	p.state.Turns += res.Turns
	p.state.StateLosses = append(p.state.StateLosses, res.StateLosses...)
	p.state.Undelivered = append(p.state.Undelivered, res.Undelivered...)
}

func (p *publicSession) Queue(msg QueuedMessage) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return false
	}
	if p.queue != nil && p.queue.Queue(msg) {
		return true
	}
	p.pending = append(p.pending, msg)
	return true
}

func (p *publicSession) Append(msg, parent IMessage) error {
	p.s.mu.Lock()
	err := p.s.tree.Add(msg, parent)
	p.s.mu.Unlock()
	if err == nil {
		p.s.events.emitNode(NodeEvent{Node: msg, Added: true})
	}
	return err
}

func (p *publicSession) SetLeaf(n IMessage) error {
	if n != nil && !p.s.tree.Owns(n) {
		return commonai.ErrNotInTree
	}
	p.mu.Lock()
	busy := p.running != nil
	p.mu.Unlock()
	if busy {
		return ErrSessionBusy
	}
	p.s.mu.Lock()
	defer p.s.mu.Unlock()
	p.s.leaf = n
	return nil
}

func (p *publicSession) Leaf() IMessage          { return p.s.Leaf() }
func (p *publicSession) Path() []IMessage        { return p.s.path() }
func (p *publicSession) Window() []IMessage      { return p.s.windowPath() }
func (p *publicSession) Messages() []Message     { return p.s.messages() }
func (p *publicSession) Node(id string) IMessage { return p.s.tree.Node(id) }

func (p *publicSession) Nodes() []IMessage {
	var out []IMessage
	for n := range p.s.tree.Nodes() {
		out = append(out, n)
	}
	return out
}

func (p *publicSession) State() SessionState {
	p.mu.Lock()
	defer p.mu.Unlock()
	st := p.state
	st.Usages = append([]Usage(nil), st.Usages...)
	st.StateLosses = append([]StateLossEvent(nil), st.StateLosses...)
	st.Undelivered = append([]Message(nil), st.Undelivered...)
	for _, n := range p.s.path() {
		if n.NodeKind() == commonai.NodeCompaction {
			st.Compactions++
		}
	}
	return st
}

func (p *publicSession) Config() SessionConfig           { return p.cfg }
func (p *publicSession) OnNode() *event.Event[NodeEvent] { return &p.s.events.OnNode }

func (p *publicSession) Close(ctx context.Context) IFuture[struct{}] {
	p.mu.Lock()
	p.closed = true
	running := p.running
	p.mu.Unlock()
	f := newFuture[struct{}](nil)
	go func() {
		if running != nil {
			if _, err := running.Wait(ctx); errors.Is(err, ctx.Err()) && ctx.Err() != nil {
				f.settle(struct{}{}, err)
				return
			}
		}
		f.settle(struct{}{}, nil)
	}()
	return f
}
