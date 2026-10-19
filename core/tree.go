package commonai

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"sync"
	"time"
)

// rootNode is the empty sentinel every tree starts from. It is never sent and never stored.
type rootNode struct{ MessageBase }

func (*rootNode) NodeKind() NodeKind { return NodeExtension }
func (*rootNode) Final() bool        { return true }

// IsRoot reports whether m is a tree's sentinel.
func IsRoot(m IMessage) bool {
	_, ok := m.(*rootNode)
	return ok
}

// Tree is a conversation: one parent per node, so a request is always one
// root-to-leaf path. An edit or a regenerate is a sibling.
type Tree struct {
	mu   sync.RWMutex
	root *rootNode
	byID map[string]IMessage
}

// NewTree is an empty conversation.
func NewTree() *Tree {
	return &Tree{root: &rootNode{}, byID: map[string]IMessage{}}
}

// Root is the sentinel.
func (t *Tree) Root() IMessage { return t.root }

// ErrNotInTree is the error of an operation on a node another tree owns.
var ErrNotInTree = errors.New("commonai: the node is not in this tree")

// NewID is a random node id.
func NewID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// Add links msg under parent. A nil parent is the root. msg keeps an id it already has.
func (t *Tree) Add(msg, parent IMessage) error {
	if msg == nil {
		return errors.New("commonai: Add of a nil node")
	}
	checkRegistered(msg)
	t.mu.Lock()
	defer t.mu.Unlock()
	if parent == nil {
		parent = t.root
	}
	if !t.ownsLocked(parent) {
		return fmt.Errorf("%w: parent %q", ErrNotInTree, parent.Base().id)
	}
	b := msg.Base()
	if b.parent != nil {
		return fmt.Errorf("commonai: node %q already has a parent", b.id)
	}
	if b.id == "" {
		b.id = NewID()
	}
	if _, dup := t.byID[b.id]; dup {
		return fmt.Errorf("commonai: a node with id %q is already in the tree", b.id)
	}
	if b.at.IsZero() {
		b.at = time.Now()
	}
	b.parent = parent
	pb := parent.Base()
	pb.children = append(pb.children, msg)
	t.byID[b.id] = msg
	return nil
}

func (t *Tree) ownsLocked(m IMessage) bool {
	if m == IMessage(t.root) {
		return true
	}
	got, ok := t.byID[m.Base().id]
	return ok && got == m
}

// Owns reports whether m is in this tree.
func (t *Tree) Owns(m IMessage) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.ownsLocked(m)
}

// Node finds a node by id.
func (t *Tree) Node(id string) IMessage {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.byID[id]
}

// Len is the number of nodes, the root not counted.
func (t *Tree) Len() uint {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return uint(len(t.byID))
}

// PathTo is the path from the root to leaf, root excluded.
func (t *Tree) PathTo(leaf IMessage) []IMessage {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var rev []IMessage
	for n := leaf; n != nil && !IsRoot(n); n = n.Base().parent {
		rev = append(rev, n)
	}
	out := make([]IMessage, len(rev))
	for i, n := range rev {
		out[len(rev)-1-i] = n
	}
	return out
}

// Nodes ranges over the whole tree, depth-first, children oldest first.
func (t *Tree) Nodes() iter.Seq[IMessage] {
	return func(yield func(IMessage) bool) {
		t.mu.RLock()
		var order []IMessage
		var walk func(IMessage)
		walk = func(n IMessage) {
			for _, c := range n.Base().children {
				order = append(order, c)
				walk(c)
			}
		}
		walk(t.root)
		t.mu.RUnlock()
		for _, n := range order {
			if !yield(n) {
				return
			}
		}
	}
}

// SetCall links a node to the call that produced it and names its producer.
func (t *Tree) SetCall(m IMessage, rec *CallRecord, p Producer) {
	t.mu.Lock()
	defer t.mu.Unlock()
	b := m.Base()
	b.call, b.producer = rec, p
}

// extension types: a host's own node types, registered with a GUID so a store can round-trip them.
var extTypes = struct {
	sync.RWMutex
	byType map[reflect.Type]ExtensionType
	byID   map[string]ExtensionType
}{byType: map[reflect.Type]ExtensionType{}, byID: map[string]ExtensionType{}}

// ExtensionType describes one registered host node type.
type ExtensionType struct {
	ID, Name string
	Type     reflect.Type
}

// RegisterMessageType registers a host node type. Such nodes are stored and never sent to a model.
func RegisterMessageType[T IMessage](id, name string) {
	rt := reflect.TypeFor[T]()
	extTypes.Lock()
	defer extTypes.Unlock()
	if prev, ok := extTypes.byID[id]; ok {
		panic(fmt.Sprintf("commonai: message type id %s is already %q", id, prev.Name))
	}
	e := ExtensionType{ID: id, Name: name, Type: rt}
	extTypes.byType[rt] = e
	extTypes.byID[id] = e
}

// ExtensionOf is the registration of a host node, if m is one.
func ExtensionOf(m IMessage) (ExtensionType, bool) {
	extTypes.RLock()
	defer extTypes.RUnlock()
	e, ok := extTypes.byType[reflect.TypeOf(m)]
	return e, ok
}

// ExtensionByID finds a registration by GUID.
func ExtensionByID(id string) (ExtensionType, bool) {
	extTypes.RLock()
	defer extTypes.RUnlock()
	e, ok := extTypes.byID[id]
	return e, ok
}

func checkRegistered(m IMessage) {
	switch m.(type) {
	case *UserMessage, *SystemMessage, *AssistantMessage, *ThinkingMessage, *ToolCallMessage,
		*ToolResultMessage, *CompactionMessage, *RawBlockMessage:
		return
	}
	if _, ok := ExtensionOf(m); !ok {
		panic(fmt.Sprintf("commonai: %T is not a registered message type; call RegisterMessageType first", m))
	}
}
