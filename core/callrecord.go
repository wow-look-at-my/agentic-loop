package commonai

import (
	"sync"
	"time"
)

// Transform is one change the encoder made to a request and to nothing else.
type Transform struct {
	NodeID string
	Field  string
	Name   string
	Reason string
}

// CallRecord is one model call, one try. Every node the call produced links to it.
type CallRecord struct {
	Turn   uint
	Status CallStatus
	// Err says why the call ended, when it did not finish.
	Err string
	// Recovery names the recovery this try ran under, such as a thinking-replay rung.
	Recovery   string
	Usages     []Usage
	Timings    []Timings
	Streamed   bool
	StopReason string
	// Model is the model the response names.
	Model   string
	Routed  string
	Started time.Time
	Ended   time.Time
	// EncodedBytes is the size of the request body the encoder sent.
	EncodedBytes uint64
	Transforms   []Transform
}

// encCache is a node's encoded wire fragment per dialect, kept once the node is final.
type encCache struct {
	mu sync.Mutex
	m  map[Dialect][]byte
}

// CachedEncoding is the fragment an encoder stored for d.
func (b *MessageBase) CachedEncoding(d Dialect) ([]byte, bool) {
	b.enc.mu.Lock()
	defer b.enc.mu.Unlock()
	v, ok := b.enc.m[d]
	return v, ok
}

// StoreEncoding keeps a final node's fragment for d.
func (b *MessageBase) StoreEncoding(d Dialect, frag []byte) {
	b.enc.mu.Lock()
	defer b.enc.mu.Unlock()
	if b.enc.m == nil {
		b.enc.m = map[Dialect][]byte{}
	}
	b.enc.m[d] = frag
}

// DropEncodings forgets every fragment, after a host edit.
func (b *MessageBase) DropEncodings() {
	b.enc.mu.Lock()
	defer b.enc.mu.Unlock()
	b.enc.m = nil
}
