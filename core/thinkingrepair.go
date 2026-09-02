package commonai

import (
	"context"
	"errors"
	"regexp"
	"strconv"
	"sync"

	"github.com/wow-look-at-my/go-containers/set"
)

// invalidThinkingSigPattern matches Anthropic's wording for a rejected
var invalidThinkingSigPattern = regexp.MustCompile(
	"(?i)messages\\.(\\d+)\\.content\\.(\\d+):\\s*Invalid `signature` in `thinking` block")

// thinkingSignatureRepair strips a rejected signature and retries,
// remembering it for the rest of the conversation.
type thinkingSignatureRepair struct {
	inner Provider

	mu  sync.Mutex
	bad set.Set[string] // rejected signature values
}

// NewThinkingSignatureRepair wraps p to strip and retry a rejected signature.
func NewThinkingSignatureRepair(p Provider) Provider {
	return &thinkingSignatureRepair{inner: p, bad: set.New[string]()}
}

// Complete implements Provider.
func (r *thinkingSignatureRepair) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	req.Messages = r.withoutBadSignatures(req.Messages)
	comp, err := r.inner.Complete(ctx, req, ev)
	// A non-nil completion means the call already streamed (see Provider), so
	// it is too late to strip anything and re-send.
	if err == nil || errors.Is(err, context.Canceled) || comp != nil {
		return comp, err
	}
	sig, ok := r.findRejectedSignature(req.Messages, err.Error())
	if !ok {
		return comp, err
	}
	r.mu.Lock()
	r.bad.Add(sig)
	r.mu.Unlock()
	req.Messages = r.withoutBadSignatures(req.Messages)
	return r.inner.Complete(ctx, req, ev)
}

// withoutBadSignatures returns a copy of msgs with every thinking block
// carrying a remembered bad signature cleared; msgs itself is never
// mutated, and msgs is returned unchanged when nothing needs stripping.
func (r *thinkingSignatureRepair) withoutBadSignatures(msgs []Message) []Message {
	if !r.anyBad(msgs) {
		return msgs
	}
	out := make([]Message, len(msgs))
	copy(out, msgs)
	for i, m := range out {
		var changed []ThinkingBlock
		for j, tb := range m.Thinking {
			if tb.Signature == "" || !r.isBad(tb.Signature) {
				continue
			}
			if changed == nil {
				changed = make([]ThinkingBlock, len(m.Thinking))
				copy(changed, m.Thinking)
			}
			changed[j].Signature = ""
		}
		if changed != nil {
			m.Thinking = changed
			out[i] = m
		}
	}
	return out
}

// anyBad reports whether any message carries a remembered bad signature.
func (r *thinkingSignatureRepair) anyBad(msgs []Message) bool {
	for _, m := range msgs {
		for _, tb := range m.Thinking {
			if tb.Signature != "" && r.isBad(tb.Signature) {
				return true
			}
		}
	}
	return false
}

func (r *thinkingSignatureRepair) isBad(sig string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.bad.Contains(sig)
}

// findRejectedSignature locates the exact signature value Anthropic
// rejected, by rebuilding the same wire content the failed call sent and
// reading the named block back out of it -- so this never re-derives the
// tool-result-folding/dropped-turn rules anWireMessages already implements.
func (r *thinkingSignatureRepair) findRejectedSignature(msgs []Message, errStr string) (string, bool) {
	m := invalidThinkingSigPattern.FindStringSubmatch(errStr)
	if m == nil {
		return "", false
	}
	msgIdx, err1 := strconv.Atoi(m[1])
	cIdx, err2 := strconv.Atoi(m[2])
	if err1 != nil || err2 != nil {
		return "", false
	}
	wire, err := anWireMessages(msgs)
	if err != nil || msgIdx < 0 || msgIdx >= len(wire) {
		return "", false
	}
	content, _ := wire[msgIdx]["content"].([]map[string]any)
	if cIdx < 0 || cIdx >= len(content) {
		return "", false
	}
	block := content[cIdx]
	if block["type"] != "thinking" {
		return "", false
	}
	sig, _ := block["signature"].(string)
	if sig == "" {
		return "", false
	}
	return sig, true
}
