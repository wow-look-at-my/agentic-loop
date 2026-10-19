package loop

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// contentIndex is every value the tree holds, keyed by node, field and revision.
type contentIndex map[string]string

func hashValue(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:8])
}

// indexNodes indexes every stream-field revision and opaque payload of nodes.
func indexNodes(nodes []IMessage) contentIndex {
	idx := contentIndex{}
	put := func(n IMessage, field string, value string) {
		if value == "" {
			return
		}
		idx[n.Base().ID()+"/"+field+"/"+hashValue(value)] = fmt.Sprintf("%s %q", field, n.Base().ID())
	}
	for _, n := range nodes {
		for i, f := range commonai.StreamFields(n) {
			for _, r := range f.Revisions() {
				put(n, fmt.Sprintf("field%d", i), r.Value)
			}
		}
		if t, ok := n.(*commonai.ThinkingMessage); ok {
			put(n, "signature", t.Signature.String())
			put(n, "encrypted", t.Encrypted.String())
			put(n, "redacted", t.Redacted.String())
			put(n, "details", string(t.Details))
		}
	}
	return idx
}

// missing names every entry of before that after lacks, sorted.
func (before contentIndex) missing(after contentIndex) []string {
	var out []string
	for k, what := range before {
		if _, ok := after[k]; !ok {
			out = append(out, what)
		}
	}
	sort.Strings(out)
	return out
}

// lossTracker makes a loss that repeats on consecutive turns of one provider chronic.
type lossTracker struct {
	lastTurn int
	provider Provider
}

// record reports a loss. It never ends the run: a run that continues can still recover, and a
// chronic loss says plainly that the endpoint drops conversation state.
func (l *lossTracker) record(cfg *Config, res *Result, loss StateLossEvent, p Provider) {
	loss.Severity = commonai.SeverityNotice
	if l.lastTurn > 0 && loss.Turn-l.lastTurn <= 1 && l.provider == p {
		loss.Severity = commonai.SeverityChronic
		loss.Detail += ". This repeats on every turn: the endpoint drops conversation state, so each " +
			"further call pays for context the model never sees. Stop using it, or report it: the bug can be in this library"
	}
	l.lastTurn, l.provider = loss.Turn, p
	res.StateLosses = append(res.StateLosses, loss)
	cfg.Events.emitStateLoss(loss)
}

// checkMonotonic compares the tree before and after a turn. A value the tree held and no longer
// holds is state the model will not see again.
func (s *session) checkMonotonic(turn int) *StateLossEvent {
	var nodes []IMessage
	for n := range s.tree.Nodes() {
		nodes = append(nodes, n)
	}
	after := indexNodes(nodes)
	s.mu.Lock()
	before := s.index
	s.index = after
	s.mu.Unlock()
	if before == nil {
		return nil
	}
	lost := before.missing(after)
	if len(lost) == 0 {
		return nil
	}
	return &StateLossEvent{Turn: turn, Local: true,
		Detail: fmt.Sprintf("the conversation no longer holds %d value(s) it held before this turn: %v", len(lost), lost)}
}
