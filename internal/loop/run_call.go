package loop

import (
	"context"
	"strings"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// modelCall is one model call's outcome: the folded completion, the nodes it added, and its record.
type modelCall struct {
	comp  *Completion
	nodes []IMessage
	rec   *commonai.CallRecord
}

// message is the call's turn as a flat message.
func (m modelCall) message() Message {
	return commonai.MessageFromNodes(m.nodes)
}

// runModelCall executes model call into the session, counting it as turn (internal retries included).
// msgs is what the call sends. Its nodes join the tree as their parts begin; firstID names the first.
func runModelCall(
	ctx context.Context, cfg *Config, s *session,
	req Request, turn int, msgs []Message, tools []ToolDecl, res *Result, el *elapsedTracker, firstID string,
) (modelCall, error) {
	r := req
	r.Messages = elapsedMessages(msgs, el.mark())
	r.Tools = tools
	if cberr := cfg.Events.emitTurnBegin(TurnBeginEvent{Turn: turn, Req: &r}); cberr != nil {
		// The call never happened; nothing to count or record.
		return modelCall{}, cberr
	}

	comp, nodes, rec, err := s.streamCall(ctx, cfg.Provider, r, turn, firstID)
	res.Turns++
	if comp != nil {
		res.Usages = append(res.Usages, comp.Usage)
	}
	mc := modelCall{comp: comp, nodes: nodes, rec: rec}
	if cberr := cfg.Events.emitTurnEnd(TurnEndEvent{Turn: turn, Comp: comp, Err: err}); cberr != nil {
		// The call happened; its data is kept, the run aborts on the sink failure.
		return mc, cberr
	}
	return mc, err
}

// fallbackOutput picks the text to surface when the loop ends without a written answer.
func fallbackOutput(m Message) string {
	if s := strings.TrimSpace(m.Content); s != "" {
		return s
	}
	var b strings.Builder
	for _, t := range m.Thinking {
		b.WriteString(t.Text)
	}
	if s := strings.TrimSpace(b.String()); s != "" {
		return s
	}
	return noOutputPlaceholder
}
