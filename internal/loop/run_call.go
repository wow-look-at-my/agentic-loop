package loop

import (
	"context"
	"strings"
)

// runModelCall executes one model call, counting it as one turn (internal retries included).
func runModelCall(
	ctx context.Context, cfg *Config,
	req Request, turn int, msgs []Message, tools []ToolDecl, res *Result,
) (*Completion, error) {
	r := req
	r.Messages = msgs
	r.Tools = tools
	if cberr := cfg.Events.emitTurnBegin(TurnBeginEvent{Turn: turn, Req: &r}); cberr != nil {
		// The call never happened; nothing to count or record.
		return nil, cberr
	}

	comp, err := cfg.Provider.Complete(ctx, r, &cfg.Events.StreamEvents)
	res.Turns++
	if comp != nil {
		res.Usages = append(res.Usages, comp.Usage)
	}
	if cberr := cfg.Events.emitTurnEnd(TurnEndEvent{Turn: turn, Comp: comp, Err: err}); cberr != nil {
		// The call happened; its data is kept, the run aborts on the sink failure.
		return comp, cberr
	}
	return comp, err
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
