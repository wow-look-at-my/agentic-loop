package loop

import (
	"context"
	"errors"
	"strings"
	"time"
)

// OneShot runs a single bounded, tool-less completion (no retry), returning the whole *Completion.
func OneShot(ctx context.Context, p Provider, req Request, timeout time.Duration) (*Completion, error) {
	if p == nil {
		return nil, badRequestErr("agentic: OneShot requires a Provider")
	}
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	r := req
	r.Tools = nil
	return p.Complete(ctx, r, nil)
}

// CompactRequestText is the instruction that triggers a compaction summary.
// It is sent as the trailing user message of the summarize call, and it is
// also the text of the stored request message so the compacted round reads
// naturally as prior history.
const CompactRequestText = "Summarize this entire conversation in detail for a future instance of yourself to pick up. Output only the summary."

// CompactResult is the outcome of a Compact call: summary, replacement round, and Completion.
type CompactResult struct {
	Summary    string
	Messages   []Message
	Completion *Completion
}

// Compact summarizes the conversation for a self-handoff, replacing history with the summary round.
func Compact(ctx context.Context, p Provider, req Request) (*CompactResult, error) {
	if p == nil {
		return nil, badRequestErr("agentic: Compact requires a Provider")
	}
	r := req
	r.Tools = nil
	msgs := make([]Message, 0, len(req.Messages)+1)
	msgs = append(msgs, req.Messages...)
	msgs = append(msgs, Message{Role: RoleUser, Content: CompactRequestText})
	r.Messages = msgs
	comp, err := p.Complete(ctx, r, nil)
	if err != nil {
		return nil, err
	}
	summary := strings.TrimSpace(comp.Message.Content)
	if summary == "" {
		return nil, errors.New("the model returned an empty summary; nothing was compacted")
	}
	return &CompactResult{
		Summary: summary,
		Messages: []Message{
			{Role: RoleUser, Content: CompactRequestText},
			{Role: RoleAssistant, Content: summary},
		},
		Completion: comp,
	}, nil
}
