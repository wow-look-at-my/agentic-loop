package agentic

import (
	"context"
	"errors"
	"strings"
	"time"
)

// OneShot runs a single bounded, tool-less completion — the title/summary
// pattern: tools are stripped from the request, the call is made exactly once
// (no retry), and a positive timeout bounds it. It returns the trimmed final
// text and the call's usage.
//
// For fire-and-forget work that must survive the parent request ending (an
// auto-title after the response streamed), pass a detached context:
//
//	text, usage, err := agentic.OneShot(context.WithoutCancel(ctx), p, req, 30*time.Second)
func OneShot(ctx context.Context, p Provider, req Request, timeout time.Duration) (string, Usage, error) {
	if p == nil {
		return "", Usage{}, badRequestErr("agentic: OneShot requires a Provider")
	}
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	r := req
	r.Tools = nil
	comp, err := p.Complete(ctx, r, nil)
	if err != nil {
		if comp != nil {
			return "", comp.Usage, err
		}
		return "", Usage{}, err
	}
	return strings.TrimSpace(comp.Message.Content), comp.Usage, nil
}

// CompactRequestText is the instruction that triggers a compaction summary.
// It is sent as the trailing user message of the summarize call, and it is
// also the text of the stored request message so the compacted round reads
// naturally as prior history.
const CompactRequestText = "Summarize this entire conversation in detail for a future instance of yourself to pick up. Output only the summary."

// CompactResult is the outcome of a Compact call: the summary text, the
// two-message replacement round that stands in for the whole history, and the
// summarize call's usage.
type CompactResult struct {
	Summary  string
	Messages []Message
	Usage    Usage
}

// Compact asks the model to summarize the conversation for a self-handoff:
// it sends req.Messages with CompactRequestText appended as the trailing user
// message and NO tools, in one call. The returned Messages are the
// replacement round — user(CompactRequestText) followed by
// assistant(summary), a valid round that ends on assistant so the next real
// prompt continues clean alternation. The caller replaces its history with
// them. req.System is passed through, so the caller chooses the summarizer's
// system prompt (what to capture, what to skip). An empty summary is an
// error and nothing should be replaced.
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
		Usage: comp.Usage,
	}, nil
}
