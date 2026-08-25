package extras

import (
	"context"
	"errors"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// noSleep is a policy whose backoff returns immediately, so a test measures what was retried.
var noSleep = RetryPolicy{Sleep: func(context.Context, time.Duration) error { return nil }}

// scriptStep is one scripted answer: what to emit through the callbacks, and
// what to return.
type scriptStep struct {
	comp *commonai.Completion
	err  error
	emit func(ev *commonai.StreamEvents)
}

// scriptProvider replays scripted responses and records every request, so a
// test can drive a decorator without an upstream.
type scriptProvider struct {
	steps []scriptStep
	reqs  []commonai.Request
}

// Complete implements commonai.Provider.
func (p *scriptProvider) Complete(_ context.Context, req commonai.Request, ev *commonai.StreamEvents) (*commonai.Completion, error) {
	p.reqs = append(p.reqs, req)
	if len(p.steps) == 0 {
		return nil, errors.New("script exhausted")
	}
	step := p.steps[0]
	p.steps = p.steps[1:]
	if step.emit != nil {
		step.emit(ev)
	}
	return step.comp, step.err
}

// assistantComp is a scripted assistant turn.
func assistantComp(content string) *commonai.Completion {
	return &commonai.Completion{
		Message:    commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: content}),
		StopReason: commonai.StopEndTurn,
	}
}

// emptyComp is a scripted assistant turn with no text, tool call, or thinking.
func emptyComp() *commonai.Completion {
	return &commonai.Completion{
		Message:    commonai.Message{Role: commonai.RoleAssistant},
		StopReason: commonai.StopEndTurn,
	}
}
