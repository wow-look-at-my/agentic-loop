package commonai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

// firstUsage is the report a call produced, or a Usage when the provider reported none.
func firstUsage(c *Completion) Usage {
	if c == nil || len(c.Usages) == 0 {
		return Usage{}
	}
	return c.Usages[0]
}

// lastTimings is the newest timings snapshot, or nil when none arrived.
func lastTimings(c *Completion) *Timings {
	if c == nil || len(c.Timings) == 0 {
		return nil
	}
	t := c.Timings[len(c.Timings)-1]
	return &t
}

// mustOpenAI builds a Provider via NewOpenAIProvider, failing the test on error.
func mustOpenAI(t *testing.T, cfg OpenAIConfig) Provider {
	t.Helper()
	p, err := NewOpenAIProvider(cfg)
	require.NoError(t, err)
	return p
}

// mustAnthropic builds a Provider via NewAnthropicProvider, failing on error.
func mustAnthropic(t *testing.T, cfg AnthropicConfig) Provider {
	t.Helper()
	p, err := NewAnthropicProvider(cfg)
	require.NoError(t, err)
	return p
}

// mustResponses builds a Provider via NewResponsesProvider, failing on error.
func mustResponses(t *testing.T, cfg ResponsesConfig) Provider {
	t.Helper()
	p, err := NewResponsesProvider(cfg)
	require.NoError(t, err)
	return p
}

// oaProvider is shorthand for an OpenAI-dialect test provider.
func oaProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL}})
}

// anProvider is shorthand for an Anthropic-dialect test provider.
func anProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL}})
}

// scriptStep is scripted answer: what to emit through the callbacks, and
// what to return.
type scriptStep struct {
	comp *Completion
	err  error
	emit func(ev *StreamEvents)
}

// scriptProvider replays scripted responses and records every request, so a
type scriptProvider struct {
	steps []scriptStep
	reqs  []Request
}

// Complete implements Provider.
func (p *scriptProvider) Complete(_ context.Context, req Request, ev *StreamEvents) (*Completion, error) {
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

// assistantComp is a scripted assistant turn with plausible usage.
func assistantComp(content string, calls ...ToolCall) *Completion {
	stop := StopEndTurn
	if len(calls) > 0 {
		stop = StopToolUse
	}
	parts := []Part{}
	if content != "" {
		parts = append(parts, TextPart{Text: content})
	}
	for _, c := range calls {
		parts = append(parts, ToolCallPart{ID: c.ID, Name: c.Name, Arguments: c.Arguments})
	}
	return &Completion{
		Message:    NewMessage(RoleAssistant, parts...),
		Usages:     []Usage{{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}},
		StopReason: stop,
	}
}

// recordingServer decodes the request body into out and answers with a minimal
// stream, so a test can check what a dialect BUILT rather than what it did
// with the answer.
func recordingServer(t *testing.T, out *map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		var decoded map[string]any
		require.NoError(t, json.Unmarshal(body, &decoded))
		*out = decoded
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)
	return srv
}
