package loop

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// turnsSSEHandler serves one scripted SSE response per request, in order.
type turnsSSEHandler struct {
	turns [][]string
	hits  int
	body  [][]byte
}

func (h *turnsSSEHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var buf strings.Builder
	_, _ = buf.ReadFrom(r.Body)
	h.body = append(h.body, []byte(buf.String()))
	payloads := h.turns[h.hits]
	h.hits++
	w.Header().Set("Content-Type", "text/event-stream")
	fl := w.(http.Flusher)
	for _, p := range payloads {
		_, _ = w.Write([]byte("data: " + p + "\n\n"))
		fl.Flush()
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	fl.Flush()
}

func toolCallChunk(args string) string {
	return `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"write_file","arguments":` + jsonString(args) + `}}]}}]}`
}

func jsonString(s string) string {
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(s) + `"`
}

// A tool that refuses from its streamed arguments ends the model call there. The refusal is the
// call's result, the tool never runs, the model never finishes the arguments, and the run goes on.
func TestPreflightRefusalEndsTheCallAndAnswersIt(t *testing.T) {
	h := &turnsSSEHandler{turns: [][]string{
		{
			toolCallChunk(`{"path": "/work/ma`),
			toolCallChunk(`in.go", "content": "pack`),
			toolCallChunk(`age main\n\nfunc main() {}\n"}`),
			`{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":40,"completion_tokens":30,"total_tokens":70}}`,
		},
		{
			`{"choices":[{"delta":{"content":"used edit_file instead"},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":5,"total_tokens":55}}`,
		},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := newProvider(mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL}}), retryTestPolicy(3))

	exec := &fakeExec{tools: []ToolDecl{{Name: "write_file"}}}
	var asked []string
	tools := Tools{WithPreflight(exec.registry()[0], func(_ context.Context, partial string) (*ToolResult, bool) {
		asked = append(asked, partial)
		if !strings.Contains(partial, `"/work/main.go"`) {
			return nil, false
		}
		return &ToolResult{Content: "write_file: main.go already exists", IsError: true}, true
	})}

	var calls []ToolCall
	var results []ToolResultEvent
	var finals []FinalizeAssistantEvent
	events := Events{}
	keep(t, &events.OnToolCall, func(ev ToolCallEvent) error { calls = append(calls, *ev.Call); return nil })
	keep(t, &events.OnToolResult, func(ev ToolResultEvent) error { results = append(results, ev); return nil })
	keep(t, &events.OnFinalizeAssistant, func(ev FinalizeAssistantEvent) error { finals = append(finals, ev); return nil })

	res, err := runTurn(context.Background(), Config{Provider: p, Tools: tools, Approver: allowAll, Events: &events},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "add main.go"}}})
	require.NoError(t, err)

	// Asked on each delta until the path was whole, then never again.
	assert.Equal(t, []string{`{"path": "/work/ma`, `{"path": "/work/main.go", "content": "pack`}, asked)
	assert.Empty(t, exec.executed, "a refused call never runs")

	// The turn is what streamed before the refusal; the rest of the arguments never arrived.
	require.Len(t, res.Messages, 4)
	assistant := res.Messages[1]
	require.Len(t, assistant.ToolCalls, 1)
	assert.Equal(t, `{"path": "/work/main.go", "content": "pack`, assistant.ToolCalls[0].Arguments)
	assert.Equal(t, Message{Role: RoleTool, Content: "write_file: main.go already exists", ToolCallID: "c1", ToolIsError: true}, res.Messages[2])
	assert.Equal(t, "used edit_file instead", res.Final.Content)

	// The host saw the call and its result the usual way; the call ended complete, not failed.
	require.Len(t, calls, 1)
	assert.Equal(t, "c1", calls[0].ID)
	require.Len(t, results, 1)
	assert.Equal(t, "write_file: main.go already exists", results[0].Result.Content)
	require.Len(t, finals, 2)
	assert.Equal(t, "complete", finals[0].Status)

	// The cut call was not re-sent, and its usage never arrived: calls, the first unreported.
	assert.Equal(t, 2, h.hits)
	require.Len(t, res.Usages, 2)
	assert.Zero(t, res.Usages[0].TotalTokens)
	assert.Equal(t, 55, res.Usages[1].TotalTokens)
	assert.Contains(t, string(h.body[1]), "main.go already exists", "the next request carries the refusal")
}

// A preflight that lets the call through changes nothing: the whole arguments arrive and Execute runs.
func TestPreflightPassLetsTheCallRunWhole(t *testing.T) {
	h := &turnsSSEHandler{turns: [][]string{
		{
			toolCallChunk(`{"path": "/work/new.go", `),
			toolCallChunk(`"content": "package new\n"}`),
			`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		},
		{`{"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL}})

	exec := &fakeExec{tools: []ToolDecl{{Name: "write_file"}}}
	decisions := 0
	tools := Tools{WithPreflight(exec.registry()[0], func(context.Context, string) (*ToolResult, bool) {
		decisions++
		return nil, true
	})}
	res, err := runTurn(context.Background(), Config{Provider: p, Tools: tools, Approver: allowAll},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, 1, decisions, "decided on the first delta, not asked again")
	require.Len(t, exec.executed, 1)
	assert.Equal(t, `{"path": "/work/new.go", "content": "package new\n"}`, exec.executed[0].Arguments)
	assert.Equal(t, "done", res.Final.Content)
}
