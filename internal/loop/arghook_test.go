package loop

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
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
	body, _ := io.ReadAll(r.Body)
	h.body = append(h.body, body)
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

// toolCallChunk is one chat-completions chunk carrying a fragment of call c1's arguments.
func toolCallChunk(args string) string {
	chunk := map[string]any{"choices": []any{map[string]any{"delta": map[string]any{
		"tool_calls": []any{map[string]any{"index": 0, "id": "c1", "type": "function",
			"function": map[string]any{"name": "write_file", "arguments": args}}},
	}}}}
	b, err := json.Marshal(chunk)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// A hook that refuses ends the model call the moment its member has streamed. The refusal is the
// call's result, the tool never runs, the rest of the arguments never arrive, and the run goes on.
func TestArgHookRefusalEndsTheCallAndAnswersIt(t *testing.T) {
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
	var paths []string
	tools := Tools{WithArgHooks(exec.registry()[0], OnArg("path", func(_ context.Context, path string) *ToolResult {
		paths = append(paths, path)
		return &ToolResult{Content: "write_file: " + path + " already exists", IsError: true}
	}))}

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

	// Called once, with the decoded value, when the path was whole: not once per delta.
	assert.Equal(t, []string{"/work/main.go"}, paths)
	assert.Empty(t, exec.executed, "a refused call never runs")

	// The turn is what streamed before the refusal; the rest of the arguments never arrived.
	require.Len(t, res.Messages, 4)
	assistant := res.Messages[1]
	require.Len(t, assistant.ToolCalls, 1)
	assert.Equal(t, `{"path": "/work/main.go", "content": "pack`, assistant.ToolCalls[0].Arguments)
	assert.Equal(t, Message{Role: RoleTool, Content: "write_file: /work/main.go already exists", ToolCallID: "c1", ToolIsError: true}, res.Messages[2])
	assert.Equal(t, "used edit_file instead", res.Final.Content)

	// The host saw the call and its result the usual way; the call ended complete, not failed.
	require.Len(t, calls, 1)
	assert.Equal(t, "c1", calls[0].ID)
	require.Len(t, results, 1)
	assert.Equal(t, "write_file: /work/main.go already exists", results[0].Result.Content)
	require.Len(t, finals, 2)
	assert.Equal(t, "complete", finals[0].Status)

	// The cut call was not re-sent, and its usage never arrived: calls, the first unreported.
	assert.Equal(t, 2, h.hits)
	require.Len(t, res.Usages, 2)
	assert.Zero(t, res.Usages[0].TotalTokens)
	assert.Equal(t, 55, res.Usages[1].TotalTokens)
	assert.Contains(t, string(h.body[1]), "main.go already exists", "the next request carries the refusal")
}

// A hook that passes changes nothing: the whole arguments arrive and Execute runs. A member no hook
// names, and a value that does not decode as the hook's type, never reach it.
func TestArgHookPassLetsTheCallRunWhole(t *testing.T) {
	h := &turnsSSEHandler{turns: [][]string{
		{
			toolCallChunk(`{"mode": 3, "path": "/work/new.go", `),
			toolCallChunk(`"content": "package new\n", "path2": 7}`),
			`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		},
		{`{"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL}})

	exec := &fakeExec{tools: []ToolDecl{{Name: "write_file"}}}
	var paths, modes []string
	tools := Tools{WithArgHooks(exec.registry()[0],
		OnArg("path", func(_ context.Context, path string) *ToolResult { paths = append(paths, path); return nil }),
		OnArg("mode", func(_ context.Context, mode string) *ToolResult { modes = append(modes, mode); return nil }),
	)}
	res, err := runTurn(context.Background(), Config{Provider: p, Tools: tools, Approver: allowAll},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, []string{"/work/new.go"}, paths)
	assert.Empty(t, modes, "a number does not decode as a string, so Execute reports it")
	require.Len(t, exec.executed, 1)
	assert.Equal(t, `{"mode": 3, "path": "/work/new.go", "content": "package new\n", "path2": 7}`, exec.executed[0].Arguments)
	assert.Equal(t, "done", res.Final.Content)
}

// A turn whose tools have no hooks scans nothing.
func TestArgHooksForIsNilWithoutHooks(t *testing.T) {
	exec := &fakeExec{tools: []ToolDecl{{Name: "a"}, {Name: "b"}}}
	assert.Nil(t, argHooksFor(exec.registry(), exec.tools))
	hooked := Tools{WithArgHooks(exec.registry()[0], ArgHook{Key: "x"}), exec.registry()[1]}
	assert.Nil(t, argHooksFor(hooked, nil), "a turn that advertises nothing runs no call")
	fn := argHooksFor(hooked, exec.tools)
	require.NotNil(t, fn)
	assert.Len(t, fn("a"), 1)
	assert.Empty(t, fn("b"))
	assert.Empty(t, fn("unknown"))
}
