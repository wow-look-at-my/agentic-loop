package loop

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// countingServer reports prompt tokens as a quarter of the message bytes it
// reads. A lossy server deletes the reasoning fields first, as an endpoint
// does when it ignores them.
func countingServer(t *testing.T, lossy bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body struct {
			Messages []map[string]any `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(raw, &body))
		if lossy {
			for _, m := range body.Messages {
				delete(m, "reasoning")
				delete(m, "reasoning_content")
				delete(m, "reasoning_details")
			}
		}
		counted, _ := json.Marshal(body.Messages)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"length\"}]}\n\n"+
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":1,\"total_tokens\":%d}}\n\n"+
			"data: [DONE]\n\n", len(counted)/4, len(counted)/4+1)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func probeAgainst(t *testing.T, lossy bool) *ReplayProbe {
	t.Helper()
	p, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: countingServer(t, lossy).URL}})
	require.NoError(t, err)
	probe, err := ProbeReasoningReplay(context.Background(), p, "m")
	require.NoError(t, err)
	require.Len(t, probe.Completions, 2, "both probe calls are surfaced for billing")
	return probe
}

func TestProbeReasoningReplayFaithfulEndpoint(t *testing.T) {
	probe := probeAgainst(t, false)
	assert.True(t, probe.Kept)
	assert.GreaterOrEqual(t, probe.WithReasoning-probe.WithoutReasoning, probe.MinGrowth)
}

func TestProbeReasoningReplayLossyEndpoint(t *testing.T) {
	probe := probeAgainst(t, true)
	assert.False(t, probe.Kept, "an endpoint that ignores the reasoning fields counts the same prompt twice")
	assert.Equal(t, probe.WithReasoning, probe.WithoutReasoning)
}

func TestProbeReasoningReplayNeedsUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	p, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL}})
	require.NoError(t, err)
	_, err = ProbeReasoningReplay(context.Background(), p, "m")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no usage")
}

func TestProbeReasoningReplayNeedsProvider(t *testing.T) {
	_, err := ProbeReasoningReplay(context.Background(), nil, "m")
	require.Error(t, err)
}

func TestProbeReasoningReplayCallFails(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	}))
	defer srv.Close()
	p, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL}})
	require.NoError(t, err)
	_, err = ProbeReasoningReplay(context.Background(), p, "m")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "call with reasoning")
}
