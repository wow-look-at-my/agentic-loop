package commonai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// syntheticModelList is an excerpt of what api.synthetic.new/v1/models answers.
const syntheticModelList = `{"data":[{"provider":"synthetic","display_name":"DeepSeek V4.1 Flash",
 "id":"hf:deepseek-ai/DeepSeek-V4.1-Flash","context_length":524288,"max_output_length":65536,
 "pricing":{"prompt":"$0.0000006","completion":"$0.0000012"}}]}`

func TestTheModelListCarriesEachModelsLimits(t *testing.T) {
	list := decoded(t, syntheticModelList)
	assert.Equal(t, Limits{ContextWindow: 524288, MaxOutput: 65536},
		list.Limits["hf:deepseek-ai/DeepSeek-V4.1-Flash"])
}

func TestEverySpellingOfALimitIsRead(t *testing.T) {
	list := decoded(t, `{"object":"list","data":[
	  {"id":"groq","context_window":131072,"max_completion_tokens":32768},
	  {"id":"lmstudio","max_context_length":32768},
	  {"id":"vllm","max_model_len":65536},
	  {"id":"anthropic","max_input_tokens":1000000,"max_tokens":128000},
	  {"id":"openrouter","context_length":200000,"top_provider":{"context_length":100000,"max_completion_tokens":8192}},
	  {"id":"provider-only","top_provider":{"context_length":100000,"max_completion_tokens":8192}},
	  {"id":"output-only","max_output_tokens":4096},
	  {"id":"string","context_length":"8192"}
	]}`)
	assert.Equal(t, map[string]Limits{
		"groq":          {ContextWindow: 131072, MaxOutput: 32768},
		"lmstudio":      {ContextWindow: 32768},
		"vllm":          {ContextWindow: 65536},
		"anthropic":     {ContextWindow: 1000000, MaxOutput: 128000},
		"openrouter":    {ContextWindow: 200000, MaxOutput: 8192},
		"provider-only": {ContextWindow: 100000, MaxOutput: 8192},
		"output-only":   {MaxOutput: 4096},
		"string":        {ContextWindow: 8192},
	}, list.Limits)
}

func TestAModelThatPublishesNoLimitIsAbsent(t *testing.T) {
	list := decoded(t, `{"object":"list","data":[{"id":"bare"},{"id":"nulls","context_length":null},
	  {"id":"zero","context_length":0}]}`)
	assert.Empty(t, list.Limits)
}

func TestALimitThatIsNotACountFailsTheDecode(t *testing.T) {
	for _, doc := range []string{
		`{"object":"list","data":[{"id":"m","context_length":"lots"}]}`,
		`{"object":"list","data":[{"id":"m","context_length":-1}]}`,
		`{"object":"list","data":[{"id":"m","context_length":1.5}]}`,
		`{"object":"list","data":[{"id":"m","context_length":true}]}`,
		`{"object":"list","data":[{"id":"m","max_tokens":1e12}]}`,
	} {
		_, err := DecodeModelList([]byte(doc))
		assert.Error(t, err, doc)
	}
}

func modelListServer(t *testing.T, body string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		hits.Add(1)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func TestEachBuiltInProviderReadsItsOwnModelListOnce(t *testing.T) {
	srv, hits := modelListServer(t, syntheticModelList)
	base := ProviderConfig{BaseURL: srv.URL + "/v1", HTTPClient: srv.Client()}
	oa, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: base})
	require.NoError(t, err)
	rp, err := NewResponsesProvider(ResponsesConfig{ProviderConfig: base})
	require.NoError(t, err)
	an, err := NewAnthropicProvider(AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()}})
	require.NoError(t, err)

	for _, p := range []Provider{oa, rp, an, NewParamStripper(oa), NewThinkingSignatureRepair(an)} {
		l, ok, err := ModelLimitsOf(context.Background(), p, "hf:deepseek-ai/DeepSeek-V4.1-Flash")
		require.NoError(t, err)
		assert.True(t, ok)
		assert.Equal(t, 524288, l.ContextWindow)
	}
	assert.Equal(t, int32(3), hits.Load(), "one read per provider; a decorator asks the provider it wraps")
}

func TestAFailedModelListReadIsAskedAgain(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if hits.Add(1) == 1 {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte(syntheticModelList))
	}))
	defer srv.Close()
	p, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()}})
	require.NoError(t, err)

	_, _, err = ModelLimitsOf(context.Background(), p, "hf:deepseek-ai/DeepSeek-V4.1-Flash")
	require.Error(t, err)
	l, _, err := ModelLimitsOf(context.Background(), p, "hf:deepseek-ai/DeepSeek-V4.1-Flash")
	require.NoError(t, err)
	assert.Equal(t, 524288, l.ContextWindow)
}

type plainProvider struct{}

func (plainProvider) Complete(context.Context, Request, *StreamEvents) (*Completion, error) {
	return nil, nil
}

func TestAProviderWithNoModelListCannotSay(t *testing.T) {
	_, ok, err := ModelLimitsOf(context.Background(), plainProvider{}, "m")
	require.NoError(t, err)
	assert.False(t, ok)
}
