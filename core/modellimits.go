package commonai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"sync"
)

// ModelLimiter is a Provider that can say what its endpoint's model list
// publishes about a model's token limits.
type ModelLimiter interface {
	// ModelLimits answers the zero Limits, not an error, for a model the list does not limit.
	ModelLimits(ctx context.Context, model string) (Limits, error)
}

// ModelLimitsOf asks p for model's limits. ok is false when p is not a ModelLimiter and so has nobody to ask.
func ModelLimitsOf(ctx context.Context, p Provider, model string) (l Limits, ok bool, err error) {
	ml, ok := p.(ModelLimiter)
	if !ok {
		return Limits{}, false, nil
	}
	l, err = ml.ModelLimits(ctx, model)
	return l, true, err
}

// modelListCache reads an endpoint's model list once. A failed read is not kept, so the next question asks again.
type modelListCache struct {
	mu   sync.Mutex
	list *ModelList
}

func (c *modelListCache) limits(ctx context.Context, cfg ProviderConfig, model string) (Limits, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.list == nil {
		list, err := FetchModelList(ctx, cfg)
		if err != nil {
			return Limits{}, err
		}
		c.list = list
	}
	return c.list.Limits[model], nil
}

func (o *openaiProvider) ModelLimits(ctx context.Context, model string) (Limits, error) {
	return o.models.limits(ctx, ProviderConfig{
		BaseURL: o.baseURL, APIKey: o.apiKey, HTTPClient: o.httpClient, UserAgent: o.userAgent, Headers: o.headers,
	}, model)
}

func (o *responsesProvider) ModelLimits(ctx context.Context, model string) (Limits, error) {
	return o.models.limits(ctx, ProviderConfig{
		BaseURL: o.baseURL, APIKey: o.apiKey, HTTPClient: o.httpClient, UserAgent: o.userAgent, Headers: o.headers,
	}, model)
}

func (a *anthropicProvider) ModelLimits(ctx context.Context, model string) (Limits, error) {
	return a.models.limits(ctx, ProviderConfig{
		BaseURL: a.baseURL, APIKey: a.apiKey, HTTPClient: a.httpClient, UserAgent: a.userAgent, Headers: a.headers,
	}, model)
}

func (s *paramStripper) ModelLimits(ctx context.Context, model string) (Limits, error) {
	l, _, err := ModelLimitsOf(ctx, s.inner, model)
	return l, err
}

func (r *thinkingSignatureRepair) ModelLimits(ctx context.Context, model string) (Limits, error) {
	l, _, err := ModelLimitsOf(ctx, r.inner, model)
	return l, err
}

// modelListLimits is every spelling a model list uses for a model's token limits. Each provider picks its own name for
// the same number, so every one is read and the first that is present wins.
type modelListLimits struct {
	// OpenRouter, Together, Fireworks, Synthetic.
	ContextLength *tokenCount `json:"context_length"`
	// Groq.
	ContextWindow *tokenCount `json:"context_window"`
	// LM Studio.
	MaxContextLength *tokenCount `json:"max_context_length"`
	// vLLM.
	MaxModelLen *tokenCount `json:"max_model_len"`
	// Anthropic.
	MaxInputTokens *tokenCount `json:"max_input_tokens"`

	// Synthetic.
	MaxOutputLength *tokenCount `json:"max_output_length"`
	// Groq.
	MaxCompletionTokens *tokenCount `json:"max_completion_tokens"`
	MaxOutputTokens     *tokenCount `json:"max_output_tokens"`
	// Anthropic.
	MaxTokens *tokenCount `json:"max_tokens"`

	// OpenRouter repeats both limits for the provider that serves the model.
	TopProvider *struct {
		ContextLength       *tokenCount `json:"context_length"`
		MaxCompletionTokens *tokenCount `json:"max_completion_tokens"`
	} `json:"top_provider"`
}

func (m modelListLimits) limits() Limits {
	var tpContext, tpOutput *tokenCount
	if m.TopProvider != nil {
		tpContext, tpOutput = m.TopProvider.ContextLength, m.TopProvider.MaxCompletionTokens
	}
	return Limits{
		ContextWindow: firstCount(m.ContextLength, m.ContextWindow, m.MaxContextLength, m.MaxModelLen, m.MaxInputTokens, tpContext),
		MaxOutput:     firstCount(m.MaxOutputLength, m.MaxCompletionTokens, m.MaxOutputTokens, m.MaxTokens, tpOutput),
	}
}

func firstCount(counts ...*tokenCount) int {
	for _, c := range counts {
		if c != nil && *c > 0 {
			return int(*c)
		}
	}
	return 0
}

// tokenCount is a token limit as a model list writes it: a JSON number, a numeric string, or null.
type tokenCount int

func (c *tokenCount) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if bytes.Equal(b, []byte("null")) {
		return nil
	}
	if len(b) > 0 && b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		b = []byte(s)
	}
	f, err := strconv.ParseFloat(string(b), 64)
	if err != nil {
		return fmt.Errorf("token limit %s is not a number", b)
	}
	if f < 0 || f != math.Trunc(f) || f > math.MaxInt32 {
		return fmt.Errorf("token limit %s is not a whole token count", b)
	}
	*c = tokenCount(f)
	return nil
}
