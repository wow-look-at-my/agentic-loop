package commonai

import (
	"context"
	"errors"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsTransient(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"408", &APIError{Status: 408}, true},
		{"429", &APIError{Status: 429}, true},
		{"500", &APIError{Status: 500}, true},
		{"524", &APIError{Status: 524}, true},
		{"400", &APIError{Status: 400}, false},
		{"400 overflow", &APIError{Status: 400, ContextOverflow: true}, false},
		{"401", &APIError{Status: 401}, false},
		{"404", &APIError{Status: 404}, false},
		{"canceled", context.Canceled, false},
		{"deadline", context.DeadlineExceeded, false},
		{"wrapped canceled", &net.OpError{Op: "read", Err: context.Canceled}, false},
		{"network", errors.New("connection reset by peer"), true},
		{"wrapped api error", wrapErr(&APIError{Status: 503}), true},
		{"request error", badRequestErr("bad config"), false},
		{"callback error", wrapCallbackErr(errors.New("sink is gone")), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, IsTransient(tc.err))
		})
	}
}

func wrapErr(err error) error {
	return &net.OpError{Op: "http", Err: err}
}

func TestIsContextOverflow(t *testing.T) {
	assert.True(t, IsContextOverflow(&APIError{Status: 400, ContextOverflow: true}))
	assert.False(t, IsContextOverflow(&APIError{Status: 400}))
	assert.False(t, IsContextOverflow(errors.New("nope")))
	assert.False(t, IsContextOverflow(nil))
}

func TestContextOverflowPatterns(t *testing.T) {
	positives := []string{
		"prompt is too long: 210000 tokens > 200000 maximum",
		"Prompt too long for this model",
		"This model's maximum context length is 8192 tokens",
		"context length exceeded",
		"the context window is full",
		"too many tokens in the request",
		"input exceeds the maximum context",
		"request exceeds token limit",
	}
	for _, s := range positives {
		assert.True(t, contextOverflowRe.MatchString(s), s)
	}
	negatives := []string{
		"invalid api key",
		"unsupported parameter: reasoning_effort",
		"model not found",
	}
	for _, s := range negatives {
		assert.False(t, contextOverflowRe.MatchString(s), s)
	}
}
