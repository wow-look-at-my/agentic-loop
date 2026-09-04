package commonai

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestDialectRefusedReadsTheEndpointTheServerNamed pins the readings that make a
// model served by another protocol on the same host recoverable.
func TestDialectRefusedReadsTheEndpointTheServerNamed(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		want   Dialect
		ok     bool
	}{
		{
			name:   "openai names responses while refusing chat completions",
			status: 400,
			body:   `{"error":{"message":"This model is not supported in v1/chat/completions. Use v1/responses instead.","type":"invalid_request_error","code":"unsupported_model"}}`,
			want:   DialectResponses,
			ok:     true,
		},
		{
			name:   "only supported in, with no second path to confuse it",
			status: 400,
			body:   `{"error":{"message":"gpt-5-pro is only supported in v1/responses"}}`,
			want:   DialectResponses,
			ok:     true,
		},
		{
			name:   "a gateway points a claude model at the messages API",
			status: 404,
			body:   `{"error":{"message":"claude-sonnet-4-5 must be called through the /v1/messages endpoint"}}`,
			want:   DialectAnthropic,
			ok:     true,
		},
		{
			name:   "the other direction: use chat completions for this one",
			status: 400,
			body:   `{"type":"error","error":{"message":"This model is not available on /v1/messages; use /v1/chat/completions."}}`,
			want:   DialectOpenAI,
			ok:     true,
		},
		{
			name:   "405 on the path itself still names the replacement",
			status: 405,
			body:   `Method Not Allowed. Use v1/responses.`,
			want:   DialectResponses,
			ok:     true,
		},
		{
			name:   "an ordinary bad request names no endpoint",
			status: 400,
			body:   `{"error":{"message":"Unsupported parameter: 'temperature'"}}`,
			want:   DialectAuto,
		},
		{
			name:   "a path mentioned as the thing REFUSED is not a direction",
			status: 400,
			body:   `{"error":{"message":"This model is not supported in v1/chat/completions."}}`,
			want:   DialectAuto,
		},
		{
			name:   "a 500 says nothing about a protocol",
			status: 500,
			body:   `upstream is confused; use v1/responses`,
			want:   DialectAuto,
		},
		{
			name:   "a credential failure is not a protocol failure",
			status: 401,
			body:   `{"error":{"message":"Use v1/responses"}}`,
			want:   DialectAuto,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := DialectRefused(&APIError{Status: tc.status, Body: tc.body})
			assert.Equal(t, tc.ok, ok)
			assert.Equal(t, tc.want, got)
		})
	}
}

// TestDialectRefusedNeedsAnAPIError keeps the reading to what an endpoint
// actually answered: a transport failure carries no body to read.
func TestDialectRefusedNeedsAnAPIError(t *testing.T) {
	_, ok := DialectRefused(errors.New("use v1/responses"))
	assert.False(t, ok, "prose in a local error is not an endpoint speaking")

	_, ok = DialectRefused(nil)
	assert.False(t, ok)
}

// TestDialectRefusedReachesThroughAWrappedError: a host wraps the library error
// in its own text, and the reading has to survive that.
func TestDialectRefusedReachesThroughAWrappedError(t *testing.T) {
	inner := &APIError{Status: 400, Body: `{"error":{"message":"only supported in v1/responses"}}`}
	got, ok := DialectRefused(fmt.Errorf("upstream: chat: %w", inner))
	assert.True(t, ok)
	assert.Equal(t, DialectResponses, got)
}
