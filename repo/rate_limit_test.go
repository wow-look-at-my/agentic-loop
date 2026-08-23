package repo

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRateLimitReportsTheCoreBudget(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/rate_limit", r.URL.Path)
		assert.Equal(t, "Bearer good-token", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"resources":{"core":{"limit":5000,"remaining":4987,"used":13,"reset":1700000000}}}`))
	}))
	t.Cleanup(srv.Close)

	res := RateLimit(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	assert.Equal(t, 5000, res.Limit)
	assert.Equal(t, 4987, res.Remaining)
	assert.Equal(t, 13, res.Used)
	assert.Equal(t, time.Unix(1700000000, 0), res.ResetAt)
}

// An empty token probes anonymously: no Authorization header at all, never a
// blank Bearer value -- GitHub buckets that request by the caller's IP, which
// is the whole point of the anonymous probe.
func TestRateLimitAnonymousProbeSendsNoAuthorizationHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Empty(t, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"resources":{"core":{"limit":60,"remaining":58,"used":2,"reset":1700000000}}}`))
	}))
	t.Cleanup(srv.Close)

	res := RateLimit(context.Background(), srv.URL, "", srv.Client())
	require.True(t, res.OK, res.Error)
	assert.Equal(t, 60, res.Limit)
	assert.Equal(t, 58, res.Remaining)
}

func TestRateLimitReportsAnUnauthorizedToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"message":"Bad credentials"}`))
	}))
	t.Cleanup(srv.Close)

	res := RateLimit(context.Background(), srv.URL, "bad-token", srv.Client())
	assert.False(t, res.OK)
	assert.Contains(t, res.Error, "Bad credentials")
}

func TestRateLimitReportsAnUnreachableServer(t *testing.T) {
	res := RateLimit(context.Background(), "http://127.0.0.1:1", "good-token", http.DefaultClient)
	assert.False(t, res.OK)
	assert.Contains(t, res.Error, "could not reach GitHub")
}
