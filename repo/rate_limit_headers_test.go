package repo

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The whole point: the numbers are already on a response the caller wanted, so
// nothing asks GitHub for them.

func rlHeader(w http.ResponseWriter, resource string, limit, remaining, used int, reset time.Time) {
	h := w.Header()
	if resource != "" {
		h.Set("X-RateLimit-Resource", resource)
	}
	h.Set("X-RateLimit-Limit", strconv.Itoa(limit))
	h.Set("X-RateLimit-Remaining", strconv.Itoa(remaining))
	h.Set("X-RateLimit-Used", strconv.Itoa(used))
	h.Set("X-RateLimit-Reset", strconv.FormatInt(reset.Unix(), 10))
}

func TestReadRateLimitTakesTheCoreBudgetOffAnyResponse(t *testing.T) {
	reset := time.Now().Add(30 * time.Minute).Truncate(time.Second)
	rec := httptest.NewRecorder()
	rlHeader(rec, "core", 5000, 4987, 13, reset)

	now := time.Now()
	s, ok := ReadRateLimit(rec.Header(), now)
	require.True(t, ok)
	assert.True(t, s.OK)
	assert.Equal(t, 5000, s.Limit)
	assert.Equal(t, 4987, s.Remaining)
	assert.Equal(t, 13, s.Used)
	assert.Equal(t, reset, s.ResetAt.UTC().Truncate(time.Second).In(reset.Location()))
	assert.Equal(t, now, s.ObservedAt, "a reader has to know how old these numbers are")
}

// Search spends a separate, much smaller budget. Reporting it as the core
// tells the reader they have requests left when they have thousands.
func TestReadRateLimitIgnoresANonCoreResource(t *testing.T) {
	rec := httptest.NewRecorder()
	rlHeader(rec, "search", 30, 1, 29, time.Now().Add(time.Minute))
	_, ok := ReadRateLimit(rec.Header(), time.Now())
	assert.False(t, ok)
}

// GitHub always names the resource; a server that does not is answering the
// core routes, which is the only thing this package calls.
func TestReadRateLimitTreatsAnAbsentResourceAsCore(t *testing.T) {
	rec := httptest.NewRecorder()
	rlHeader(rec, "", 60, 42, 18, time.Now().Add(time.Minute))
	s, ok := ReadRateLimit(rec.Header(), time.Now())
	require.True(t, ok)
	assert.Equal(t, 42, s.Remaining)
}

func TestReadRateLimitReportsNothingWhenTheHeadersAreAbsent(t *testing.T) {
	_, ok := ReadRateLimit(http.Header{}, time.Now())
	assert.False(t, ok, "absent is not zero remaining")
}

// Used is derived when GitHub omits it, so the field is never a bare
// standing in for a number nobody sent.
func TestReadRateLimitDerivesUsedWhenItIsMissing(t *testing.T) {
	h := http.Header{}
	h.Set("X-RateLimit-Limit", "5000")
	h.Set("X-RateLimit-Remaining", "4000")
	s, ok := ReadRateLimit(h, time.Now())
	require.True(t, ok)
	assert.Equal(t, 1000, s.Used)
}

// An ordinary read reports the quota it spent, naming the credential that
// spent it -- which is what makes a separate probe unnecessary.
func TestAnOrdinaryReadReportsTheQuotaItSpent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer tok-a" {
			rlHeader(w, "core", 5000, 4321, 679, time.Now().Add(time.Hour))
		} else {
			rlHeader(w, "core", 60, 7, 53, time.Now().Add(time.Hour))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"default_branch":"main"}`))
	}))
	defer srv.Close()

	var seen []RateLimitObservation
	e := NewGitHub(GitHubConfig{
		HTTPClient:  srv.Client(),
		APIBaseURL:  srv.URL,
		Tokens:      []GitHubToken{{ID: "t1", Name: "work PAT", Token: "tok-a"}},
		OnRateLimit: func(o RateLimitObservation) { seen = append(seen, o) },
	})
	_, err := e.DefaultBranch(context.Background(), "org", "repo")
	require.NoError(t, err)

	require.Len(t, seen, 1, "one call, one observation")
	assert.Equal(t, "t1", seen[0].CredentialID)
	assert.Equal(t, "work PAT", seen[0].CredentialName)
	assert.False(t, seen[0].Anonymous)
	assert.Equal(t, 4321, seen[0].Status.Remaining)
}

// The unauthenticated bucket is a credential of its own, and the whose
// exhaustion a host most needs to see coming.
func TestAnAnonymousReadReportsTheAnonymousBucket(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rlHeader(w, "core", 60, 7, 53, time.Now().Add(time.Hour))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"default_branch":"main"}`))
	}))
	defer srv.Close()

	var seen []RateLimitObservation
	e := NewGitHub(GitHubConfig{
		HTTPClient:  srv.Client(),
		APIBaseURL:  srv.URL,
		OnRateLimit: func(o RateLimitObservation) { seen = append(seen, o) },
	})
	_, err := e.DefaultBranch(context.Background(), "org", "repo")
	require.NoError(t, err)

	require.Len(t, seen, 1)
	assert.True(t, seen[0].Anonymous)
	assert.Empty(t, seen[0].CredentialID)
	assert.Equal(t, 7, seen[0].Status.Remaining)
}

// The token test already asked GitHub something; its answer carries the quota.
func TestTestTokenCarriesTheQuotaItsOwnProbeReported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rlHeader(w, "core", 5000, 4900, 100, time.Now().Add(time.Hour))
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/user":
			_, _ = w.Write([]byte(`{"login":"someone"}`))
		default:
			_, _ = w.Write([]byte(`[]`))
		}
	}))
	defer srv.Close()

	got := TestToken(context.Background(), srv.URL, "tok", srv.Client())
	require.True(t, got.OK, got.Error)
	assert.True(t, got.RateLimit.OK)
	assert.Equal(t, 4900, got.RateLimit.Remaining)
}

// A failed test still reports the quota: a token rejected while the budget is
// spent is a different problem from rejected with budget to spare.
func TestTestTokenCarriesTheQuotaEvenWhenTheTokenIsRejected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		rlHeader(w, "core", 5000, 11, 4989, time.Now().Add(time.Hour))
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	got := TestToken(context.Background(), srv.URL, "tok", srv.Client())
	require.False(t, got.OK)
	assert.Equal(t, 11, got.RateLimit.Remaining)
}
