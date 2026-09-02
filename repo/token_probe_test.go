package repo

import (
	"context"
	"fmt"
	"github.com/wow-look-at-my/agentic-loop/internal/jsontest"
	"github.com/wow-look-at-my/go-containers/set"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTestTokenReportsTheAuthenticatedLogin(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer good-token", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch r.URL.Path {
		case "/user":
			_, _ = w.Write([]byte(`{"login":"octocat"}`))
		case "/user/repos", "/user/orgs":
			_, _ = w.Write([]byte(`[]`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	assert.Equal(t, "octocat", res.Login)
	assert.Empty(t, res.Warning)
	assert.Empty(t, res.Repos)
	assert.Empty(t, res.Orgs)
}

func TestTestTokenListsVisibleReposWithPermissions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch r.URL.Path {
		case "/user":
			_, _ = w.Write([]byte(`{"login":"octocat"}`))
		case "/user/repos":
			assert.Equal(t, "1", r.URL.Query().Get("page"))
			_, _ = w.Write([]byte(`[
				{"full_name":"octo/admin-repo","private":true,"permissions":{"admin":true,"maintain":true,"push":true,"triage":true,"pull":true}},
				{"full_name":"octo/read-only-repo","private":false,"permissions":{"admin":false,"maintain":false,"push":false,"triage":false,"pull":true}}
			]`))
		case "/user/orgs":
			_, _ = w.Write([]byte(`[]`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	require.Empty(t, res.ReposError)
	require.False(t, res.ReposTruncated)
	require.Len(t, res.Repos, 2)
	assert.Equal(t, TokenTestRepo{FullName: "octo/admin-repo", Private: true, Admin: true, Maintain: true, Push: true, Triage: true, Pull: true}, res.Repos[0])
	assert.Equal(t, TokenTestRepo{FullName: "octo/read-only-repo", Private: false, Pull: true}, res.Repos[1])
}

func TestTestTokenPaginatesReposUpToTheCapAndReportsTruncation(t *testing.T) {
	repo := `{"full_name":"octo/repo","private":false,"permissions":{"admin":false,"maintain":false,"push":false,"triage":false,"pull":true}}`
	var fullPage []byte
	fullPage = append(fullPage, '[')
	for i := 0; i < OwnerReposPerPage; i++ {
		if i > 0 {
			fullPage = append(fullPage, ',')
		}
		fullPage = append(fullPage, repo...)
	}
	fullPage = append(fullPage, ']')

	var pagesSeen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch r.URL.Path {
		case "/user":
			_, _ = w.Write([]byte(`{"login":"octocat"}`))
		case "/user/repos":
			pagesSeen = append(pagesSeen, r.URL.Query().Get("page"))
			_, _ = w.Write(fullPage)
		case "/user/orgs":
			_, _ = w.Write([]byte(`[]`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	require.Empty(t, res.ReposError)
	assert.True(t, res.ReposTruncated, "a full page every time means more remain past the cap")
	assert.Len(t, res.Repos, OwnerReposMaxPages*OwnerReposPerPage)
	assert.Len(t, pagesSeen, OwnerReposMaxPages, "stops at the page cap instead of paginating forever")
}

func TestTestTokenReportsARepoListingFailureWithoutFailingTheTokenItself(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/user":
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"login":"octocat"}`))
		case "/user/repos":
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"message":"access blocked"}`))
		case "/user/orgs":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[]`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, "the token itself is fine; only listing its repos failed")
	assert.Equal(t, "octocat", res.Login)
	assert.Empty(t, res.Repos)
	assert.Contains(t, res.ReposError, "access blocked")
}

func TestTestTokenReportsANearExpiryWarningOnSuccess(t *testing.T) {
	expiry := time.Now().Add(2 * 24 * time.Hour)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("GitHub-Authentication-Token-Expiration", expiry.UTC().Format(githubTokenExpirationLayout))
		w.WriteHeader(http.StatusOK)
		if r.URL.Path == "/user/repos" || r.URL.Path == "/user/orgs" {
			_, _ = w.Write([]byte(`[]`))
			return
		}
		_, _ = w.Write([]byte(`{"login":"octocat"}`))
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	assert.Contains(t, res.Warning, "rotate it")
}

func TestTestTokenReportsRejectedCredential(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"message":"Bad credentials"}`))
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "bad-token", srv.Client())
	assert.False(t, res.OK)
	assert.Contains(t, res.Error, "rejected this token (401)")
	assert.Contains(t, res.Error, `"Bad credentials"`)
}

func TestTestTokenReportsRateLimitAsTransient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-RateLimit-Remaining", "0")
		w.Header().Set("X-RateLimit-Reset", "9999999999")
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"message":"API rate limit exceeded"}`))
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "tok", srv.Client())
	assert.False(t, res.OK)
	assert.Contains(t, res.Error, "TRANSIENT")
	assert.NotContains(t, res.Error, "denied", "a rate limit is not a verdict on the token itself")
}

// This never rotates to another token or falls back to anonymous — it must
// test exactly the credential it was given, even when that credential fails.
func TestTestTokenNeverFallsBackToAnonymous(t *testing.T) {
	var authSeen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authSeen = append(authSeen, r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"message":"Bad credentials"}`))
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "bad-token", srv.Client())
	assert.False(t, res.OK)
	assert.Equal(t, []string{"Bearer bad-token"}, authSeen, "exactly one request, with exactly the given token")
}

// --- organization enumeration -----------------------------------------------
//
// "list every repo you can see" has to mean every ORG you can see too: a
// fine-grained PAT scoped to read/write all of an org's contents can still
// fail to self-report that org membership via /user/orgs, and /user/repos'
// organization_member affiliation is membership-based, not grant-based. These
// prove the org sweep actually runs and actually plugs that gap, rather than
// just trusting /user/repos to have caught everything.

func TestTestTokenListsOrganizationsDiscoveredViaUserOrgs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch {
		case r.URL.Path == "/user":
			_, _ = w.Write([]byte(`{"login":"octocat"}`))
		case r.URL.Path == "/user/repos":
			_, _ = w.Write([]byte(`[]`))
		case r.URL.Path == "/user/orgs":
			_, _ = w.Write([]byte(`[{"login":"PazerOP"}]`))
		case r.URL.Path == "/orgs/PazerOP/repos":
			_, _ = w.Write([]byte(`[{"full_name":"PazerOP/UE553","private":true,"permissions":{"admin":false,"maintain":false,"push":true,"triage":true,"pull":true}}]`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	require.Empty(t, res.OrgsError)
	require.Len(t, res.Orgs, 1)
	assert.Equal(t, "PazerOP", res.Orgs[0].Login)
	require.Len(t, res.Orgs[0].Repos, 1)
	assert.Equal(t, "PazerOP/UE553", res.Orgs[0].Repos[0].FullName)

	// A repo the flat /user/repos listing missed still lands in the Repos union.
	require.Len(t, res.Repos, 1)
	assert.Equal(t, "PazerOP/UE553", res.Repos[0].FullName)
}

// The exact gap this feature exists to close: /user/orgs fails outright (the
// common case for a fine-grained PAT scoped only to repository contents), but
// the org still gets swept because an Organization-type repo owner from the
// flat listing names it.
func TestTestTokenFallsBackToRepoOwnersWhenUserOrgsFails(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/user":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"login":"octocat"}`))
		case r.URL.Path == "/user/repos":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[{"full_name":"PazerOP/UE553","private":true,"owner":{"login":"PazerOP","type":"Organization"},"permissions":{"admin":false,"maintain":false,"push":true,"triage":true,"pull":true}}]`))
		case r.URL.Path == "/user/orgs":
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"message":"Resource not accessible by personal access token"}`))
		case r.URL.Path == "/orgs/PazerOP/repos":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[
				{"full_name":"PazerOP/UE553","private":true,"permissions":{"admin":false,"maintain":false,"push":true,"triage":true,"pull":true}},
				{"full_name":"PazerOP/second-repo","private":true,"permissions":{"admin":false,"maintain":false,"push":false,"triage":false,"pull":true}}
			]`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	assert.NotEmpty(t, res.OrgsError, "the token itself is fine, but it could not self-report org memberships")

	require.Len(t, res.Orgs, 1, "PazerOP was still swept, discovered from the repo owner instead of /user/orgs")
	assert.Equal(t, "PazerOP", res.Orgs[0].Login)
	assert.Empty(t, res.Orgs[0].Error)
	require.Len(t, res.Orgs[0].Repos, 2)

	// PazerOP/-repo was found only by the org sweep.
	fullNames := []string{res.Repos[0].FullName, res.Repos[1].FullName}
	assert.ElementsMatch(t, []string{"PazerOP/UE553", "PazerOP/second-repo"}, fullNames)
}

func TestTestTokenReportsAPerOrgListingFailureWithoutFailingTheToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/user":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"login":"octocat"}`))
		case r.URL.Path == "/user/repos":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[]`))
		case r.URL.Path == "/user/orgs":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[{"login":"blocked-org"}]`))
		case r.URL.Path == "/orgs/blocked-org/repos":
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"message":"SAML enforcement"}`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	require.Len(t, res.Orgs, 1)
	assert.Equal(t, "blocked-org", res.Orgs[0].Login)
	assert.Empty(t, res.Orgs[0].Repos)
	assert.Contains(t, res.Orgs[0].Error, "SAML enforcement")
}

func TestTestTokenCapsTheNumberOfOrganizationsSwept(t *testing.T) {
	const orgCount = orgSweepMaxOrgs + 5
	var orgs jsontest.Arr
	for i := 0; i < orgCount; i++ {
		orgs = append(orgs, jsontest.Obj{"login": fmt.Sprintf("org%02d", i)})
	}
	orgsBody := jsontest.Must(orgs)

	orgReposSeen := set.New[string]()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch {
		case r.URL.Path == "/user":
			_, _ = w.Write([]byte(`{"login":"octocat"}`))
		case r.URL.Path == "/user/repos":
			_, _ = w.Write([]byte(`[]`))
		case r.URL.Path == "/user/orgs":
			_, _ = w.Write([]byte(orgsBody))
		default:
			orgReposSeen.Add(r.URL.Path)
			_, _ = w.Write([]byte(`[]`))
		}
	}))
	t.Cleanup(srv.Close)

	res := TestToken(context.Background(), srv.URL, "good-token", srv.Client())
	require.True(t, res.OK, res.Error)
	assert.True(t, res.OrgsTruncated)
	assert.Len(t, res.Orgs, orgSweepMaxOrgs)
	assert.Equal(t, orgSweepMaxOrgs, orgReposSeen.Len(), "only the capped set is actually swept, not every discovered org")
}
