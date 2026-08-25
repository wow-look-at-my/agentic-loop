package repo

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/wow-look-at-my/go-containers/set"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// TestToken probes exactly one credential's real health against GitHub's /user endpoint, plus its repos and orgs.
type TokenTestResult struct {
	OK             bool            `json:"ok"`
	Login          string          `json:"login,omitempty"`   // authenticated GitHub login, when ok
	Warning        string          `json:"warning,omitempty"` // e.g. a near-expiry note, present even when ok
	Error          string          `json:"error,omitempty"`   // human-readable reason, when not ok
	Repos          []TokenTestRepo `json:"repos,omitempty"`
	ReposTruncated bool            `json:"repos_truncated,omitempty"` // more repos exist past the page cap
	ReposError     string          `json:"repos_error,omitempty"`     // the token itself is OK, but listing its repos failed
	Orgs           []TokenTestOrg  `json:"orgs,omitempty"`
	OrgsError      string          `json:"orgs_error,omitempty"`     // GET /user/orgs failed; Orgs may still be non-empty from repo owners
	OrgsTruncated  bool            `json:"orgs_truncated,omitempty"` // more organizations exist past orgSweepMaxOrgs
}

// TokenTestRepo is one repository visible to the tested token, with the
// permission levels GitHub reports for it in a repo-list response's per-repo
// "permissions" object (shared by /user/repos and /orgs/{org}/repos).
type TokenTestRepo struct {
	FullName string `json:"full_name"`
	Private  bool   `json:"private"`
	Admin    bool   `json:"admin"`
	Maintain bool   `json:"maintain"`
	Push     bool   `json:"push"`
	Triage   bool   `json:"triage"`
	Pull     bool   `json:"pull"`
}

// TokenTestOrg is one organization the tested token can see, with that org's
// own repository listing (a direct GET /orgs/{org}/repos, not an inference
// from the flat Repos list).
type TokenTestOrg struct {
	Login     string          `json:"login"`
	Repos     []TokenTestRepo `json:"repos,omitempty"`
	Truncated bool            `json:"truncated,omitempty"`
	Error     string          `json:"error,omitempty"` // this org's own repos could not be listed
}

// TestToken issues the probe. apiBase and httpClient mirror GitHubConfig
// (empty apiBase defaults to https://api.github.com) so a caller outside this
// package's tool wiring — a Settings API handler — can use it without
// building a client over the user's whole token list.
func TestToken(ctx context.Context, apiBase, token string, httpClient *http.Client) TokenTestResult {
	e := NewGitHub(GitHubConfig{HTTPClient: httpClient, APIBaseURL: apiBase})
	now := time.Now()
	res, err := e.doGet(ctx, e.base+"/user", token, "application/vnd.github+json")
	if err != nil {
		return TokenTestResult{Error: "could not reach GitHub: " + err.Error()}
	}
	if rl, limited := classifyRateLimit(res, now); limited {
		kind := "rate limit"
		if rl.secondary {
			kind = "secondary rate limit"
		}
		return TokenTestResult{Error: fmt.Sprintf("GitHub %s exceeded (%d) — TRANSIENT, not a problem with this token. %s.", kind, res.status, rl.waitAdvice())}
	}
	switch {
	case res.status >= 200 && res.status < 300:
		var user struct {
			Login string `json:"login"`
		}
		_ = json.Unmarshal(res.body, &user)
		result := TokenTestResult{OK: true, Login: user.Login, Warning: strings.TrimSpace(tokenExpiryDetail(res, now))}
		repos, orgOwners, truncated, reposErr := e.listVisibleRepos(ctx, token)
		result.Repos, result.ReposTruncated = repos, truncated
		if reposErr != "" {
			result.ReposError = reposErr
		}
		e.sweepOrgs(ctx, token, orgOwners, &result)
		return result
	case res.status == http.StatusUnauthorized:
		return TokenTestResult{Error: "GitHub rejected this token (401) — invalid, revoked, or expired; the response doesn't say which." + authRejectionDetail(res, now)}
	case res.status == http.StatusForbidden:
		return TokenTestResult{Error: "GitHub accepted this token but denied the request (403)." + missingPermissionDetail(res, now)}
	default:
		if msg := GitHubErrorMessage(res.body); msg != "" {
			return TokenTestResult{Error: fmt.Sprintf("GitHub returned %d: %s", res.status, msg)}
		}
		return TokenTestResult{Error: fmt.Sprintf("GitHub returned status %d", res.status)}
	}
}

// orgSweepMaxOrgs caps how many organizations get their own /orgs/{org}/repos sweep.
const orgSweepMaxOrgs = 20

// listVisibleRepos enumerates every repository the token can see via
// GET /user/repos, paginated up to OwnerReposMaxPages full pages of
// OwnerReposPerPage each — the same page size and cap OwnerRepos uses for
// an owner listing. It reports truncated=true when more pages remain past
// the cap, and a non-empty error string when a page could not be fetched (the
// token itself already passed the /user check, so this is reported alongside
// OK rather than failing the whole test). orgOwners collects the distinct
// Organization-type repo owners seen, in first-seen order, as a fallback
// source of organizations to sweep when the token can't self-report its org
// memberships via /user/orgs.
func (e *GitHub) listVisibleRepos(ctx context.Context, token string) (repos []TokenTestRepo, orgOwners []string, truncated bool, errMsg string) {
	now := time.Now()
	seenOrg := set.New[string]()
	for page := 1; page <= OwnerReposMaxPages; page++ {
		target := fmt.Sprintf("%s/user/repos?per_page=%d&page=%d&sort=full_name", e.base, OwnerReposPerPage, page)
		res, err := e.doGet(ctx, target, token, "application/vnd.github+json")
		if err != nil {
			return repos, orgOwners, false, "could not reach GitHub: " + err.Error()
		}
		if res.status < 200 || res.status >= 300 {
			return repos, orgOwners, false, explainFailure("list", "the repositories visible to this token", res, 1, now)
		}
		var batch []ghUserRepo
		if err := json.Unmarshal(res.body, &batch); err != nil {
			return repos, orgOwners, false, "could not parse GitHub's repository list: " + err.Error()
		}
		for _, r := range batch {
			repos = append(repos, r.tokenTestRepo())
			if r.Owner.Type == "Organization" && r.Owner.Login != "" && seenOrg.Add(strings.ToLower(r.Owner.Login)) {
				orgOwners = append(orgOwners, r.Owner.Login)
			}
		}
		if len(batch) < OwnerReposPerPage {
			return repos, orgOwners, false, ""
		}
	}
	return repos, orgOwners, true, ""
}

// sweepOrgs discovers every organization the token can see — GET /user/orgs,
// unioned with orgOwners (the Organization-type owners already found in the
// flat repo listing, which needs no org-level permission at all) — and gives
// each its own GET /orgs/{org}/repos listing in result.Orgs. A repo that
// sweep finds but the flat /user/repos listing missed is folded into
// result.Repos too (deduplicated by full_name), so Repos stays the complete
// union regardless of which source actually saw it.
func (e *GitHub) sweepOrgs(ctx context.Context, token string, orgOwners []string, result *TokenTestResult) {
	discovered, orgsErr := e.listVisibleOrgs(ctx, token)
	if orgsErr != "" {
		result.OrgsError = orgsErr
	}

	orgs := mergeOrgLogins(discovered, orgOwners)
	if len(orgs) > orgSweepMaxOrgs {
		orgs = orgs[:orgSweepMaxOrgs]
		result.OrgsTruncated = true
	}

	seenRepo := set.New[string](len(result.Repos))
	for _, r := range result.Repos {
		seenRepo.Add(strings.ToLower(r.FullName))
	}

	for _, org := range orgs {
		orgRepos, truncated, orgErr := e.listOrgRepos(ctx, token, org)
		result.Orgs = append(result.Orgs, TokenTestOrg{Login: org, Repos: orgRepos, Truncated: truncated, Error: orgErr})
		for _, r := range orgRepos {
			if seenRepo.Add(strings.ToLower(r.FullName)) {
				result.Repos = append(result.Repos, r)
			}
		}
	}
}

// mergeOrgLogins unions two organization-login lists, case-insensitively
// deduplicated, preferring discovered's casing (it comes straight from
// GitHub's own org listing) and preserving first-seen order.
func mergeOrgLogins(discovered, fromRepos []string) []string {
	seen := set.New[string](len(discovered) + len(fromRepos))
	var out []string
	for _, login := range discovered {
		if seen.Add(strings.ToLower(login)) {
			out = append(out, login)
		}
	}
	for _, login := range fromRepos {
		if seen.Add(strings.ToLower(login)) {
			out = append(out, login)
		}
	}
	sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i]) < strings.ToLower(out[j]) })
	return out
}

// listVisibleOrgs lists the organizations the token can self-report via
// GET /user/orgs, paginated the same way as listVisibleRepos. Many
// fine-grained PATs scoped only to repository contents cannot read their own
// org memberships at all — that failure is expected and non-fatal here
// (sweepOrgs still covers those orgs via the repo owners it already found),
// so it is reported through errMsg rather than treated as the token failing.
func (e *GitHub) listVisibleOrgs(ctx context.Context, token string) (logins []string, errMsg string) {
	now := time.Now()
	for page := 1; page <= OwnerReposMaxPages; page++ {
		target := fmt.Sprintf("%s/user/orgs?per_page=%d&page=%d", e.base, OwnerReposPerPage, page)
		res, err := e.doGet(ctx, target, token, "application/vnd.github+json")
		if err != nil {
			return logins, "could not reach GitHub: " + err.Error()
		}
		if res.status < 200 || res.status >= 300 {
			return logins, explainFailure("list", "the organizations visible to this token", res, 1, now)
		}
		var batch []struct {
			Login string `json:"login"`
		}
		if err := json.Unmarshal(res.body, &batch); err != nil {
			return logins, "could not parse GitHub's organization list: " + err.Error()
		}
		for _, o := range batch {
			logins = append(logins, o.Login)
		}
		if len(batch) < OwnerReposPerPage {
			return logins, ""
		}
	}
	return logins, ""
}

// listOrgRepos lists one organization's repositories directly (GET
// /orgs/{org}/repos, type=all so it covers everything the token itself can
// see rather than only public ones), paginated the same way as
// listVisibleRepos.
func (e *GitHub) listOrgRepos(ctx context.Context, token, org string) (repos []TokenTestRepo, truncated bool, errMsg string) {
	now := time.Now()
	for page := 1; page <= OwnerReposMaxPages; page++ {
		target := fmt.Sprintf("%s/orgs/%s/repos?type=all&per_page=%d&page=%d&sort=full_name", e.base, url.PathEscape(org), OwnerReposPerPage, page)
		res, err := e.doGet(ctx, target, token, "application/vnd.github+json")
		if err != nil {
			return repos, false, "could not reach GitHub: " + err.Error()
		}
		if res.status < 200 || res.status >= 300 {
			return repos, false, explainFailure("list", org+"'s repositories", res, 1, now)
		}
		var batch []ghUserRepo
		if err := json.Unmarshal(res.body, &batch); err != nil {
			return repos, false, "could not parse GitHub's repository list: " + err.Error()
		}
		for _, r := range batch {
			repos = append(repos, r.tokenTestRepo())
		}
		if len(batch) < OwnerReposPerPage {
			return repos, false, ""
		}
	}
	return repos, true, ""
}

// ghUserRepo is one repository from a repo-list response (/user/repos or
// /orgs/{org}/repos), including the permissions object GitHub reports for the
// authenticated credential and the owner GitHub attributes it to.
type ghUserRepo struct {
	FullName string `json:"full_name"`
	Private  bool   `json:"private"`
	Owner    struct {
		Login string `json:"login"`
		Type  string `json:"type"` // "Organization" or "User"
	} `json:"owner"`
	Permissions struct {
		Admin    bool `json:"admin"`
		Maintain bool `json:"maintain"`
		Push     bool `json:"push"`
		Triage   bool `json:"triage"`
		Pull     bool `json:"pull"`
	} `json:"permissions"`
}

func (r ghUserRepo) tokenTestRepo() TokenTestRepo {
	return TokenTestRepo{
		FullName: r.FullName, Private: r.Private,
		Admin: r.Permissions.Admin, Maintain: r.Permissions.Maintain,
		Push: r.Permissions.Push, Triage: r.Permissions.Triage, Pull: r.Permissions.Pull,
	}
}
