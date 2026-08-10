package agentic

import (
	"context"
	"fmt"
	"net/http"
	"strings"
)

// Why an exhausted credential rotation failed.
//
// GitHub answers 404 for a repository a token cannot see, for one it cannot
// write to, and for an object that is not in it. The rotation itself cannot
// tell those apart -- it only knows every credential was refused -- so it used
// to announce the first as fact. That told a user holding admin on the
// repository that none of their PATs could write to it, and sent them to fix a
// setting that was already correct. See docs/tools/repo-tools.md.

// moreInformativeAuth folds one credential's failure into the best failure so
// far, so an exhausted rotation reports what it established rather than
// whichever credential happens to be configured last. The read side already
// works this way (failureRank); a write rotation kept the LAST attempt, which
// made the reported cause depend on the order of Settings -> github.
//
// That is not cosmetic: explainExhaustedPush recognizes a vanished branch
// point only by a 404 that names an object, so one later token rejecting its
// own credentials erased the finding and put the push back on the
// write-access guess this file exists to retire.
func moreInformativeAuth(best, next GitHubAuthError) GitHubAuthError {
	if best.status == 0 {
		return next
	}
	if authFailureRank(next) >= authFailureRank(best) {
		return next
	}
	return best
}

// authFailureRank scores an exhausted-rotation failure by how much of the
// cause it pins down. A 404 naming an object is the only one that identifies
// something about the REPOSITORY -- the object is not there -- while every
// other status describes one credential.
func authFailureRank(a GitHubAuthError) int {
	switch {
	case a.status == http.StatusNotFound && a.object != "":
		return 30 // "this commit/branch is gone" -- a cause, not a credential
	case a.status == http.StatusForbidden:
		return 20 // recognized and denied: a real signal about access
	case a.status == http.StatusUnauthorized:
		return 5 // "this credential was rejected" -- true and useless
	default:
		return 10
	}
}

// explainExhaustedWrite says why every credential failed instead of guessing.
// A 404 from GitHub covers a repository the token cannot see, a repository it
// cannot write to, and an object that is not there; announcing the first as
// the cause sends a user to Settings to fix a token that was never the
// problem. One extra read separates "cannot see it" from the rest, and a 404
// on a step that read one named object reports that object.
func (e *repoTools) explainExhaustedWrite(ctx context.Context, toolName, cacheKey string, order []tokenAttempt, bestAuth GitHubAuthError) string {
	repo := cacheKey
	if repo == "" {
		repo = "this repository"
	}
	if !e.repoReadableWith(ctx, order, cacheKey) {
		return fmt.Sprintf("%s failed with every configured GitHub token (%s), and none of them can read %s at all — add or update a token with access to it in Settings -> github.",
			toolName, bestAuth.Error(), repo)
	}
	if bestAuth.status == http.StatusNotFound && bestAuth.object != "" {
		return fmt.Sprintf("%s failed: %s is not in %s (every configured token can read the repository, so this is not a permissions problem).",
			toolName, bestAuth.object, repo)
	}
	return fmt.Sprintf("%s failed with every configured GitHub token (%s). They can read %s but none of them could write to it — grant one \"Contents: read & write\" on that repository in Settings -> github.",
		toolName, bestAuth.Error(), repo)
}

// repoReadableWith reports whether any of these credentials can see the
// repository. Best-effort: a transport failure answers "yes", since the only
// thing this decides is whether to blame the credentials, and a request that
// never arrived is no evidence against them.
func (e *repoTools) repoReadableWith(ctx context.Context, order []tokenAttempt, cacheKey string) bool {
	org, repo, ok := strings.Cut(cacheKey, "/")
	if !ok {
		return true
	}
	for _, att := range order {
		res, err := e.gh.doGet(ctx, e.repoURL(org, repo), att.token, "application/vnd.github+json")
		if err != nil {
			return true
		}
		if res.status >= 200 && res.status < 300 {
			return true
		}
	}
	return false
}
