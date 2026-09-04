package commonai

import (
	"errors"
	"regexp"
	"strings"
)

// A model can be served by one protocol and refused by another on the same
// host. see docs/dialect-refusal.md

// dialectRefusalRe finds the endpoint an error body DIRECTS the caller to.
// see docs/dialect-refusal.md
var dialectRefusalRe = regexp.MustCompile(`(?i)(not|n't|never|no longer)?\s*(?:` +
	// Told to go somewhere.
	`use|switch to|call|try|post to|send (?:it )?to|belongs (?:on|in)|` +
	// Told where it IS served, which is the same direction stated the other way.
	`(?:must |should |can |may )?(?:be )?(?:called|requested|sent|invoked) (?:through|on|via|against)|` +
	`(?:only |exclusively )?(?:supported|available|works?|served) (?:only )?(?:in|on|with|via|through)` +
	`)\s+(?:the\s+)?["'` + "`" + `]?/?(v\d+/(?:responses|messages|chat/completions))`)

// dialectOfPath maps an API path to the dialect that speaks it.
func dialectOfPath(path string) Dialect {
	switch p := strings.ToLower(path); {
	case strings.HasSuffix(p, "/responses"):
		return DialectResponses
	case strings.HasSuffix(p, "/messages"):
		return DialectAnthropic
	case strings.HasSuffix(p, "/chat/completions"):
		return DialectOpenAI
	}
	return DialectAuto
}

// DialectRefused reports the protocol an endpoint NAMED as the one to use for
// the request it just refused. see docs/dialect-refusal.md
func DialectRefused(err error) (Dialect, bool) {
	var ae *APIError
	if !errors.As(err, &ae) {
		return DialectAuto, false
	}
	// A refusal of THIS request. A 5xx says nothing about a protocol, and a 401
	// is about the credential.
	switch ae.Status {
	case 400, 404, 405, 422:
	default:
		return DialectAuto, false
	}
	for _, m := range dialectRefusalRe.FindAllStringSubmatch(ae.Body, -1) {
		// A NEGATED phrase names the path that was refused, not the one to use:
		// "not supported in v1/chat/completions" says only that this failed.
		if m[1] != "" {
			continue
		}
		if d := dialectOfPath(m[2]); d != DialectAuto {
			return d, true
		}
	}
	return DialectAuto, false
}
