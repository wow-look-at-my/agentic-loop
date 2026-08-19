package commonai

// Which wire protocol an endpoint speaks, established by asking it rather than
// by asking the user.
//
// The two dialects answer the same question -- "what models do you have?" --
// with structurally different documents, and that difference is a fact about
// the server rather than a guess about its hostname:
//
//	OpenAI:    {"object":"list","data":[{"id":"gpt-x","object":"model",...}]}
//	Anthropic: {"data":[{"type":"model","id":"claude-x",...}],"has_more":false}
//
// So one GET identifies most endpoints, including a self-hosted Anthropic
// gateway on a domain no hostname rule would recognize.
//
// What it cannot see is DialectResponses: an OpenAI endpoint answers the same
// model list whether or not it also serves /v1/responses, so nothing in the
// document distinguishes them. Detection reports the chat-completions dialect
// there, and using the Responses API is an explicit choice a caller makes.
//
// What it cannot settle, and why callers keep an override: this reads the
// MODELS endpoint and infers the CHAT endpoint from it. A gateway is free to
// serve those two independently -- an OpenAI-shaped model list in front of a
// /v1/messages chat endpoint is a legal thing to build -- and the only way to
// prove the chat dialect is to post to it, which either spends tokens or
// deliberately sends a malformed request to read the error shape back. Neither
// is worth doing on every save, so detection answers with what it saw and the
// host may overrule it.

// Dialect names a wire protocol.
type Dialect string

const (
	// DialectAuto asks the endpoint (DetectDialect). It is the zero value, so
	// a config that predates the field detects rather than assuming.
	DialectAuto Dialect = ""
	// DialectOpenAI is the OpenAI-compatible chat-completions API.
	DialectOpenAI Dialect = "openai"
	// DialectAnthropic is the native Anthropic Messages API.
	DialectAnthropic Dialect = "anthropic"
	// DialectResponses is the OpenAI Responses API. Detection NEVER returns
	// it: an endpoint serving /v1/responses serves the same /v1/models
	// document as one serving /v1/chat/completions, so nothing about the
	// model list distinguishes them. It is a deliberate choice -- reasoning
	// carried across tool calls, at the cost of an API only some servers
	// implement -- and a choice is exactly what an explicit setting is for.
	DialectResponses Dialect = "responses"
)

// Valid reports whether d is a dialect this library can speak.
func (d Dialect) Valid() bool {
	switch d {
	case DialectAuto, DialectOpenAI, DialectAnthropic, DialectResponses:
		return true
	}
	return false
}

// Label is how a dialect reads on a screen. It lives here so a host's UI does
// not have to carry its own copy of the vocabulary -- a second list that goes
// stale the moment a dialect is added.
func (d Dialect) Label() string {
	switch d {
	case DialectAnthropic:
		return "anthropic messages"
	case DialectOpenAI:
		return "openai-compatible"
	case DialectResponses:
		return "openai responses"
	case DialectAuto:
		return "detect"
	}
	return string(d)
}

// Dialects is every dialect a host may offer, in the order it should present
// them: the default first.
func Dialects() []Dialect {
	return []Dialect{DialectAuto, DialectOpenAI, DialectAnthropic, DialectResponses}
}
