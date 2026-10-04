package commonai

// Detection asks the endpoint which protocol it speaks; a gateway may serve the separately; callers may override.

// Dialect names a wire protocol.
type Dialect string

const (
	// DialectAuto asks the endpoint (DetectDialect); it is the value, so a config that predates the field detects.
	DialectAuto Dialect = ""
	// DialectOpenAI is the OpenAI-compatible chat-completions API.
	DialectOpenAI Dialect = "openai"
	// DialectAnthropic is the native Anthropic Messages API.
	DialectAnthropic Dialect = "anthropic"
	// DialectResponses is the OpenAI Responses API; Detection never returns it, so it is a deliberate explicit choice.
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
// not have to carry its own copy of the vocabulary -- a list that goes
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
// them: the default.
func Dialects() []Dialect {
	return []Dialect{DialectAuto, DialectOpenAI, DialectAnthropic, DialectResponses}
}
