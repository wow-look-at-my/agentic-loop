// Package agentic provides a reusable agentic loop for chat-model APIs:
// provider adapters for OpenAI-compatible and Anthropic Messages endpoints
// (built with NewOpenAIProvider / NewAnthropicProvider, both hidden behind
// the Provider interface), a tool-calling loop (Run) with approval seams
// (ToolExecutor, Approver), transient-failure retry on by default in every
// provider (ProviderConfig.Retry, RetryPolicy),
// rejected-parameter
// recovery (NewParamStripper), prompt caching on both dialects, streaming
// callbacks that can abort the call by returning an error (StreamEvents,
// Events), provider-reported timings passthrough (Timings), conversation
// compaction (Compact, OneShot), and two optional built-in tool executors —
// a sub-agent tool (NewSubagentExecutor) and a web-fetch tool
// (NewWebFetchExecutor) — composed like any other ToolExecutor via
// NewComposite. It is extracted from an internal chat application so the
// same loop can be embedded in other hosts.
//
// # Layering
//
// Two layers, and the split decides where anything new belongs. The loop
// (Run) is high-level: it asks the model, runs the tools the model asks for,
// feeds results back, and repeats. It knows nothing about HTTP, status codes,
// or backoff, so an error that reaches it is treated as REAL and PERMANENT
// and the run stops — the layer whose job was to make the call happen has
// already given up. The loop never retries and exposes no retry knob.
//
// The Provider is where those instructions are carried out. When the loop
// says "complete this request", making that true — across a 429, a 502, a
// dropped connection, a rejected parameter — is the provider's
// responsibility; those are implementation details of doing the thing, not
// outcomes to propagate. It errors only when the operation genuinely cannot
// be completed. Hence retry (ProviderConfig.Retry) and NewParamStripper are
// both provider-side.
//
// The runtime depends only on the standard library; all I/O goes through an
// injectable *http.Client, and the package reads no environment variables.
package agentic
