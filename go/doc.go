// Package agentic provides a reusable agentic loop for chat-model APIs:
// provider adapters for OpenAI-compatible and Anthropic Messages endpoints
// (built with NewOpenAIProvider / NewAnthropicProvider, both hidden behind
// the Provider interface), a tool-calling loop (Run) with approval seams
// (ToolExecutor, Approver), transient-failure retry (RetryPolicy),
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
// The runtime depends only on the standard library; all I/O goes through an
// injectable *http.Client, and the package reads no environment variables.
package agentic
