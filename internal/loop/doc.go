// Package loop provides the reusable agentic loop for chat-model APIs:
// provider adapters for OpenAI-compatible and Anthropic Messages endpoints
// (built with NewOpenAIProvider / NewAnthropicProvider, both hidden behind
// the Provider interface), a tool-calling loop (Run) over a flat set of
// individual tools with approval seams (Tool, Tools, Approver),
// transient-failure retry on by default in every provider
// (ProviderConfig.Retry, RetryPolicy), rejected-parameter recovery
// (NewParamStripper), prompt caching on both dialects, streaming callbacks
// that can abort the call by returning an error (StreamEvents, Events),
// provider-reported timings passthrough (Timings), conversation compaction
// (Compact, OneShot). Optional tool families live in sibling packages (vfs,
// repo, subagent, webfetch, todo, resources) and return Tools values a host
// appends itself. It is extracted from an internal chat application so the
// same loop can be embedded in other hosts.
package loop
