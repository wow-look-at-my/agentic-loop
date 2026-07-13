// Package agentic provides a reusable agentic loop for chat-model APIs:
// provider adapters for OpenAI-compatible and Anthropic endpoints, tool
// registration and execution, approval gating for sensitive tool calls, and
// streaming callbacks for tokens, tool activity, and usage. It is extracted
// from an internal chat application so the same loop can be embedded in
// other hosts.
package agentic
