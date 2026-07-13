// Package agentic provides a reusable agentic loop for chat-model APIs:
// provider adapters for OpenAI-compatible (OpenAI) and Anthropic Messages
// (Anthropic) endpoints, a tool-calling loop (Run) with approval seams
// (ToolExecutor, Approver), transient-failure retry (RetryPolicy),
// rejected-parameter recovery (NewParamStripper), prompt caching on both
// dialects, streaming callbacks (StreamEvents, Events), and conversation
// compaction (Compact, OneShot). It is extracted from an internal chat
// application so the same loop can be embedded in other hosts.
//
// The runtime depends only on the standard library; all I/O goes through an
// injectable *http.Client, and the package reads no environment variables.
package agentic
