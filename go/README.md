# agentic (Go)

A reusable agentic loop for chat-model APIs, with two provider dialects —
OpenAI-compatible chat completions and the Anthropic Messages API — plus the
machinery a production tool loop needs: streaming callbacks, tool execution
with approval gating, transient-failure retry, rejected-parameter recovery,
prompt caching on both dialects, and conversation compaction.

The runtime is **standard library only**. All I/O goes through an injectable
`*http.Client`, and the package reads **no environment variables** — every
endpoint, key, and knob is explicit configuration.

## Install

```sh
go get github.com/wow-look-at-my/agentic-loop/go
```

```go
import agentic "github.com/wow-look-at-my/agentic-loop/go"
```

## Quick start — OpenAI-compatible

```go
provider := &agentic.OpenAI{
	BaseURL: "https://api.openai.com/v1", // any OpenAI-compatible endpoint
	APIKey:  "YOUR_API_KEY",
}

res, err := agentic.Run(ctx, agentic.Config{
	Provider: agentic.NewParamStripper(provider),
	Tools:    myExecutor, // a ToolExecutor, or nil for tool-less
	Events: agentic.Events{
		StreamEvents: agentic.StreamEvents{
			OnText: func(s string) { fmt.Print(s) },
		},
	},
}, agentic.Request{
	Model:  "your-model-id",
	System: "You are a helpful assistant.",
	Messages: []agentic.Message{
		{Role: agentic.RoleUser, Content: "What changed in the last release?"},
	},
	Extra: map[string]any{"reasoning_effort": "high"}, // verbatim passthrough
})
if err != nil { /* res may still carry the partial transcript */ }
fmt.Println(res.Final.Content)
```

## Quick start — Anthropic

```go
provider := &agentic.Anthropic{
	BaseURL: "https://api.anthropic.com",
	APIKey:  "YOUR_API_KEY",
	// Version defaults to "2023-06-01".
}

res, err := agentic.Run(ctx, agentic.Config{Provider: provider, Tools: myExecutor},
	agentic.Request{
		Model:     "your-model-id",
		System:    "You are a helpful assistant.",
		Messages:  []agentic.Message{{Role: agentic.RoleUser, Content: "Hi!"}},
		MaxTokens: 4096, // REQUIRED on this dialect — the call fails fast without it
	})
```

## The pieces

### Provider

```go
type Provider interface {
	Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error)
}
```

One streaming model call. `StreamEvents` carries optional callbacks (`OnText`,
`OnReasoning`, `OnUsage`, `OnProgress`); all are nil-safe. On a mid-stream
failure or cancellation **after data arrived**, `Complete` returns the partial
`*Completion` **alongside** the error (both non-nil) so nothing streamed is
lost. Both built-in providers are safe for concurrent use.

`Request.Extra` is a verbatim top-level passthrough (`temperature`,
`reasoning_effort`, `num_ctx`, `thinking`, ...). It is merged first, so the
typed core fields always win; the library never interprets, gates, or filters
what you put there.

### Tools and approval

```go
type ToolExecutor interface {
	Tools() []Tool
	Execute(ctx context.Context, call ToolCall) (ToolResult, error)
	NeedsApproval(name string) bool
}
type Approver interface {
	Ask(ctx context.Context, call ToolCall) (bool, error)
}
```

Combinators mirror the source application's semantics:

- `NewComposite(execs...)` — one deterministic tool list across executors;
  first registration of a name wins; unknown calls become recoverable
  `unknown tool: <name>` results. Returns nil when no tools remain.
- `ReadonlyView(e)` — only tools with `Tool.Readonly`; anything else is
  refused. The toolset to hand a sub-agent.
- `SubsetView(e, names)` — only the explicitly named tools.

### The loop

`Run(ctx, cfg, req)` drives the turn loop: call the model, execute the
requested tools, feed the results back, repeat. Key behaviors:

- **MaxTurns** caps model calls (default `DefaultMaxTurns` = 10). On the final
  permitted turn **tools are withheld** so the model must answer instead of
  requesting another never-executed call.
- **Tool failures never abort the loop.** An `Execute` error becomes a
  `tool execution failed: ...` error result; a denied approval records
  exactly `DeniedMessage` ("The user denied permission to run this tool.");
  a hallucinated call with no executor gets `unknown tool: ...`. In every
  case the loop continues so the model can react.
- **Approval**: a tool whose `NeedsApproval(name)` is true pauses for
  `Approver.Ask`. A nil `Approver` fails closed (denies). If `Ask` returns an
  error (the decision never arrived), the run ends with the pending batch
  cleared — the assistant message keeps its content and reasoning but loses
  its tool calls, and that batch's already-appended results are dropped — so
  the returned transcript is replayable with no orphan tool calls.
- **Stall fallback**: if the loop ends with no written answer (a
  thinking-only turn, or the cap hit mid-research) and tools were in play,
  one extra tool-less wrap-up turn forces the model to synthesize its answer
  from what it gathered; failing that, the final content falls back to the
  accumulated reasoning, then to a clear placeholder.
- **Result.Usages** holds one entry per model call, in order — deliberately
  **not summed**: successive prompts overlap (each turn re-sends the growing
  transcript), so summing would double-count the shared prefix.

### Retry and error classification

`RetryPolicy` (default `DefaultRetry`: 4 attempts, 500ms base, delay =
base × 2^(n−1), no jitter) retries only what `IsTransient` allows: HTTP 408,
429, any 5xx, and network/transport errors. Context cancellation and other
4xx are permanent. A 400 whose body says the prompt exceeded the context
window is flagged — check with `IsContextOverflow(err)` — and never retried.
Anthropic in-stream `error` events (an HTTP 200 whose stream carries the
error — how `overloaded_error` typically arrives) are mapped onto the same
`APIError` using Anthropic's documented error-type → status table, so an
in-stream overload (529) or rate limit (429) retries exactly like its
non-2xx counterpart.

Inside `Run`, a model call is re-attempted **only when the failed attempt
streamed nothing**; once data arrived, the partial assistant message is
finalized into the transcript and the partial `Result` is returned with the
error.

### Rejected-parameter recovery

```go
provider := agentic.NewParamStripper(&agentic.OpenAI{...})
```

Some upstreams 400 on a parameter others accept. The stripper parses the
offending parameter name out of the error body (four provider phrasings,
matched case-insensitively and by normalized name, so a camelCase report
matches your snake_case key), removes that one key from `Extra`, and retries
once. The strip is remembered for later calls through the same stripper. It
never fires on context cancellation or after data streamed.

### Prompt caching

- **Anthropic**: always on unless `DisableCaching` — exactly two ephemeral
  `cache_control` breakpoints per request: a static one on the system block
  (which also covers the tools array via the tools → system → messages
  hierarchy) and a moving one on the last content block of the last message.
  Both are applied to per-request wire structures only; **your `Messages` are
  never mutated**, so the stored transcript stays marker-free.
- **OpenAI**: hosted caching is automatic server-side; the provider defaults
  `stream_options: {"include_usage": true}` (unless your `Extra` already sets
  `stream_options`) so usage — including cached-token counts — arrives on the
  stream. Set `CacheKey` to send `prompt_cache_key` (a routing hint), and set
  `SelfHosted` to add `cache_prompt: true` for llama.cpp-style servers
  (never send it to hosted OpenAI/Azure — they 400 on unknown fields).

### Usage accounting

`Usage.PromptTokens` is always the **full prompt** including cached tokens
(the Anthropic provider adds `input_tokens + cache_read + cache_creation`,
because Anthropic's `input_tokens` excludes cached tokens; OpenAI's
`prompt_tokens` already includes them). `CacheReadTokens`/`CacheWriteTokens`
are tri-state pointers: nil means the provider reported no cache info; a
value — including an explicit 0 — is a real report. `CachedTokens()` gives
the normalized cache-hit count. Streamed usage snapshots are merged
newest-wins (xAI attaches a cumulative snapshot to every chunk; they are
never summed), and the finalized total is floored at prompt+completion while
preserving a genuine surplus (reasoning tokens).

### Compaction and one-off calls

```go
cr, err := agentic.Compact(ctx, provider, agentic.Request{
	Model:    "your-model-id",
	System:   "You are compacting a conversation ...", // your summarizer brief
	Messages: history,
})
// cr.Messages is the replacement round:
//   user(agentic.CompactRequestText), assistant(cr.Summary)
```

`Compact` sends the whole history with the summarize instruction as the
trailing user message and no tools; you replace your history with the
returned two-message round. `OneShot(ctx, p, req, timeout)` is the bounded
tool-less single call (titles, micro-summaries); compose it with
`context.WithoutCancel(parent)` for fire-and-forget work that must survive
the parent request ending.

## Concurrency

`OpenAI` and `Anthropic` values are read-only during `Complete` and safe to
share across goroutines. The `NewParamStripper` wrapper is also safe for
concurrent use (its strip memory is mutex-guarded). One `Run` drives one
conversation; run concurrent conversations by calling `Run` concurrently,
sharing the provider.

## Testing

The test suite runs entirely against `httptest` fake servers and an
in-process scripted provider — no network, no credentials.
