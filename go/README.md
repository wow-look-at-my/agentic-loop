# agentic (Go)

A reusable agentic loop for chat-model APIs, with two provider dialects —
OpenAI-compatible chat completions and the Anthropic Messages API — plus the
machinery a production tool loop needs: streaming callbacks that can abort
the call, tool execution with approval gating, transient-failure retry,
rejected-parameter recovery, prompt caching on both dialects, conversation
compaction, and two optional built-in tools (a sub-agent and a web fetcher).

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
provider, err := agentic.NewOpenAIProvider(agentic.OpenAIConfig{
	ProviderConfig: agentic.ProviderConfig{
		BaseURL: "https://api.openai.com/v1", // any OpenAI-compatible endpoint
		APIKey:  "YOUR_API_KEY",
	},
})
if err != nil { /* misconfiguration */ }

res, err := agentic.Run(ctx, agentic.Config{
	Provider: agentic.NewParamStripper(provider),
	Tools:    myTools,   // an agentic.Tools slice; empty for tool-less
	Events: agentic.Events{
		StreamEvents: agentic.StreamEvents{
			OnText: func(s string) error { fmt.Print(s); return nil },
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
provider, err := agentic.NewAnthropicProvider(agentic.AnthropicConfig{
	ProviderConfig: agentic.ProviderConfig{
		BaseURL: "https://api.anthropic.com",
		APIKey:  "YOUR_API_KEY",
	},
	// Version (the anthropic-version header) defaults to "2023-06-01".
})
if err != nil { /* misconfiguration */ }

res, err := agentic.Run(ctx, agentic.Config{Provider: provider, Tools: myTools},
	agentic.Request{
		Model:     "your-model-id",
		System:    "You are a helpful assistant.",
		Messages:  []agentic.Message{{Role: agentic.RoleUser, Content: "Hi!"}},
		MaxTokens: 4096, // REQUIRED on this dialect — the call fails fast without it
	})
```

## Layering — which layer owns what

Two layers, and the split decides where anything new belongs. Get this
backwards and you end up with the same concern implemented twice, fighting
each other.

**The loop (`Run`) is a high-level construct.** It asks the model something,
runs the tools the model asks for, feeds the results back, and repeats until
there is an answer. It knows nothing about HTTP, connections, status codes,
or backoff — those words do not appear in it. Its contract with the layer
below is simply *"complete this request."*

Consequently **an error that reaches the loop is treated as real and
permanent, and the loop stops.** It is entitled to that assumption: by the
time a failure has surfaced this far, the layer whose job was to make the
call happen has already given up. The loop does not second-guess it, does
not retry, and has no retry knob to configure — deliberately.

**The provider is where the loop's instructions are actually carried out.**
When the loop says "complete this request", making that true is the
provider's responsibility, including everything that can go transiently
wrong in the attempt: a 429, a 502, a dropped connection, a rejected
parameter. Those are implementation details of *doing the thing*, not
outcomes worth propagating. The provider surfaces an error only when the
operation genuinely cannot be completed — at which point it is, by
construction, permanent.

So: the provider owns **how** a model call gets made and everything that
can transiently go wrong doing it; the loop owns **what** calls to make and
what to do with the results.

That is why `ProviderConfig.Retry` is the library's one retry knob, why
`NewParamStripper` is a provider decorator rather than a loop feature, and
why `Config`/`SubagentConfig` carry no retry policy. It is also why retry
lives where it can see whether a call streamed anything — the condition
that decides whether re-sending is even safe — which the loop cannot see.

The one thing the provider must NOT do is hide the wait: see
`StreamEvents.OnRetry` under [Retry](#retry-and-error-classification).

## The pieces

### Provider

```go
type Provider interface {
	Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error)
}

func NewOpenAIProvider(cfg OpenAIConfig) (Provider, error)
func NewAnthropicProvider(cfg AnthropicConfig) (Provider, error)
```

The per-dialect constructors are the only way to build the two dialect
providers — the implementations are unexported, so consumers hold nothing
but the `Provider` interface. `ProviderConfig` is the shared connection
base — required `BaseURL`, plus `APIKey`, `HTTPClient`, `UserAgent`,
`Headers` — embedded in each dialect's config: `OpenAIConfig` adds
`SelfHosted` (the `cache_prompt` opt-in for llama.cpp-style servers), and
`AnthropicConfig` adds `Version` (the `anthropic-version` header) and
`DisableCaching`. An empty `BaseURL` fails fast with a permanent error.

One streaming model call. `StreamEvents` carries optional callbacks —
`OnText`, `OnReasoning`, `OnUsage`, `OnProgress`, `OnTimings`, `OnRetry` —
all nil-safe,
and **every callback returns an error**: a non-nil return aborts the stream
read immediately, and `Complete` returns the partial `*Completion` (content,
reasoning, tool calls, usage so far) together with that error. The error
surfaces unwrapped or `%w`-wrapped, so `errors.Is` against your sentinel
holds; it is never converted into an `*APIError` and never classified
transient, so neither the retry policy nor the param stripper re-sends a call
whose sink failed. On any other mid-stream failure or cancellation **after
data arrived**, `Complete` likewise returns the partial `*Completion`
alongside the error. Both dialects are safe for concurrent use.

`Request.Extra` is a verbatim top-level passthrough (`temperature`,
`reasoning_effort`, `num_ctx`, `thinking`, ...). It is merged first, so the
typed core fields always win; the library never interprets, gates, or filters
what you put there.

### Timings

llama.cpp-style upstreams (llama.cpp, ollama) attach a `timings` object to
streamed chunks. The OpenAI dialect decodes it wire-faithfully into

```go
type Timings struct {
	PromptN     int     `json:"prompt_n,omitempty"`
	PromptMS    float64 `json:"prompt_ms,omitempty"`
	PredictedN  int     `json:"predicted_n,omitempty"`
	PredictedMS float64 `json:"predicted_ms,omitempty"`
}
```

Each reported snapshot fires `StreamEvents.OnTimings`, and the **last** one
lands on `Completion.Timings` — a pointer that stays nil when the provider
never reported timings (tri-state, like the usage cache fields). The library
only surfaces what the provider said: it never synthesizes timings from
wall-clock time (tok/s fallbacks stay the caller's concern). The Anthropic
dialect has no equivalent and never fires it.

### Tools and approval

```go
type Tool interface {
	Decl() ToolDecl
	Execute(ctx context.Context, args json.RawMessage) (ToolResult, error)
	NeedsApproval() bool
}
type Tools []Tool // the flat set one run offers
type Approver interface {
	Ask(ctx context.Context, call ToolCall) (bool, error)
}
```

A tool is an individual thing, and nothing groups them. Every tool a run
offers -- a built-in below, or one a host discovered on an MCP server -- is one
`Tool` in a flat `Tools` slice that `Config.Tools` takes; the loop resolves a
requested name against it and can no more tell the kinds apart than the model
can. There is no executor and no routing table. Concatenating two sources is
`append`, and the first tool to claim a name answers it.

The id of the call being answered is NOT an argument -- almost no tool wants
it. `Run` puts it on the context, and the one built-in that needs it (the
sub-agent tool, to stamp its telemetry) reads it back with `ToolCallID(ctx)`.

A `ToolResult` is `Content` (the text the MODEL is fed), `IsError`, and
`Parts []ToolContentPart` — structured content for the HOST: images, audio,
embedded files, or a block a tool and its host agree on. Parts never reach the
model, so a result can hand a front end a megabyte of image while costing the
context only what `Content` says. `Run` passes the whole result to
`Events.OnToolResult`; nothing else in the loop reads them.

Restricting a toolset is filtering a slice, so `Tools` carries the four
helpers the loop and the sub-agent tool need:

- `Decls()` / `Names()` — the advertised declarations and names, in order.
  An unnamed tool is skipped rather than sent to a provider that rejects it.
- `Find(name)` — resolve a requested name; a miss is the loop's recoverable
  `unknown tool: <name>` result.
- `Readonly()` — the tools with `ToolDecl.Readonly`. The toolset to hand a
  sub-agent: a mutating tool is ABSENT from it, not merely refused by it.
- `Subset(names)` — the named tools, in the registry's order (the advertised
  list is part of the prompt-cache prefix, so it must not depend on the
  caller's argument order).

`NewTool(decl, run)` is the shorthand for a plain function tool; a host that
gates a tool implements `Tool` itself, since only it knows its own settings.

### Built-in tools (the plugin pattern)

The `Tool` seam **is** the plugin interface: a "plugin" is nothing more than a
value implementing it, appended to your toolset. The library ships the optional
tools below; callers who don't append them are unaffected.

```go
tools := append(agentic.Tools{},
	myTools...,
)
tools = append(tools,
	agentic.NewWebFetchTool(agentic.WebFetchConfig{Provider: provider, Model: model}),
)
tools = append(tools, agentic.NewSubagentTool(agentic.SubagentConfig{
	Provider: provider, Model: model,
	Tools: tools,                 // the FULL parent toolset — grants select from it
	Gate:  agentic.NewGate(1),    // serialize sub-agents (the source app's choice)
	OnActivity: func(a agentic.SubagentActivity) { /* live telemetry */ },
}))
```

- **`NewSubagentTool(SubagentConfig)`** — the `run_subagent` tool: the
  model offloads a focused task to a sub-agent running its own in-memory
  loop (this package's `Run`) and gets back only the distilled final report.
  By default the sub-agent sees only the read-only subset
  (`cfg.Tools.Readonly()`); the orchestrator can pin it to — and thereby
  explicitly grant, including non-read-only tools — an exact set via the
  `allowed_tools` argument (exact advertised names, with an unambiguous
  bare-name fallback for `server__tool` prefixes; `run_subagent` itself is
  never grantable, so no recursion). The advertised schema embeds the
  grantable names as an enum, flagging non-read-only ones
  `(modifies state)`. `share_context` optionally shares parent history —
  `none` (default) / `full` / `last_n` / `messages` / `summary` (one bounded
  briefing call) / `custom` — rendered as plain text
  (`RenderTranscript`/`SelectLastN`/`SelectByEndIndices`) and folded into
  one delimited task message. Misuse (a bad mode, an unknown grant) is a
  recoverable teaching error listing the valid options. A shared `*Gate`
  (`NewGate(n)`, a context-cancellable cap-n semaphore; nil = unlimited)
  bounds concurrency, and `OnActivity` streams live
  `SubagentActivity{CallID, Kind, Turn, Tool, Detail, Content, IsError}`
  steps (kinds `turn`/`tool_call`/`tool_result`/`text`/`thinking`). `Detail`
  is the one-line preview — whitespace-flattened, capped at 160 runes —
  and `Content` is the same thing WHOLE: the full arguments, full tool
  output, full answer or full reasoning, so a host can show what the
  sub-agent actually read and said rather than a truncated hint. The
  sub-run's streaming never leaks into the parent's `StreamEvents` — the
  parent sees only the final report and the activity feed.
  **`Runs *SubagentRuns` makes the tool ASYNCHRONOUS**: the call returns
  `SubagentLaunchReceipt(description)` the moment the sub-agent starts, so a
  model can fan several out in one turn and keep working, and each report is
  delivered between turns instead of blocking the call. `Config.Subagents`
  (the same registry) is what makes `Run` keep that promise: a turn that
  would otherwise END while sub-agents are out waits for the next report and
  appends it as a user message (`FormatSubagentDelivery`). Every failure past
  the JSON parse — an unconfigured model, a misused argument — reaches the
  model as that report rather than as the launch's result. A nil `Runs` keeps
  the call synchronous: it blocks until the sub-agent answers. A final message
  that is really a leaked tool-call envelope (a backend that did not parse
  the model's tool-call template) is never passed off as findings: it is cut
  at the envelope and returned as an error result carrying
  `SubagentCutOffNote`, or `SubagentNoReportText` when nothing usable
  survives.
- **`NewWebFetchTool(WebFetchConfig)`** — the `web_fetch` tool: one
  unauthenticated, plain HTTP GET (http/https only, userinfo rejected, 5 MiB
  body cap, 45 s default client timeout), cleaned to readable text (built-in
  HTML cleanup, or an optional Apache Tika server via `TikaURL` with silent
  fallback) and rune-capped at 200 000 with an explicit truncation note. An
  optional `summary_prompt` argument runs one bounded, tool-less call to the
  configured `Provider`/`Model` to summarize the cleaned content
  (`OneShot`, 2 min timeout). `BlockURL func(url string) string` is the
  injectable refusal seam: return a non-empty teaching message to refuse a
  fetch (the source application used it to redirect fetches of its
  workspace repository); the library ships the hook, not the policy. The
  tool is `Readonly`, so a sub-agent's default toolset includes it.
- **`NewTodoTool(TodoConfig)`** — the `todo_write` tool: the model's own
  task list, so a long job is legible to the user while it runs. Every call
  REPLACES the list whole, so a task carries no id to track across turns and
  ordering never has to be reconciled; `TodoConfig.Write(ctx, []Todo)` is
  where the host keeps and displays it, and an empty list arrives as an
  empty (non-nil) one because clearing the list is a real instruction. The
  result also carries the list itself as a `TodoListPartType` content part, so
  a host can draw the plan mid-turn instead of parsing it back out of the text.
  The library owns the tool's semantics — the schema's `pending`/`in_progress`/
  `done` enum, the caps (100 tasks, 200-rune titles), the per-item teaching
  errors naming `todos[<i>]`, and `RenderTodos` (exported, so a host renders
  the same text the model is answered with). A `Write` that returns an error
  is reported to the model as a failure: a list that was not stored is never
  reported as stored. NOT `Readonly` — it writes state the host shows, and a
  sub-agent inheriting it would overwrite its parent's plan.

No built-in tool is approval-gated (`NeedsApproval` is always false) —
approval wiring stays the caller's concern; wrap the tool if launching
sub-agents, fetching, or writing the task list should be gated.

The two model-calling built-ins use the `Provider` you hand them exactly as given — the
library never wraps it. In the source application every one of these model
calls (the sub-agent's nested loop, the context-summary briefing, and the
web summary) went through its rejected-parameter recovery; to reproduce
that, pass a `NewParamStripper`-wrapped provider in
`SubagentConfig`/`WebFetchConfig` — the same wrapped value you give
`Config.Provider`.

### The loop

`Run(ctx, cfg, req)` drives the turn loop: call the model, execute the
requested tools, feed the results back, repeat. Key behaviors:

- **There is no turn cap.** The loop runs until the model stops asking for
  tools. A counted cap cannot tell a model looping uselessly from one deep in
  a hard task, so it fires at the worst possible moment: after the run has
  spent every call gathering context and just before the model writes any of
  it down. Bound a run with the two mechanisms that judge the right thing —
  `ErrStuck` below (evidence the model stopped progressing) and your own
  `ctx` (wall-clock and spend, without discarding work in flight).
- **A stuck model is caught, not waited out.** A turn whose tool calls are
  byte-identical to the previous turn's cannot learn anything new — the same
  calls return the same results, which produce the same turn again. The
  third identical turn in a row (`StuckNudgeAt`) gets one nudge appended
  after its tool results; the sixth (`StuckFailAt`) ends the run with
  `ErrStuck` (match it with `errors.Is`) rather than letting it spin
  — the failing batch is never executed and its tool calls are
  cleared, so the partial transcript stays replayable. Any change in what
  the model asks for clears the count; call IDs are excluded from the
  comparison, since providers mint a fresh one per call.
- **Tool failures never abort the loop.** An `Execute` error becomes a
  `tool execution failed: ...` error result; a denied approval records
  exactly `DeniedMessage` ("The user denied permission to run this tool.");
  a call to a name the run does not offer gets `unknown tool: ...`. In every
  case the loop continues so the model can react.
- **Callbacks can abort.** `Events.OnToolCall`/`OnToolResult` (like the
  stream callbacks) return an error; a non-nil return ends the run the way a
  cancellation does — the pending batch is cleared (the assistant message
  keeps its content and reasoning but loses its tool calls, and the batch's
  already-appended results are dropped) so the transcript stays replayable
  with no orphan tool calls, and the partial `Result` is returned together
  with your error.
- **Per-turn hooks.** `Events.OnTurnBegin(turn, req)` fires before each
  numbered model call (turns are numbered from 1; the stall wrap-up call
  fires one past the turn that stalled) with the 1-based turn number and a
  pointer to the per-call `Request`: mutate it (append a wind-down message,
  tweak `System`, add `Extra`) and the change applies to that one call only,
  never the persistent transcript. `Events.OnTurnEnd(turn, comp, err)` fires
  after each call with the `Completion` (nil when the call produced none) and
  the call's error. A non-nil return from either aborts the run like any
  other callback error: `OnTurnBegin` aborts before the call, `OnTurnEnd`
  after it with the completed data kept. The internal subagent telemetry
  hook is unaffected.
- **Approval**: a tool whose `NeedsApproval(name)` is true pauses for
  `Approver.Ask`. A nil `Approver` fails closed (denies). If `Ask` returns an
  error (the decision never arrived), the run ends with the pending batch
  cleared exactly as above.
- **Stall fallback**: if the loop ends with no written answer (a
  thinking-only turn, or the cap hit mid-research) and tools were in play,
  one extra tool-less wrap-up turn forces the model to synthesize its answer
  from what it gathered; failing that, the final content falls back to the
  accumulated reasoning, then to a clear placeholder.
- **Result.Usages** holds one entry per model call, in order — deliberately
  **not summed**: successive prompts overlap (each turn re-sends the growing
  transcript), so summing would double-count the shared prefix.

### Retry and error classification

`RetryPolicy` (default `DefaultRetry`: **10 attempts**, 500ms base, delay =
base × 2^(n−1), no jitter, no cap) retries only what `IsTransient`
allows: HTTP 408,
429, any 5xx, and network/transport errors. Context cancellation, other
4xx, and **errors your own callbacks returned** are permanent. A 400 whose
body says the prompt exceeded the context window is flagged — check with
`IsContextOverflow(err)` — and never retried. Anthropic in-stream `error`
events (an HTTP 200 whose stream carries the error — how `overloaded_error`
typically arrives) are mapped onto the same `APIError` using Anthropic's
documented error-type → status table, so an in-stream overload (529) or rate
limit (429) retries exactly like its non-2xx counterpart.

**Retry is on by default and belongs to the Provider**, not to `Run`: every
provider a dialect constructor builds already retries, so there is nothing
to opt into and nothing to remember. Tune or disable it per provider:

```go
// The default — retries transient failures.
p, _ := agentic.NewOpenAIProvider(agentic.OpenAIConfig{
    ProviderConfig: agentic.ProviderConfig{BaseURL: url},
})

// Explicitly off.
p, _ = agentic.NewOpenAIProvider(agentic.OpenAIConfig{
    ProviderConfig: agentic.ProviderConfig{
        BaseURL: url,
        Retry:   &agentic.RetryPolicy{MaxAttempts: 1},
    },
})
```

The provider is the right home because it is the layer that knows whether a
call streamed anything — the condition that decides whether re-sending is
safe. A call is re-attempted **only when the failed attempt streamed
nothing**; once data arrived the error surfaces with the partial completion
attached. A provider marks data as seen **before** each emit, so even a
callback that fails on the very first delta yields a partial completion —
the call is never re-sent into a dead sink. That single signal, the non-nil
completion, is the whole mechanism: nothing watches the callbacks to
second-guess it.

**Retrying is not silent.** Ten attempts of uncapped backoff is minutes of
wall-clock, so every retry fires `StreamEvents.OnRetry` *before* its
backoff, carrying which attempt failed, of how many, the delay about to be
waited, and the error — show it, don't leave the user staring at nothing:

```go
ev := &agentic.StreamEvents{
    OnRetry: func(a agentic.RetryAttempt) error {
        log.Printf("attempt %d/%d failed (%v), retrying in %s", a.Attempt, a.Of, a.Err, a.Delay)
        return nil // returning an error stops the retrying
    },
}
```

`OnRetry` fires from the retry layer, not from a dialect provider, and does
**not** count as "streamed something" — a notification about a failed
attempt cannot make the next one unsafe.

`Run` therefore has no retry knob, and a retried call is one turn: `Run`
only ever sees the outcome. A custom `Provider` implementation is
responsible for its own retry.

### Rate limiting

`ProviderConfig.RateLimiter`, when set, throttles the provider's outgoing
request starts to the limiter's fixed rate — a provider's per-minute request
cap, spaced evenly:

```go
limiter := agentic.NewRateLimiter(30) // at most 30 requests/minute
p, _ := agentic.NewOpenAIProvider(agentic.OpenAIConfig{
    ProviderConfig: agentic.ProviderConfig{
        BaseURL:     url,
        RateLimiter: limiter,
    },
})
```

Like retry, the throttle belongs to the Provider: staying under an upstream's
rate limit is part of making the call happen. The limiter is wired in as an
`http.RoundTripper` on the provider's client, so **retries ride the same
gate** — a re-sent attempt is a request too. Only request **starts** are
counted (a slow call never pushes the average over the limit), and because
consecutive starts are at least one interval apart, no 60-second window can
contain more than the configured number of started requests.

**Share one limiter to throttle many providers together.** `RateLimiter` is
safe for concurrent use; a single instance passed to every provider of a run
(e.g. the concurrent jobs of a benchmark) keeps them under one per-endpoint
cap — per-provider limiters would multiply it by the number of callers.

### Rejected-parameter recovery

```go
provider := agentic.NewParamStripper(inner)
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
  (never send it to hosted OpenAI/Azure -- they 400 on unknown fields). Set
  `PromptCache` to add the two Anthropic-style ephemeral `cache_control`
  breakpoints in openai shape (a static one on the leading system message
  and a moving one on the tail content block) for Anthropic-fronting
  gateways that pass them through; keep it off for strict OpenAI-compatible
  servers, which 400 on the unknown marker.

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

`Completion.UsageReported` is true iff at least one usage snapshot was
merged during the call: `Usage` is a value type, so this is how a caller
reading only the returned `Completion` distinguishes an upstream that
reported all-zero usage from one that reported none at all (common on local
servers) — check it before persisting or displaying `Usage`.

`Completion.RawUsage` carries the provider's usage object verbatim (the raw
wire JSON on the openai dialect, the merged wire-shaped object on Anthropic)
for logging and for extracting provider extras the normalized `Usage` drops:
`Completion.ReasoningTokens` (openai
`usage.completion_tokens_details.reasoning_tokens`) and
`Completion.CostUsd` (`usage.cost`, falling back to `usage.estimated_cost`),
each a tri-state pointer present only when the upstream reported it.

`Completion.Streamed` records whether the response actually arrived as an SSE
stream. A 200 that is NOT `text/event-stream` is read as a plain JSON body
and reassembled into a Completion with `Streamed` false -- a server that
ignores `stream: true` is accepted transparently, and the flag preserves the
truth of how the call was transported.

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

Providers built by the dialect constructors are read-only during `Complete`
and safe to share across goroutines. The `NewParamStripper` wrapper is also safe for
concurrent use (its strip memory is mutex-guarded). One `Run` drives one
conversation; run concurrent conversations by calling `Run` concurrently,
sharing the provider. A shared `Gate` bounds concurrent sub-agents.

## Testing

The test suite runs entirely against `httptest` fake servers and an
in-process scripted provider — no network, no credentials.

### Resource watching (optional)

A resource is not advertised in the model's context the way a tool is, so a
model never learns one exists. `NewResourceWatcher(ResourceWatchConfig)` closes
that gap: at each turn boundary it re-reads every watched resource, hashes it,
records what moved, and hands back a `ResourcePoll` a host renders with
`FormatResourceNotice` — names and change ids, never content, because dumping a
changed resource into the thread would cost its full size on every later turn.

Nothing in it knows what MCP is. A host supplies two seams:

- **`ResourceSource`** — `ID`/`Name`/`List`/`Read`. One thing that publishes
  resources (an MCP server, a directory, anything listable).
- **`ResourceSnapshots`** — where the watcher keeps what it saw, and where it
  records each change. `RecordChange` returns the id, so the host owns id
  generation (it is the side that resolves one later).

`NewResourceDiffTool(ResourceChanges)` is the matching tool: it resolves a
change id to the before/after captured AT THAT MOMENT, from the change record's
own copy — a resource that has moved three times since still answers the first
notice with the first change. Failures are recoverable results that list the
real ids; a source that could not be listed becomes a WARNING, never a removal
(absent-because-unreachable is not absent). Binary content is watched by hash
and size and reported as such, never rendered as a diff.

`UnifiedDiff`/`CountLineChanges`/`HumanSize` are exported for hosts rendering
their own changes with the same words.

### Filesystem tools (optional)

`NewFileTools(FileToolsConfig)` returns the seven-tool file vocabulary —
`list_dir`, `read_file`, `find_files`, `grep`, `write_file`, `edit_file`,
`delete_file` — over whatever a host mounts. The library owns what a file tool
IS: the names, the model-facing descriptions, the argument schemas, the caps
and every word of the rendering. A host owns what is behind a mount, and
nothing about its storage reaches here. Mount nothing and you get no tools at
all, rather than tools that can only fail.

A host mounts `Folder`s under virtual prefixes:

```go
tools := agentic.NewFileTools(agentic.FileToolsConfig{
	Folders: map[string]agentic.Folder{
		"repos":     repoFolder,      // /repos/<org>/<repo>[@<ref>]/<path>
		"workspace": workspaceFolder, // an agentic.WritableFolder
	},
	MountsBlurb: "/repos is read-only; /workspace is editable.",
})
```

- **`Folder`** — `Display`/`List`/`Read`/`Find`/`Grep`. Every method receives
  the WHOLE virtual path as the model wrote it, because only the folder knows
  its own grammar: `/repos/<org>/<repo>@<ref>/<path>` is the repository host's
  business, not the tool layer's.
- **`WritableFolder`** adds `Writable`/`Create`/`Replace`/`Remove`. A folder
  that is not one gets the three write tools' refusals for free, and a
  `ReadOnlyExplainer` lets it say *why* a particular path is read-only.
- **`PathGuard`** blocks a path before any folder sees it, with the reason the
  model is shown — how a host redirects `/repos` writes at an attached
  workspace.

Two properties are the module's whole point, and a host cannot opt out of
either. **A cap that bites is announced**: a truncated listing, a `find_files`
that stopped at its limit, a `grep` that hit `MaxHits` all say so, because a
partial result that reads as complete is worse than no result. And **an empty
`grep` is a real negative**: every line in scope was read, so no matches means
the text is not there — which is why a scope the mount does not hold is an
error rather than an empty result, and why `GrepResult.Note` exists for a
folder that could only cover part of what the path named.
