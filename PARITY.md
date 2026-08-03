# PARITY.md — the specification for the TypeScript port

This document is the contract for `ts/`: a TypeScript implementation with
**full behavioral parity** with `go/`. It captures every rule the Go library
implements — including the exact strings, regexes, and edge-case orderings a
reimplementation gets wrong first. Where the Go code and this document ever
disagree, the Go code (and its tests) wins; update this file.

The semantics originate from three applications (an OpenAI-compatible chat
server in Go, a browser agent with a native Anthropic dialect, and a
benchmark harness with retry/caching plumbing); the Go library is the
distilled, dependency-free extraction.

---

## 1. Data model

| Type | Fields | Notes |
|---|---|---|
| `Role` | `"system" \| "user" \| "assistant" \| "tool"` | |
| `ThinkingBlock` | `text`, `signature`, `redacted` | `redacted` holds an opaque `redacted_thinking` payload; when set, the other two are empty. |
| `ToolCall` | `id`, `name`, `arguments` | `arguments` is the RAW JSON object **text** (OpenAI: concatenated streamed fragments; Anthropic: accumulated `input_json_delta.partial_json`). |
| `Message` | `role`, `content`, `thinking[]`, `toolCalls[]`, `toolCallID`, `toolIsError` | thinking/toolCalls assistant-only; toolCallID/toolIsError tool-only. |
| `Tool` | `name`, `description`, `inputSchema`, `readonly` | nil/absent schema marshals as `{"type":"object"}`. `readonly` is never sent to the upstream. |
| `ToolResult` | `content`, `isError` | Text-only by design (see cuts). |
| `Usage` | `promptTokens`, `completionTokens`, `totalTokens`, `cacheReadTokens?`, `cacheWriteTokens?` | The two cache fields are **tri-state**: absent/undefined = provider reported no cache info; a number (including 0) = a real report. Never zero-fill, never estimate. |
| `PromptProgress` | `processed`, `total`, `cache`, `timeMS` | Wire-faithful to the upstream `prompt_progress` object (`{total, cache, processed, time_ms}`). |
| `Timings` | `promptN`, `promptMS`, `predictedN`, `predictedMS` | Wire-faithful to the llama.cpp-style chunk `timings` object (`{prompt_n, prompt_ms, predicted_n, predicted_ms}`; the `_ms` fields are floats). Decode only — the library NEVER synthesizes timings from wall-clock time. |
| `Request` | `model`, `system`, `messages`, `tools`, `maxTokens`, `extra`, `cacheKey` | See §3/§4 for per-dialect handling. |
| `Completion` | `message`, `usage`, `usageReported`, `timings?`, `rawUsage`, `reasoningTokens?`, `costUsd?`, `streamed`, `stopReason` | `usageReported` = at least one usage snapshot was merged (usage is a value, so this is the "reported vs absent" tri-state for the whole struct). `timings` = the LAST reported snapshot, absent when the provider never reported one; the Anthropic dialect never sets it. `rawUsage` = the provider's usage object verbatim (the raw wire JSON on the openai dialect; the merged wire-shaped object on Anthropic), absent when no usage was reported. `reasoningTokens` = openai `usage.completion_tokens_details.reasoning_tokens`, `costUsd` = openai `usage.cost` / `usage.estimated_cost` -- each a tri-state pointer present only when the upstream reported it (Anthropic never sets either). `streamed` = whether the response actually arrived as an SSE stream (a non-SSE 200 is read as plain JSON and reassembled with `streamed` false). |
| `Result` | `messages`, `final`, `usages[]`, `turns` | `usages` = one entry per model call IN ORDER, never summed (successive prompts overlap; summing double-counts the shared prefix). |

Seams (interfaces): `Provider.complete(req, events) -> Completion`
(streaming under the hood), `ToolExecutor {tools(), execute(call),
needsApproval(name)}`, `Approver {ask(call) -> boolean}` (throw = decision
never arrived).

**Provider construction is one factory function per dialect over a shared
config base.** The dialect implementations are internal — NOT exported, no
exported dialect classes, and no dialect string enum: consumers call the
dialect's factory (Go: `NewOpenAIProvider(config)` /
`NewAnthropicProvider(config)`) and hold only the `Provider` interface. The
shared connection-config shape — the required `baseURL` plus
`apiKey`/`httpClient`/`userAgent`/`headers` — is embedded/extended by the
per-dialect configs: the OpenAI config adds `selfHosted`, the Anthropic
config adds `version` (the anthropic-version header) and `disableCaching`.
An empty baseURL fails fast with a permanent (never-retried) error. There
are no per-dialect base-URL defaults: baseURL is always explicit.

**Callback error contract** (every callback below): `StreamEvents.onText /
onReasoning / onUsage / onProgress / onTimings / onRetry` and
`Events.onToolCall / onToolResult` all return/throw an error to signal
failure. Semantics to
reproduce exactly:

- A stream-callback error ABORTS the upstream read immediately; `complete`
  returns the partial Completion (content/reasoning/tool-calls/usage
  accumulated so far — state is accumulated BEFORE each emit, so the failing
  delta is included) together with that error. The caller's error object
  stays reachable (`errors.Is` in Go; identity/`cause` chain in TS) — it is
  never converted into an `APIError` and NEVER classified transient, so
  neither retry nor the param stripper re-sends a call whose sink failed.
- A provider marks data as seen BEFORE invoking the callback, so a callback
  that fails on the very FIRST delta still yields a partial completion. That
  completion IS the "streamed something" signal the retry guard (§8) and the
  param stripper (§7) read; neither watches the callbacks. Pin: exactly one
  upstream request.
- A tool-callback error (`onToolCall`/`onToolResult`) ends the run via the
  same batch-clearing finalization as an approval interruption (§8): the
  assistant message keeps content/thinking, loses its toolCalls, the batch's
  already-appended results are dropped, and the partial Result returns with
  the error. Note `onToolResult` fires after execution — the tool DID run;
  only its delivery failed.
- Nil/absent callbacks never fire; the empty-delta guards (`onText`/
  `onReasoning` skip empty strings) are unchanged.

Normalized stop reasons: `end_turn`, `tool_use`, `max_tokens`. OpenAI
mapping: `stop → end_turn`, `tool_calls → tool_use`, `length → max_tokens`,
anything else raw. Post-stream inference (both dialects): missing stop
reason becomes `tool_use` if calls were assembled, else `end_turn`.

## 1a. Layering — loop vs provider

The responsibility split the whole design rests on. A port that blurs it
ends up implementing the same concern twice.

- **The loop (`run`) is high-level.** Ask the model, run the tools it asks
  for, feed results back, repeat. It MUST know nothing about HTTP, status
  codes, or backoff.
- **An error reaching the loop is REAL and PERMANENT: the loop STOPS.** It
  is entitled to that assumption — the layer whose job was to make the call
  happen has already given up. The loop MUST NOT retry and MUST NOT expose
  a retry knob (§6).
- **The provider carries out what the loop asks.** When the loop says
  "complete this request", making that true across transient failure (429,
  502, dropped connection, rejected parameter) is the provider's
  responsibility — implementation details of doing the thing, not outcomes
  to propagate. It surfaces an error only when the operation genuinely
  cannot be completed.

Provider owns HOW a call is made and everything transient in the attempt;
the loop owns WHAT calls to make and what to do with results. Hence retry
(§6) and the param stripper (§7) are BOTH provider-side, and neither the
loop config nor the sub-agent config carries a retry policy.

## 2. Executor combinators

- **Composite**: skip nil executors and empty tool names; **first
  registration of a name wins**; advertised order = iteration order across
  executors; zero tools ⇒ the combinator itself is nil/absent ("no tools").
  Unknown call ⇒ recoverable error result `unknown tool: <name>` (never a
  thrown error). `needsApproval` routes to the owning executor; unknown ⇒
  false.
- **ReadonlyView**: only `readonly` tools; nil inner or zero survivors ⇒
  nil. Refusal text: `tool not available to subagent (read-only tools
  only): <name>`. `needsApproval` = allowed && inner's flag.
- **SubsetView(names)**: only named tools; nil inner / empty names / no
  matches ⇒ nil. Refusal text: `tool not in the sub-agent's allowed set:
  <name>`.

## 3. OpenAI dialect

**Endpoint**: POST `BaseURL + "/chat/completions"` (BaseURL includes `/v1`;
default `https://api.openai.com/v1`; trim trailing slashes before joining).
Headers: `Content-Type: application/json`, `Accept: text/event-stream`,
`Authorization: Bearer <key>` when a key is set, `User-Agent` when set,
then caller headers (which may override).

**Body build order (precedence is load-bearing)**:
1. Merge `extra` first, skipping the reserved keys `messages`, `model`,
   `stream`, `tools` (an "evil" extra cannot break routing).
2. Set `model`, `messages`, `stream: true` (always forced true), and
   `tools` only when non-empty. **No `tool_choice` is ever sent.**
3. `max_tokens` only when `maxTokens > 0` (overriding an extra value);
   0 leaves the field to extra or the provider default.
4. `stream_options: {"include_usage": true}` **only when extra did not
   supply a `stream_options` key** (a caller's `{"include_usage": false}`
   must survive verbatim).
5. `prompt_cache_key` when `cacheKey` non-empty.
6. `cache_prompt: true` when `SelfHosted` — NEVER otherwise (hosted
   OpenAI/Azure 400 on unknown fields). SelfHosted is explicit
   configuration in this library (the TS reference derived it from the URL;
   don't).

**Message wire mapping**:
- `system` (when non-empty) is prepended as `{role:"system", content}`.
- **The content presence rule** (the single most-regressed detail): the
  `content` field is ALWAYS emitted — even when empty — EXCEPT for an
  assistant message with empty content that carries `tool_calls` (there the
  spec makes content optional and the model produced none, so it is
  omitted). An empty tool result must serialize as
  `{"role":"tool","content":"","tool_call_id":"..."}` exactly; dropping the
  empty `content` makes upstreams reject the request with
  `invalid message content type: <nil>` / 400, failing the whole turn.
- Assistant `toolCalls` replay as
  `{id, type:"function", function:{name, arguments}}` with `arguments` the
  raw string.
- `thinking` is NOT replayed on this dialect (no wire field);
  `toolIsError` has no wire equivalent.
- Tools: `{type:"function", function:{name, description?, parameters}}`
  with `parameters` = schema or `{"type":"object"}`.

**SSE decode**:
- Scanner processes only `data:` lines (comments/`event:`/blank skipped);
  trim after the prefix; empty payloads skipped; `[DONE]` ends the stream;
  buffer sizes **64 KiB initial / 8 MiB max line** (large tool-call
  argument deltas WILL exceed small defaults).
- Unparseable payload JSON is silently ignored (keep-alive tolerance).
- `choices[].delta.content` → OnText + accumulate.
- Reasoning arrives under TWO field names: `reasoning_content`
  (OpenAI/DeepSeek) and `reasoning` (Ollama); `reasoning_content` wins when
  both are present on one delta. Accumulated reasoning becomes ONE
  `ThinkingBlock{text}` on the final message.
- `choices[].delta.tool_calls` accumulate **by `index`**: first delta for
  an index creates the slot (type defaults `"function"`) and records
  first-appearance order; `id`/`type`/`function.name` overwrite when
  non-empty; `function.arguments` CONCATENATES. Output in first-appearance
  order.
- `choices[].finish_reason` (last non-empty wins).
- `prompt_progress` rides a choices-less chunk → OnProgress, and nothing
  else is read from that chunk.
- `usage` may arrive once on a final chunk (OpenAI) or as a cumulative
  snapshot on EVERY chunk (xAI). See §5 for the merge.
- `timings` (llama.cpp/ollama) may ride any chunk: each occurrence fires
  OnTimings and REPLACES the held snapshot (last wins — never merged, never
  summed); the last one becomes `completion.timings`. No timings ⇒ the
  field stays absent. Anthropic has no equivalent (§4 never fires it).
- **Plain-JSON fallback**: a 200 whose `Content-Type` is NOT
  `text/event-stream` is read as the non-streaming response body and
  reassembled into a `Completion` with `streamed` FALSE -- a server that
  ignores `stream:true` (or a proxy that buffered it) is accepted
  transparently. Reassembly: `choices[0].message` maps onto the message
  (content string or null; `reasoning` becomes a single thinking block;
  `tool_calls` become calls with `arguments` the raw string),
  `choices[0].finish_reason` is normalized and post-inferred like the
  streamed path, and `usage` is normalized as in section 5 with
  `rawUsage`/`reasoningTokens`/`costUsd` captured. A malformed body or a
  response with no choices is a permanent (never-retried) error. The SSE
  path always sets `streamed` TRUE.
- **`PromptCache`** (openai config flag, default false): adds the two
  Anthropic-style ephemeral `cache_control` breakpoints in openai shape --
  a static one on the leading system message (its string content becomes a
  marked one-block array) and a moving one on the last content block of the
  last message (string content becomes a marked one-block array; a block
  array gets the marker on its last block; empty content passes through
  unmarked). Applied to the per-request wire copy only; the caller's
  transcript is never mutated. Intended for Anthropic-fronting gateways
  that pass cache_control through; strict OpenAI-compatible servers 400 on
  the marker.
- **`ReplayReasoning`** (openai config flag, default false): when set, each
  assistant message's accumulated reasoning replays as `message.reasoning`
  (the gateway-extension behavior that keeps a model seeing its own
  chain-of-thought on this dialect); empty or redacted-only thinking
  produces no field. Default false, so strict OpenAI-compatible servers
  never see the unknown field.

## 4. Anthropic dialect

**Endpoint**: POST `BaseURL + "/v1/messages"` (default
`https://api.anthropic.com`). Headers: `Content-Type`, `Accept`,
`anthropic-version` (default `2023-06-01`), `x-api-key` when set,
`User-Agent` when set, then caller headers. Do NOT send the browser-only
`anthropic-dangerous-direct-browser-access` header from a server-side
library.

**Validation**: `maxTokens` must be positive — the Messages API requires
`max_tokens` on every request; fail fast before any I/O, and classify the
failure as permanent (never retried).

**Body build order**: merge `extra` first, skipping reserved keys `model`,
`max_tokens`, `stream`, `system`, `messages`, `tools`; then set the core
fields. The library does NO model gating of `thinking`/`temperature` — what
to send is the caller's job via `extra`.

**System**: emitted as a content-block ARRAY
`[{type:"text", text: system}]` (cache_control lives on blocks, not string
bodies); omitted entirely when empty (and then there is no static cache
breakpoint — the moving one still covers the prefix).

**Message mapping** (build fresh wire structures per request — NEVER mutate
the caller's messages):
- Role: assistant stays assistant; everything else rides as `user`
  (`system` in the transcript is a degenerate case — `Request.system` is
  the system channel on this dialect).
- User messages: plain string `content`.
- Assistant messages: block array in this exact order —
  1. thinking blocks FIRST (required, or tool-use continuations 400):
     `{type:"thinking", thinking, signature}` with signatures replayed
     unchanged, or `{type:"redacted_thinking", data}` for redacted blocks
     (NOTE: redacted_thinking replay is an addition beyond the TS reference
     app, which never handled it; the Go library replays it verbatim per
     the API docs — port this),
  2. `{type:"text", text}` when content non-empty,
  3. `{type:"tool_use", id, name, input}` per call, where `input` is the
     **PARSED JSON object** (invalid, empty, or non-object arguments → `{}`).
  A degenerate assistant message with no blocks at all degrades to string
  content.
- Tool messages: **consecutive** RoleTool messages fold into ONE user
  message whose content is a list of
  `{type:"tool_result", tool_use_id, is_error, content}` blocks (string
  content; `is_error` always present, false by default).
- Tools: `{name, description, input_schema}` (NOT the function wrapper;
  `input_schema`, not `parameters`); nil schema → `{"type":"object"}`. No
  tool_choice.

**Prompt-cache breakpoints** (unless disabled): EXACTLY two ephemeral
markers per request (API max 4):
1. STATIC on the (last) system block — the cache hierarchy is tools →
   system → messages, so this one marker covers the tools array too.
2. MOVING on the **last content block of the last message**: a NON-EMPTY
   string content becomes a one-block text array carrying the marker; a
   non-empty block array gets the marker on a copy of its last block; empty
   strings, empty arrays, and unrecognized shapes pass through UNMARKED
   (caching is an optimization, never a correctness requirement). The
   empty-string case is a deliberate deviation from the TS reference (which
   converted `""` too): the API rejects empty text blocks, and this
   library's own Run can leave a transcript ending on an EMPTY assistant
   message (a turn cancelled after only tool-call deltas streamed, tool
   calls then cleared) — marking that tail would turn a replay of it into a
   guaranteed 400.
Markers exist only in the per-request wire copy; the persistent transcript
must remain marker-free (pin with a test that calls twice and deep-equals
the caller's messages before/after). `DisableCaching` removes both markers.

**SSE decode** (the payload's `type` field discriminates; the SSE `event:`
name is redundant):
- `message_start`: `message.usage` → `input_tokens`, `output_tokens`,
  tri-state `cache_read_input_tokens` / `cache_creation_input_tokens`.
- `content_block_start`: seed block state by `index` — `id`/`name` for
  tool_use, `thinking`/`signature` for thinking, `data` for
  redacted_thinking.
- `content_block_delta` by `delta.type`: `text_delta.text` → OnText +
  accumulate; `thinking_delta.thinking` → OnReasoning + accumulate on the
  block; `signature_delta.signature` accumulates; `input_json_delta.
  partial_json` accumulates the raw JSON string.
- `content_block_stop`: finalize the block — tool_use →
  `ToolCall{id, name, arguments: accumulatedJSON}` (arguments stays the raw
  string); thinking → `ThinkingBlock{text, signature}`; redacted_thinking →
  `ThinkingBlock{redacted: data}`. Blocks that never see stop (a cut-off
  turn) are dropped from the result.
- `message_delta`: `delta.stop_reason` (values `end_turn`, `tool_use`,
  `max_tokens`, `stop_sequence` pass through); `usage.output_tokens` is a
  cumulative snapshot — OVERWRITE, never sum.
- `message_stop`: done. `ping`: ignored. Unparseable payloads: ignored.
- `error`: abort the stream with the dialect-agnostic `APIError` — status
  from Anthropic's documented error-type table (`invalid_request_error` →
  400, `authentication_error` → 401, `permission_error` / `billing_error` →
  403, `not_found_error` → 404, `request_too_large` → 413,
  `rate_limit_error` → 429, `api_error` → 500, `overloaded_error` → 529;
  anything unrecognized → 500, so unknown server aborts stay retryable),
  body = the RAW event JSON. This makes an in-stream overload/rate-limit
  (HTTP 200 + error event) classify as transient exactly like its non-2xx
  counterpart, and a 400-mapped body is still checked against the overflow
  regex. The error event does NOT count as streamed data, so an error event
  that arrives before any content leaves the call retryable; after data, the
  partial completion rides alongside the error and the loop's
  nothing-streamed guard prevents the re-send.

**Usage normalization**: Anthropic `input_tokens` EXCLUDES cached tokens —
`promptTokens = input + cache_read + cache_creation` (so both dialects
report the FULL prompt); `completionTokens = output_tokens`;
`totalTokens = prompt + completion`. Tri-state cache fields pass through as
reported.

**Plain-JSON fallback**: a 200 whose `Content-Type` is NOT
`text/event-stream` is read as the non-streaming Messages body and
reassembled into a `Completion` with `streamed` FALSE. Reassembly: content
blocks map onto the same fields as the stream path (text concatenated;
thinking and redacted_thinking collected with signatures/data; tool_use
calls with their input object re-serialized to the raw argument string),
`stop_reason` is post-inferred like the streamed path, and `usage` is
normalized identically. `completion.rawUsage` is the merged wire-shaped
usage object `{input_tokens, output_tokens, cache_read_input_tokens?,
cache_creation_input_tokens?}` (input_tokens EXCLUDES cached tokens; each
cache field included only when reported); the stream path synthesizes the
same shape from the merged message_start/message_delta state.
`reasoningTokens`/`costUsd` never exist on this dialect. A malformed body
is a permanent (never-retried) error. The SSE path always sets `streamed`
TRUE.

## 5. Usage merging, flooring, cache normalization

- **Merge rule** (per model call): streamed usage objects are monotonic
  snapshots. `merge(prev, next)`: next absent → prev; prev absent OR
  `evidence(next) >= evidence(prev)` → next; else prev. `evidence(u) =
  max(totalTokens, promptTokens + completionTokens)`. Consequences to pin:
  never summed; a regressing final snapshot (zeros) is discarded; **equal
  evidence lets the LATER snapshot win** (it may carry richer cache
  detail); evidence uses max(total, parts) so a total that omits reasoning
  tokens still compares correctly.
- **Finalize-time floor**: `totalTokens = max(totalTokens, prompt +
  completion)`; a genuine surplus (xAI: total = prompt + completion +
  reasoning) is PRESERVED. Applied only when the call finalizes.
- **OpenAI cache decode**: the cache-read count is the LARGEST signal
  present among `prompt_tokens_details.cached_tokens` (OpenAI/vLLM/
  OpenRouter), `prompt_cache_hit_tokens` (DeepSeek), and
  `cache_read_input_tokens` (compat layers) — tri-state (absent when none
  of the fields appeared; an explicit 0 is a report). When cache info was
  reported, `cacheWriteTokens` is an explicit 0 (OpenAI has no separate
  write class); when none was, both stay absent. (The TS reference set
  write=0 unconditionally on any usage object; this library keys it to a
  cache report to honor the tri-state contract — match the library.)
- **CachedTokens()** accessor: cacheRead or 0; clamped to promptTokens when
  promptTokens > 0; floored at 0.

## 6. Errors, retry, overflow

- `APIError {status, body, contextOverflow}` — body is the first 4 KiB of
  the non-2xx response (falling back to the HTTP status text when empty)
  and MUST be embedded in the error's message text: the param-strip
  regexes run against it.
- **Transient** = status 408, 429, or ≥ 500; or any network/transport
  error. NEVER context cancellation/deadline, NEVER any other 4xx
  (including overflow), NEVER the library's own validation errors.
- **Context overflow**: a 400 whose body matches (case-insensitive)
  `prompt (is )?too long|context (length|window)|maximum context|too many tokens|exceeds?.{0,20}(context|token)`
  is flagged `contextOverflow` — surfaced explicitly, never retried.
- **RetryPolicy**: defaults **10 attempts** (1 try + 9 retries), 500ms base,
  `delay = base × 2^(attempt−1)`, no jitter, no cap (the 9 delays sum to
  ~255s — which is why retrying MUST be observable, below). Retry only
  transient
  failures; the final attempt's error surfaces regardless; a sleep
  interrupted by cancellation stops retrying and surfaces the last fn
  error.
- **Retry lives in the PROVIDER and is ON BY DEFAULT.** Both dialect
  constructors MUST apply it to every Provider they build, configured by
  `ProviderConfig.retry` (absent ⇒ the defaults above; a policy capped at
  one attempt ⇒ retry off, and the port SHOULD return the dialect provider
  unwrapped rather than pay for a wrapper that can never fire). Retry is
  deliberately NOT something a caller opts into: an opt-in retry is one a
  caller forgets to enable. The provider is also the only layer that knows
  whether a call streamed anything.
- **The retry condition**: re-attempt ONLY when the failed attempt produced
  no partial completion AND delivered no stream event AND the error is
  transient.
- **Retrying MUST be observable.** Before each backoff, fire
  `StreamEvents.onRetry` with `{attempt (1-based), of, delay, err}` — the
  delay actually about to be waited. Minutes of silent backoff is not an
  acceptable user experience; the host has to be able to show the failure
  and the wait. It fires from the retry layer, NOT from a dialect provider;
  it must NOT mark the call as having "streamed something" (i.e. it is
  not a stream event and never a reason to withhold a retry), and a non-nil return stops the
  retrying and surfaces THAT error in place of the upstream's. No event
  fires for an attempt that succeeds, or for a failure that is not retried.
- **The loop has NO retry knob.** `Run`'s config MUST NOT carry a retry
  policy, and neither must the sub-agent executor's config; both inherit
  whatever the Provider does. A retried call counts as ONE turn, trivially,
  because the loop only sees the outcome. A custom Provider implementation
  owns its own retry.

## 7. Rejected-parameter strip middleware

Wraps a Provider. On a failure that (a) is not a context cancellation,
(b) returned NO partial completion (the §1 streamed-something signal), and (c) whose
error text matches one of four phrasings, the named parameter — matched
against `extra` keys by normalized form — is removed and the call retried
ONCE. The strip is REMEMBERED: subsequent calls through the same wrapper
drop the key up front (the source mutated the shared map in place; the
library keeps its own memory and never mutates the caller's map — either
way the key stays gone for later turns). No match, or the named param not
in extra ⇒ surface the original error, no retry.

The four patterns (case-insensitive; each captures the name):

```
does not support parameter\s+["'`]?([^"'`\s,.;:)]+)
unsupported parameter:?\s+["'`]?([^"'`\s,.;:)]+)
unknown field\s+["'`]?([^"'`\s,.;:)]+)
unrecognized request argument:?\s+["'`]?([^"'`\s,.;:)]+)
```

Captured token cleanup: trim the characters `` space \t " ' ` . , : ; ) ``
from both ends. Normalization: lowercase + remove underscores (so
`reasoningEffort` ≡ `reasoning_effort`).

## 8. The loop (Run)

Shape: the **subagent-style** loop (see the asymmetry note below).

- `maxTurns <= 0` → default **10**.
- Each turn advertises `tools.tools()`; `request.tools` is ignored and
  overwritten (nil executor ⇒ no tools advertised).
- `lastTurn = (turn == maxTurns − 1)`: the request carries **no tools** on
  the last permitted turn.
  - **Asymmetry vs the parent app's loop**: the source chat server's
    parent `Run` still ADVERTISES tools on its capped last turn (its
    lastTurn flag only suppresses persisting the never-executed calls and
    ends the loop); its `RunSubagent` — and this library — WITHHOLD them so
    the model must answer. Port the library's (subagent) behavior.
- Continue looping while the executor is non-nil… actually: while the turn
  produced tool calls AND !lastTurn. Per call, in order:
  1. fire OnToolCall — a thrown/returned error ⇒ the batch-clearing
     finalization below, partial Result + that error;
  2. approval gate: only if `needsApproval(name)` — nil Approver ⇒ deny
     with `DeniedMessage`; `ask` false ⇒ deny with `DeniedMessage` (exact
     string: `The user denied permission to run this tool.`), tool NOT
     executed, loop continues; `ask` THROWS ⇒ **approval-cancel
     finalization**: truncate the transcript back to the assistant
     message (dropping this batch's already-appended tool results), clear
     that assistant message's toolCalls (content/thinking preserved),
     return the partial Result together with the error — the transcript
     stays replayable with no orphans;
  3. execute; an executor throw/error becomes
     `tool execution failed: <message>` with isError (the loop NEVER
     aborts on tool failure);
  4. fire OnToolResult — a thrown/returned error ⇒ the same batch-clearing
     finalization (the tool ran; its executed result is dropped from the
     transcript); else append
     `{role:"tool", content, toolCallID, toolIsError}`.
- **Stuck detection** (on the tool branch, before the batch executes):
  fingerprint the turn's tool calls as their names + raw arguments in
  order — call IDs are EXCLUDED (providers mint a fresh one per call, so
  including them makes every batch unique and the detector dead code).
  Identical to the previous turn's fingerprint ⇒ increment the repeat
  count; anything else ⇒ reset it to 1.
  - repeats >= `StuckFailAt` (6): end the run — the batch is NOT executed,
    the assistant message is appended with its toolCalls cleared (the
    mid-stream-cancel shape), and the partial Result rides alongside an
    error matching `ErrStuck` whose text is
    `agentic: model is stuck repeating the same tool calls: <n> identical turns in a row`.
    The port's equivalent must be identifiable the same way (a sentinel or
    subclass, not a string match) and must classify as PERMANENT.
  - repeats == `StuckNudgeAt` (3): after that batch's tool results, append
    ONE user turn with stuckNudgeInstruction, verbatim:
    `You have now requested the same tool calls several times in a row and received the same results each time. Repeating them again cannot tell you anything new. Do something different: act on the results you already have, call a different tool, or write your final answer now. Another identical request ends this run.`
  - Both thresholds are CONSTANTS, not config: a verbatim repeat is never
    the model working, so there is nothing to tune.
- Internal turn hook: the loop exposes a package-internal per-turn hook
  (1-based turn number, fired as each NUMBERED turn begins; the stall
  wrap-up call is not a numbered turn). It is not public API — it exists
  solely so the built-in subagent executor (§10) can emit its `turn`
  activity at exactly the source's emission points. The port needs an
  equivalent internal seam.
- **Public per-turn hooks** (in addition to the internal seam, which stays
  byte-for-byte): `Events.onTurnBegin(turn, req)` fires before each model
  call (numbered turns 1..maxTurns, the stall wrap-up as maxTurns+1), with
  a pointer to the per-call request the hook may MUTATE (the change applies
  to that one call only, never the persistent transcript; wind-down prompt
  injection rides this). `Events.onTurnEnd(turn, completion, err)` fires
  after each call (completion nil when the call produced none). A non-nil
  return aborts the run like any other callback error: onTurnBegin aborts
  before the call (nothing counted), onTurnEnd after it (the completed data
  is kept, finalized like a mid-stream break).
- With a nil executor, a hallucinated call gets the teaching result
  `unknown tool: <name>` and the loop continues (deviation from the source
  subagent, which ended the loop; the teaching behavior is deliberate).
- **Retry** is NOT the loop's concern (see §6): the Provider re-attempts
  internally, so a retried call counts as ONE turn here trivially — the
  loop only sees the outcome.
- **Mid-stream break/cancel**: the provider returns a partial completion +
  error; the loop appends the partial assistant message with its
  **toolCalls cleared** (they never executed) and returns the partial
  Result + error. The partial call's usage snapshot IS recorded.
- **Ending with text**: trimmed non-empty content ends the loop; the final
  assistant message is appended with any dangling (capped last turn)
  toolCalls cleared.
- **Stall fallback** (content empty at loop end):
  - with an executor: one extra TOOL-LESS wrap-up call on transcript +
    user(wrapUpInstruction). The stalling turn's assistant message is NOT
    in that request (it was never appended — only the tool branch
    appends), so the wrap-up cannot be rejected for an unanswered tool
    call. Non-empty wrap-up content becomes the final (transcript gains
    the wrap-up user message + the answer); a wrap-up error is swallowed
    and falls through. Note the wrap-up is an EXTRA call: turns can reach
    maxTurns + 1.
  - last resort: final content = trimmed content, else the concatenated
    thinking text, else the placeholder `(subagent produced no output)`.
  - wrapUpInstruction, verbatim:
    `Stop researching and write your final answer now, using only the information already gathered above. Do not call any tools and do not keep thinking -- output the complete, self-contained report that directly answers the task.`
- `usages`: one entry per model call that produced a completion (success or
  partial), in order; a fully-failed call contributes none. `turns` counts
  logical model calls (retries collapse).
- On error, Run returns the Result-so-far (non-nil) alongside the error,
  except for the nil-Provider misconfiguration.

## 9. Compaction and OneShot

- `CompactRequestText`, verbatim:
  `Summarize this entire conversation in detail for a future instance of yourself to pick up. Output only the summary.`
- `Compact`: append `user(CompactRequestText)` to the history (per-call
  copy), send NO tools, ONE call, `request.system` passed through (the
  summarizer brief is the caller's — the reference app's system prompt was
  app-specific: identity paragraph + "capture the conversational context"
  bullets: goals/requests in order, stylistic intent, decisions and
  dead-ends, unresolved items + next step, "output ONLY the summary").
  Trimmed empty summary ⇒ error (`the model returned an empty summary;
  nothing was compacted` — the Go error string is lowercased per Go
  convention; the TS reference threw the same sentence capitalized).
  Result messages = exactly
  `[user(CompactRequestText), assistant(summary)]` — a valid round ending
  on assistant so the next real prompt continues clean alternation.
- `OneShot(p, req, timeout)`: strip tools, apply the timeout when > 0, ONE
  attempt (never retried), return the trimmed final text + usage. Callers
  compose with a detached context for fire-and-forget.

## 10. Built-in tool executors (plugins)

The `ToolExecutor` seam IS the plugin interface: the built-ins are ordinary
executors composed via the Composite; callers who don't use them see zero
behavior change. Both are ports from the source application; the strings
below are contract.

Both executors use `config.provider` exactly as given — never wrapped
implicitly. The source application routed every one of these model calls
(the sub-agent's nested loop, the context-summary briefing, and the web
summary) through its param-strip recovery layer (§7), so a caller/port
reproduces the source exactly by handing the built-ins a
param-stripper-wrapped Provider — the same wrapped value given to the
loop's own config.

### 10a. run_subagent (`NewSubagentExecutor(SubagentConfig)`)

Config: `provider`, `model`, `maxTokens`, `extra`, `tools` (the parent's
FULL executor; there is deliberately NO retry field — see §6), `parentSystem` +
`parentMessages` (the share_context source), `maxTurns` (<= 0 ⇒ the loop
default 10), `gate?`, `systemPrompt?` (empty ⇒
`DefaultSubagentSystemPrompt`), `onActivity?`.

- Tool name exactly `run_subagent`; `readonly` FALSE (so ReadonlyView drops
  it from any sub-agent's default toolset) and the tool is excluded from
  its own grantable set — a sub-agent can never spawn another.
  `needsApproval` is always false (approval wiring is the caller's).
- **Default sub toolset** = `ReadonlyView(tools)`. **`allowed_tools`**
  selects from the FULL grantable set (every parent tool except
  run_subagent, empty names skipped), so naming a non-read-only tool
  explicitly GRANTS it (`SubsetView` over the full executor). Matching:
  exact advertised name, else unambiguous bare-name fallback — request `x`
  matches a single tool ending in `__x`. Blank entries are skipped.
- **Advertised schema**: the static schema (the Go literal is the source of
  truth) with, when grantable tools exist, `allowed_tools.description`
  rewritten to list them — each non-read-only name suffixed
  ` (modifies state)` — and `allowed_tools.items` = `{type:"string",
  enum:[exact names]}`. No grantable tools ⇒ the static schema verbatim.
- **Execution**: unknown name ⇒ `unknown tool: <name>`; no provider/model ⇒
  `run_subagent is unavailable: no model is configured for the sub-agent`;
  bad JSON ⇒ `invalid run_subagent arguments: <err>`; blank prompt ⇒
  `run_subagent requires a non-empty prompt describing the task`; gate
  acquisition failure ⇒ `run_subagent was cancelled before it could start:
  <err>`. The nested run is this library's §8 loop — one user message (the
  composed task), the subagent system prompt, the config's
  model/maxTokens/extra, and an approve-everything Approver (the
  source loop never consulted the approval flow: the explicit grant IS the
  authorization). A nested-run error ⇒ `sub-agent failed: <err>`; success ⇒
  the TRIMMED final content as the tool result (the nested loop's stall
  wrap-up and `(subagent produced no output)` placeholder apply).
- **allowed_tools teaching errors** (exact):
  - `run_subagent: the sub-agent has no tools available, so allowed_tools cannot be applied -- omit it.`
  - `run_subagent: allowed_tools names no available tool: <unknown, list>. Available tools: <available, list>. Use these exact names, or omit allowed_tools to allow every read-only tool.`
  - `run_subagent: allowed_tools contained no usable tool names. Available tools: <available, list>.`
- **share_context** (case-insensitive, trimmed): `none`/empty ⇒ prompt
  alone; `custom` ⇒ the trimmed `custom_context` (blank ⇒
  `share_context=custom requires custom_context text`); `full` ⇒
  `RenderTranscript` of the parent context (parentSystem prepended as a
  system message, then parentMessages — selection indices count over that
  combined list); `last_n` ⇒ the last `context_message_count` (missing ⇒
  `share_context=last_n requires context_message_count (a positive
  integer)`); `messages` ⇒ `context_message_indices`, 1-based from the most
  recent, de-duplicated, chronological (missing ⇒ `share_context=messages
  requires context_message_indices (1 = the most recent message)`);
  `summary` ⇒ one bounded (2 min) tool-less OneShot briefing call — empty
  parent context shares nothing with NO model call; a failure ⇒ `failed to
  summarize the parent conversation: <err>`; anything else ⇒ `unknown
  share_context mode "<mode>" (want none, full, last_n, messages, summary,
  or custom)` (mode quoted).
- **Task composition**: a non-blank block is folded as
  `Context shared from the parent conversation (background only):\n\n` +
  block + `\n\n----------------------------------------\n\nYour task:\n\n`
  + prompt; a blank block leaves the prompt untouched.
- **Transcript rendering** (`RenderTranscript`): per message, `<Label>:\n`
  (System/User/Assistant/`Tool result`/`Message` for empty role, else the
  role with its first letter upper-cased), the trimmed content on its own
  line when non-empty, then one `[requested tool <name> with <args>]` line
  per tool call (` with <args>` omitted for blank args), a blank line
  between messages; the whole render is trimmed and TAIL-capped at 200 000
  runes behind `[... earlier shared context truncated ...]\n\n`.
  `SelectLastN`: n <= 0 ⇒ empty; n >= len ⇒ all. `SelectByEndIndices`:
  out-of-range ignored, duplicates collapse, chronological output.
- **Summary prompts** (verbatim): system = `You condense a conversation
  into a briefing for a sub-agent that will carry out a related task.
  Capture the salient facts, decisions, constraints, and any specific
  identifiers, names, or values the sub-agent will need to act correctly.
  Be faithful and concise; omit pleasantries and meta-commentary. Output
  only the briefing.`; user = `Conversation to brief the sub-agent on:\n\n`
  + transcript.
- **DefaultSubagentSystemPrompt** (verbatim): `You are a sub-agent launched
  by another assistant to carry out one focused, read-only task. You cannot
  modify anything; you have only read-only tools (web and repository
  access) to gather information. Work autonomously — you cannot ask
  follow-up questions, and you do not see the parent conversation, only the
  task below. Use the available tools as needed, then return a single,
  self-contained final report that directly answers the task: give the
  concrete findings the calling assistant needs, not a narration of your
  process. Be concise and factual.`
- **Tool description**: ported verbatim from the source EXCEPT one
  deliberate adaptation — the source's CAPABILITIES sentence enumerated its
  own app's tools ("fetch a web page (web_fetch), read GitHub repositories
  (repo_read: ...), and any read-only MCP tools that are enabled"); the
  library drops that enumeration ("By DEFAULT it may use only read-only
  tools and causes no side effect."). The Go literal is the contract.
- **Gate**: `NewGate(n)` = capacity-n semaphore (n < 1 clamps to 1); `nil`
  gate = unlimited; `acquire(ctx)` blocks until a slot or ctx-done and
  returns a release func. The source app used capacity 1 process-wide;
  the capacity is the caller's choice here.
- **Activity telemetry** (`onActivity`): steps
  `{callID, kind, turn, tool, detail, isError}` with kinds `turn` (1-based,
  at each numbered nested-turn start), `tool_call` (before execution,
  detail = argument preview), `tool_result` (after, detail = result-text
  preview, isError from the recorded result). `callID` = the parent
  run_subagent call's id. Previews are whitespace-flattened
  (`fields`-join) and capped at 160 runes (`157 + "..."`). Telemetry only —
  never fed back to any model, and the nested run's stream events NEVER
  reach the parent's StreamEvents.

### 10b. web_fetch (`NewWebFetchExecutor(WebFetchConfig)`)

Config: `httpClient?` (nil ⇒ a 45 s-timeout client), `userAgent?` (sent on
the tool's outbound requests), `tikaURL?`, `provider`/`model`/`maxTokens`/
`extra` (the summary path), `blockURL?: (url) => string`.

- Tool name exactly `web_fetch`; `readonly` TRUE; `needsApproval` false.
  Description (verbatim): `Fetches one http/https URL with an
  unauthenticated, plain HTTP GET and returns cleaned page content.
  Optionally provide summary_prompt to have the same model summarize the
  cleaned content before it is returned.` Schema: the Go literal
  (`url` required, `summary_prompt` optional).
- **Validation** (exact error texts): blank ⇒ `url is required`; parse
  failure ⇒ `invalid url: <err>`; non-http(s) ⇒ `web_fetch only supports
  http and https URLs`; hostless ⇒ `web_fetch URL must include a host`;
  userinfo ⇒ `web_fetch rejects URLs containing userinfo credentials`.
  Unknown tool name ⇒ `unknown tool: <name>`; bad JSON ⇒
  `invalid web_fetch arguments: <err>`.
- **Block seam**: `blockURL(validatedURL)` returning non-empty text refuses
  the fetch with exactly that text as a recoverable error result (the
  source used this slot for its workspace-repository redirect; the hook is
  the contract, the policy is not).
- **Fetch**: plain GET; non-2xx ⇒ `web_fetch GET failed: status <status>`;
  transport error ⇒ `web_fetch GET failed: <err>`; body cap 5 MiB
  (5242880) ⇒ `web_fetch response exceeds <n> bytes`.
- **Cleaning**: when `tikaURL` is set, PUT the raw bytes to
  `<tika>/tika` (`Accept: text/plain`, fetched Content-Type forwarded,
  extracted cap 800 000 bytes); success with non-empty text ⇒ normalized
  Tika text; ANY failure or empty output falls back SILENTLY to the
  built-in HTML cleanup (strip comments; drop
  script/style/noscript/template/svg/canvas/head subtrees; `<br>` and
  closing block tags ⇒ newlines; strip remaining tags; unescape entities;
  collapse intra-line whitespace and blank-line runs). Result rune-capped
  at 200 000 ⇒ prefix note `Note: cleaned content was truncated to 200000
  runes.\n`; blank result ⇒ `(no extractable content)`.
- **Result shapes**: `URL: <url>\n` (+ optional truncation note) + `\n` +
  cleaned; with `summary_prompt`: same prefix + `\nSummary:\n` + summary.
- **Summary path**: no provider/model ⇒ `web_fetch summary requested, but
  no model is available for summarization`; one bounded (2 min) tool-less
  OneShot with system (verbatim) `You summarize cleaned web content for
  another assistant in the same conversation. Follow the provided summary
  instructions. If the content is thin, blocked, or unrelated, say so
  plainly.` and user = `Fetched URL:\n<url>\n\nSummary instructions:\n` +
  (trimmed instructions, else `Produce a concise, faithful summary of the
  fetched content.`) + `\n\nCleaned fetched content:\n` + cleaned. A call
  failure ⇒ `web_fetch summary failed: <err>`; empty output ⇒
  `web_fetch summary returned empty output`.

## 11. Deliberate cuts (do NOT implement in ts/ either)

- DB/persistence (message trees, leaf advancement, statuses) — the library
  works on a flat in-memory transcript.
- Server-side SSE fan-out (the source's meta/token/done/error event
  protocol and sinks). The subagent activity feed is a plain callback, not
  an event protocol.
- The source's per-user tool settings (`disabled_tools` / `ask_tools`
  lists, tool-name namespacing `<Server>__<tool>`): the built-in executors
  are composed or not composed; gating them is caller-side wrapping.
- Title sanitization (`sanitizeTitle`, `<think>`-stripping, length caps) —
  `OneShot` returns raw trimmed text.
- Structured tool-result content parts (multimodal blocks); `ToolResult` is
  text-only.
- Model-gated thinking/temperature/effort tables (the reference app's
  per-model gates) — callers own `extra`.
- Pricing/cost computation.
- Steering (mid-turn user-message injection).
- Model-id namespacing (`<upstream>/<model>` stripping) — callers pass the
  bare model id.
- Timings tok/s wall-clock synthesis (the DECODE is implemented — §3; the
  fallback synthesis stays the consumer's).
- The `isHostedOpenAI` URL heuristic — replaced by the explicit
  `SelfHosted` flag.
- Web-fetch failure logging (the source warned on Tika failures via its
  logger; the library falls back silently — no logger dependency).

## 12. Test pins the ts/ suite must replicate

- The tool message JSON pin: `{"role":"tool","content":"","tool_call_id":"call_1"}`.
- Merge table incl. equal-evidence-later-wins and regression-discard;
  floor-preserves-surplus.
- Anthropic: two-breakpoint placement (system + tail), transcript
  unchanged after two calls, thinking replayed first with signature,
  redacted_thinking round-trip, tool_result folding with is_error,
  max_tokens fail-fast, tri-state cache fields, full-prompt normalization,
  DisableCaching strips all markers, empty-string tail left unmarked (and
  never converted to a text block), in-stream error events mapped to
  APIError per the type→status table (529/429 transient, 400 checked for
  overflow, raw event JSON as body; before any data the call stays
  retryable — pinned end to end by a Run-over-httptest retry — while after
  data the partial completion is returned alongside the error).
- OpenAI: both reasoning field names (reasoning_content precedence),
  split tool-call deltas by index, cumulative usage on every chunk,
  prompt_progress, stream_options default + caller override,
  cache_prompt/SelfHosted, prompt_cache_key, reserved-extra protection.
- Loop: denied string, Ask-error batch clearing, final-turn withheld
  tools, stall wrap-up request shape (tool-less; instruction as the last
  user message; stalled assistant absent), hallucinated-call teaching
  error, no-retry-after-partial, retried-call-is-one-turn, plus
  retry-on-by-default straight from a constructor with no policy set.
- Param stripper: all four phrasings, camelCase↔snake_case normalization,
  retry-once, memory across calls, never on cancel/after delivery.
- Provider factories: one per dialect building that dialect's
  implementation, required baseURL failure (permanent) on both.
- Callback errors: per-callback abort (text/reasoning/usage/progress/
  timings), the caller's sentinel reachable from the returned error, never
  transient/never APIError, partial completion carries the state so far
  (failing delta included), first-delta failure still means exactly ONE
  upstream request end-to-end through Run, onToolCall/onToolResult abort
  with the batch-clearing finalization (executed result dropped).
- Timings: wire decode of all four fields, last-snapshot-wins, per-snapshot
  OnTimings, absent ⇒ `completion.timings` absent and no synthesis;
  Anthropic never fires it. `usageReported` true for a reported all-zero
  snapshot, false when no usage arrived.
- run_subagent: default toolset = read-only subset; allowed_tools exact +
  unambiguous bare-name + ambiguous/unknown teaching errors (exact texts,
  listing the available tools); non-read-only grant executes (no approval
  prompt inside the nested run); run_subagent absent from schema enum and
  never grantable; every share_context mode's composed task message (exact
  delimiters) + its misuse errors; summary-mode briefing request shape
  (system prompt, tool-less, `Conversation to brief the sub-agent on:`
  prefix) and no-call-on-empty-context; gate serialization + cancelled
  acquire text; activity sequence turn/tool_call/tool_result with stamped
  callID and 160-rune flattened previews; nested-run failure ⇒
  `sub-agent failed: ...`; placeholder pass-through.
- web_fetch: advertisement (readonly, exact description), HTML cleanup
  pipeline output, truncation note, 5 MiB body cap text, non-2xx/transport
  error texts, all URL-validation texts, block-hook refusal (hook text IS
  the result; hook sees the validated URL), Tika request shape
  (PUT /tika, Accept text/plain, forwarded Content-Type) + silent
  fallback on failure/empty, summary request shape (system prompt, user
  input layout, tool-less) + `no model available` / `failed` / `empty
  output` texts, `(no extractable content)` placeholder.
- Per-turn hooks (`onTurnBegin`/`onTurnEnd`): numbered 1..maxTurns in order,
  the wrap-up as maxTurns+1; a begin-hook mutation of the per-call request
  reaches the provider for that call only; begin-error aborts before the
  call (no turn counted, provider never called), end-error aborts after it
  (completed data kept); the internal turnHook still fires once per numbered
  turn.
- Completion extras + streaming: `rawUsage` verbatim (unknown provider
  fields survive), `reasoningTokens` from
  `completion_tokens_details.reasoning_tokens`, `costUsd` from `cost` (wins
  over `estimated_cost`), all tri-state absent; plain-JSON fallback on both
  dialects (non-SSE 200 makes `streamed` false, exact reassembly incl. tool
  calls/thinking/usage, malformed body permanent); `streamed` true on the
  SSE path.
- openai `PromptCache`: exactly two `cache_control` breakpoints (static
  system + moving tail; a role:"tool" tail is marked in openai shape), the
  stored transcript marker-free, absent by default.
- openai `ReplayReasoning`: assistant thinking replays as `message.reasoning`
  only when enabled; empty/redacted thinking emits no field; absent by
  default.
