---
name: transcript-decay
description: Check a chat host, gateway or dialect for transcript decay (a model's reasoning lost between a response and the next request), or fix it. Run as /transcript-decay check or /transcript-decay fix.
argument-hint: check|fix
user-invocable: true
---

# /transcript-decay

The argument is `$ARGUMENTS`. With `check`, run the check below and report each finding with its file and line. With `fix`, run the check, then fix every finding by the matching recipe, add the test that proves it, and push. With no argument, run `check`.

Transcript decay is a field of a model's own output that does not come back on the next request. The field is almost always reasoning: thinking text, a signature, an encrypted payload. Every call succeeds and nothing logs an error. The model just gets worse over a long task, because each turn it cannot see what it thought before. `docs/postmortem-transcript-decay.md` holds the incident.

## The one question

For each hop between the code and the provider, ask: **does every reasoning field the response carried appear in the next request?** One hop that drops a field is enough. A gateway is a hop. A store that the host reads the transcript back from is a hop.

## What each provider sends and reads

| Provider and API | Where thinking arrives | What it reads on replay |
| --- | --- | --- |
| Ollama, OpenAI-compatible chat (local and ollama.com) | `reasoning` | `reasoning` only. `reasoning_content` is ignored. |
| Ollama, Responses | the summary and `encrypted_content` | the summary only |
| OpenRouter, chat completions | `reasoning`, plus `reasoning_details` | `reasoning_details` for encrypted reasoning and signatures |
| OpenRouter, Responses, a non-OpenAI model | a reasoning item with `content:[{type:"reasoning_text"}]`, an empty summary, no `encrypted_content` | the item's `content` |
| DeepSeek-style chat completions | `reasoning_content` | `reasoning_content` |
| Anthropic Messages | a `thinking` block with a `signature`, or `redacted_thinking` | the block, signature intact |

## check

Work through every hop. For each, find the code that writes the outbound request and the code that reads the response, then look for these shapes.

1. **One spelling on a chat-completions write.** An assistant message that sets `reasoning_content` and no `reasoning`, or the reverse. Ollama drops the first. Search the request structs:

   ```
   rg -n 'json:"reasoning_content' --type go --type rust --type ts
   rg -n 'json:"reasoning"' --type go --type rust --type ts
   ```

   A struct with one tag and not the other is a finding, unless the provider it targets rejects the other name.

2. **`reasoning_details` read and never written.** The response parser keeps it, but the request builder, or a gateway's conversion from its internal type, and never copies it out. Search `reasoning_details` and `ReasoningDetails` and follow each value from the response to the request.

3. **A reasoning item dropped for its payload shape.** A Responses converter that skips an item when `encrypted_content` is empty, or the summary is empty, or both. An item with `reasoning_text` content and neither of those is still the model's thinking. Search for the skip:

   ```
   rg -n 'encrypted_content|EncryptedContent|reasoning_text' --type go --type rust --type ts
   ```

   Read each condition around a `continue`, a `skip` or a filter.

4. **A thinking block dropped for a missing signature.** A projection or a store that keeps a block only when it has a signature or a redacted payload. That is correct for Anthropic and incorrect for a `reasoning_text` block, which has neither. In agentic-loop the block carries `Raw: true`. Look for a condition on `Signature == ""` and check whether it also tests `Raw`.

5. **A store that keeps a subset of the fields.** A host that writes thinking to its own struct or columns and reads it back later. Compare that struct field by field with `core.ThinkingBlock` (`Text`, `Signature`, `ID`, `Redacted`, `Raw`). The `reasoning_details` payload is `ThinkingMessage.Details` on the document node, so a host that stores the document must keep it too. A missing field is a finding. A host that stores the whole `agentic.ThinkingBlock` as JSON gets every field for free.

6. **Prove it on the wire.** Static reading misses a hop. For each hop, a fake upstream answers turn one with reasoning in the provider's own shape, and records the body of turn two. Assert that the reasoning is in that body under the name the provider reads. In agentic-loop this is `core/conservation_test.go`. `TestTranscriptConservation` covers every dialect, and `TestResponsesConservesReasoningText` covers the `reasoning_text` case. Run it with `go-toolchain` from the repository root. A host that has no such test is a finding by itself.

Report each finding as: the hop, the file and line, the field that is lost, and the provider that loses it. Say which hops you checked and found clean, so the report covers the whole chain.

## fix

Fix the hop that drops the field. Never work around it in the next hop.

- **One spelling.** Write every name the response carried: `reasoning_content` and `reasoning` and `reasoning_details`. Narrow the set only for a provider that rejects a name, and only after that rejection names it. Grok-build's `ChatMessageProfile` does this with `accepts_reasoning_content` and `accepts_reasoning`.
- **`reasoning_details` not forwarded.** Copy it from the stored assistant message to the outbound one, verbatim. It is an opaque array. Never rebuild it.
- **A dropped `reasoning_text` item.** Read it as the thinking itself and mark it raw. Replay it as `{type:"reasoning", id, summary:[], content:[{type:"reasoning_text", text}]}`. In agentic-loop, `respItem.reasoningText` reads it and `respReasoningItem` writes it.
- **A signature check that drops raw thinking.** Keep the block when `Raw` is set, as well as when it has a signature or a redacted payload.
- **A store that loses a field.** Add the field to the stored struct and to the read-back. Store the library's own type where the format allows.

Each fix gets a conservation test at the hop it fixes: response in, next request out, field present under the provider's name. A fix with no such test is not done.

When the fix is in another repository, add that repository with `add_repo`, fix it there, and open its pull request. State the merge order when the fix spans repositories.
