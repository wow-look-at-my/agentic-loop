# The completions dialect

`DialectCompletions` speaks the legacy OpenAI `POST /completions` API. The request carries one `prompt` string. The response streams `choices[].text`. There is no message list, no tools field and no image field.

## Why it exists

Some models are served on this path and refused on chat completions. OpenAI's instruct models answer a chat call with a refusal that names `v1/completions`, and `DialectRefused` reads that direction. A self-hosted server can serve a base model with no chat template only here. A gateway such as OpenRouter advertises `/api/v1/completions` per model beside the chat APIs. So a host that routes per model needs a provider for this path.

## The prompt

The transcript becomes labelled turns, separated by a blank line, and ends on an open `Assistant:` for the model to continue:

```
System: <system prompt>

User: <text>

Assistant: (reasoning) <reasoning text>
<text>
(tool call <name> <id>) <arguments>

Tool result (<id>): <content>

Assistant:
```

The default stop sequences are `"\nUser:"` and `"\nTool result"`. The model stops where it starts to write the next turn. A caller's own `stop` param replaces them. The leading space every answer to `Assistant:` starts with is trimmed from the result.

## What it keeps, what it refuses

- **Kept as text:** the system prompt, every user and assistant text, reasoning text, tool calls (name, id, arguments) and tool results. A transcript that started on another dialect keeps every fact on the way in.
- **Dropped, declared in `TestTranscriptConservation`:** a reasoning signature and a reasoning item id. A prompt string has no field for either, and neither is readable text.
- **Refused:** a declared tool, and an image. The request fails with `UnsupportedError` before anything is sent. A host that runs such a model without tools removes them itself, out loud.

## Params and limits

Params are read from the `openai` namespace (`ParamsFor(DialectOpenAI)`): the same vendor, the same names (`temperature`, `top_p`, `seed`, `stop`, ...). `prompt`, `model`, `stream`, `messages` and `tools` are the core's.

A request with no `max_tokens` gets `DefaultCompletionsMaxTokens`. The API's own default is tokens, which cuts off almost every answer. `stream_options.include_usage` is sent unless the caller sets its own. The last chunk carries usage.
