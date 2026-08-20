package commonai

// The Messages API wire vocabulary on the way OUT: the transcript as content
// blocks, the two prompt-cache breakpoints, and the shapes that carry an
// image. The transport and the stream decoding live in anthropic.go.

import (
	"encoding/json"
	"strings"
)

// anWireMessages maps the neutral transcript onto Messages API messages,
// building fresh wire structures every call (the caller's Messages are never
// touched). Assistant messages become content-block arrays with thinking
// blocks replayed FIRST (required, or tool-use continuations 400 — signatures
// and redacted payloads replayed verbatim), then the text, then tool_use
// blocks whose input is the PARSED argument object. Consecutive RoleTool
// messages fold into ONE user message of tool_result blocks. Everything else
// (user, and any stray system message — Request.System is the system channel
// on this dialect) rides as a user message with string content.
func anWireMessages(msgs []Message) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(msgs))
	for i := 0; i < len(msgs); i++ {
		m := msgs[i]
		switch {
		case m.Role == RoleTool:
			var blocks []map[string]any
			for ; i < len(msgs) && msgs[i].Role == RoleTool; i++ {
				t := msgs[i]
				blocks = append(blocks, map[string]any{
					"type":        "tool_result",
					"tool_use_id": t.ToolCallID,
					"is_error":    t.ToolIsError,
					"content":     t.Content,
				})
			}
			i-- // the outer loop increments past the run's last message
			out = append(out, map[string]any{"role": "user", "content": blocks})
		case m.Role == RoleAssistant:
			content := anAssistantContent(m)
			if content == nil {
				// The turn says nothing this dialect can carry. An empty text
				// block fails the whole request ("text content blocks must
				// contain non-whitespace text"), and the API combines
				// consecutive same-role turns, so the drop changes nothing
				// else. The Responses dialect omits the same turn already.
				continue
			}
			out = append(out, map[string]any{"role": "assistant", "content": content})
		default:
			content, err := anUserContent(m)
			if err != nil {
				return nil, err
			}
			out = append(out, map[string]any{"role": "user", "content": content})
		}
	}
	return out, nil
}

// anUserContent builds a user message's content: the plain string when it is
// only text, and the block array when it carries an image. The source keeps
// the form it was supplied in -- base64 for inline bytes, url for a reference
// -- because converting between them means either fetching a URL the caller
// did not ask this library to fetch, or inventing one for bytes.
func anUserContent(m Message) (any, error) {
	if !hasImage(m) {
		return m.Content, nil
	}
	var blocks []map[string]any
	for _, p := range m.EffectiveParts() {
		switch v := p.(type) {
		case TextPart:
			if v.Text != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": v.Text})
			}
		case ImagePart:
			source, err := anImageSource(v)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, map[string]any{"type": "image", "source": source})
		}
	}
	return blocks, nil
}

// anImageSource is the Messages API's source object for one image.
func anImageSource(i ImagePart) (map[string]any, error) {
	switch {
	case i.Src != "":
		return map[string]any{"type": "url", "url": i.Src}, nil
	case i.inline():
		return map[string]any{
			"type": "base64", "media_type": i.MediaType, "data": i.Data,
		}, nil
	}
	if _, err := imageRef(DialectAnthropic, i); err != nil {
		return nil, err
	}
	return nil, Unsupported(DialectAnthropic, "this image", "it carries neither a source nor any bytes")
}

// anAssistantContent builds an assistant message's content blocks in replay
// order: thinking first, then text, then tool_use. It returns nil for a turn
// that produces no block at all -- no text, no tool call, and no replayable
// thinking. The caller drops such a turn.
func anAssistantContent(m Message) any {
	blocks := make([]map[string]any, 0, len(m.Thinking)+1+len(m.ToolCalls))
	for _, tb := range m.Thinking {
		if tb.Redacted != "" {
			blocks = append(blocks, map[string]any{"type": "redacted_thinking", "data": tb.Redacted})
			continue
		}
		if tb.Signature == "" {
			// Not replayable, so not replayed. The signature is what makes a
			// thinking block the model's own; Anthropic rejects one without a
			// valid signature, so sending it would fail the whole turn rather
			// than lose one block. A caller whose transcript carries text-only
			// reasoning -- from another dialect, or from storage that kept the
			// text and dropped the signature -- must not have every turn 400.
			continue
		}
		blocks = append(blocks, map[string]any{"type": "thinking", "thinking": tb.Text, "signature": tb.Signature})
	}
	if m.Content != "" {
		blocks = append(blocks, map[string]any{"type": "text", "text": m.Content})
	}
	for _, tc := range m.ToolCalls {
		blocks = append(blocks, map[string]any{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Name,
			"input": parseToolInput(tc.Arguments),
		})
	}
	if len(blocks) == 0 {
		return nil
	}
	return blocks
}

// parseToolInput parses a tool call's raw argument JSON into the object the
// Messages API expects; invalid, empty, or non-object arguments become {}.
func parseToolInput(args string) map[string]any {
	s := strings.TrimSpace(args)
	if s == "" {
		return map[string]any{}
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return map[string]any{}
	}
	if obj, ok := v.(map[string]any); ok {
		return obj
	}
	return map[string]any{}
}

// markTranscriptTail places the moving prompt-cache breakpoint on the last
// content block of the last message, so the next request cache-hits
// everything through this one's tail. It operates on the per-request wire
// structures only (freshly built by anWireMessages), which is what keeps the
// caller's transcript marker-free — old moving markers can never accumulate.
// A non-empty string content becomes a one-block array; a non-empty block
// array gets the marker on its last block; empty strings, empty arrays, and
// unrecognized shapes pass through unmarked — caching is an optimization,
// never a correctness requirement. The empty-string case matters: the API
// rejects an empty text block, and a user turn can carry empty content, so
// converting "" into an empty marked text block would turn a valid request
// into a 400.
func markTranscriptTail(msgs []map[string]any) {
	if len(msgs) == 0 {
		return
	}
	last := msgs[len(msgs)-1]
	switch c := last["content"].(type) {
	case string:
		if c == "" {
			return
		}
		last["content"] = []map[string]any{{"type": "text", "text": c, "cache_control": cacheEphemeral}}
	case []map[string]any:
		if len(c) == 0 {
			return
		}
		c[len(c)-1]["cache_control"] = cacheEphemeral
	}
}
