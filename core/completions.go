// completions.go holds the OpenAI completions dialect.

package commonai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/wow-look-at-my/go-containers/set"
)

// The API takes one prompt string. This dialect writes the transcript as labelled text. see docs/completions.md

// DefaultCompletionsMaxTokens is the max_tokens a request with none gets.
const DefaultCompletionsMaxTokens = 4096

// completionsStop ends the answer where the model starts to write the next turn.
var completionsStop = []string{"\nUser:", "\nTool result"}

// cmplReserved are the Extra keys the typed core always overrides.
var cmplReserved = set.Of("prompt", "model", "stream", "messages", "tools")

// completionsProvider speaks POST /completions; read-only fields make it concurrent-safe.
type completionsProvider struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	userAgent  string
	headers    map[string]string
}

// Complete sends the call.
func (c *completionsProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	body, err := c.buildBody(req)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(c.baseURL, "/") + "/completions"
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, badRequestErr("completions: build request: " + err.Error())
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	if c.apiKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	if c.userAgent != "" {
		hreq.Header.Set("User-Agent", c.userAgent)
	}
	for k, v := range c.headers {
		hreq.Header.Set(k, v)
	}
	client := c.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("completions: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, readAPIError(resp)
	}

	st := &cmplStream{ev: ev}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		raw, rerr := io.ReadAll(resp.Body)
		if rerr != nil {
			return nil, badRequestErr("completions: read non-streaming response: " + rerr.Error())
		}
		if err := st.onData(raw); err != nil {
			return nil, err
		}
		return st.completion(false), nil
	}
	if scanErr := scanSSE(resp.Body, st.onData); scanErr != nil {
		wrapped := fmt.Errorf("completions: %w", scanErr)
		if st.sawData {
			return st.completion(true), wrapped
		}
		return nil, wrapped
	}
	return st.completion(true), nil
}

// buildBody writes the request. A declared tool or an image FAILS: this API has no field for either.
func (c *completionsProvider) buildBody(req Request) ([]byte, error) {
	if len(req.Tools) > 0 {
		return nil, Unsupported(DialectCompletions, "a tool declaration", "the completions API has no tools field")
	}
	prompt, err := completionsPrompt(req)
	if err != nil {
		return nil, err
	}
	out := map[string]any{}
	for k, v := range req.ParamsFor(DialectOpenAI) {
		if !cmplReserved.Contains(k) {
			out[k] = v
		}
	}
	out["model"] = req.Model
	out["prompt"] = prompt
	out["stream"] = true
	if req.MaxTokens > 0 {
		out["max_tokens"] = req.MaxTokens
	} else if _, ok := out["max_tokens"]; !ok {
		out["max_tokens"] = DefaultCompletionsMaxTokens
	}
	if _, ok := out["stop"]; !ok {
		out["stop"] = completionsStop
	}
	if _, ok := out["stream_options"]; !ok {
		out["stream_options"] = map[string]any{"include_usage": true}
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, badRequestErr("completions: encode request: " + err.Error())
	}
	return b, nil
}

// completionsPrompt writes the transcript as labelled turns and ends on an open
// "Assistant:" for the model to continue.
func completionsPrompt(req Request) (string, error) {
	var b strings.Builder
	turn := func(label, text string) {
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(label)
		b.WriteString(": ")
		b.WriteString(text)
	}
	if req.System != "" {
		turn("System", req.System)
	}
	for _, m := range req.Messages {
		switch m.Role {
		case RoleTool:
			turn("Tool result ("+m.ToolCallID+")", m.Content)
			continue
		case RoleSystem:
			turn("System", m.Content)
			continue
		}
		text, err := completionsTurnText(m)
		if err != nil {
			return "", err
		}
		label := "User"
		if m.Role == RoleAssistant {
			label = "Assistant"
		}
		turn(label, text)
	}
	if b.Len() > 0 {
		b.WriteString("\n\n")
	}
	b.WriteString("Assistant:")
	return b.String(), nil
}

// completionsTurnText is one turn's parts as text: reasoning, text, then tool calls by name, id and arguments.
func completionsTurnText(m Message) (string, error) {
	var parts []string
	for _, p := range m.EffectiveParts() {
		switch v := p.(type) {
		case TextPart:
			parts = append(parts, v.Text)
		case ThinkingPart:
			if v.Text != "" {
				parts = append(parts, "(reasoning) "+v.Text)
			}
		case ToolCallPart:
			parts = append(parts, fmt.Sprintf("(tool call %s %s) %s", v.Name, v.ID, replayToolArgs(v.Arguments)))
		case ImagePart:
			return "", Unsupported(DialectCompletions, "an image", "a completions prompt is text only")
		case RedactedThinkingPart:
			// Opaque to every reader but Anthropic; it has no text to write.
		}
	}
	return strings.Join(parts, "\n"), nil
}

// cmplChunk is one SSE payload, or the whole non-streaming body.
type cmplChunk struct {
	Choices []struct {
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *oaUsage `json:"usage"`
	Model string   `json:"model"`
}

// cmplStream folds chunks into one Completion.
type cmplStream struct {
	ev      *StreamEvents
	text    strings.Builder
	stop    string
	model   string
	usages  []Usage
	sawData bool
}

// onData folds one chunk.
func (s *cmplStream) onData(data []byte) error {
	var ch cmplChunk
	if err := json.Unmarshal(data, &ch); err != nil {
		return badRequestErr("completions: decode chunk: " + err.Error())
	}
	s.sawData = true
	if ch.Model != "" {
		s.model = ch.Model
	}
	for _, c := range ch.Choices {
		if c.FinishReason != "" {
			s.stop = normalizeStopReason(c.FinishReason)
		}
		if c.Text == "" {
			continue
		}
		s.text.WriteString(c.Text)
		if err := s.ev.EmitText(c.Text); err != nil {
			return err
		}
	}
	if ch.Usage != nil {
		u := ch.Usage.toUsage()
		var raw struct {
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(data, &raw); err == nil {
			u.Raw = raw.Usage
		}
		u.CostUsd = ch.Usage.costUsd()
		u.ReasoningTokens = ch.Usage.reasoningTokens()
		s.usages = append(s.usages, u)
		if s.ev != nil && s.ev.OnUsage != nil {
			if err := wrapCallbackErr(s.ev.OnUsage(u)); err != nil {
				return err
			}
		}
	}
	return nil
}

// completion is what the chunks so far said. The leading space every answer to "Assistant:" starts with is trimmed.
func (s *cmplStream) completion(streamed bool) *Completion {
	text := strings.TrimPrefix(s.text.String(), " ")
	var m Message
	if text != "" {
		m = NewMessage(RoleAssistant, TextPart{Text: text})
	} else {
		m = Message{Role: RoleAssistant}
	}
	return &Completion{Message: m, Usages: s.usages, Streamed: streamed, StopReason: s.stop, Model: s.model}
}
