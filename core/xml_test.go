package commonai

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// roundTripRequest encodes, validates against the schema, and decodes.
func roundTripRequest(t *testing.T, req Request) Request {
	t.Helper()
	data, err := EncodeRequestBytes(req)
	require.NoError(t, err)
	require.NoError(t, Validate(data), "document:\n%s", data)
	got, err := DecodeRequest(data)
	require.NoError(t, err)
	return got
}

func TestRequestRoundTrip(t *testing.T) {
	req := Request{
		Model:     "claude-x",
		System:    "You are helpful.",
		MaxTokens: 4096,
		CacheKey:  "abc",
		Messages: []Message{
			NewMessage(RoleUser,
				TextPart{Text: "what is in this image?"},
				ImagePart{MediaType: "image/png", Data: "iVBORw0KGgo="},
			),
			NewMessage(RoleAssistant,
				ThinkingPart{Text: "The user wants a description.", Signature: "sig-1"},
				TextPart{Text: "It shows "},
				ToolCallPart{ID: "call_1", Name: "grep", Arguments: `{"q":"x"}`},
			),
			{Role: RoleTool, ToolCallID: "call_1", Content: "3 matches"},
		},
	}
	got := roundTripRequest(t, req)

	assert.Equal(t, req.Model, got.Model)
	assert.Equal(t, req.MaxTokens, got.MaxTokens)
	assert.Equal(t, req.CacheKey, got.CacheKey)
	assert.Equal(t, req.System, got.System)
	require.Len(t, got.Messages, 3)
	assert.Equal(t, req.Messages[0].Parts, got.Messages[0].Parts)
	assert.Equal(t, req.Messages[1].Parts, got.Messages[1].Parts)
	assert.Equal(t, "3 matches", got.Messages[2].Content)
	assert.Equal(t, "call_1", got.Messages[2].ToolCallID)
}

// Order is the whole point of parts: a reply whose text brackets a thinking
// block has to come back the same way round.
func TestPartOrderSurvives(t *testing.T) {
	req := Request{Model: "m", Messages: []Message{
		NewMessage(RoleAssistant,
			TextPart{Text: "before"},
			ThinkingPart{Text: "middle", Signature: "s"},
			TextPart{Text: "after"},
		),
	}}
	got := roundTripRequest(t, req)
	require.Len(t, got.Messages, 1)
	require.Len(t, got.Messages[0].Parts, 3)
	assert.Equal(t, TextPart{Text: "before"}, got.Messages[0].Parts[0])
	assert.Equal(t, ThinkingPart{Text: "middle", Signature: "s"}, got.Messages[0].Parts[1])
	assert.Equal(t, TextPart{Text: "after"}, got.Messages[0].Parts[2])
	assert.Equal(t, "beforeafter", got.Messages[0].Content)
}

// A tool result can be arbitrary bytes. NUL has no literal spelling and no
// standard character reference, so the writer emits &#0; and our own parser
// reads it back.
func TestNulByteSurvives(t *testing.T) {
	req := Request{Model: "m", Messages: []Message{
		NewMessage(RoleTool, TextPart{Text: "before\x00after\x01end"}),
	}}
	data, err := EncodeRequestBytes(req)
	require.NoError(t, err)
	assert.Contains(t, string(data), "&#0;")
	assert.Contains(t, string(data), "&#1;")

	got, err := DecodeRequest(data)
	require.NoError(t, err)
	assert.Equal(t, "before\x00after\x01end", got.Messages[0].Content)
}

func TestToolSchemaRoundTrip(t *testing.T) {
	schema := `{"type":"object","properties":{"q":{"type":"string","description":"the query"},` +
		`"n":{"type":"number","default":1.50}},"required":["q"]}`
	req := Request{Model: "m", Tools: []ToolDecl{{
		Name:        "grep",
		Description: "search files",
		Readonly:    true,
		InputSchema: json.RawMessage(schema),
	}}}
	got := roundTripRequest(t, req)
	require.Len(t, got.Tools, 1)
	assert.Equal(t, "grep", got.Tools[0].Name)
	assert.True(t, got.Tools[0].Readonly)
	// Byte-for-byte: member order and the 1.50 literal both survive, which is
	// what makes the tree a lossless mapping rather than a lossy one.
	assert.Equal(t, schema, string(got.Tools[0].InputSchema))
}

func TestDialectParamsRoundTrip(t *testing.T) {
	req := Request{
		Model: "m",
		Extra: map[string]any{"temperature": 0.7},
		DialectExtra: map[Dialect]map[string]any{
			DialectAnthropic: {
				"top_k":    40,
				"thinking": map[string]any{"type": "enabled", "budget_tokens": 1024},
			},
			DialectOpenAI: {"reasoning_effort": "high"},
		},
	}
	data, err := EncodeRequestBytes(req)
	require.NoError(t, err)
	require.NoError(t, Validate(data), "document:\n%s", data)
	// The scalar rides as a qualified attribute; the object rides as a
	// namespaced element, because an attribute cannot hold an object.
	assert.Contains(t, string(data), `anthropic:top-k="40"`)
	assert.Contains(t, string(data), `openai:reasoning-effort="high"`)
	assert.Contains(t, string(data), "<anthropic:params>")

	got, err := DecodeRequest(data)
	require.NoError(t, err)
	assert.Equal(t, "40", got.DialectExtra[DialectAnthropic]["top_k"])
	assert.Equal(t, "high", got.DialectExtra[DialectOpenAI]["reasoning_effort"])
	thinking, ok := got.DialectExtra[DialectAnthropic]["thinking"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "enabled", thinking["type"])
	// Dialect-agnostic params stay dialect-agnostic.
	assert.Contains(t, got.Extra, "temperature")
}

// ParamsFor is what stops one dialect's parameters reaching another's upstream.
func TestParamsForKeepsDialectsApart(t *testing.T) {
	req := Request{
		Extra: map[string]any{"temperature": 0.7},
		DialectExtra: map[Dialect]map[string]any{
			DialectAnthropic: {"top_k": 40},
			DialectOpenAI:    {"seed": 42},
		},
	}
	an := req.ParamsFor(DialectAnthropic)
	assert.Equal(t, 40, an["top_k"])
	assert.Equal(t, 0.7, an["temperature"])
	assert.NotContains(t, an, "seed")

	oa := req.ParamsFor(DialectOpenAI)
	assert.Equal(t, 42, oa["seed"])
	assert.NotContains(t, oa, "top_k")
}

func TestResponseRoundTrip(t *testing.T) {
	read := 100
	comp := &Completion{
		Message: NewMessage(RoleAssistant,
			ThinkingPart{Text: "thinking", Signature: "sig"},
			TextPart{Text: "answer"},
			ToolCallPart{ID: "c1", Name: "grep", Arguments: `{"q":"x"}`},
		),
		Usages: []Usage{{
			PromptTokens: 123, CompletionTokens: 45, TotalTokens: 168,
			CacheReadTokens: &read,
			Raw:             json.RawMessage(`{"prompt_tokens":123,"cost":0.0021}`),
		}},
		Timings:    []Timings{{PromptN: 12, PromptMS: 8.4}},
		StopReason: StopToolUse,
		Streamed:   true,
	}
	data, err := EncodeResponseBytes(comp)
	require.NoError(t, err)
	require.NoError(t, Validate(data), "document:\n%s", data)

	got, err := DecodeResponse(data)
	require.NoError(t, err)
	assert.Equal(t, comp.Message.Parts, got.Message.Parts)
	assert.Equal(t, StopToolUse, got.StopReason)
	assert.True(t, got.Streamed)
	require.Len(t, got.Usages, 1)
	assert.Equal(t, 123, got.Usages[0].PromptTokens)
	require.NotNil(t, got.Usages[0].CacheReadTokens)
	assert.Equal(t, 100, *got.Usages[0].CacheReadTokens)
	assert.JSONEq(t, string(comp.Usages[0].Raw), string(got.Usages[0].Raw))
	require.Len(t, got.Timings, 1)
	assert.Equal(t, 12, got.Timings[0].PromptN)
}

// A provider that reported nothing and one that reported zeros are different
// facts, and attribute presence is how the format keeps them apart.
func TestCacheTokensStayTriState(t *testing.T) {
	zero := 0
	data, err := EncodeResponseBytes(&Completion{
		Usages: []Usage{{PromptTokens: 10, CacheReadTokens: &zero}},
	})
	require.NoError(t, err)
	got, err := DecodeResponse(data)
	require.NoError(t, err)
	require.NotNil(t, got.Usages[0].CacheReadTokens)
	assert.Equal(t, 0, *got.Usages[0].CacheReadTokens)
	assert.Nil(t, got.Usages[0].CacheWriteTokens)
}

// The streamed document and the buffered one are the same bytes. If they ever
// diverge, a consumer that reads one cannot trust the other.
func TestStreamedDocumentMatchesBuffered(t *testing.T) {
	comp := &Completion{
		Message: NewMessage(RoleAssistant,
			TextPart{Text: "Hello, world"},
			ToolCallPart{ID: "c1", Name: "grep", Arguments: `{"q":"x"}`},
		),
		Usages:     []Usage{{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}},
		StopReason: StopToolUse,
		Streamed:   true,
	}
	buffered, err := EncodeResponseBytes(comp)
	require.NoError(t, err)

	var streamed bytes.Buffer
	rw := NewResponseWriter(&streamed, RoleAssistant)
	require.NoError(t, rw.Text("Hello, "))
	require.NoError(t, rw.Text("world"))
	require.NoError(t, rw.Part(ToolCallPart{ID: "c1", Name: "grep", Arguments: `{"q":"x"}`}))
	require.NoError(t, rw.Usage(comp.Usages[0]))
	require.NoError(t, rw.Close(StopToolUse, true))

	assert.Equal(t, string(buffered), streamed.String())
	require.NoError(t, Validate(streamed.Bytes()))
}

// A cut stream leaves a document whose root never closed. The content that did
// arrive is still there, and that is the point: it is the partial completion.
func TestTruncatedStreamKeepsWhatArrived(t *testing.T) {
	var buf bytes.Buffer
	rw := NewResponseWriter(&buf, RoleAssistant)
	require.NoError(t, rw.Text("half an ans"))

	_, err := DecodeResponse(buf.Bytes())
	require.Error(t, err, "an unclosed document must not read as a complete one")
	assert.Contains(t, buf.String(), "half an ans")
}

// A call that fails after producing output says both, in one document.
func TestFailureAfterOutputIsOneDocument(t *testing.T) {
	var buf bytes.Buffer
	rw := NewResponseWriter(&buf, RoleAssistant)
	require.NoError(t, rw.Text("partial"))
	require.NoError(t, rw.Fail(&APIError{Status: 429, Body: "slow down"}))

	require.NoError(t, Validate(buf.Bytes()), "document:\n%s", buf.String())
	got, err := DecodeResponse(buf.Bytes())
	require.Error(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "partial", got.Message.Content)
	assert.True(t, IsTransient(err), "a 429 stays retryable across the wire")
}

func TestConversationRoundTrip(t *testing.T) {
	req := Request{Model: "m", Messages: []Message{NewMessage(RoleUser, TextPart{Text: "hi"})}}
	var buf bytes.Buffer
	require.NoError(t, EncodeConversation(&buf, "sess-1", req))
	require.NoError(t, Validate(buf.Bytes()), "document:\n%s", buf.String())

	id, got, err := DecodeConversation(buf.Bytes())
	require.NoError(t, err)
	assert.Equal(t, "sess-1", id)
	require.Len(t, got.Messages, 1)
	assert.Equal(t, "hi", got.Messages[0].Content)
}

// The schema is not decoration: a misspelled provider attribute has no
// declaration to match, so it fails here instead of at the upstream.
func TestSchemaRejectsUnknownQualifiedAttribute(t *testing.T) {
	doc := `<?xml version="1.1" encoding="UTF-8"?>` +
		`<request xmlns="` + NS + `" xmlns:anthropic="` + NSAnthropic + `" model="m" anthropic:bugdet="1"/>`
	err := Validate([]byte(doc))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unexpected attribute")
}

func TestSchemaRejectsUnknownParamType(t *testing.T) {
	doc := `<?xml version="1.1" encoding="UTF-8"?>` +
		`<request xmlns="` + NS + `" model="m"><params><param name="x" type="nonsense">1</param></params></request>`
	err := Validate([]byte(doc))
	require.Error(t, err)
}

func TestDecodeRejectsWrongRoot(t *testing.T) {
	_, err := DecodeRequest([]byte(`<?xml version="1.1"?><response xmlns="` + NS + `"/>`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected a <request> document")
}

func TestDecodeRejectsForeignNamespace(t *testing.T) {
	_, err := DecodeRequest([]byte(`<?xml version="1.1"?><request xmlns="http://example.invalid/other"/>`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "namespace")
}

func TestParamTreePreservesNumberLiterals(t *testing.T) {
	p, err := ParamFromJSON("v", []byte(`{"a":1.50,"b":1e3,"c":[1,"two",null,true]}`))
	require.NoError(t, err)
	back, err := p.JSON()
	require.NoError(t, err)
	assert.Equal(t, `{"a":1.50,"b":1e3,"c":[1,"two",null,true]}`, string(back))
}

func TestParamTreeRejectsNonNumberInNumber(t *testing.T) {
	_, err := Param{Type: ParamNumber, Value: "not-a-number"}.JSON()
	require.Error(t, err)
}

func TestEscapesInvalidUTF8(t *testing.T) {
	var buf bytes.Buffer
	rw := NewResponseWriter(&buf, RoleAssistant)
	require.NoError(t, rw.Text("ok\xff!"))
	require.NoError(t, rw.Close(StopEndTurn, false))
	assert.Contains(t, buf.String(), "&#255;")
	require.NoError(t, Validate(buf.Bytes()))
}

func TestEffectivePartsFallsBackToFlatFields(t *testing.T) {
	// A caller that only ever set Content -- the shape every existing consumer
	// uses -- still encodes correctly.
	m := Message{Role: RoleAssistant, Content: "hi", ToolCalls: []ToolCall{{ID: "c", Name: "n", Arguments: "{}"}}}
	parts := m.EffectiveParts()
	require.Len(t, parts, 2)
	assert.Equal(t, TextPart{Text: "hi"}, parts[0])
	assert.Equal(t, ToolCallPart{ID: "c", Name: "n", Arguments: "{}"}, parts[1])
}

func TestSchemaFilesEmbedded(t *testing.T) {
	entries, err := schemaFS.ReadDir("schema")
	require.NoError(t, err)
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	assert.ElementsMatch(t,
		[]string{"common-ai-api.xsd", "anthropic.xsd", "openai.xsd", "responses.xsd"},
		names)
}

func TestValidateRejectsGarbage(t *testing.T) {
	require.Error(t, Validate([]byte("not xml at all")))
	require.Error(t, Validate([]byte(`<?xml version="1.1"?><nope/>`)))
}

func TestDecodeResponseWithoutRoleDefaultsToAssistant(t *testing.T) {
	doc := `<?xml version="1.1" encoding="UTF-8"?><response xmlns="` + NS + `"><text>hi</text></response>`
	got, err := DecodeResponse([]byte(doc))
	require.NoError(t, err)
	assert.Equal(t, RoleAssistant, got.Message.Role)
	assert.Equal(t, "hi", got.Message.Content)
}

func TestUnsupportedErrorReadsPlainly(t *testing.T) {
	err := Unsupported(DialectOpenAI, "an image by URL", "the dialect takes inline bytes only")
	assert.True(t, IsUnsupported(err))
	assert.Contains(t, err.Error(), "openai cannot express an image by URL")
	assert.False(t, IsTransient(err) && false)
	assert.True(t, strings.HasPrefix(err.Error(), "commonai: "))
}

// Reasoning deltas coalesce into one <thinking> element the same way text
// deltas coalesce into one <text>, and a part written afterwards closes it.
func TestResponseWriterReasoningDeltas(t *testing.T) {
	var buf bytes.Buffer
	rw := NewResponseWriter(&buf, RoleAssistant)
	require.NoError(t, rw.Reasoning("weigh"))
	require.NoError(t, rw.Reasoning("ing it"))
	require.NoError(t, rw.Text("answer"))
	require.NoError(t, rw.Close(StopEndTurn, true))

	require.NoError(t, Validate(buf.Bytes()), "document:\n%s", buf.String())
	assert.Contains(t, buf.String(), "<thinking>weighing it</thinking>")

	comp, err := DecodeResponse(buf.Bytes())
	require.NoError(t, err)
	require.Len(t, comp.Message.Parts, 2)
	assert.Equal(t, PartKindThinking, comp.Message.Parts[0].Kind())
	assert.Equal(t, "answer", comp.Message.Content)

	// An empty delta is not a part: a stream that sends one has said nothing.
	var empty bytes.Buffer
	rw = NewResponseWriter(&empty, RoleAssistant)
	require.NoError(t, rw.Reasoning(""))
	require.NoError(t, rw.Close(StopEndTurn, true))
	assert.NotContains(t, empty.String(), "<thinking")
}

// A listing says what a store holds without inlining it.
func TestConversationListRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, EncodeConversationIDs(&buf, []string{"c1", "c2"}))
	require.NoError(t, Validate(buf.Bytes()), "document:\n%s", buf.String())

	ids, err := DecodeConversationIDs(buf.Bytes())
	require.NoError(t, err)
	assert.Equal(t, []string{"c1", "c2"}, ids)

	var empty bytes.Buffer
	require.NoError(t, EncodeConversationIDs(&empty, nil))
	require.NoError(t, Validate(empty.Bytes()))
	ids, err = DecodeConversationIDs(empty.Bytes())
	require.NoError(t, err)
	assert.Empty(t, ids)
}

// A call that failed before it had any answer to give is still a document, and
// the failure keeps its kind and status across the wire.
func TestErrorDocumentRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, EncodeError(&buf, &APIError{Status: 429, Body: "slow down"}))
	require.NoError(t, Validate(buf.Bytes()), "document:\n%s", buf.String())

	err := DecodeError(buf.Bytes())
	require.Error(t, err)
	assert.True(t, IsTransient(err), "a 429 stays retryable across the wire")
	var ae *APIError
	require.ErrorAs(t, err, &ae)
	assert.Equal(t, 429, ae.Status)

	var refused bytes.Buffer
	require.NoError(t, EncodeError(&refused, BadRequest("commonai: model is required")))
	require.NoError(t, Validate(refused.Bytes()))
	err = DecodeError(refused.Bytes())
	require.Error(t, err)
	assert.True(t, IsBadRequest(err), "whose fault it was survives too")
	assert.Equal(t, ErrorKindRequest, ErrorKind(err))
}
