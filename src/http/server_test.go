package httpapi

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"
	"github.com/wow-look-at-my/agentic-loop/go/session"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeProvider answers with a scripted completion, optionally emitting events
// first so a test can drive the streamed path.
type fakeProvider struct {
	emit func(ev *commonai.StreamEvents) error
	comp *commonai.Completion
	err  error
	reqs []commonai.Request
}

func (p *fakeProvider) Complete(_ context.Context, req commonai.Request, ev *commonai.StreamEvents) (*commonai.Completion, error) {
	p.reqs = append(p.reqs, req)
	if p.emit != nil {
		if err := p.emit(ev); err != nil {
			return p.comp, err
		}
	}
	return p.comp, p.err
}

// answer is a completion that a streamed call would also have produced.
func answer(text string) *commonai.Completion {
	return &commonai.Completion{
		Message:    commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: text}),
		Usages:     []commonai.Usage{{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}},
		Streamed:   true,
		StopReason: commonai.StopEndTurn,
	}
}

func serve(t *testing.T, cfg Config) *httptest.Server {
	t.Helper()
	s, err := NewServer(cfg)
	require.NoError(t, err)
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, url, body string) (*http.Response, []byte) {
	t.Helper()
	resp, err := http.Post(url, contentType, strings.NewReader(body))
	require.NoError(t, err)
	t.Cleanup(func() { _ = resp.Body.Close() })
	data, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return resp, data
}

func requestDoc(t *testing.T, req commonai.Request) string {
	t.Helper()
	data, err := commonai.EncodeRequestBytes(req)
	require.NoError(t, err)
	return string(data)
}

func ask(t *testing.T, text string) string {
	t.Helper()
	return requestDoc(t, commonai.Request{
		Model:    "m",
		Messages: []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: text})},
	})
}

func TestCompleteAnswersWithAResponseDocument(t *testing.T) {
	p := &fakeProvider{
		comp: answer("hello"),
		emit: func(ev *commonai.StreamEvents) error {
			if err := ev.OnText("hel"); err != nil {
				return err
			}
			if err := ev.OnText("lo"); err != nil {
				return err
			}
			if err := ev.OnPart(commonai.TextPart{Text: "hello"}); err != nil {
				return err
			}
			return ev.OnUsage(commonai.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12})
		},
	}
	srv := serve(t, Config{Provider: p})

	resp, data := post(t, srv.URL+"/v1/complete", ask(t, "hi"))
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)

	comp, err := commonai.DecodeResponse(data)
	require.NoError(t, err)
	assert.Equal(t, "hello", comp.Message.Content, "the deltas are the answer, written once")
	assert.Equal(t, commonai.StopEndTurn, comp.StopReason)
	require.Len(t, comp.Usages, 1)
	assert.Equal(t, 12, comp.Usages[0].TotalTokens)

	require.Len(t, p.reqs, 1)
	assert.Equal(t, "m", p.reqs[0].Model)
	assert.Equal(t, "hi", p.reqs[0].Messages[0].Content)
}

// The document a streamed call writes says the same thing as the one a
// buffered call writes. That is the whole claim of writing the answer
// progressively rather than inventing a second vocabulary for it.
func TestStreamedAndBufferedDocumentsAgree(t *testing.T) {
	comp := &commonai.Completion{
		Message: commonai.NewMessage(commonai.RoleAssistant,
			commonai.ThinkingPart{Text: "weighing it", Signature: "sig-1"},
			commonai.TextPart{Text: "here"},
			commonai.ToolCallPart{ID: "c1", Name: "grep", Arguments: `{"q":"x"}`},
		),
		Usages:     []commonai.Usage{{PromptTokens: 9, CompletionTokens: 3, TotalTokens: 12}},
		Streamed:   true,
		StopReason: commonai.StopToolUse,
	}
	streamed := &fakeProvider{comp: comp, emit: func(ev *commonai.StreamEvents) error {
		for _, p := range comp.Message.Parts {
			if tp, ok := p.(commonai.TextPart); ok {
				for _, chunk := range []string{"he", "re"} {
					if err := ev.OnText(chunk); err != nil {
						return err
					}
				}
				_ = tp
			}
			if err := ev.OnPart(p); err != nil {
				return err
			}
		}
		return ev.OnUsage(comp.Usages[0])
	}}
	// The same answer from a server that ignored stream:true: no events at all.
	buffered := &fakeProvider{comp: &commonai.Completion{
		Message: comp.Message, Usages: comp.Usages, StopReason: comp.StopReason,
	}}

	_, live := post(t, serve(t, Config{Provider: streamed}).URL+"/v1/complete", ask(t, "hi"))
	_, whole := post(t, serve(t, Config{Provider: buffered}).URL+"/v1/complete", ask(t, "hi"))
	require.NoError(t, commonai.Validate(live), "document:\n%s", live)
	require.NoError(t, commonai.Validate(whole), "document:\n%s", whole)

	fromLive, err := commonai.DecodeResponse(live)
	require.NoError(t, err)
	fromWhole, err := commonai.DecodeResponse(whole)
	require.NoError(t, err)

	assert.Equal(t, fromWhole.Message.Parts, fromLive.Message.Parts)
	assert.Equal(t, fromWhole.Usages, fromLive.Usages)
	assert.Equal(t, fromWhole.StopReason, fromLive.StopReason)
	assert.True(t, fromLive.Streamed)
	assert.False(t, fromWhole.Streamed, "how it was transported is part of what happened")
}

// A call that fails before it has anything to say gets a status and an <error>
// document; one that fails after says both, in one document.
func TestFailureBeforeAndAfterOutput(t *testing.T) {
	before := &fakeProvider{err: &commonai.APIError{Status: 429, Body: "slow down"}}
	resp, data := post(t, serve(t, Config{Provider: before}).URL+"/v1/complete", ask(t, "hi"))
	assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	err := commonai.DecodeError(data)
	require.Error(t, err)
	assert.True(t, commonai.IsTransient(err))

	after := &fakeProvider{
		comp: &commonai.Completion{
			Message:  commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: "partial"}),
			Streamed: true,
		},
		err: &commonai.APIError{Status: 500, Body: "upstream died"},
		emit: func(ev *commonai.StreamEvents) error {
			return ev.OnText("partial")
		},
	}
	resp, data = post(t, serve(t, Config{Provider: after}).URL+"/v1/complete", ask(t, "hi"))
	assert.Equal(t, http.StatusOK, resp.StatusCode, "the status was spent before the failure happened")
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	comp, err := commonai.DecodeResponse(data)
	require.Error(t, err, "the document says it failed")
	require.NotNil(t, comp)
	assert.Equal(t, "partial", comp.Message.Content)
}

func TestRejectsADocumentTheSchemaDoesNotAllow(t *testing.T) {
	srv := serve(t, Config{Provider: &fakeProvider{comp: answer("hi")}})
	for _, body := range []string{
		`<?xml version="1.1"?><request xmlns="https://github.com/wow-look-at-my/common-ai-api/schema/v1"><nope/></request>`,
		`<?xml version="1.1"?><response xmlns="https://github.com/wow-look-at-my/common-ai-api/schema/v1"/>`,
		`not xml at all`,
	} {
		resp, data := post(t, srv.URL+"/v1/complete", body)
		assert.Equal(t, http.StatusBadRequest, resp.StatusCode, body)
		require.NoError(t, commonai.Validate(data), "even a refusal is a document:\n%s", data)
	}
}

func TestConversationLifecycle(t *testing.T) {
	p := &fakeProvider{comp: answer("hello"), emit: func(ev *commonai.StreamEvents) error {
		if err := ev.OnText("hello"); err != nil {
			return err
		}
		return ev.OnPart(commonai.TextPart{Text: "hello"})
	}}
	srv := serve(t, Config{Provider: p, Store: session.NewMemory()})

	resp, data := post(t, srv.URL+"/v1/conversations", requestDoc(t, commonai.Request{Model: "m", System: "be brief"}))
	require.Equal(t, http.StatusCreated, resp.StatusCode)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	id, stored, err := commonai.DecodeConversation(data)
	require.NoError(t, err)
	require.NotEmpty(t, id)
	assert.Equal(t, "be brief", stored.System)

	resp, data = post(t, srv.URL+"/v1/conversations/"+id+"/turns", ask(t, "first question"))
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)

	// The turn ran over the stored defaults plus the whole transcript.
	require.Len(t, p.reqs, 1)
	assert.Equal(t, "be brief", p.reqs[0].System, "the conversation's own system prompt")
	require.Len(t, p.reqs[0].Messages, 1)
	assert.Equal(t, "first question", p.reqs[0].Messages[0].Content)

	// Both sides of the turn are in the stored transcript, so the next one
	// sees what was said.
	resp2, err := http.Get(srv.URL + "/v1/conversations/" + id)
	require.NoError(t, err)
	body, err := io.ReadAll(resp2.Body)
	require.NoError(t, err)
	require.NoError(t, resp2.Body.Close())
	require.NoError(t, commonai.Validate(body), "document:\n%s", body)
	_, conv, err := commonai.DecodeConversation(body)
	require.NoError(t, err)
	require.Len(t, conv.Messages, 2)
	assert.Equal(t, commonai.RoleUser, conv.Messages[0].Role)
	assert.Equal(t, "hello", conv.Messages[1].Content)

	resp2, err = http.Get(srv.URL + "/v1/conversations")
	require.NoError(t, err)
	body, err = io.ReadAll(resp2.Body)
	require.NoError(t, err)
	require.NoError(t, resp2.Body.Close())
	ids, err := commonai.DecodeConversationIDs(body)
	require.NoError(t, err)
	assert.Equal(t, []string{id}, ids)

	req, err := http.NewRequest(http.MethodDelete, srv.URL+"/v1/conversations/"+id, nil)
	require.NoError(t, err)
	resp2, err = http.DefaultClient.Do(req)
	require.NoError(t, err)
	require.NoError(t, resp2.Body.Close())
	assert.Equal(t, http.StatusNoContent, resp2.StatusCode)

	resp2, err = http.Get(srv.URL + "/v1/conversations/" + id)
	require.NoError(t, err)
	require.NoError(t, resp2.Body.Close())
	assert.Equal(t, http.StatusNotFound, resp2.StatusCode)
}

// A turn may raise max-tokens for one question without that becoming what the
// conversation is.
func TestTurnOverridesAreForThatTurnOnly(t *testing.T) {
	p := &fakeProvider{comp: answer("ok")}
	srv := serve(t, Config{Provider: p, Store: session.NewMemory()})
	_, data := post(t, srv.URL+"/v1/conversations", requestDoc(t, commonai.Request{Model: "m", MaxTokens: 100}))
	id, _, err := commonai.DecodeConversation(data)
	require.NoError(t, err)

	_, _ = post(t, srv.URL+"/v1/conversations/"+id+"/turns", requestDoc(t, commonai.Request{
		MaxTokens: 4000,
		Messages:  []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: "long one"})},
	}))
	require.Len(t, p.reqs, 1)
	assert.Equal(t, 4000, p.reqs[0].MaxTokens)
	assert.Equal(t, "m", p.reqs[0].Model, "what the turn did not state stays the conversation's")

	resp, err := http.Get(srv.URL + "/v1/conversations/" + id)
	require.NoError(t, err)
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	_, conv, err := commonai.DecodeConversation(body)
	require.NoError(t, err)
	assert.Equal(t, 100, conv.MaxTokens, "the override was for that turn")
}

// A server with no store does not have sessions to serve, and says so rather
// than answering an empty list that reads as "you have none".
func TestStatefulRoutesWithoutAStore(t *testing.T) {
	srv := serve(t, Config{Provider: &fakeProvider{comp: answer("hi")}})
	resp, data := post(t, srv.URL+"/v1/conversations", requestDoc(t, commonai.Request{Model: "m"}))
	assert.Equal(t, http.StatusNotImplemented, resp.StatusCode)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	assert.Error(t, commonai.DecodeError(data))
}

func TestServerRequiresAProvider(t *testing.T) {
	_, err := NewServer(Config{})
	require.Error(t, err)
	assert.True(t, commonai.IsBadRequest(err))
}
