package socket

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"
	"github.com/wow-look-at-my/agentic-loop/go/session"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptProvider answers with whatever the test hands it, so a case can say
// exactly which events fired before the call ended and how.
type scriptProvider struct {
	comp *commonai.Completion
	err  error
	emit func(*commonai.StreamEvents) error
	reqs []commonai.Request
}

func (p *scriptProvider) Complete(_ context.Context, req commonai.Request, ev *commonai.StreamEvents) (*commonai.Completion, error) {
	p.reqs = append(p.reqs, req)
	if p.emit != nil {
		if err := p.emit(ev); err != nil {
			return nil, err
		}
	}
	return p.comp, p.err
}

// ask sends one document and reads the whole answer back.
func ask(t *testing.T, cfg Config, doc []byte) []byte {
	t.Helper()
	conn, err := net.Dial("unix", listen(t, cfg))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))
	_, err = conn.Write(doc)
	require.NoError(t, err)
	data, err := commonai.ReadDocument(bufio.NewReader(conn))
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	return data
}

// A call that breaks AFTER it produced output says both things in the one
// document it had already started -- the partial answer, then why it stopped.
// There is no status left to fail with: the peer has seen the content.
func TestFailureAfterOutputRidesInsideTheDocument(t *testing.T) {
	p := &scriptProvider{
		comp: &commonai.Completion{
			Message:  commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: "partial"}),
			Streamed: true,
		},
		err:  &commonai.APIError{Status: 500, Body: "upstream died"},
		emit: func(ev *commonai.StreamEvents) error { return ev.OnText("partial") },
	}
	data := ask(t, Config{Provider: p}, askDoc(t, "hi"))

	comp, err := commonai.DecodeResponse(data)
	require.Error(t, err, "the document says it failed")
	require.NotNil(t, comp)
	assert.Equal(t, "partial", comp.Message.Content)
}

// A completion whose parts no event announced is still written: OnPart is how
// a part is normally delivered, not the only way one can exist, and a provider
// that answers without streaming must not lose its content on the way out.
func TestPartsNoEventAnnouncedAreStillWritten(t *testing.T) {
	p := &scriptProvider{
		comp: &commonai.Completion{
			Message: commonai.NewMessage(commonai.RoleAssistant,
				commonai.TextPart{Text: "written once"},
				commonai.ToolCallPart{ID: "c1", Name: "grep", Arguments: `{"q":"x"}`}),
			Usages:     []commonai.Usage{{PromptTokens: 7, CompletionTokens: 2, TotalTokens: 9}},
			Timings:    []commonai.Timings{{PredictedN: 2, PredictedMS: 40}},
			StopReason: commonai.StopToolUse,
		},
		// One text delta, and nothing else: the tool call and the usage have to
		// come from the completion at the end.
		emit: func(ev *commonai.StreamEvents) error { return ev.OnText("written once") },
	}
	data := ask(t, Config{Provider: p}, askDoc(t, "hi"))

	comp, err := commonai.DecodeResponse(data)
	require.NoError(t, err)
	assert.Equal(t, "written once", comp.Message.Content, "the streamed text is not repeated")
	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, "grep", comp.Message.ToolCalls[0].Name)
	require.Len(t, comp.Usages, 1, "a call that did not stream reports its usage at the end")
	assert.Equal(t, 9, comp.Usages[0].TotalTokens)
	require.Len(t, comp.Timings, 1)
	assert.Equal(t, commonai.StopToolUse, comp.StopReason)
}

// A streamed call already announced its usage, so the completion's copy is not
// written a second time.
func TestStreamedUsageIsNotWrittenTwice(t *testing.T) {
	u := commonai.Usage{PromptTokens: 4, CompletionTokens: 1, TotalTokens: 5}
	p := &scriptProvider{
		comp: &commonai.Completion{
			Message:    commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: "hi"}),
			Usages:     []commonai.Usage{u},
			Streamed:   true,
			StopReason: commonai.StopEndTurn,
		},
		emit: func(ev *commonai.StreamEvents) error {
			if err := ev.OnText("hi"); err != nil {
				return err
			}
			if err := ev.OnPart(commonai.TextPart{Text: "hi"}); err != nil {
				return err
			}
			return ev.OnUsage(u)
		},
	}
	data := ask(t, Config{Provider: p}, askDoc(t, "hi"))

	comp, err := commonai.DecodeResponse(data)
	require.NoError(t, err)
	require.Len(t, comp.Usages, 1)
	assert.Equal(t, "hi", comp.Message.Content)
}

// A turn's own fields win over the conversation's, and only for that turn:
// what is stored keeps the defaults the conversation was created with.
func TestTurnOverridesTheConversationsDefaults(t *testing.T) {
	p := &scriptProvider{comp: &commonai.Completion{
		Message:    commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: "ok"}),
		StopReason: commonai.StopEndTurn,
	}}
	store := session.NewMemory()
	cfg := Config{Provider: p, Store: store}
	conn, err := net.Dial("unix", listen(t, cfg))
	require.NoError(t, err)
	defer conn.Close()
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))
	r := bufio.NewReader(conn)

	turn := func(req commonai.Request) {
		buf, err := encodeConversation(req, "work")
		require.NoError(t, err)
		_, err = conn.Write(buf)
		require.NoError(t, err)
		data, err := commonai.ReadDocument(r)
		require.NoError(t, err)
		require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	}

	turn(commonai.Request{
		Model: "base", System: "be brief", MaxTokens: 100, CacheKey: "k1",
		Messages: []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: "one"})},
	})
	turn(commonai.Request{
		Model: "override", System: "be verbose", MaxTokens: 500, CacheKey: "k2",
		Tools:    []commonai.ToolDecl{{Name: "grep", Description: "search"}},
		Extra:    map[string]any{"temperature": 0.5},
		Messages: []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: "two"})},
	})
	// A turn that states nothing inherits everything the conversation was
	// created with -- not the previous turn's overrides.
	turn(commonai.Request{
		Messages: []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: "three"})},
	})

	require.Len(t, p.reqs, 3)
	assert.Equal(t, "base", p.reqs[0].Model)

	assert.Equal(t, "override", p.reqs[1].Model)
	assert.Equal(t, "be verbose", p.reqs[1].System)
	assert.Equal(t, 500, p.reqs[1].MaxTokens)
	assert.Equal(t, "k2", p.reqs[1].CacheKey)
	require.Len(t, p.reqs[1].Tools, 1)
	// A number crosses the document as the literal that was written, not as a
	// float that has been through a conversion nobody asked for.
	assert.Equal(t, "0.5", fmt.Sprint(p.reqs[1].Extra["temperature"]))

	assert.Equal(t, "base", p.reqs[2].Model, "the override was for its own turn only")
	assert.Equal(t, "be brief", p.reqs[2].System)
	assert.Equal(t, 100, p.reqs[2].MaxTokens)
	assert.Empty(t, p.reqs[2].Tools)
}

// Without a Store, a <conversation> is refused rather than quietly run as a
// stateless call: an answer that forgets what it was asked to remember is
// worse than no answer.
func TestConversationWithoutAStoreIsRefused(t *testing.T) {
	buf, err := encodeConversation(commonai.Request{
		Model:    "m",
		Messages: []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: "hi"})},
	}, "work")
	require.NoError(t, err)

	p := &scriptProvider{comp: &commonai.Completion{Message: commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: "hi"})}}
	data := ask(t, Config{Provider: p}, buf)

	got := commonai.DecodeError(data)
	require.Error(t, got)
	assert.True(t, commonai.IsBadRequest(got))
	assert.Empty(t, p.reqs, "the call was never made")
}

// A document that stops mid-element is reported rather than ignored: the peer
// believes it sent something, and silence would let it wait forever.
func TestATruncatedDocumentIsAnsweredWithAnError(t *testing.T) {
	conn, err := net.Dial("unix", listen(t, Config{Provider: &scriptProvider{}}))
	require.NoError(t, err)
	defer conn.Close()
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))

	doc := askDoc(t, "hi")
	_, err = conn.Write(doc[:len(doc)/2])
	require.NoError(t, err)
	require.NoError(t, conn.(*net.UnixConn).CloseWrite())

	data, err := commonai.ReadDocument(bufio.NewReader(conn))
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	assert.Error(t, commonai.DecodeError(data))
}
