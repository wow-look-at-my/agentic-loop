package commonai

import (
	"encoding/json"
	"strings"
)

// The streamed event types this dialect acts on. The rest is lifecycle chatter.
const (
	respEvItemAdded       = "response.output_item.added"
	respEvOutputTextDelta = "response.output_text.delta"
	respEvSummaryDelta    = "response.reasoning_summary_text.delta"
	respEvArgsDelta       = "response.function_call_arguments.delta"
	respEvItemDone        = "response.output_item.done"
	respEvCompleted       = "response.completed"
	respEvIncomplete      = "response.incomplete"
	respEvFailed          = "response.failed"
	respEvError           = "error"
)

// respStream writes a streamed response into a CallSink.
type respStream struct {
	out    CallSink
	byItem map[string]IMessage
	nodes  []IMessage
	usages []Usage

	haveCall bool
	stop     string
	failure  error
	sawData  bool
}

func newRespStream(out CallSink) *respStream {
	return &respStream{out: out, byItem: map[string]IMessage{}}
}

func (st *respStream) start(itemID string, n IMessage) IMessage {
	st.byItem[itemID] = n
	st.nodes = append(st.nodes, n)
	st.out.Start(n)
	return n
}

// nodeFor is the node of an item, made from the item when no added event announced it.
func (st *respStream) nodeFor(itemID string, item *respItem, kind string) IMessage {
	if n := st.byItem[itemID]; n != nil {
		return n
	}
	switch kind {
	case respItemReasoning:
		t := NewThinking(DialectResponses)
		t.ItemID = ResponsesItemID(itemID)
		return st.start(itemID, t)
	case respItemFuncCall:
		tc := NewToolCall("", "")
		if item != nil {
			tc.CallID, tc.Name = CallID(item.CallID), item.Name
		}
		st.haveCall = true
		return st.start(itemID, tc)
	default:
		return st.start(itemID, NewAssistant())
	}
}

func (st *respStream) onData(data []byte) error {
	var e respEvent
	if err := json.Unmarshal(data, &e); err != nil {
		return nil
	}
	switch e.Type {
	case respEvItemAdded:
		if e.Item == nil {
			return nil
		}
		st.sawData = true
		st.nodeFor(e.Item.ID, e.Item, e.Item.Type)
		return st.out.Err()

	case respEvOutputTextDelta:
		if e.Delta == "" {
			return nil
		}
		st.sawData = true
		st.out.Delta(st.nodeFor(e.ItemID, nil, respItemMessage), FieldText, e.Delta)
		return st.out.Err()

	case respEvSummaryDelta:
		if e.Delta == "" {
			return nil
		}
		st.sawData = true
		st.out.Delta(st.nodeFor(e.ItemID, nil, respItemReasoning), FieldText, e.Delta)
		return st.out.Err()

	case respEvArgsDelta:
		if e.Delta == "" {
			return nil
		}
		st.sawData = true
		st.out.Delta(st.nodeFor(e.ItemID, nil, respItemFuncCall), FieldArguments, e.Delta)
		return st.out.Err()

	case respEvItemDone:
		if e.Item != nil {
			st.sawData = true
			st.finishItem(*e.Item)
			return st.out.Err()
		}
		return nil

	case respEvCompleted, respEvIncomplete:
		if e.Response == nil {
			return nil
		}
		st.sawData = true
		return st.finishResponse(data, *e.Response)

	case respEvFailed:
		st.failure = respFailure(e.Response)
		return nil

	case respEvError:
		st.failure = respEventFailure(e.Error)
		return nil
	}
	return nil
}

// finishItem ends an item's node with the values output_item.done carries. Those values are
// authoritative: a field that differs from what streamed keeps the streamed text as a revision.
func (st *respStream) finishItem(item respItem) {
	switch item.Type {
	case respItemMessage:
		var b strings.Builder
		for _, c := range item.Content {
			b.WriteString(c.Text)
		}
		n := st.nodeFor(item.ID, &item, respItemMessage).(*AssistantMessage)
		if final := b.String(); final != "" || n.Text.String() == "" {
			WriterOf(n.Text).Finish(final)
		}
		st.out.Done(n)
	case respItemFuncCall:
		n := st.nodeFor(item.ID, &item, respItemFuncCall).(*ToolCallMessage)
		n.CallID, n.Name = CallID(item.CallID), item.Name
		WriterOf(n.Arguments).Finish(item.Arguments)
		st.out.Done(n)
	case respItemReasoning:
		n := st.nodeFor(item.ID, &item, respItemReasoning).(*ThinkingMessage)
		var b strings.Builder
		for _, s := range item.Summary {
			b.WriteString(s.Text)
		}
		if final := b.String(); final != "" || n.Text.String() == "" {
			WriterOf(n.Text).Finish(final)
		}
		n.Encrypted = NewEncryptedReasoning(item.EncryptedContent)
		st.out.Done(n)
	}
}

// finishResponse reads the terminal Response: the usage with its wire form, and the stop reason.
// Items are not read again from response.output: each already arrived as output_item.done.
func (st *respStream) finishResponse(raw []byte, r respResponse) error {
	st.stop = respStopReason(r, st.haveCall)
	if r.Usage == nil {
		return nil
	}
	u := r.Usage.toUsage()
	var envelope struct {
		Response struct {
			Usage json.RawMessage `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil {
		u.Raw = envelope.Response.Usage
	}
	u.ReasoningTokens = r.Usage.reasoningTokens()
	st.usages = append(st.usages, u)
	st.out.Usage(u)
	return st.out.Err()
}

// completion ends every open node and gives the call as a Completion, partial on a broken stream.
func (st *respStream) completion() *Completion {
	for _, n := range st.nodes {
		// An item that never got output_item.done keeps what it received but is not announced.
		FinishNode(n)
	}
	stop := st.stop
	if stop == "" {
		stop = respStopFromShape(st.haveCall)
	}
	return &Completion{
		Message:    MessageFromNodes(st.nodes),
		Usages:     st.usages,
		Streamed:   true,
		StopReason: stop,
	}
}
