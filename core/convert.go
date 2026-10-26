// Conversions between the flat Message and the node tree.

package commonai

// Conversions between the flat Message and the node tree. They exist while callers still pass
// []Message; the session replaces both directions.

// MessageFromNodes is the assistant turn one call produced.
func MessageFromNodes(nodes []IMessage) Message {
	var parts []Part
	for _, n := range nodes {
		if p := partOf(n); p != nil {
			parts = append(parts, p)
		}
	}
	m := Message{Role: RoleAssistant, Parts: parts}
	m.SyncViews()
	return m
}

// partOf is the flat part of one model-output node, or nil.
func partOf(n IMessage) Part {
	switch v := n.(type) {
	case *ThinkingMessage:
		if v.Redacted != nil {
			return RedactedThinkingPart{Data: v.Redacted.String()}
		}
		return ThinkingPart{Text: v.Text.String(), Signature: thinkingBlob(v), ID: string(v.ItemID), Raw: v.Raw}
	case *AssistantMessage:
		if s := v.Text.String(); s != "" {
			return TextPart{Text: s}
		}
	case *ToolCallMessage:
		return ToolCallPart{ID: string(v.CallID), Name: v.Name, Arguments: toolArgs(v.Arguments.String())}
	}
	return nil
}

// thinkingBlob is the opaque value the flat Message carried for a reasoning
// block.
func thinkingBlob(t *ThinkingMessage) string {
	switch {
	case t.Signature != nil:
		return t.Signature.String()
	case t.Encrypted != nil:
		return t.Encrypted.String()
	case len(t.Details) > 0:
		return string(t.Details)
	}
	return ""
}

// ThinkingFromBlock is a reasoning node for a flat block, with its blob in d's field.
func ThinkingFromBlock(text, blob, id string, d Dialect) *ThinkingMessage {
	t := &ThinkingMessage{Text: FinishedString(text), ItemID: ResponsesItemID(id), Dialect: d}
	if blob == "" {
		return t
	}
	switch d {
	case DialectAnthropic:
		t.Signature = NewEncryptedReasoning(blob)
	case DialectResponses:
		t.Encrypted = NewEncryptedReasoning(blob)
	case DialectOpenAI:
		t.Details = ReasoningDetails(blob)
	default:
		t.Signature = NewEncryptedReasoning(blob)
	}
	return t
}

// CompactionHandoffPrefix heads a compaction summary on the wire. A summary is never sent as a live
// instruction: a model reading one as the newest ask answers with another summary.
const CompactionHandoffPrefix = "The conversation so far exceeded the context window and was summarized. " +
	"This summary replaces the earlier messages; treat it as what you already know, not as a request. " +
	"Continue the work it describes.\n\n"

// MessagesFromPath is a window path as flat messages. Consecutive model-output nodes of one call form
// one assistant message. A host's own node types are skipped: no dialect knows their wire shape.
func MessagesFromPath(path []IMessage) []Message {
	var out []Message
	var group []IMessage
	var groupCall *CallRecord
	flush := func() {
		if len(group) == 0 {
			return
		}
		m := MessageFromNodes(group)
		m.ID = group[0].Base().HostID()
		out = append(out, m)
		group, groupCall = nil, nil
	}
	for _, n := range path {
		if n.Base().superseded {
			continue
		}
		switch v := n.(type) {
		case *ThinkingMessage, *AssistantMessage, *ToolCallMessage, *RawBlockMessage:
			if len(group) > 0 && (n.Base().call != groupCall || groupCall == nil) {
				flush()
			}
			group = append(group, n)
			groupCall = n.Base().call
		case *UserMessage:
			flush()
			out = append(out, flatContent(RoleUser, v.Base(), v.Content))
		case *SystemMessage:
			flush()
			out = append(out, flatContent(RoleUser, v.Base(), v.Content))
		case *ToolResultMessage:
			flush()
			m := flatContent(RoleTool, v.Base(), v.Content)
			if v.Call != nil {
				m.ToolCallID = string(v.Call.CallID)
			}
			m.ToolIsError = v.IsError
			out = append(out, m)
		case *CompactionMessage:
			flush()
			out = append(out, Message{Role: RoleUser, ID: v.HostID(), Kind: v.kind, Content: CompactionHandoffPrefix + v.Summary.String()})
		}
	}
	flush()
	return out
}

func flatContent(role Role, b *MessageBase, content []IContent) Message {
	var parts []Part
	for _, c := range content {
		switch v := c.(type) {
		case *TextContent:
			parts = append(parts, TextPart{Text: v.Text})
		case *ImageContent:
			p := ImagePart{MediaType: v.MediaType, Src: v.Src, Data: v.inline}
			if v.Blob != nil && p.Data == "" {
				if b64, err := v.Blob.Base64(); err == nil {
					p.Data = b64
				}
			}
			parts = append(parts, p)
		}
	}
	m := Message{Role: role, ID: b.HostID(), Kind: b.kind}
	// One text block is the plain Content a flat message carries; anything else keeps its parts.
	if len(parts) == 1 {
		if t, ok := parts[0].(TextPart); ok {
			m.Content = t.Text
			return m
		}
	}
	m.Parts = parts
	m.SyncViews()
	return m
}

// NodesFromMessage is a flat message as nodes, its opaque blobs placed as dialect d reads them.
func NodesFromMessage(m Message, d Dialect) []IMessage {
	var out []IMessage
	add := func(n IMessage) {
		// The first node takes the message's id; the rest get their own.
		if len(out) == 0 {
			n.Base().id = m.ID
		}
		n.Base().kind = m.Kind
		out = append(out, n)
	}
	switch m.Role {
	case RoleUser, RoleSystem:
		var content []IContent
		for _, p := range m.EffectiveParts() {
			switch v := p.(type) {
			case TextPart:
				content = append(content, Text(v.Text))
			case ImagePart:
				content = append(content, &ImageContent{MediaType: v.MediaType, Src: v.Src, inline: v.Data})
			}
		}
		if m.Kind != "" && m.Role == RoleUser && m.Kind != CompactionKind {
			add(&SystemMessage{Content: content})
		} else {
			add(&UserMessage{Content: content})
		}
	case RoleTool:
		call := &ToolCallMessage{CallID: CallID(m.ToolCallID), Arguments: FinishedString("{}")}
		var content []IContent
		for _, p := range m.EffectiveParts() {
			switch v := p.(type) {
			case TextPart:
				content = append(content, Text(v.Text))
			case ImagePart:
				content = append(content, &ImageContent{MediaType: v.MediaType, Src: v.Src, inline: v.Data})
			}
		}
		add(&ToolResultMessage{Call: call, Content: content, IsError: m.ToolIsError})
	case RoleAssistant:
		// One record for the turn keeps its parts one wire message.
		rec := &CallRecord{Status: CallDone}
		defer func() {
			for _, n := range out {
				n.Base().call = rec
			}
		}()
		for _, p := range m.EffectiveParts() {
			switch v := p.(type) {
			case ThinkingPart:
				t := ThinkingFromBlock(v.Text, v.Signature, v.ID, d)
				t.Raw = v.Raw
				add(t)
			case RedactedThinkingPart:
				add(&ThinkingMessage{Text: FinishedString(""), Redacted: NewEncryptedReasoning(v.Data), Dialect: d})
			case TextPart:
				add(NewAssistantText(v.Text))
			case ToolCallPart:
				add(NewToolCallArgs(CallID(v.ID), v.Name, v.Arguments))
			}
		}
	}
	return out
}
