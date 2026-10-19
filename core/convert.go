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
		return ThinkingPart{Text: v.Text.String(), Signature: thinkingBlob(v), ID: string(v.ItemID)}
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

// NodesFromMessage is a flat message as nodes, its opaque blobs placed as dialect d reads them.
func NodesFromMessage(m Message, d Dialect) []IMessage {
	var out []IMessage
	add := func(n IMessage) {
		n.Base().id, n.Base().kind = m.ID, m.Kind
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
		for _, p := range m.EffectiveParts() {
			switch v := p.(type) {
			case ThinkingPart:
				add(ThinkingFromBlock(v.Text, v.Signature, v.ID, d))
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
