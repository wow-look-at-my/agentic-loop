package commonai

import (
	"bytes"
	"io"
	"strconv"
)

// Element and attribute names. They are constants because they are the format:
// a rename here is a breaking change to every consumer, not a refactor.
const (
	elRequest      = "request"
	elResponse     = "response"
	elConversation = "conversation"
	// elConversations lists what a store holds, by id: inlining every
	// transcript would make "what is there" cost as much as reading it all.
	elConversations  = "conversations"
	elConversationID = "conversation-id"
	elError          = "error"
	elMessages       = "messages"
	elMessage        = "message"
	elSystem         = "system"
	elTools          = "tools"
	elTool           = "tool"
	elInputSchema    = "input-schema"
	elParams         = "params"
	elParam          = "param"
	elText           = "text"
	elImage          = "image"
	elThinking       = "thinking"
	elRedacted       = "redacted-thinking"
	elToolCall       = "tool-call"
	elArguments      = "arguments"
	elUsage          = "usage"
	elRaw            = "raw"
	elResult         = "result"
	elTimings        = "timings"
)

// EncodeRequest writes req as a <request> document.
func EncodeRequest(w io.Writer, req Request) error {
	x := newWriter(w)
	x.start(elRequest, requestAttrs(req)...)
	writeRequestBody(x, req)
	x.end()
	return x.err
}

// EncodeConversation writes a stored conversation: an id plus the same body a
// request has, which is what a session actually is -- the defaults and the
// transcript a next turn will be appended to.
func EncodeConversation(w io.Writer, id string, req Request) error {
	x := newWriter(w)
	attrs := append([]attr{{name: "id", value: id}}, requestAttrs(req)...)
	x.start(elConversation, attrs...)
	writeRequestBody(x, req)
	x.end()
	return x.err
}

// EncodeError writes err as a standalone <error> document, for a transport
// answering a call that failed before it had any answer to give.
func EncodeError(w io.Writer, err error) error {
	if err == nil {
		return nil
	}
	x := newWriter(w)
	writeError(x, err, attr{name: "xmlns", value: NS})
	return x.err
}

// EncodeConversationIDs writes a <conversations> listing.
func EncodeConversationIDs(w io.Writer, ids []string) error {
	x := newWriter(w)
	x.start(elConversations, attr{name: "xmlns", value: NS})
	for _, id := range ids {
		x.start(elConversationID, attr{name: "id", value: id})
		x.end()
	}
	x.end()
	return x.err
}

// EncodeResponse writes comp as a <response> document. It produces exactly what
// a ResponseWriter produces for the same completion, so a caller cannot tell
// from the bytes whether the call was streamed -- only from what the document
// says.
func EncodeResponse(w io.Writer, comp *Completion) error {
	rw := NewResponseWriter(w, comp.Message.Role)
	for _, p := range comp.Message.EffectiveParts() {
		if err := rw.Part(p); err != nil {
			return err
		}
	}
	for _, u := range comp.Usages {
		if err := rw.Usage(u); err != nil {
			return err
		}
	}
	for _, t := range comp.Timings {
		if err := rw.Timings(t); err != nil {
			return err
		}
	}
	return rw.Close(comp.StopReason, comp.Streamed)
}

// requestAttrs is the root element's attributes: the typed scalars, then every
// namespace declaration the body will need, then the provider parameters that
// have a declared attribute of their own.
func requestAttrs(req Request) []attr {
	attrs := []attr{{name: "xmlns", value: NS}}
	for _, d := range KnownDialects() {
		if len(req.DialectExtra[d]) > 0 {
			attrs = append(attrs, attr{name: "xmlns:" + dialectPrefix[d], value: dialectNS[d]})
		}
	}
	attrs = append(attrs, optAttr("model", req.Model)...)
	if req.MaxTokens > 0 {
		attrs = append(attrs, intAttr("max-tokens", req.MaxTokens))
	}
	attrs = append(attrs, optAttr("cache-key", req.CacheKey)...)
	return append(attrs, qualifiedParamAttrs(req.DialectExtra)...)
}

// qualifiedParamAttrs renders the provider parameters that have a declared
// attribute: a scalar whose name the dialect's table knows. Everything else is
// left for the dialect's <params> tree.
func qualifiedParamAttrs(extra map[Dialect]map[string]any) []attr {
	var out []attr
	for _, d := range KnownDialects() {
		params := extra[d]
		if len(params) == 0 {
			continue
		}
		table := knownAttrs[d]
		for _, k := range sortedKeys(params) {
			local, known := table[k]
			if !known {
				continue
			}
			text, ok := scalarText(params[k])
			if !ok {
				continue
			}
			out = append(out, attr{name: dialectPrefix[d] + ":" + local, value: text})
		}
	}
	return out
}

// scalarText renders a value that can live in an attribute, reporting false for
// anything object-shaped -- which belongs in a nested node instead.
func scalarText(v any) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, true
	case bool:
		return boolText(t), true
	case int:
		return strconv.Itoa(t), true
	case int64:
		return strconv.FormatInt(t, 10), true
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64), true
	}
	return "", false
}

// writeRequestBody writes the children shared by <request> and <conversation>.
func writeRequestBody(x *writer, req Request) {
	if parts := req.EffectiveSystemParts(); len(parts) > 0 {
		x.start(elSystem)
		writeParts(x, parts)
		x.end()
	}
	if len(req.Messages) > 0 {
		x.start(elMessages)
		for _, m := range req.Messages {
			writeMessage(x, m)
		}
		x.end()
	}
	if len(req.Tools) > 0 {
		x.start(elTools)
		for _, t := range req.Tools {
			writeTool(x, t)
		}
		x.end()
	}
	writeParamsElement(x, elParams, req.Extra)
	for _, d := range KnownDialects() {
		writeDialectParams(x, d, req.DialectExtra[d])
	}
}

// writeMessage writes one transcript entry.
func writeMessage(x *writer, m Message) {
	attrs := []attr{{name: "role", value: string(m.Role)}}
	attrs = append(attrs, optAttr("tool-call-id", m.ToolCallID)...)
	if m.ToolIsError {
		attrs = append(attrs, attr{name: "tool-is-error", value: "true"})
	}
	x.start(elMessage, attrs...)
	writeParts(x, m.EffectiveParts())
	x.end()
}

// writeParts writes a message's ordered content.
func writeParts(x *writer, parts []Part) {
	for _, p := range parts {
		writePart(x, p)
	}
}

// writePart writes one content part.
func writePart(x *writer, p Part) {
	switch v := p.(type) {
	case TextPart:
		x.start(elText)
		x.text(v.Text)
		x.end()
	case ImagePart:
		attrs := optAttr("media-type", v.MediaType)
		attrs = append(attrs, optAttr("src", v.Src)...)
		x.element(elImage, v.Data, attrs...)
	case ThinkingPart:
		attrs := optAttr("signature", v.Signature)
		attrs = append(attrs, optAttr("id", v.ID)...)
		x.element(elThinking, v.Text, attrs...)
	case RedactedThinkingPart:
		x.element(elRedacted, v.Data)
	case ToolCallPart:
		attrs := optAttr("id", v.ID)
		attrs = append(attrs, optAttr("name", v.Name)...)
		x.start(elToolCall, attrs...)
		x.start(elArguments)
		x.text(v.Arguments)
		x.end()
		x.end()
	}
}

// writeTool writes one advertised tool. The input schema is a param tree, not
// a JSON string: a schema nothing can validate is not a schema.
func writeTool(x *writer, t ToolDecl) {
	attrs := []attr{{name: "name", value: t.Name}}
	attrs = append(attrs, optAttr("description", t.Description)...)
	if t.Readonly {
		attrs = append(attrs, attr{name: "readonly", value: "true"})
	}
	// A nil pointer is written as an ABSENT attribute, never as "false": the
	// two mean different things here, and destructive/open-world default to
	// true when absent.
	attrs = append(attrs, optBoolAttr("destructive", t.Destructive)...)
	if t.Idempotent {
		attrs = append(attrs, attr{name: "idempotent", value: "true"})
	}
	attrs = append(attrs, optBoolAttr("open-world", t.OpenWorld)...)
	if t.Unvouched {
		attrs = append(attrs, attr{name: "unvouched", value: "true"})
	}
	x.start(elTool, attrs...)
	params, err := ParamsFromJSONObject(t.schema())
	if err != nil {
		if x.err == nil {
			x.err = err
		}
		x.end()
		return
	}
	x.start(elInputSchema)
	writeParams(x, params)
	x.end()
	x.end()
}

// writeParamsElement writes a <params> tree for a map of values, skipping the
// element entirely when there is nothing to say.
func writeParamsElement(x *writer, name string, values map[string]any) {
	params, err := paramsFromMap(values)
	if err != nil {
		if x.err == nil {
			x.err = err
		}
		return
	}
	if len(params) == 0 {
		return
	}
	x.start(name)
	writeParams(x, params)
	x.end()
}

// writeDialectParams writes the provider parameters that did NOT become
// qualified attributes: the object-shaped ones, and any name the dialect's
// table does not know.
func writeDialectParams(x *writer, d Dialect, values map[string]any) {
	if len(values) == 0 {
		return
	}
	table := knownAttrs[d]
	rest := make(map[string]any, len(values))
	for k, v := range values {
		if _, known := table[k]; known {
			if _, scalar := scalarText(v); scalar {
				continue
			}
		}
		rest[k] = v
	}
	writeParamsElement(x, dialectPrefix[d]+":"+elParams, rest)
}

// paramsFromMap converts a Go value map into params, in a deterministic order.
// The values go through JSON because that is what they are: whatever the caller
// would have handed the provider verbatim.
func paramsFromMap(values map[string]any) ([]Param, error) {
	if len(values) == 0 {
		return nil, nil
	}
	keys := sortedKeys(values)
	out := make([]Param, 0, len(keys))
	for _, k := range keys {
		raw, err := marshalValue(values[k])
		if err != nil {
			return nil, err
		}
		p, err := ParamFromJSON(k, raw)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

// writeParams writes a param tree.
func writeParams(x *writer, params []Param) {
	for _, p := range params {
		writeParam(x, p)
	}
}

// writeParam writes one param node.
func writeParam(x *writer, p Param) {
	attrs := optAttr("name", p.Name)
	attrs = append(attrs, attr{name: "type", value: p.Type})
	switch p.Type {
	case ParamObject, ParamArray:
		if len(p.Children) == 0 {
			x.empty(elParam, attrs...)
			return
		}
		x.start(elParam, attrs...)
		writeParams(x, p.Children)
		x.end()
	case ParamNull:
		x.empty(elParam, attrs...)
	default:
		x.start(elParam, attrs...)
		x.text(p.Value)
		x.end()
	}
}

// EncodeRequestBytes renders a request document to bytes.
func EncodeRequestBytes(req Request) ([]byte, error) {
	var b bytes.Buffer
	if err := EncodeRequest(&b, req); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// EncodeResponseBytes renders a response document to bytes.
func EncodeResponseBytes(comp *Completion) ([]byte, error) {
	var b bytes.Buffer
	if err := EncodeResponse(&b, comp); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}
