package commonai

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/wow-look-at-my/xml-validator/validator"
)

// Decoding goes through xml-validator rather than encoding/xml for two
// reasons: it is the parser that accepts `&#0;`, and it is the one that can
// check a document against the schema. Everything that arrives from outside is
// validated before anything acts on it -- see Validate.

// DecodeRequest reads a <request> document.
func DecodeRequest(data []byte) (Request, error) {
	root, err := parseRoot(data, elRequest)
	if err != nil {
		return Request{}, err
	}
	return requestFrom(root)
}

// DecodeConversation reads a <conversation> document, returning its id and the
// request body it carries.
func DecodeConversation(data []byte) (string, Request, error) {
	root, err := parseRoot(data, elConversation)
	if err != nil {
		return "", Request{}, err
	}
	id, _ := root.Attr("id")
	req, err := requestFrom(root)
	return id, req, err
}

// DecodeError reads a standalone <error> document back into the error it
// describes, so a failure keeps its kind and its status across a transport
// instead of arriving as a string a caller has to match on.
func DecodeError(data []byte) error {
	root, err := parseRoot(data, elError)
	if err != nil {
		return err
	}
	return errorFrom(root)
}

// DecodeConversationIDs reads a <conversations> listing.
func DecodeConversationIDs(data []byte) ([]string, error) {
	root, err := parseRoot(data, elConversations)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, el := range root.ChildElements() {
		if el.Local != elConversationID {
			continue
		}
		id, _ := el.Attr("id")
		ids = append(ids, id)
	}
	return ids, nil
}

// DecodeResponse reads a <response> document. A document whose root never
// closed is a stream that was cut: the parts that did arrive come back with an
// error saying so, because throwing away output the caller already watched
// arrive helps nobody.
func DecodeResponse(data []byte) (*Completion, error) {
	root, err := parseRoot(data, elResponse)
	if err != nil {
		return nil, err
	}
	return completionFrom(root)
}

// parseRoot parses a document and checks its root element.
func parseRoot(data []byte, want string) (*validator.Element, error) {
	doc, err := validator.ParseTree(strings.NewReader(string(data)))
	if err != nil {
		return nil, fmt.Errorf("commonai: parsing the document: %w", err)
	}
	if doc.Root == nil {
		return nil, fmt.Errorf("commonai: the document has no root element")
	}
	if doc.Root.Local != want {
		return nil, fmt.Errorf("commonai: expected a <%s> document, got <%s>", want, doc.Root.Local)
	}
	if ns := doc.Root.Namespace; ns != "" && ns != NS {
		return nil, fmt.Errorf("commonai: <%s> is in namespace %q, not %q", want, ns, NS)
	}
	return doc.Root, nil
}

// requestFrom builds a Request from a <request> or <conversation> element.
func requestFrom(root *validator.Element) (Request, error) {
	var req Request
	req.Model = attrOf(root, "model")
	req.CacheKey = attrOf(root, "cache-key")
	if v := attrOf(root, "max-tokens"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Request{}, fmt.Errorf("commonai: max-tokens %q is not a number", v)
		}
		req.MaxTokens = n
	}
	extra, err := qualifiedParams(root)
	if err != nil {
		return Request{}, err
	}
	req.DialectExtra = extra
	for _, child := range root.ChildElements() {
		switch {
		case child.Local == elSystem && isCore(child):
			parts, err := partsFrom(child)
			if err != nil {
				return Request{}, err
			}
			req.SystemParts = parts
			req.System = flattenText(parts)
		case child.Local == elMessages && isCore(child):
			for _, mel := range child.ChildElements() {
				m, err := messageFrom(mel)
				if err != nil {
					return Request{}, err
				}
				req.Messages = append(req.Messages, m)
			}
		case child.Local == elTools && isCore(child):
			for _, tel := range child.ChildElements() {
				t, err := toolFrom(tel)
				if err != nil {
					return Request{}, err
				}
				req.Tools = append(req.Tools, t)
			}
		case child.Local == elParams && isCore(child):
			values, err := valuesFromParams(child)
			if err != nil {
				return Request{}, err
			}
			req.Extra = values
		case child.Local == elParams:
			d, ok := dialectOfNS(child.Namespace)
			if !ok {
				return Request{}, fmt.Errorf("commonai: <params> in unknown namespace %q", child.Namespace)
			}
			values, err := valuesFromParams(child)
			if err != nil {
				return Request{}, err
			}
			mergeInto(req.DialectExtra, d, values)
		}
	}
	return req, nil
}

// completionFrom builds a Completion from a <response> element.
func completionFrom(root *validator.Element) (*Completion, error) {
	comp := &Completion{}
	role := Role(attrOf(root, "role"))
	if role == "" {
		role = RoleAssistant
	}
	var parts []Part
	var failure error
	for _, child := range root.ChildElements() {
		switch {
		case child.Local == elUsage && isCore(child):
			u, err := usageFrom(child)
			if err != nil {
				return nil, err
			}
			comp.Usages = append(comp.Usages, u)
		case child.Local == elTimings && child.Namespace == NSOpenAI:
			comp.Timings = append(comp.Timings, timingsFrom(child))
		case child.Local == elResult && isCore(child):
			comp.StopReason = attrOf(child, "stop-reason")
			comp.Streamed = attrOf(child, "streamed") == "true"
		case child.Local == elError && isCore(child):
			failure = errorFrom(child)
		default:
			p, err := partFrom(child)
			if err != nil {
				return nil, err
			}
			if p != nil {
				parts = append(parts, p)
			}
		}
	}
	comp.Message = Message{Role: role, Parts: parts}
	comp.Message.SyncViews()
	return comp, failure
}

// messageFrom builds one transcript entry.
func messageFrom(el *validator.Element) (Message, error) {
	if el.Local != elMessage {
		return Message{}, fmt.Errorf("commonai: <messages> holds <%s>, which is not a message", el.Local)
	}
	m := Message{
		Role:        Role(attrOf(el, "role")),
		ToolCallID:  attrOf(el, "tool-call-id"),
		ToolIsError: attrOf(el, "tool-is-error") == "true",
	}
	parts, err := partsFrom(el)
	if err != nil {
		return Message{}, err
	}
	m.Parts = parts
	m.SyncViews()
	return m, nil
}

// partsFrom reads an element's content parts.
func partsFrom(el *validator.Element) ([]Part, error) {
	var out []Part
	for _, child := range el.ChildElements() {
		p, err := partFrom(child)
		if err != nil {
			return nil, err
		}
		if p != nil {
			out = append(out, p)
		}
	}
	return out, nil
}

// partFrom reads one content part, returning nil for an element that is not
// one.
func partFrom(el *validator.Element) (Part, error) {
	if !isCore(el) {
		return nil, nil
	}
	switch el.Local {
	case elText:
		return TextPart{Text: el.TextContent()}, nil
	case elImage:
		return ImagePart{
			MediaType: attrOf(el, "media-type"),
			Src:       attrOf(el, "src"),
			Data:      el.TextContent(),
		}, nil
	case elThinking:
		return ThinkingPart{
			Text:      el.TextContent(),
			Signature: attrOf(el, "signature"),
			ID:        attrOf(el, "id"),
		}, nil
	case elRedacted:
		return RedactedThinkingPart{Data: el.TextContent()}, nil
	case elToolCall:
		call := ToolCallPart{ID: attrOf(el, "id"), Name: attrOf(el, "name")}
		for _, c := range el.ChildElements() {
			if c.Local == elArguments {
				call.Arguments = c.TextContent()
			}
		}
		return call, nil
	}
	return nil, nil
}

// toolFrom reads one advertised tool.
func toolFrom(el *validator.Element) (ToolDecl, error) {
	if el.Local != elTool {
		return ToolDecl{}, fmt.Errorf("commonai: <tools> holds <%s>, which is not a tool", el.Local)
	}
	t := ToolDecl{
		Name:        attrOf(el, "name"),
		Description: attrOf(el, "description"),
		Readonly:    attrOf(el, "readonly") == "true",
	}
	for _, c := range el.ChildElements() {
		if c.Local != elInputSchema {
			continue
		}
		params, err := paramsFrom(c)
		if err != nil {
			return ToolDecl{}, err
		}
		raw, err := ParamsJSON(params)
		if err != nil {
			return ToolDecl{}, err
		}
		t.InputSchema = raw
	}
	return t, nil
}

// usageFrom reads one usage report.
func usageFrom(el *validator.Element) (Usage, error) {
	u := Usage{
		PromptTokens:     intAttrOf(el, "prompt-tokens"),
		CompletionTokens: intAttrOf(el, "completion-tokens"),
		TotalTokens:      intAttrOf(el, "total-tokens"),
		CacheReadTokens:  ptrIntAttrOf(el, "cache-read-tokens"),
		CacheWriteTokens: ptrIntAttrOf(el, "cache-write-tokens"),
		ReasoningTokens:  ptrIntAttrOfNS(el, NSOpenAI, "reasoning-tokens"),
	}
	if v, ok := attrOfNS(el, NSOpenAI, "cost-usd"); ok {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return Usage{}, fmt.Errorf("commonai: cost-usd %q is not a number", v)
		}
		u.CostUsd = &f
	}
	for _, c := range el.ChildElements() {
		if c.Local != elRaw {
			continue
		}
		params, err := paramsFrom(c)
		if err != nil {
			return Usage{}, err
		}
		raw, err := ParamsJSON(params)
		if err != nil {
			return Usage{}, err
		}
		u.Raw = raw
	}
	return u, nil
}

// timingsFrom reads one timings snapshot.
func timingsFrom(el *validator.Element) Timings {
	return Timings{
		PromptN:     intAttrOf(el, "prompt-n"),
		PromptMS:    floatAttrOf(el, "prompt-ms"),
		PredictedN:  intAttrOf(el, "predicted-n"),
		PredictedMS: floatAttrOf(el, "predicted-ms"),
	}
}

// errorFrom rebuilds the failure an <error> element describes. An api error
// comes back as an *APIError so IsTransient and IsContextOverflow keep working
// on the far side of the wire.
func errorFrom(el *validator.Element) error {
	msg := el.TextContent()
	switch attrOf(el, "kind") {
	case ErrorKindAPI, ErrorKindOverflow:
		return &APIError{
			Status:          intAttrOf(el, "status"),
			Body:            msg,
			ContextOverflow: attrOf(el, "context-overflow") == "true",
		}
	case ErrorKindRequest:
		return badRequestErr(msg)
	case ErrorKindUnsupported:
		return &UnsupportedError{What: msg}
	}
	return fmt.Errorf("%s", msg)
}

// paramsFrom reads a param tree from an element's children.
func paramsFrom(el *validator.Element) ([]Param, error) {
	var out []Param
	for _, c := range el.ChildElements() {
		if c.Local != elParam {
			continue
		}
		p, err := paramFrom(c)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

// paramFrom reads one param node.
func paramFrom(el *validator.Element) (Param, error) {
	p := Param{Name: attrOf(el, "name"), Type: attrOf(el, "type")}
	switch p.Type {
	case ParamObject, ParamArray:
		children, err := paramsFrom(el)
		if err != nil {
			return Param{}, err
		}
		p.Children = children
	case ParamString, ParamNumber, ParamBoolean:
		p.Value = el.TextContent()
	case ParamNull:
	default:
		return Param{}, fmt.Errorf("commonai: param %q has unknown type %q", p.Name, p.Type)
	}
	return p, nil
}

// valuesFromParams reads a <params> element into the Go value map the dialects
// pass through.
func valuesFromParams(el *validator.Element) (map[string]any, error) {
	params, err := paramsFrom(el)
	if err != nil {
		return nil, err
	}
	if len(params) == 0 {
		return nil, nil
	}
	out := make(map[string]any, len(params))
	for _, p := range params {
		raw, err := p.JSON()
		if err != nil {
			return nil, err
		}
		v, err := unmarshalValue(raw)
		if err != nil {
			return nil, err
		}
		out[p.Name] = v
	}
	return out, nil
}

// qualifiedParams reads the provider parameters that ride as qualified
// attributes on an element.
func qualifiedParams(el *validator.Element) (map[Dialect]map[string]any, error) {
	out := map[Dialect]map[string]any{}
	for _, a := range el.Attrs {
		if a.Namespace == "" || a.Namespace == NS {
			continue
		}
		d, ok := dialectOfNS(a.Namespace)
		if !ok {
			return nil, fmt.Errorf("commonai: attribute %q is in unknown namespace %q", a.Local, a.Namespace)
		}
		wire, known := attrToWire[d][a.Local]
		if !known {
			return nil, fmt.Errorf("commonai: %s has no parameter %q", d, a.Local)
		}
		if out[d] == nil {
			out[d] = map[string]any{}
		}
		out[d][wire] = a.Value
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

// mergeInto folds a dialect's <params> values into the map, without letting
// them overwrite a qualified attribute that named the same parameter.
func mergeInto(dst map[Dialect]map[string]any, d Dialect, values map[string]any) {
	if len(values) == 0 {
		return
	}
	if dst[d] == nil {
		dst[d] = map[string]any{}
	}
	for k, v := range values {
		if _, exists := dst[d][k]; !exists {
			dst[d][k] = v
		}
	}
}

// dialectOfNS maps a namespace back to its dialect.
func dialectOfNS(ns string) (Dialect, bool) {
	switch ns {
	case NSAnthropic:
		return DialectAnthropic, true
	case NSOpenAI:
		return DialectOpenAI, true
	case NSResponses:
		return DialectResponses, true
	}
	return DialectAuto, false
}

// isCore reports whether an element is in the core vocabulary. An element with
// no namespace counts: a document that declares no default namespace is still
// readable, and the schema is what decides whether it was valid.
func isCore(el *validator.Element) bool {
	return el.Namespace == NS || el.Namespace == ""
}

// attrOf returns an unqualified attribute's value, or "".
func attrOf(el *validator.Element, name string) string {
	for _, a := range el.Attrs {
		if a.Local == name && (a.Namespace == "" || a.Namespace == NS) {
			return a.Value
		}
	}
	return ""
}

// attrOfNS returns a qualified attribute's value.
func attrOfNS(el *validator.Element, ns, name string) (string, bool) {
	for _, a := range el.Attrs {
		if a.Local == name && a.Namespace == ns {
			return a.Value, true
		}
	}
	return "", false
}

// intAttrOf reads an int attribute, defaulting to 0.
func intAttrOf(el *validator.Element, name string) int {
	n, _ := strconv.Atoi(attrOf(el, name))
	return n
}

// floatAttrOf reads a float attribute, defaulting to 0.
func floatAttrOf(el *validator.Element, name string) float64 {
	f, _ := strconv.ParseFloat(attrOf(el, name), 64)
	return f
}

// ptrIntAttrOf reads a tri-state int attribute: absent stays nil, because
// "the provider said nothing" and "the provider said zero" are different
// facts.
func ptrIntAttrOf(el *validator.Element, name string) *int {
	for _, a := range el.Attrs {
		if a.Local == name && (a.Namespace == "" || a.Namespace == NS) {
			if n, err := strconv.Atoi(a.Value); err == nil {
				return &n
			}
		}
	}
	return nil
}

// ptrIntAttrOfNS is ptrIntAttrOf for a qualified attribute.
func ptrIntAttrOfNS(el *validator.Element, ns, name string) *int {
	if v, ok := attrOfNS(el, ns, name); ok {
		if n, err := strconv.Atoi(v); err == nil {
			return &n
		}
	}
	return nil
}

// flattenText is the text view of a part list.
func flattenText(parts []Part) string {
	var b strings.Builder
	for _, p := range parts {
		if t, ok := p.(TextPart); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}
