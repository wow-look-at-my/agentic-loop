package commonai

import (
	"io"
	"strconv"
)

// The stream IS the response document: text deltas write directly into the open <text> element.

// ResponseWriter writes a response document incrementally. Its methods are not
// safe for concurrent use: one call streams from one goroutine.
type ResponseWriter struct {
	x *writer
	// open is the part element accepting text, so same-kind deltas extend it.
	open string
	// flusher is called after every delta; a buffered document is not streaming.
	flusher interface{ Flush() }
}

// NewResponseWriter starts a <response> document for a message of the given
// role.
func NewResponseWriter(w io.Writer, role Role) *ResponseWriter {
	if role == "" {
		role = RoleAssistant
	}
	x := newWriter(w)
	x.start(elResponse, attr{name: "xmlns", value: NS},
		attr{name: "xmlns:" + prefixOpenAI, value: NSOpenAI},
		attr{name: "role", value: string(role)})
	rw := &ResponseWriter{x: x}
	if f, ok := w.(interface{ Flush() }); ok {
		rw.flusher = f
	}
	return rw
}

// Text appends a content delta, opening a <text> element if one is not already
// open.
func (rw *ResponseWriter) Text(delta string) error {
	if delta == "" {
		return rw.x.err
	}
	rw.openPart(elText)
	rw.x.text(delta)
	return rw.flush()
}

// Reasoning appends a delta, opening a <thinking> element; a signed block goes through Part.
func (rw *ResponseWriter) Reasoning(delta string) error {
	if delta == "" {
		return rw.x.err
	}
	rw.openPart(elThinking)
	rw.x.text(delta)
	return rw.flush()
}

// Part writes a complete content part, closing whatever was open.
func (rw *ResponseWriter) Part(p Part) error {
	rw.closePart()
	writePart(rw.x, p)
	return rw.flush()
}

// Usage writes one provider usage report, exactly as reported.
func (rw *ResponseWriter) Usage(u Usage) error {
	rw.closePart()
	writeUsage(rw.x, u)
	return rw.flush()
}

// Timings writes one provider timings snapshot.
func (rw *ResponseWriter) Timings(t Timings) error {
	rw.closePart()
	attrs := []attr{
		intAttr("prompt-n", t.PromptN),
		{name: "prompt-ms", value: strconv.FormatFloat(t.PromptMS, 'g', -1, 64)},
		intAttr("predicted-n", t.PredictedN),
		{name: "predicted-ms", value: strconv.FormatFloat(t.PredictedMS, 'g', -1, 64)},
	}
	rw.x.empty(prefixOpenAI+":"+elTimings, attrs...)
	return rw.flush()
}

// Fail records a failure in the document itself and closes it, keeping output and error together.
func (rw *ResponseWriter) Fail(err error) error {
	rw.closePart()
	writeError(rw.x, err)
	rw.x.end()
	return rw.flush()
}

// Close writes the trailing <result/> and closes the document.
func (rw *ResponseWriter) Close(stopReason string, streamed bool) error {
	rw.closePart()
	attrs := optAttr("stop-reason", stopReason)
	attrs = append(attrs, attr{name: "streamed", value: boolText(streamed)})
	rw.x.empty(elResult, attrs...)
	rw.x.end()
	return rw.flush()
}

// openPart opens the named part element unless it is already the open one.
func (rw *ResponseWriter) openPart(name string) {
	if rw.open == name {
		return
	}
	rw.closePart()
	rw.x.start(name)
	rw.open = name
}

// closePart closes the open part element, if any.
func (rw *ResponseWriter) closePart() {
	if rw.open == "" {
		return
	}
	rw.x.end()
	rw.open = ""
}

// flush pushes what has been written so far and reports the first error.
func (rw *ResponseWriter) flush() error {
	if rw.flusher != nil {
		rw.flusher.Flush()
	}
	return rw.x.err
}

// writeUsage writes one usage report: the counts the provider sent, the two
// provider extras worth naming, and its verbatim object as a param tree.
func writeUsage(x *writer, u Usage) {
	attrs := []attr{
		intAttr("prompt-tokens", u.PromptTokens),
		intAttr("completion-tokens", u.CompletionTokens),
		intAttr("total-tokens", u.TotalTokens),
	}
	attrs = append(attrs, ptrIntAttr("cache-read-tokens", u.CacheReadTokens)...)
	attrs = append(attrs, ptrIntAttr("cache-write-tokens", u.CacheWriteTokens)...)
	attrs = append(attrs, ptrIntAttr(prefixOpenAI+":reasoning-tokens", u.ReasoningTokens)...)
	attrs = append(attrs, ptrFloatAttr(prefixOpenAI+":cost-usd", u.CostUsd)...)
	if len(u.Raw) == 0 {
		x.empty(elUsage, attrs...)
		return
	}
	params, err := ParamsFromJSONObject(u.Raw)
	if err != nil {
		// The provider's own usage object did not parse as an object. Say so
		// rather than dropping the numbers it came with.
		if x.err == nil {
			x.err = err
		}
		x.empty(elUsage, attrs...)
		return
	}
	x.start(elUsage, attrs...)
	x.start(elRaw)
	writeParams(x, params)
	x.end()
	x.end()
}

// writeError writes an <error> element, saying what kind of failure it was so a
// reader can tell a rejected request from a dropped connection without parsing
// prose.
// lead is prepended to the attributes, for the standalone document form where
// the element is the root and has to declare the namespace.
func writeError(x *writer, err error, lead ...attr) {
	if err == nil {
		return
	}
	attrs := append(append([]attr{}, lead...), attr{name: "kind", value: errorKind(err)})
	var ae *APIError
	if asAPIError(err, &ae) {
		attrs = append(attrs, intAttr("status", ae.Status))
		if ae.ContextOverflow {
			attrs = append(attrs, attr{name: "context-overflow", value: "true"})
		}
	}
	if IsTransient(err) {
		attrs = append(attrs, attr{name: "transient", value: "true"})
	}
	x.element(elError, err.Error(), attrs...)
}
