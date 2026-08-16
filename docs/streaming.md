# Streaming

**The stream is the response document, written as it arrives.** There is no
second vocabulary: no event names, no delta frames, no reassembly rule.

```
<?xml version="1.1" encoding="UTF-8"?>
<response xmlns="..." role="assistant"><text>Hel      <- flush
lo, wor                                            <- flush (7 bytes on the wire)
ld</text><tool-call id="call_1" name="grep">...    <- flush
</tool-call><usage prompt-tokens="12" .../><result stop-reason="tool_use" streamed="true"/></response>
```

A text delta is character data appended inside the open `<text>` element: two
bytes of model output are two bytes on the wire. As an independent frame the
same delta costs about a hundred, because a standalone document inherits
nothing and has to carry its own root and namespace declarations. Stop
redeclaring them and the frames stop being documents -- they are fragments of
one document, which is this design.

The deltas cannot disagree with the result, because they are the result. A
consumer in any language already has a parser for this: `Decoder.Token()`,
expat and SAX are pull parsers over a progressively-arriving document. Event
frames would need a framing layer *underneath* the parser to split them first.

## What rides at the end

`<result stop-reason="..." streamed="..."/>` is the last child, not an
attribute on the root, because the root is written before the call knows
either fact. An attribute that has to be written first cannot hold something
learned last.

## Deltas versus parts

`StreamEvents` carries both, and the split is not redundant.

- `OnText` / `OnReasoning` are deltas: what the text is, as soon as it exists.
- `OnPart` fires when a part is COMPLETE, in message order, exactly once per
  part.

A thinking block is what forces this. Its signature arrives after its text and
belongs on the element -- and by then a host streaming deltas has long since
written `<thinking>` and cannot go back and add an attribute to it. So a host
streams `OnText` for the live view, takes everything else from `OnPart`, and
what it writes is the message the call returns rather than a reordered
approximation of it.

`go/http`'s `streamWriter` is the worked example, and
`TestStreamedAndBufferedDocumentsAgree` pins the property: for the same call,
the streamed document and the buffered one are the same bytes.

## Truncation, and failing mid-answer

A connection that drops leaves a document whose root never closed. That is
exactly the partial completion the `Provider` contract already defines -- the
content that arrived is readable, and the unclosed root is what says the rest
never came. `ReadDocument` (`go/core/docread.go`) reads one document off a
stream by tracking depth, so it returns what arrived rather than blocking for
a close that is not coming.

A call that fails *after* producing output appends a declared `<error>`
element and closes the root: the partial answer and the failure are one
readable document. A call that fails before producing anything never opens the
root at all, so a transport can still answer with a status code and a
standalone `<error>` document. Once the root is open the status code is spent,
which is honest -- the caller has already seen output.

## No framing layer, anywhere

Documents are self-delimiting: a parser knows one ended when its root closes.
So HTTP sends the document as a chunked `application/xml` body (curl shows the
answer appearing live), the unix socket carries documents back-to-back in both
directions, and a websocket sends one message per flush. No SSE envelope to
strip, no length prefixes, no double framing.
