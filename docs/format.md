# The format

Namespace `https://github.com/wow-look-at-my/common-ai-api/schema/v1`, with
`/anthropic`, `/openai` and `/responses` under it. Root elements: `<request>`,
`<response>`, `<conversation>`, `<conversations>`, `<error>`.

The schema is `go/core/schema/`, symlinked to `schema/` at the repo root, and
embedded in the core package so validation needs nothing on disk.

## What losslessness means here

**Input → document → the same input, exactly.** Whatever a caller supplied
comes back the way it was supplied: an image's bytes stay bytes and a URI stays
a URI, a tool schema keeps its key order and its numeric literals, and a
provider parameter nobody enumerated still has somewhere to live.

Cross-dialect output is a different question and deliberately NOT round-trip:
one provider's answer expressed for another provider is a translation, and
where a target dialect cannot express something the answer is an error, never a
silent omission.

## Content parts

A message's content is an ordered list: `<text>`, `<image>`, `<thinking>`,
`<redacted-thinking>`, `<tool-call>`. Order is the meaning -- a reply whose
text brackets a thinking block reads the way the model wrote it, and flattening
it to a string would lose that.

`<text>` holds characters plus character references. `<image>` holds either
inline bytes (`media-type` plus base64 content) or a `src` URI, whichever was
supplied. `<thinking>` carries the `signature` that makes a block replayable;
without it the block is prose ABOUT the reasoning, which is why a dialect that
needs the signature drops an unsigned block rather than sending a paraphrase of
the model to itself.

## Provider-specific data

Everything common is an ordinary element or an unqualified attribute.
Everything provider-specific is namespaced:

- a scalar rides as a qualified attribute on the core element it belongs to
  (`<request openai:seed="42">`, `<usage openai:cost-usd="0.0021">`);
- anything object-shaped rides as a namespaced element
  (`<anthropic:thinking type="enabled" budget-tokens="1024"/>`), whose own
  attributes are unqualified -- the element already carries the namespace.

Each dialect's schema declares its own, and the core schema references them by
`ref`. There is no wildcard anywhere, so a misspelled provider attribute is a
validation error here rather than a 400 from an upstream, and a reader knows
what every attribute in a document means.

A parameter no dialect schema has named goes in a `<params>` tree instead --
under the dialect's namespace when it is addressed to one dialect, unqualified
when it is addressed to whoever runs the request.

## The param tree

The general JSON mapping, used for tool input schemas, a provider's verbatim
usage object, and any parameter not enumerated:

```xml
<param name="stream_options" type="object">
  <param name="include_usage" type="boolean">true</param>
</param>
```

`type` is one of `string|number|boolean|null|object|array`; array items are
`<param>` with no name; a scalar's value is the element's text VERBATIM, so
`1.50` does not come back as `1.5`. Child order carries object key order, which
is what makes a captured provider object reproducible byte for byte.

JSON text in an element would have been shorter to write and impossible to
check: a second format inside the first, with its own escaping rules and
nothing validating it.

## Usage

One `<usage>` element per report the provider sent, in order, exactly as sent.
A call can produce several: OpenAI emits one final snapshot,
usage-on-every-chunk upstreams attach a cumulative one to each, Anthropic sends
fragments. Reconciling them is a policy and the format has none -- the document
says what the provider said, and `go/client` folds them for a Go caller.

An absent `cache-read-tokens` means the provider reported no cache information,
which is not the same as reporting zero. Attribute presence is the tri-state.
