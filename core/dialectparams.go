package commonai

import "sort"

// Provider params live in the dialect's namespace: known scalars as attributes, the rest in the declared params tree.

// dialectNS maps a dialect to its namespace.
var dialectNS = map[Dialect]string{
	DialectAnthropic: NSAnthropic,
	DialectOpenAI:    NSOpenAI,
	DialectResponses: NSResponses,
}

// dialectPrefix maps a dialect to the prefix the writer emits for it.
var dialectPrefix = map[Dialect]string{
	DialectAnthropic: prefixAnthropic,
	DialectOpenAI:    prefixOpenAI,
	DialectResponses: prefixResponses,
}

// knownAttrs is, per dialect, the scalar parameters that ride as qualified
// attributes: the wire name a provider expects, mapped to the attribute's
// local name. The two differ only in spelling -- underscores are not how XML
// names read -- and the mapping is what keeps the round trip exact.
var knownAttrs = map[Dialect]map[string]string{
	DialectAnthropic: {
		"top_k":       "top-k",
		"top_p":       "top-p",
		"temperature": "temperature",
	},
	DialectOpenAI: {
		"reasoning_effort":    "reasoning-effort",
		"seed":                "seed",
		"temperature":         "temperature",
		"top_p":               "top-p",
		"frequency_penalty":   "frequency-penalty",
		"presence_penalty":    "presence-penalty",
		"parallel_tool_calls": "parallel-tool-calls",
	},
	DialectResponses: {
		"reasoning_effort": "reasoning-effort",
		"temperature":      "temperature",
		"top_p":            "top-p",
		"truncation":       "truncation",
	},
}

// attrToWire is the reverse of knownAttrs, built once so decoding does not scan.
var attrToWire = buildAttrToWire()

// buildAttrToWire inverts knownAttrs.
func buildAttrToWire() map[Dialect]map[string]string {
	out := make(map[Dialect]map[string]string, len(knownAttrs))
	for d, m := range knownAttrs {
		rev := make(map[string]string, len(m))
		for wire, local := range m {
			rev[local] = wire
		}
		out[d] = rev
	}
	return out
}

// KnownDialects is every dialect that has a namespace in the format, in a
// stable order.
func KnownDialects() []Dialect {
	return []Dialect{DialectAnthropic, DialectOpenAI, DialectResponses}
}

// sortedKeys returns a map's keys in a deterministic order; Go map iteration is randomized.
func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
