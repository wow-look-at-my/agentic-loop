// Package client is the Go way to speak the common AI API.
//
// The format is the definition; this package is client of it, and the
// reference. Everything here is expressible as a document, which is what
// keeps a client in another language able to do the same thing from the schema
// alone -- but a Go caller pays no serialization for the privilege: a call
// through this package is a struct call into the dialect provider, not a
// document round trip.
//
// What it adds over the core package is what a Go caller wants and a wire
// format has no business deciding:
//
// - [Completion] per call, with the usage reports folded into the single
// figure a caller bills against. Core reports every snapshot a provider
// sent, in order, because that is what the provider said; deciding that the
// newest wins is a policy, and this is where it lives.
// - Retry and rate limiting wired in through [ProviderConfig], so a call is
// ridden through a transient failure without the caller remembering to ask.
package client
