package commonai

import (
	"encoding/base64"
	"fmt"
)

// ReasoningEncoding is how an EncryptedReasoning rebuilds the exact text the provider sent.
type ReasoningEncoding uint8

const (
	ReasoningBase64Std ReasoningEncoding = iota
	ReasoningBase64RawStd
	ReasoningBase64URL
	ReasoningBase64RawURL
	// ReasoningVerbatim holds text that is not canonical base64, as it arrived.
	ReasoningVerbatim
)

func (e ReasoningEncoding) String() string {
	switch e {
	case ReasoningBase64Std:
		return "padded standard base64"
	case ReasoningBase64RawStd:
		return "unpadded standard base64"
	case ReasoningBase64URL:
		return "padded URL-safe base64"
	case ReasoningBase64RawURL:
		return "unpadded URL-safe base64"
	case ReasoningVerbatim:
		return "verbatim"
	default:
		return fmt.Sprintf("ReasoningEncoding(%d)", uint8(e))
	}
}

var reasoningEncodings = []struct {
	enc ReasoningEncoding
	b64 *base64.Encoding
}{
	{ReasoningBase64Std, base64.StdEncoding},
	{ReasoningBase64RawStd, base64.RawStdEncoding},
	{ReasoningBase64URL, base64.URLEncoding},
	{ReasoningBase64RawURL, base64.RawURLEncoding},
}

// EncryptedReasoning is an opaque reasoning payload: a signature,
// encrypted_content, redacted thinking data.
type EncryptedReasoning struct {
	Data     []byte
	Encoding ReasoningEncoding
}

// NewEncryptedReasoning keeps the first encoding that gives the wire text back byte for byte.
// Text that no base64 form round-trips is kept verbatim. Empty text gives nil.
func NewEncryptedReasoning(wire string) *EncryptedReasoning {
	if wire == "" {
		return nil
	}
	for _, e := range reasoningEncodings {
		data, err := e.b64.DecodeString(wire)
		if err == nil && e.b64.EncodeToString(data) == wire {
			return &EncryptedReasoning{Data: data, Encoding: e.enc}
		}
	}
	return &EncryptedReasoning{Data: []byte(wire), Encoding: ReasoningVerbatim}
}

func (e *EncryptedReasoning) b64() *base64.Encoding {
	for _, c := range reasoningEncodings {
		if c.enc == e.Encoding {
			return c.b64
		}
	}
	if e.Encoding != ReasoningVerbatim {
		panic(fmt.Sprintf("commonai: %v has no wire form", e.Encoding))
	}
	return nil
}

// String is the exact wire text. A nil value is "".
func (e *EncryptedReasoning) String() string {
	if e == nil {
		return ""
	}
	if b := e.b64(); b != nil {
		return b.EncodeToString(e.Data)
	}
	return string(e.Data)
}

// Len is the length of the wire text, without building it.
func (e *EncryptedReasoning) Len() uint {
	if e == nil {
		return 0
	}
	if b := e.b64(); b != nil {
		return uint(b.EncodedLen(len(e.Data)))
	}
	return uint(len(e.Data))
}

// Equal reports whether payloads give the same wire text.
func (e *EncryptedReasoning) Equal(o *EncryptedReasoning) bool {
	return e.String() == o.String()
}
