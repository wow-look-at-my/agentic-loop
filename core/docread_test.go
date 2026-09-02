package commonai

import (
	"bufio"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Documents ride back-to-back with nothing between them, which is what makes a
// framing layer unnecessary rather than merely unfashionable.
func TestReadDocumentSplitsAStream(t *testing.T) {
	stream := `<?xml version="1.1"?><a><b>one</b></a>` +
		"\n" + `<?xml version="1.1"?><c/>` +
		`<?xml version="1.1"?><d attr="&gt; not a tag"><!-- </d> --><![CDATA[</d>]]>two</d>`
	r := bufio.NewReader(strings.NewReader(stream))

	first, err := ReadDocument(r)
	require.NoError(t, err)
	assert.Equal(t, `<?xml version="1.1"?><a><b>one</b></a>`, string(first))

	second, err := ReadDocument(r)
	require.NoError(t, err)
	assert.Equal(t, `<?xml version="1.1"?><c/>`, string(second),
		"a self-closing root is a whole document")

	third, err := ReadDocument(r)
	require.NoError(t, err)
	assert.Contains(t, string(third), "two</d>", "a </d> inside a comment, a CDATA section or an attribute is text")
	assert.True(t, strings.HasSuffix(string(third), "</d>"))

	_, err = ReadDocument(r)
	assert.ErrorIs(t, err, io.EOF, "a clean end between documents is not a failure")
}

// A stream cut mid-document must not read as a document: half an answer that
// parses is worse than that says it is half.
func TestReadDocumentRejectsATruncatedStream(t *testing.T) {
	r := bufio.NewReader(strings.NewReader(`<?xml version="1.1"?><a><b>one`))
	_, err := ReadDocument(r)
	require.Error(t, err)
	assert.NotErrorIs(t, err, io.EOF)
	assert.Contains(t, err.Error(), "inside a document")
}

// The documents this format actually carries, read back off a stream.
func TestReadDocumentOverRealDocuments(t *testing.T) {
	req, err := EncodeRequestBytes(Request{
		Model:    "m",
		Messages: []Message{NewMessage(RoleUser, TextPart{Text: "hi <there> &amp;"})},
	})
	require.NoError(t, err)
	resp, err := EncodeResponseBytes(&Completion{
		Message:    NewMessage(RoleAssistant, TextPart{Text: "an answer"}),
		StopReason: StopEndTurn,
	})
	require.NoError(t, err)

	r := bufio.NewReader(strings.NewReader(string(req) + string(resp)))
	got, err := ReadDocument(r)
	require.NoError(t, err)
	assert.Equal(t, string(req), string(got))
	back, err := DecodeRequest(got)
	require.NoError(t, err)
	assert.Equal(t, "hi <there> &amp;", back.Messages[0].Content)

	got, err = ReadDocument(r)
	require.NoError(t, err)
	assert.Equal(t, string(resp), string(got))
}
