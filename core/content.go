package commonai

// IContent is one block of a user, system or tool-result message.
type IContent interface {
	Meta() *Meta
	contentBlock()
}

// TextContent is text exactly as it arrived, control characters included.
type TextContent struct {
	Text string
	meta Meta
}

// Text is a text block.
func Text(s string) *TextContent { return &TextContent{Text: s} }

func (c *TextContent) Meta() *Meta { return &c.meta }
func (*TextContent) contentBlock() {}

// BlobRef is binary content held on disk as raw bytes: the bytes never stay on the heap as base64.
type BlobRef struct {
	// Hash is the hex SHA-256 of the bytes. It names the file in the blob store.
	Hash      string
	MediaType string
	Size      uint64
	store     *BlobStore
}

// ImageContent is an image, by reference to a blob or by URL. It is never re-encoded.
type ImageContent struct {
	MediaType string
	Src       string
	Blob      *BlobRef
	meta      Meta
}

func (c *ImageContent) Meta() *Meta { return &c.meta }
func (*ImageContent) contentBlock() {}

// TextOf joins the text blocks of a content list.
func TextOf(content []IContent) string {
	switch len(content) {
	case 0:
		return ""
	case 1:
		if t, ok := content[0].(*TextContent); ok {
			return t.Text
		}
	}
	var n int
	for _, c := range content {
		if t, ok := c.(*TextContent); ok {
			n += len(t.Text)
		}
	}
	b := make([]byte, 0, n)
	for _, c := range content {
		if t, ok := c.(*TextContent); ok {
			b = append(b, t.Text...)
		}
	}
	return string(b)
}
