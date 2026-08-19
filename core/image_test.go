package commonai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// withImage is a user turn holding text and an inline image, the shape a
// caller gets from "here is a screenshot, what is this?".
func withImage() Request {
	return Request{
		Model:     "m",
		MaxTokens: 64,
		Messages: []Message{NewMessage(RoleUser,
			TextPart{Text: "what is this?"},
			ImagePart{MediaType: "image/png", Data: "iVBORw0KGgo="},
		)},
	}
}

// bodyOf runs a call against a recording server and returns the request body
// the dialect built.
func bodyOf(t *testing.T, build func(baseURL string) Provider, req Request) map[string]any {
	t.Helper()
	var body map[string]any
	srv := recordingServer(t, &body)
	_, err := build(srv.URL).Complete(context.Background(), req, nil)
	require.NoError(t, err)
	return body
}

func TestOpenAISendsAnInlineImage(t *testing.T) {
	body := bodyOf(t, func(u string) Provider {
		return mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: u}})
	}, withImage())

	msgs, ok := body["messages"].([]any)
	require.True(t, ok)
	content, ok := msgs[0].(map[string]any)["content"].([]any)
	require.True(t, ok, "a message with an image takes the block form")
	require.Len(t, content, 2)
	assert.Equal(t, "what is this?", content[0].(map[string]any)["text"])
	img := content[1].(map[string]any)
	assert.Equal(t, "image_url", img["type"])
	assert.Equal(t, "data:image/png;base64,iVBORw0KGgo=",
		img["image_url"].(map[string]any)["url"])
}

// A text-only message keeps the plain string form, which is what every
// OpenAI-compatible server takes -- including the ones that never implemented
// the block array.
func TestOpenAIKeepsTextOnlyMessagesPlain(t *testing.T) {
	body := bodyOf(t, func(u string) Provider {
		return mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: u}})
	}, Request{Model: "m", Messages: []Message{NewMessage(RoleUser, TextPart{Text: "hi"})}})

	msgs := body["messages"].([]any)
	assert.Equal(t, "hi", msgs[0].(map[string]any)["content"])
}

func TestAnthropicSendsAnInlineImage(t *testing.T) {
	body := bodyOf(t, func(u string) Provider {
		return mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: u}})
	}, withImage())

	msgs := body["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	require.Len(t, content, 2)
	img := content[1].(map[string]any)
	assert.Equal(t, "image", img["type"])
	source := img["source"].(map[string]any)
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "image/png", source["media_type"])
	assert.Equal(t, "iVBORw0KGgo=", source["data"])
}

// An image supplied as a reference stays a reference: fetching it would be
// this library making a request to a host the caller never named.
func TestAnthropicSendsAnImageReferenceAsAReference(t *testing.T) {
	req := withImage()
	req.Messages = []Message{NewMessage(RoleUser, ImagePart{Src: "https://example.com/a.png"})}
	body := bodyOf(t, func(u string) Provider {
		return mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: u}})
	}, req)

	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	source := content[0].(map[string]any)["source"].(map[string]any)
	assert.Equal(t, "url", source["type"])
	assert.Equal(t, "https://example.com/a.png", source["url"])
}

func TestResponsesSendsAnInlineImage(t *testing.T) {
	body := bodyOf(t, func(u string) Provider {
		return mustResponses(t, ResponsesConfig{ProviderConfig: ProviderConfig{BaseURL: u}})
	}, withImage())

	input := body["input"].([]any)
	content := input[0].(map[string]any)["content"].([]any)
	require.Len(t, content, 2)
	assert.Equal(t, "input_text", content[0].(map[string]any)["type"])
	img := content[1].(map[string]any)
	assert.Equal(t, "input_image", img["type"])
	assert.Equal(t, "data:image/png;base64,iVBORw0KGgo=", img["image_url"])
}

// An image nobody can send is a loud failure, not a request quietly stripped
// of the thing the caller asked about.
func TestAnImageWithNothingInItFailsLoudly(t *testing.T) {
	for _, tc := range []struct {
		name string
		img  ImagePart
		want string
	}{
		{"no media type", ImagePart{Data: "iVBORw0KGgo="}, "media type"},
		{"nothing at all", ImagePart{}, "empty image"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := Request{Model: "m", MaxTokens: 64,
				Messages: []Message{NewMessage(RoleUser, tc.img)}}
			for name, build := range map[string]func(string) Provider{
				"openai": func(u string) Provider {
					return mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: u}})
				},
				"anthropic": func(u string) Provider {
					return mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: u}})
				},
				"responses": func(u string) Provider {
					return mustResponses(t, ResponsesConfig{ProviderConfig: ProviderConfig{BaseURL: u}})
				},
			} {
				var body map[string]any
				srv := recordingServer(t, &body)
				_, err := build(srv.URL).Complete(context.Background(), req, nil)
				require.Error(t, err, name)
				assert.True(t, IsUnsupported(err), name)
				assert.Contains(t, err.Error(), tc.want, name)
				assert.Nil(t, body, "nothing was sent: %s", name)
			}
		})
	}
}

// An image survives the format itself, which is what makes any of the above
// reachable from a document.
func TestImageRoundTripsThroughTheDocument(t *testing.T) {
	req := withImage()
	data, err := EncodeRequestBytes(req)
	require.NoError(t, err)
	require.NoError(t, Validate(data), "document:\n%s", data)

	back, err := DecodeRequest(data)
	require.NoError(t, err)
	require.Len(t, back.Messages[0].Parts, 2)
	img, ok := back.Messages[0].Parts[1].(ImagePart)
	require.True(t, ok)
	assert.Equal(t, "image/png", img.MediaType)
	assert.Equal(t, "iVBORw0KGgo=", img.Data)

}
