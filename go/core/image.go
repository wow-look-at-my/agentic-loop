package commonai

// An image is held the way it was supplied -- inline bytes with a media type,
// or a URI -- and it is never converted between the two. The format cannot
// fetch a URL (that would be this library making a request the caller did not
// ask for, to a host the caller did not name), and it cannot invent a URL for
// bytes. So a dialect that cannot express the form it was given says so.

// dataURI is an inline image as the data: URI the OpenAI dialects take.
func (i ImagePart) dataURI() string {
	return "data:" + i.MediaType + ";base64," + i.Data
}

// inline reports whether the image carries its own bytes.
func (i ImagePart) inline() bool { return i.Data != "" && i.MediaType != "" }

// imageRef is the reference form of an image for a dialect that takes a URI:
// the supplied URI, or a data: URI built from the supplied bytes.
func imageRef(d Dialect, i ImagePart) (string, error) {
	switch {
	case i.Src != "":
		return i.Src, nil
	case i.inline():
		return i.dataURI(), nil
	case i.Data != "":
		return "", Unsupported(d, "an image with no media type",
			"inline bytes need one to be sent, and guessing it would announce the image as something it may not be")
	}
	return "", Unsupported(d, "an empty image", "it carries neither a source nor any bytes")
}

// hasImage reports whether a message holds one, which is what decides between
// the plain-text and the block form of a message on the dialects that have
// both.
func hasImage(m Message) bool {
	for _, p := range m.EffectiveParts() {
		if p.Kind() == PartKindImage {
			return true
		}
	}
	return false
}
