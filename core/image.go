package commonai

// An image is held as supplied -- inline bytes or a URI -- and never converted between the.

// dataURI is an inline image as the data: URI the OpenAI dialects take.
func (i ImagePart) dataURI() string {
	return "data:" + i.MediaType + ";base64," + i.Data
}

// inline reports whether the image carries its own bytes.
func (i ImagePart) inline() bool { return i.Data != "" && i.MediaType != "" }

// imageRef is the reference form of an image: the supplied URI, or a data: URI.
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

// hasImage reports whether a message holds, deciding plain-text vs block form.
func hasImage(m Message) bool {
	for _, p := range m.EffectiveParts() {
		if p.Kind() == PartKindImage {
			return true
		}
	}
	return false
}
