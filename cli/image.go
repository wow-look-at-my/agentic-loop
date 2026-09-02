package cli

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// imagePart reads an image file into the part that carries it. The media type
// is sniffed from the bytes rather than taken from the extension: a file named
//.png that is really a JPEG would otherwise be announced as something it is
// not, and the upstream is the that finds out.
func imagePart(path string) (commonai.Part, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading image %s: %w", path, err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("image %s is empty", path)
	}
	mediaType := http.DetectContentType(data)
	if !strings.HasPrefix(mediaType, "image/") {
		return nil, fmt.Errorf("%s is %s, not an image", filepath.Base(path), mediaType)
	}
	return commonai.ImagePart{
		MediaType: mediaType,
		Data:      base64.StdEncoding.EncodeToString(data),
	}, nil
}
