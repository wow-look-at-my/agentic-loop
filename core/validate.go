package commonai

import (
	"embed"
	"fmt"
	"path"

	"github.com/wow-look-at-my/xml-validator/validator"
)

// The schema is the normative format definition, embedded so validation needs nothing on disk.

//go:embed schema/*.xsd
var schemaFS embed.FS

// SchemaFS exposes the schema files, for a caller that wants to write them out
// or serve them.
func SchemaFS() embed.FS { return schemaFS }

// schemaMain is the entry point that imports the dialect schemas.
const schemaMain = "schema/common-ai-api.xsd"

// Validate checks a document against the schema before anything acts on it.
func Validate(data []byte) error {
	main, err := schemaFS.ReadFile(schemaMain)
	if err != nil {
		return fmt.Errorf("commonai: reading the schema: %w", err)
	}
	if err := validator.ValidateWithSchemaResolver(data, main, embeddedResolver); err != nil {
		return fmt.Errorf("commonai: the document does not match the schema: %w", err)
	}
	return nil
}

// embeddedResolver resolves an xs:import schemaLocation against the embedded
// files.
func embeddedResolver(_, location string) ([]byte, error) {
	data, err := schemaFS.ReadFile(path.Join("schema", location))
	if err != nil {
		return nil, fmt.Errorf("commonai: the schema imports %q, which is not embedded: %w", location, err)
	}
	return data, nil
}
