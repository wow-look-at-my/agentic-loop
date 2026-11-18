package client

import _ "embed"

//go:embed testdata/openai_plain_usage.json
var openAIPlainUsageBody string

//go:embed testdata/anthropic_content_json.json
var anthropicContentJSONBody string
