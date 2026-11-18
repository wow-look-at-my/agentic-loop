package commonai

import _ "embed"

// The response bodies these tests decode live in testdata/, verbatim.

//go:embed testdata/synthetic_model_list.json
var syntheticModelList string

//go:embed testdata/limit_spellings.json
var limitSpellingsList string

//go:embed testdata/openrouter_model_list.json
var openRouterModelList string

//go:embed testdata/crofai_model_list.json
var crofAIModelList string

//go:embed testdata/anthropic_model_list.json
var anthropicModelList string

//go:embed testdata/anthropic_priced_model_list.json
var anthropicPricedModelList string

//go:embed testdata/openai_nonstream_response.json
var openAINonStreamResponse string

//go:embed testdata/anthropic_nonstream_response.json
var anthropicNonStreamResponse string

//go:embed testdata/openai_served_model.json
var openAIServedModelResponse string

//go:embed testdata/served_model_list.json
var servedModelList string

//go:embed testdata/openai_routed_model.json
var openAIRoutedModelResponse string

//go:embed testdata/thinking_repair_stream.yml
var thinkingRepairStream string

//go:embed testdata/gzip_anthropic_stream.yml
var gzipAnthropicStream string
