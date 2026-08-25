// Package cli holds cai's commands, one per file, each registering itself.
package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"github.com/wow-look-at-my/agentic-loop/client"
	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/session"
)

// root is the command tree. Subcommands add themselves to it from their own
// files, so adding one is adding a file.
var root = &cobra.Command{
	Use:   "cai",
	Short: "Talk to a model",
	Long: "cai talks to a model over the common AI API.\n\n" +
		"Endpoint, dialect and credentials come from flags and the environment " +
		"of this process -- never from anything a model or a stored conversation " +
		"said.",
	SilenceUsage:  true,
	SilenceErrors: true,
}

// Connection flags, shared by every command that makes a call.
var (
	flagEndpoint string
	flagDialect  string
	flagModel    string
	flagSystem   string
	flagMaxTok   int
	flagTemp     float64
	flagImages   []string
	flagSessions string
)

func init() {
	pf := root.PersistentFlags()
	pf.StringVar(&flagEndpoint, "endpoint", "", "API base URL (default $CAI_ENDPOINT)")
	pf.StringVar(&flagDialect, "dialect", "", "wire dialect: openai, anthropic or responses (default $CAI_DIALECT, else detected)")
	pf.StringVar(&flagModel, "model", "", "model name (default $CAI_MODEL)")
	pf.StringVar(&flagSystem, "system", "", "system prompt")
	pf.IntVar(&flagMaxTok, "max-tokens", 0, "cap the answer's length")
	pf.Float64Var(&flagTemp, "temperature", -1, "sampling temperature; unset leaves it to the provider")
	pf.StringArrayVar(&flagImages, "image", nil, "attach an image file (repeatable)")
	pf.StringVar(&flagSessions, "sessions", "", "directory holding stored conversations (default $CAI_SESSIONS, else ~/.cai/sessions)")
}

// Execute runs the command tree.
func Execute() error { return root.Execute() }

// env reads a setting from a flag, falling back to the environment.
func env(flag, key string) string {
	if flag != "" {
		return flag
	}
	return os.Getenv(key)
}

// newProvider builds the provider the flags describe. The API key comes from
// the environment only: a key on a command line is a key in the shell history
// and in every process listing on the machine.
func newProvider(cmd *cobra.Command) (client.Provider, error) {
	endpoint := env(flagEndpoint, "CAI_ENDPOINT")
	if endpoint == "" {
		return nil, fmt.Errorf("no endpoint: pass --endpoint or set CAI_ENDPOINT")
	}
	cfg := client.ProviderConfig{BaseURL: endpoint, APIKey: os.Getenv("CAI_API_KEY")}

	dialect := commonai.Dialect(env(flagDialect, "CAI_DIALECT"))
	if dialect == "" {
		list, err := client.FetchModelList(cmd.Context(), cfg)
		if err == nil && list.Dialect == commonai.DialectAuto {
			err = fmt.Errorf("the model list matches neither dialect")
		}
		if err != nil {
			return nil, fmt.Errorf("cannot tell which dialect %s speaks (pass --dialect): %w", endpoint, err)
		}
		dialect = list.Dialect
	}
	switch dialect {
	case client.DialectOpenAI:
		return client.NewOpenAIProvider(client.OpenAIConfig{ProviderConfig: cfg})
	case client.DialectAnthropic:
		return client.NewAnthropicProvider(client.AnthropicConfig{ProviderConfig: cfg})
	case client.DialectResponses:
		return client.NewResponsesProvider(client.ResponsesConfig{ProviderConfig: cfg})
	}
	return nil, fmt.Errorf("unknown dialect %q: openai, anthropic or responses", dialect)
}

// buildRequest turns the flags plus a prompt into the call to make.
func buildRequest(prompt string) (commonai.Request, error) {
	model := env(flagModel, "CAI_MODEL")
	if model == "" {
		return commonai.Request{}, fmt.Errorf("no model: pass --model or set CAI_MODEL")
	}
	parts := []commonai.Part{}
	for _, path := range flagImages {
		p, err := imagePart(path)
		if err != nil {
			return commonai.Request{}, err
		}
		parts = append(parts, p)
	}
	if prompt != "" {
		parts = append(parts, commonai.TextPart{Text: prompt})
	}
	if len(parts) == 0 {
		return commonai.Request{}, fmt.Errorf("nothing to ask: give a prompt, pipe one in, or attach an image")
	}
	req := commonai.Request{
		Model:     model,
		System:    flagSystem,
		MaxTokens: flagMaxTok,
		Messages:  []commonai.Message{commonai.NewMessage(commonai.RoleUser, parts...)},
	}
	if flagTemp >= 0 {
		req.Extra = map[string]any{"temperature": flagTemp}
	}
	return req, nil
}

// store opens the conversation store the flags point at.
func store() (session.Store, error) {
	dir := env(flagSessions, "CAI_SESSIONS")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("no home directory to keep sessions in (pass --sessions): %w", err)
		}
		dir = filepath.Join(home, ".cai", "sessions")
	}
	return session.NewFile(dir)
}

// answerText is what a completion says; reasoning is excluded so it never silently becomes part of piped data.
func answerText(comp *client.Completion) string {
	if comp == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range comp.Message.EffectiveParts() {
		if tp, ok := p.(commonai.TextPart); ok {
			b.WriteString(tp.Text)
		}
	}
	return b.String()
}
