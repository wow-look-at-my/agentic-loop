package cmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"
)

var sessionCmd = &cobra.Command{
	Use:   "session",
	Short: "Look at and remove stored conversations",
}

var sessionListCmd = &cobra.Command{
	Use:   "list",
	Short: "List stored conversations",
	Args:  cobra.NoArgs,
	RunE: func(cmd *cobra.Command, _ []string) error {
		st, err := store()
		if err != nil {
			return err
		}
		ids, err := st.List()
		if err != nil {
			return err
		}
		for _, id := range ids {
			conv, err := st.Get(id)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "%s\t%d messages\t%s\n", id, len(conv.Messages), conv.Model)
		}
		return nil
	},
}

var sessionShowCmd = &cobra.Command{
	Use:   "show <name>",
	Short: "Print a stored conversation as text",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		st, err := store()
		if err != nil {
			return err
		}
		conv, err := st.Get(args[0])
		if err != nil {
			return err
		}
		out := cmd.OutOrStdout()
		if conv.System != "" {
			fmt.Fprintf(out, "system: %s\n\n", conv.System)
		}
		for _, m := range conv.Messages {
			fmt.Fprintf(out, "%s: %s\n\n", m.Role, messageText(m))
		}
		return nil
	},
}

var sessionRemoveCmd = &cobra.Command{
	Use:     "rm <name>",
	Aliases: []string{"remove"},
	Short:   "Forget a stored conversation",
	Args:    cobra.ExactArgs(1),
	RunE: func(_ *cobra.Command, args []string) error {
		st, err := store()
		if err != nil {
			return err
		}
		return st.Delete(args[0])
	},
}

func init() {
	sessionCmd.AddCommand(sessionListCmd, sessionShowCmd, sessionRemoveCmd)
	root.AddCommand(sessionCmd)
}

// messageText renders one transcript entry for a human. A tool call shows the
// call rather than nothing: a transcript that silently omits the turns where
// the model asked for something reads as if it never did.
func messageText(m commonai.Message) string {
	var b strings.Builder
	for _, p := range m.EffectiveParts() {
		switch v := p.(type) {
		case commonai.TextPart:
			b.WriteString(v.Text)
		case commonai.ImagePart:
			fmt.Fprintf(&b, "[image %s]", v.MediaType)
		case commonai.ThinkingPart:
			fmt.Fprintf(&b, "[thinking: %s]", v.Text)
		case commonai.RedactedThinkingPart:
			b.WriteString("[thinking, redacted by the provider]")
		case commonai.ToolCallPart:
			fmt.Fprintf(&b, "[calls %s %s]", v.Name, v.Arguments)
		}
	}
	return b.String()
}
