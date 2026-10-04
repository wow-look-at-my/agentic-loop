package cli

import (
	"errors"
	"fmt"

	"github.com/spf13/cobra"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/session"
)

var chatSession string

var chatCmd = &cobra.Command{
	Use:   "chat [prompt]",
	Short: "Ask a question in a stored conversation",
	Long: "Ask a question in a stored conversation, so the model sees what was " +
		"said before.\n\nThe conversation is named by --session and kept as one " +
		"document per session under --sessions.",
	Args: cobra.ArbitraryArgs,
	RunE: runChat,
}

func init() {
	chatCmd.Flags().StringVar(&chatSession, "session", "default", "which conversation to continue")
	root.AddCommand(chatCmd)
}

func runChat(cmd *cobra.Command, args []string) error {
	prompt, err := promptFrom(cmd, args)
	if err != nil {
		return err
	}
	turn, err := buildRequest(prompt)
	if err != nil {
		return err
	}
	st, err := store()
	if err != nil {
		return err
	}

	req, err := continueConversation(st, chatSession, turn)
	if err != nil {
		return err
	}
	p, err := newProvider(cmd)
	if err != nil {
		return err
	}
	comp, callErr := p.Complete(cmd.Context(), req, printing(cmd.OutOrStdout()))
	if comp != nil {
		// What the caller has already seen belongs in the transcript even when the call failed partway.
		if _, err := st.Append(chatSession, comp.Message); err != nil {
			return fmt.Errorf("keeping the answer: %w", err)
		}
	}
	return report(cmd, comp, callErr)
}

// continueConversation appends the turn to the stored conversation, starting
// when the name is new, and returns the call to make over the whole
// transcript.
func continueConversation(st session.Store, name string, turn commonai.Request) (commonai.Request, error) {
	stored, err := st.Append(name, turn.Messages...)
	if errors.Is(err, session.ErrNotFound) {
		if err := st.Put(name, turn); err != nil {
			return commonai.Request{}, err
		}
		return turn, nil
	}
	if err != nil {
		return commonai.Request{}, err
	}
	// The flags govern this turn; what they did not state stays as the conversation was.
	out := stored
	out.Model = turn.Model
	if turn.System != "" {
		out.System = turn.System
	}
	if turn.MaxTokens > 0 {
		out.MaxTokens = turn.MaxTokens
	}
	if len(turn.Extra) > 0 {
		out.Extra = turn.Extra
	}
	return out, nil
}
