package cli

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/wow-look-at-my/agentic-loop/client"
	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

var askCmd = &cobra.Command{
	Use:   "ask [prompt]",
	Short: "Ask one question and print the answer",
	Long: "Ask one question and print the answer as it arrives.\n\n" +
		"With no prompt argument, the prompt is read from stdin, so a pipe " +
		"works:\n    echo 'what is this?' | cai ask",
	Args: cobra.ArbitraryArgs,
	RunE: runAsk,
}

func init() { root.AddCommand(askCmd) }

func runAsk(cmd *cobra.Command, args []string) error {
	prompt, err := promptFrom(cmd, args)
	if err != nil {
		return err
	}
	req, err := buildRequest(prompt)
	if err != nil {
		return err
	}
	p, err := newProvider(cmd)
	if err != nil {
		return err
	}
	comp, callErr := p.Complete(cmd.Context(), req, printing(cmd.OutOrStdout()))
	return report(cmd, comp, callErr)
}

// promptFrom takes the prompt from the arguments, or from stdin when it is a
// pipe. A terminal is never read from: a command that appears to hang while
// waiting for input nobody knew to type is worse than one that says what it
// wanted.
func promptFrom(cmd *cobra.Command, args []string) (string, error) {
	if len(args) > 0 {
		return strings.Join(args, " "), nil
	}
	in := cmd.InOrStdin()
	if f, ok := in.(*os.File); ok {
		info, err := f.Stat()
		if err != nil {
			return "", err
		}
		if info.Mode()&os.ModeCharDevice != 0 {
			return "", nil
		}
	}
	data, err := io.ReadAll(in)
	if err != nil {
		return "", fmt.Errorf("reading the prompt: %w", err)
	}
	return strings.TrimSpace(string(data)), nil
}

// printing writes the answer's text to w as it arrives. Only text: a caller
// piping cai into something else asked for the answer.
func printing(w io.Writer) *commonai.StreamEvents {
	return &commonai.StreamEvents{
		OnText: func(delta string) error {
			_, err := io.WriteString(w, delta)
			return err
		},
	}
}

// report ends a call: a newline if the answer did not end with one, and the
// failure if there was one -- after whatever text arrived, because output the
// caller already saw is theirs to keep.
func report(cmd *cobra.Command, comp *client.Completion, callErr error) error {
	text := answerText(comp)
	if text != "" && !strings.HasSuffix(text, "\n") {
		_, _ = io.WriteString(cmd.OutOrStdout(), "\n")
	}
	if callErr == nil {
		return nil
	}
	if text != "" {
		return fmt.Errorf("the answer was cut off: %w", callErr)
	}
	return callErr
}
