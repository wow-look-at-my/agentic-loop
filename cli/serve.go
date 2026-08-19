package cli

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/wow-look-at-my/agentic-loop/client"
	httpapi "github.com/wow-look-at-my/agentic-loop/http"
	"github.com/wow-look-at-my/agentic-loop/socket"
)

var (
	serveHTTP   string
	serveSocket string
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Serve the API over HTTP and a unix socket",
	Long: "Serve the API to other programs on this machine.\n\n" +
		"The endpoint and credentials are this process's -- a client of this " +
		"server never sends them, and never chooses them.",
	Args: cobra.NoArgs,
	RunE: runServe,
}

func init() {
	serveCmd.Flags().StringVar(&serveHTTP, "http", "", "serve HTTP on this address (e.g. :8080)")
	serveCmd.Flags().StringVar(&serveSocket, "socket", "", "serve on this unix socket path")
	root.AddCommand(serveCmd)
}

func runServe(cmd *cobra.Command, _ []string) error {
	if serveHTTP == "" && serveSocket == "" {
		return fmt.Errorf("nothing to serve on: pass --http, --socket, or both")
	}
	p, err := newProvider(cmd)
	if err != nil {
		return err
	}
	st, err := store()
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errs := make(chan error, 2)
	started := 0
	if serveSocket != "" {
		l, err := listenUnix(serveSocket)
		if err != nil {
			return err
		}
		defer l.Close()
		srv, err := socket.NewServer(socket.Config{Provider: client.Unwrap(p), Store: st})
		if err != nil {
			return err
		}
		fmt.Fprintf(cmd.ErrOrStderr(), "cai: serving on %s\n", serveSocket)
		started++
		go func() { errs <- srv.Serve(ctx, l) }()
	}
	if serveHTTP != "" {
		api, err := httpapi.NewServer(httpapi.Config{Provider: client.Unwrap(p), Store: st})
		if err != nil {
			return err
		}
		ws, err := socket.NewServer(socket.Config{Provider: client.Unwrap(p), Store: st})
		if err != nil {
			return err
		}
		mux := http.NewServeMux()
		mux.Handle("/v1/ws", ws.WebSocketHandler())
		mux.Handle("/", api)
		hs := &http.Server{Addr: serveHTTP, Handler: mux}
		fmt.Fprintf(cmd.ErrOrStderr(), "cai: serving on %s\n", serveHTTP)
		started++
		go func() {
			err := hs.ListenAndServe()
			if errors.Is(err, http.ErrServerClosed) {
				err = nil
			}
			errs <- err
		}()
		go func() {
			<-ctx.Done()
			_ = hs.Shutdown(context.Background())
		}()
	}

	for range started {
		select {
		case err := <-errs:
			if err != nil {
				return err
			}
		case <-ctx.Done():
			return nil
		}
	}
	return nil
}

// listenUnix binds a unix socket, refusing to remove something that is already
// there: a path in use is either another server or a file somebody wants, and
// unlinking it silently is how one of those disappears.
func listenUnix(path string) (net.Listener, error) {
	if _, err := os.Stat(path); err == nil {
		return nil, fmt.Errorf("%s already exists: another cai may be serving there", path)
	}
	l, err := net.Listen("unix", path)
	if err != nil {
		return nil, fmt.Errorf("listening on %s: %w", path, err)
	}
	return l, nil
}
