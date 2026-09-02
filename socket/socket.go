// Package socket serves the common AI API over a unix socket and over a
// websocket.
//
// Both carry the same documents as every other way in. A document is
// self-delimiting -- it ends when its root element closes -- so the unix
// socket needs no framing layer at all: documents ride back-to-back in both
// directions, and a reader knows where each ends because XML already says
// so. The websocket has framing whether we want it or not, so each flush of
// the answer is text message, and a client concatenates them into the same
// document a unix-socket client reads byte for byte.
//
// operations, told apart by the root element the client sends:
//
// - <request> runs call and answers with a <response>.
// - <conversation id="..."> appends its messages to the stored conversation
// of that id (creating it when the id is new), runs the call over the whole
// transcript, and answers with a <response>. The assistant's turn is
// appended, so the next sees it.
//
// A call that fails before it produced anything answers with an <error>
// document; that fails after says both, in the <response> it had
// already started.
package socket

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/session"
)

// Config builds a Server.
type Config struct {
	// Provider runs the calls and is required; it is the format's own Provider, not the Go client's.
	Provider commonai.Provider
	// Store holds conversations; nil serves <request> only and errors on <conversation>.
	Store session.Store
}

// Server answers documents on a connection.
type Server struct {
	provider commonai.Provider
	store    session.Store
}

// NewServer checks the configuration up front rather than failing per
// connection later.
func NewServer(cfg Config) (*Server, error) {
	if cfg.Provider == nil {
		return nil, commonai.BadRequest("socket: a Provider is required")
	}
	return &Server{provider: cfg.Provider, store: cfg.Store}, nil
}

// Serve accepts connections until l is closed, handling each in its own
// goroutine.
func (s *Server) Serve(ctx context.Context, l net.Listener) error {
	for {
		conn, err := l.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) || ctx.Err() != nil {
				return nil
			}
			return err
		}
		go func() {
			defer conn.Close()
			s.Handle(ctx, conn)
		}()
	}
}

// Handle reads documents from conn and answers each, until the peer stops
// sending or the context ends.
func (s *Server) Handle(ctx context.Context, conn io.ReadWriter) {
	r := bufio.NewReader(conn)
	for {
		if ctx.Err() != nil {
			return
		}
		data, err := commonai.ReadDocument(r)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				// A stream that ended mid-document is worth saying out loud.
				_ = commonai.EncodeError(conn, commonai.BadRequest(err.Error()))
			}
			return
		}
		if !s.answer(ctx, conn, data) {
			return
		}
	}
}

// answer handles document, reporting whether the connection is still good
// for another.
func (s *Server) answer(ctx context.Context, w io.Writer, data []byte) bool {
	if err := commonai.Validate(data); err != nil {
		return commonai.EncodeError(w, commonai.BadRequest(err.Error())) == nil
	}
	req, id, err := s.decode(data)
	if err != nil {
		return commonai.EncodeError(w, err) == nil
	}

	stream := newStreamWriter(w)
	comp, callErr := s.provider.Complete(ctx, req, stream.events())
	if !stream.started() {
		if callErr != nil {
			return commonai.EncodeError(w, callErr) == nil
		}
		stream.begin(comp.Message.Role)
	}
	// Record BEFORE closing the document: a caller told the turn is done reads
	// the conversation next, and must not find the answer missing.
	if id != "" && comp != nil {
		_, _ = s.store.Append(id, comp.Message)
	}
	stream.finish(comp, callErr)
	return true
}

// decode turns an inbound document into the call to make, and the conversation
// id to append the answer to (empty for a stateless call).
func (s *Server) decode(data []byte) (commonai.Request, string, error) {
	if req, err := commonai.DecodeRequest(data); err == nil {
		return req, "", nil
	}
	id, turn, err := commonai.DecodeConversation(data)
	if err != nil {
		return commonai.Request{}, "", commonai.BadRequest(err.Error())
	}
	if s.store == nil {
		return commonai.Request{}, "", commonai.BadRequest("socket: this server keeps no conversations")
	}
	stored, err := s.store.Append(id, turn.Messages...)
	if err != nil {
		if !errors.Is(err, session.ErrNotFound) {
			return commonai.Request{}, "", err
		}
		// An id nobody has used yet names the conversation this turn starts.
		if err := s.store.Put(id, turn); err != nil {
			return commonai.Request{}, "", err
		}
		stored = turn
	}
	return overlay(stored, turn), id, nil
}

// overlay applies a turn's own fields over the conversation's defaults.
func overlay(stored, turn commonai.Request) commonai.Request {
	out := stored
	if turn.Model != "" {
		out.Model = turn.Model
	}
	if turn.System != "" || len(turn.SystemParts) > 0 {
		out.System, out.SystemParts = turn.System, turn.SystemParts
	}
	if turn.MaxTokens > 0 {
		out.MaxTokens = turn.MaxTokens
	}
	if turn.CacheKey != "" {
		out.CacheKey = turn.CacheKey
	}
	if len(turn.Tools) > 0 {
		out.Tools = turn.Tools
	}
	if len(turn.Extra) > 0 {
		out.Extra = turn.Extra
	}
	if len(turn.DialectExtra) > 0 {
		out.DialectExtra = turn.DialectExtra
	}
	return out
}
