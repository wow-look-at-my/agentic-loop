// Package httpapi serves the common AI API over HTTP: one stateless endpoint
// that runs a call, and a set of stateful ones that keep the conversation.
//
// There is no envelope and no second vocabulary. A request IS a <request>
// document, an answer IS the <response> document, and a streamed answer is
// that same document written progressively down a chunked body -- so `curl`
// shows the answer appearing, and a consumer in any language parses it with
// the XML parser it already has.
package httpapi

import (
	"errors"
	"io"
	"net/http"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/session"
)

// maxBody caps an inbound document. A model call's transcript is large but
// bounded; anything past this is not a conversation.
const maxBody = 32 << 20

// contentType is what every document is served and accepted as.
const contentType = "application/xml; charset=utf-8"

// Config builds a Server.
type Config struct {
	// Provider runs the calls and is required.
	//
	// It is the format's own Provider, not the Go client's: what the document
	// says the provider reported has to be what the provider reported, and a
	// server that folded the usage reports on the way out would be answering
	// with its own arithmetic instead.
	Provider commonai.Provider
	// Store holds conversations for the stateful endpoints. Nil serves the
	// stateless endpoint only, and answers the rest with 501.
	Store session.Store
}

// Server is the HTTP handler set. Build it with NewServer and mount it
// wherever it belongs; it is an http.Handler.
type Server struct {
	provider commonai.Provider
	store    session.Store
	mux      *http.ServeMux
}

// NewServer wires the routes. It fails on a missing Provider rather than
// serving 500s later.
func NewServer(cfg Config) (*Server, error) {
	if cfg.Provider == nil {
		return nil, commonai.BadRequest("httpapi: a Provider is required")
	}
	s := &Server{provider: cfg.Provider, store: cfg.Store, mux: http.NewServeMux()}
	s.mux.HandleFunc("POST /v1/complete", s.handleComplete)
	s.mux.HandleFunc("POST /v1/conversations", s.handleCreate)
	s.mux.HandleFunc("GET /v1/conversations", s.handleList)
	s.mux.HandleFunc("GET /v1/conversations/{id}", s.handleGet)
	s.mux.HandleFunc("DELETE /v1/conversations/{id}", s.handleDelete)
	s.mux.HandleFunc("POST /v1/conversations/{id}/turns", s.handleTurn)
	return s, nil
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

// handleComplete runs one call and streams the response document back.
func (s *Server) handleComplete(w http.ResponseWriter, r *http.Request) {
	req, err := s.readRequest(r)
	if err != nil {
		writeError(w, err)
		return
	}
	s.run(w, r, req, nil, "")
}

// handleCreate stores a new conversation and answers with the document that
// now stands for it, id included.
func (s *Server) handleCreate(w http.ResponseWriter, r *http.Request) {
	if !s.haveStore(w) {
		return
	}
	req, err := s.readRequest(r)
	if err != nil {
		writeError(w, err)
		return
	}
	id, err := s.store.Create(req)
	if err != nil {
		writeError(w, err)
		return
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(http.StatusCreated)
	_ = commonai.EncodeConversation(w, id, req)
}

// handleList answers with the ids the store holds.
func (s *Server) handleList(w http.ResponseWriter, _ *http.Request) {
	if !s.haveStore(w) {
		return
	}
	ids, err := s.store.List()
	if err != nil {
		writeError(w, err)
		return
	}
	w.Header().Set("Content-Type", contentType)
	_ = commonai.EncodeConversationIDs(w, ids)
}

// handleGet answers with a stored conversation.
func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	if !s.haveStore(w) {
		return
	}
	id := r.PathValue("id")
	req, err := s.store.Get(id)
	if err != nil {
		writeError(w, err)
		return
	}
	w.Header().Set("Content-Type", contentType)
	_ = commonai.EncodeConversation(w, id, req)
}

// handleDelete forgets a stored conversation.
func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	if !s.haveStore(w) {
		return
	}
	if err := s.store.Delete(r.PathValue("id")); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleTurn appends the posted messages to a stored conversation, runs the
// call over the whole transcript, and appends the answer.
//
// The posted document's other fields override the stored defaults for this
// turn only: a caller raising max-tokens for one question has not changed what
// the conversation is.
func (s *Server) handleTurn(w http.ResponseWriter, r *http.Request) {
	if !s.haveStore(w) {
		return
	}
	id := r.PathValue("id")
	turn, err := s.readRequest(r)
	if err != nil {
		writeError(w, err)
		return
	}
	stored, err := s.store.Append(id, turn.Messages...)
	if err != nil {
		writeError(w, err)
		return
	}
	s.run(w, r, overlay(stored, turn), s.store, id)
}

// run makes the call and writes the response document as it arrives. When a
// store is given, the assistant turn is appended to it once the call is done.
func (s *Server) run(w http.ResponseWriter, r *http.Request, req commonai.Request, store session.Store, id string) {
	stream := newStreamWriter(w)
	comp, err := s.provider.Complete(r.Context(), req, stream.events())

	// Nothing was written yet, so the failure can still be an HTTP status and
	// an <error> document of its own rather than a truncated answer.
	if !stream.started() {
		if err != nil {
			writeError(w, err)
			return
		}
		stream.begin(comp.Message.Role)
	}
	stream.finish(comp, err)

	if store != nil && comp != nil {
		// A turn the caller has already seen belongs in the transcript even
		// when the call failed partway: the next turn has to know what was
		// said, and a caller cannot append it themselves without guessing what
		// arrived.
		_, _ = store.Append(id, comp.Message)
	}
}

// readRequest reads, validates and decodes an inbound <request> document.
func (s *Server) readRequest(r *http.Request) (commonai.Request, error) {
	data, err := io.ReadAll(http.MaxBytesReader(nil, r.Body, maxBody))
	if err != nil {
		return commonai.Request{}, commonai.BadRequest("httpapi: reading the request body: " + err.Error())
	}
	if err := commonai.Validate(data); err != nil {
		return commonai.Request{}, commonai.BadRequest(err.Error())
	}
	req, err := commonai.DecodeRequest(data)
	if err != nil {
		return commonai.Request{}, commonai.BadRequest(err.Error())
	}
	return req, nil
}

// haveStore answers the stateful routes when no store was configured. A
// server without one does not have sessions to serve, and saying so is better
// than an empty list that reads as "you have none".
func (s *Server) haveStore(w http.ResponseWriter) bool {
	if s.store != nil {
		return true
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(http.StatusNotImplemented)
	_ = commonai.EncodeError(w, commonai.BadRequest("httpapi: this server keeps no conversations"))
	return false
}

// overlay applies a turn's own fields over the conversation's defaults. Only
// what the turn actually stated is taken; the transcript is the stored one,
// which already has the turn's messages appended.
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

// writeError answers with an <error> document and the status that goes with
// it. It is only ever used before any of the answer has been written; once the
// response document is open, a failure is appended inside it instead.
func writeError(w http.ResponseWriter, err error) {
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(statusFor(err))
	_ = commonai.EncodeError(w, err)
}

// statusFor maps a failure onto the status a caller should act on.
func statusFor(err error) int {
	var apiErr *commonai.APIError
	switch {
	case errors.Is(err, session.ErrNotFound):
		return http.StatusNotFound
	case errors.As(err, &apiErr):
		// The upstream's own status, when it is one a client can act on.
		if apiErr.Status >= 400 && apiErr.Status < 600 {
			return apiErr.Status
		}
		return http.StatusBadGateway
	case commonai.IsUnsupported(err):
		return http.StatusUnprocessableEntity
	case commonai.IsBadRequest(err):
		return http.StatusBadRequest
	case commonai.ErrorKind(err) == commonai.ErrorKindCanceled:
		// The caller went away, or their deadline did. Nothing here failed,
		// and nothing is owed a body.
		return httpStatusClientClosed
	}
	return http.StatusInternalServerError
}

// httpStatusClientClosed is nginx's 499, which has no net/http constant. It is
// the only status that says "the caller stopped waiting", and answering 500
// would blame the server for the caller's own cancellation.
const httpStatusClientClosed = 499
