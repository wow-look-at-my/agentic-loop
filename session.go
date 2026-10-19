package agentic

import "github.com/wow-look-at-my/agentic-loop/internal/loop"

// The session API: a conversation owned as a tree. The node types live in the core package.
type (
	ISession       = loop.ISession
	ITurn          = loop.ITurn
	IFuture[T any] = loop.IFuture[T]
	IMessage       = loop.IMessage
	SessionConfig  = loop.SessionConfig
	SessionState   = loop.SessionState
	NodeEvent      = loop.NodeEvent
	Severity       = loop.Severity
)

// NewSession is an empty conversation.
var NewSession = loop.NewSession

// ErrSessionBusy and ErrSessionClosed are the errors of a Send the session cannot take.
var (
	ErrSessionBusy   = loop.ErrSessionBusy
	ErrSessionClosed = loop.ErrSessionClosed
)

// SeverityNotice marks a one-off state loss; SeverityChronic one that repeats turn after turn.
const (
	SeverityNotice  = loop.SeverityNotice
	SeverityChronic = loop.SeverityChronic
)
