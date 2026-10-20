package loop

import "context"

// Run is the one-shot loop: req.Messages seeded into a fresh session, one turn,
// the answer's Result. Hosts that keep a conversation hold an ISession and call
// Send; Run stays for callers that hold no session and want one answer.
func Run(ctx context.Context, cfg Config, req Request) (*Result, error) {
	s, err := NewSession(SessionConfig{Loop: cfg, Request: req})
	if err != nil {
		return nil, err
	}
	turn := s.Send(ctx, req.Messages...)
	if err := turn.Wait(ctx); err != nil {
		return turn.Result(), err
	}
	_ = s.Close(ctx)
	return turn.Result(), nil
}
