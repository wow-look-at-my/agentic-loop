package loop

import "context"

// runTurn runs one session turn over req.Messages: the shape most loop tests need.
func runTurn(ctx context.Context, cfg Config, req Request) (*Result, error) {
	if cfg.Events == nil {
		cfg.Events = &Events{}
	}
	s := newSession(cfg.Events)
	for _, m := range req.Messages {
		s.appendFlat(m)
	}
	return runOn(ctx, cfg, req, s)
}
