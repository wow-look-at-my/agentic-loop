package loop

// shouldCompact reports whether the loop should auto-compact the transcript
// after the turn that just completed. The trigger is:
//
//   - Request.AutoCompact is positive (0 disables the feature),
//   - Config.ContextWindow is positive (without a window the fraction has
//     no denominator), and
//   - the turn's prompt tokens have reached AutoCompact * ContextWindow.
//
// Prompt tokens are the full prompt the provider received — system, transcript,
// and cached tokens included — so the fraction is against the real context
// size, not a partial estimate. A completion that reported no usage (a local
// server that omits usage) never triggers, which is correct: without a token
// count the loop cannot know the context is full, and compacting blindly would
// discard history on a server that may have plenty of room.
func shouldCompact(req Request, cfg Config, comp *Completion) bool {
	if req.AutoCompact <= 0 || cfg.ContextWindow <= 0 || comp == nil {
		return false
	}
	if !comp.UsageReported {
		return false
	}
	threshold := int(req.AutoCompact * float64(cfg.ContextWindow))
	return comp.Usage.PromptTokens >= threshold
}
