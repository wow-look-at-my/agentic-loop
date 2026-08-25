package loop

// shouldCompact auto-compacts when prompt tokens reach AutoCompact * ContextWindow.
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
