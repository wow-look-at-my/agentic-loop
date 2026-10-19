package loop

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// ErrContextWindowConfig ends a run whose context-window settings contradict each other or the server.
var ErrContextWindowConfig = errors.New("agentic: context window settings")

var errNoWindowList = errors.New("agentic: the provider has no model list to read a context window from")

// checkWindowConfig refuses settings that can never be valid, before any call.
func checkWindowConfig(cfg Config) error {
	switch {
	case cfg.ContextWindowFallback < 0 || cfg.ContextWindowOverride < 0:
		return fmt.Errorf("%w: ContextWindowFallback and ContextWindowOverride must not be negative", ErrContextWindowConfig)
	case cfg.ContextWindowFallback > 0 && cfg.ContextWindowOverride > 0:
		return fmt.Errorf("%w: ContextWindowFallback and ContextWindowOverride are both set. The fallback is "+
			"for a server that reports no window. The override caps a window the server reports. Set one",
			ErrContextWindowConfig)
	}
	return nil
}

// reportedWindow asks the provider's model list for model's window. Zero comes with the reason.
func reportedWindow(ctx context.Context, p Provider, model string) (int, error) {
	l, ok, err := ModelLimitsOf(ctx, p, model)
	switch {
	case !ok:
		return 0, errNoWindowList
	case err != nil:
		return 0, fmt.Errorf("agentic: reading the context window for %q: %w", model, err)
	case l.ContextWindow <= 0:
		return 0, fmt.Errorf("agentic: the model list publishes no context window for %q", model)
	}
	return l.ContextWindow, nil
}

// applyWindow sets the run's window from what the server reported and both host settings. A
// fallback never outranks a reported window. An override may only shrink one.
func applyWindow(cfg *Config, reported int, reportErr error, served string) error {
	ev := ContextWindowEvent{Served: served, Reported: reported}
	switch {
	case reported > 0 && cfg.ContextWindowOverride > reported:
		return fmt.Errorf("%w: ContextWindowOverride is %d, larger than the %d tokens the server reports for %q",
			ErrContextWindowConfig, cfg.ContextWindowOverride, reported, windowModel(served))
	case reported > 0 && cfg.ContextWindowOverride > 0:
		ev.Window, ev.Overridden = cfg.ContextWindowOverride, true
	case reported > 0:
		ev.Window, ev.FromModelList = reported, true
	case cfg.ContextWindowOverride > 0:
		ev.Window, ev.Overridden = cfg.ContextWindowOverride, true
		ev.Warning = fmt.Sprintf("the server reports no context window (%v), so ContextWindowOverride "+
			"is used as the window and nothing checks it", reportErr)
	case cfg.ContextWindowFallback > 0:
		ev.Window, ev.FromFallback = cfg.ContextWindowFallback, true
		ev.Warning = fmt.Sprintf("the server reports no context window (%v), so ContextWindowFallback is used", reportErr)
	default:
		ev.Err = reportErr
	}
	cfg.window, cfg.windowReported = ev.Window, reported > 0
	cfg.Events.emitContextWindow(ev)
	return nil
}

func windowModel(served string) string {
	if served == "" {
		return "the requested model"
	}
	return served
}

// ErrContextWindowExceeded ends a run whose next request would carry more than the run's window.
var ErrContextWindowExceeded = errors.New("agentic: the next request exceeds the context window")

// estimateCharsPerToken over-counts the tokens a transcript gains, so the cap compacts early rather than late.
const estimateCharsPerToken = 3

// nextPromptEstimate is the server's last prompt count plus the transcript the run appended since that call.
// Zero means no evidence: no previous call, or one that reported no usage.
func nextPromptEstimate(last *Completion, appended []Message) int {
	if last == nil || !last.UsageReported {
		return 0
	}
	chars := 0
	for _, m := range appended {
		chars += outputChars(m)
	}
	return last.Usage.PromptTokens + chars/estimateCharsPerToken
}

// capToWindow keeps the next request within the run's window. It compacts when auto-compaction is armed.
// Otherwise, or when compaction fails, the run ends rather than send more than the window.
func capToWindow(ctx context.Context, cfg *Config, req Request, transcript []Message, last *Completion,
	sent int, res *Result, deduper *OutputDeduper) ([]Message, error) {
	if cfg.window <= 0 || sent > len(transcript) {
		return nil, nil
	}
	est := nextPromptEstimate(last, transcript[sent:])
	if est < cfg.window {
		return nil, nil
	}
	if req.AutoCompact > 0 {
		if next, ok := compactNow(ctx, cfg, req, transcript, res, deduper); ok {
			return next, nil
		}
	}
	return nil, fmt.Errorf("%w: the next request is about %d tokens, and the window is %d",
		ErrContextWindowExceeded, est, cfg.window)
}

// wantsWindow reports whether the run needs a window: to compact, or to cap a request by a host setting.
func wantsWindow(cfg *Config, req Request) bool {
	return req.AutoCompact > 0 || cfg.ContextWindowOverride > 0 || cfg.ContextWindowFallback > 0
}

// resolveContextWindow sets the run's window before the first call. A run that needs none asks nothing.
func resolveContextWindow(ctx context.Context, cfg *Config, req Request) error {
	if !wantsWindow(cfg, req) {
		return nil
	}
	reported, err := reportedWindow(ctx, cfg.Provider, req.Model)
	return applyWindow(cfg, reported, err, "")
}

// resolveServedWindow asks again by the model a response names, while no server has reported a window.
// An endpoint that serves an alias lists the model it served, not the alias. The list id a gateway routed to
// is asked before the body's model name, and each set of names is asked once.
func resolveServedWindow(ctx context.Context, cfg *Config, req Request, comp *Completion, asked *string) error {
	if !wantsWindow(cfg, req) || cfg.windowReported || comp == nil {
		return nil
	}
	names := servedNames(req.Model, comp)
	key := strings.Join(names, "\n")
	if len(names) == 0 || key == *asked {
		return nil
	}
	*asked = key
	var errs []error
	for _, served := range names {
		reported, err := reportedWindow(ctx, cfg.Provider, served)
		if reported > 0 {
			return applyWindow(cfg, reported, nil, served)
		}
		if errors.Is(err, errNoWindowList) {
			return nil
		}
		errs = append(errs, err)
	}
	// Nothing new is known: keep the window the run has.
	if cfg.window == 0 {
		cfg.Events.emitContextWindow(ContextWindowEvent{Served: names[0], Err: errors.Join(errs...)})
	}
	return nil
}

// servedNames is every name a response gives the model that answered, most exact first, without the requested name.
func servedNames(requested string, comp *Completion) []string {
	var names []string
	for _, name := range []string{comp.Routed, comp.Model} {
		if name != "" && name != requested && !slices.Contains(names, name) {
			names = append(names, name)
		}
	}
	return names
}
