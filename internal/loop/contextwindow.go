package loop

import (
	"context"
	"errors"
	"fmt"
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

// resolveContextWindow sets the run's window before the first call. A run that does not auto-compact asks nothing.
func resolveContextWindow(ctx context.Context, cfg *Config, req Request) error {
	if req.AutoCompact <= 0 {
		return nil
	}
	reported, err := reportedWindow(ctx, cfg.Provider, req.Model)
	return applyWindow(cfg, reported, err, "")
}

// resolveServedWindow asks again by the model a response names, while no server has reported a window.
// An endpoint that serves an alias lists the model it served, not the alias. Each served name is asked once.
func resolveServedWindow(ctx context.Context, cfg *Config, req Request, comp *Completion, asked *string) error {
	if req.AutoCompact <= 0 || cfg.windowReported || comp == nil {
		return nil
	}
	served := comp.Model
	if served == "" || served == req.Model || served == *asked {
		return nil
	}
	*asked = served
	reported, err := reportedWindow(ctx, cfg.Provider, served)
	if reported == 0 {
		// Nothing new is known: keep the window the run has, and stay quiet when there is no list to ask.
		if cfg.window > 0 || errors.Is(err, errNoWindowList) {
			return nil
		}
		cfg.Events.emitContextWindow(ContextWindowEvent{Served: served, Err: err})
		return nil
	}
	return applyWindow(cfg, reported, nil, served)
}
