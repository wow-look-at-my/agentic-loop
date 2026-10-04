package loop

import (
	"runtime"
	"testing"

	"github.com/wow-look-at-my/go-containers/event"
)

// keep subscribes fn and holds it until the test ends.
func keep[T event.EventArgs](t *testing.T, e *event.Event[T], fn func(T) error) {
	t.Helper()
	e.Subscribe(&fn)
	t.Cleanup(func() { runtime.KeepAlive(&fn) })
}
