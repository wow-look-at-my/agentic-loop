package loop

import (
	"strings"
	"time"
)

// ElapsedKind marks the elapsed-time notice, for a host reading the per-call Request.
const ElapsedKind = "elapsed_time"

// The notice wraps the rendered gap; the header keeps the model from reading it as the user.
const (
	elapsedNoticeHeader = "[automated notice -- time since the previous request in this conversation: "
	elapsedNoticeTail   = "; this is not a message from the user]"
	elapsedInstant      = "less than a second"
)

// ElapsedTime, on Config, states the gap since the previous request on EVERY
// model call. The notice rides that request only: a stored one lies on replay.
type ElapsedTime struct {
	// Since is when the previous request was made; zero says nothing on the run's first call.
	Since time.Time

	// Now is the clock; nil is time.Now.
	Now func() time.Time
}

// elapsedTracker is one run's live view of ElapsedTime: when the last call was made.
type elapsedTracker struct {
	now  func() time.Time
	prev time.Time
}

// newElapsedTracker builds the tracker for a run; a nil setting yields a nil tracker.
func newElapsedTracker(e *ElapsedTime) *elapsedTracker {
	if e == nil {
		return nil
	}
	now := e.Now
	if now == nil {
		now = time.Now
	}
	return &elapsedTracker{now: now, prev: e.Since}
}

// mark stamps this call and returns its notice, or "" with no previous call to
// measure from. A clock that went backwards reports zero, never a negative age.
func (t *elapsedTracker) mark() string {
	if t == nil {
		return ""
	}
	at := t.now()
	prev := t.prev
	t.prev = at
	if prev.IsZero() {
		return ""
	}
	d := at.Sub(prev)
	if d < 0 {
		d = 0
	}
	return FormatElapsedNotice(d)
}

// FormatElapsedNotice renders one gap as the text the model reads.
func FormatElapsedNotice(d time.Duration) string {
	return elapsedNoticeHeader + FormatElapsed(d) + elapsedNoticeTail
}

// elapsedUnits are the units a gap is rendered in, largest first.
var elapsedUnits = []struct {
	size time.Duration
	one  string
	many string
}{
	{24 * time.Hour, "day", "days"},
	{time.Hour, "hour", "hours"},
	{time.Minute, "minute", "minutes"},
	{time.Second, "second", "seconds"},
}

// FormatElapsed renders a gap as its two largest non-zero units, e.g. "2 hours
// 14 minutes". Under a second is named rather than rounded to "0 seconds",
// which reads as a broken clock.
func FormatElapsed(d time.Duration) string {
	if d < time.Second {
		return elapsedInstant
	}
	var parts []string
	for _, u := range elapsedUnits {
		n := int64(d / u.size)
		if n == 0 {
			continue
		}
		d -= time.Duration(n) * u.size
		parts = append(parts, plural(int(n), u.one, u.many))
		if len(parts) == 2 {
			break
		}
	}
	return strings.Join(parts, " ")
}

// elapsedMessages returns msgs plus this call's notice, leaving msgs untouched.
func elapsedMessages(msgs []Message, note string) []Message {
	if note == "" {
		return msgs
	}
	out := make([]Message, len(msgs), len(msgs)+1)
	copy(out, msgs)
	return append(out, Message{Role: RoleUser, Kind: ElapsedKind, Content: note})
}
