package loop

import (
	"strconv"
	"strings"
	"time"
)

// ElapsedKind marks the time notice, for a host reading the per-call Request.
const ElapsedKind = "elapsed_time"

// The notice's shape: the wall clock, then the gap when there is to state.
const (
	elapsedNoticeHead = "Current time is "
	elapsedNoticeTail = " have passed"
	elapsedTimeLayout = "3:04 PM on 1/2/2006"
	elapsedInstant    = "<1sec"
)

// ElapsedTime, on Config, states the clock and the gap since the previous
// request on EVERY model call. The notice rides that request: a stored lies.
type ElapsedTime struct {
	// Since is when the previous request was made; states the time alone.
	Since time.Time

	// Now is the clock; nil is time.Now.
	Now func() time.Time
}

// elapsedTracker is run's live view of ElapsedTime: when the last call was made.
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

// mark stamps this call and returns its notice.
func (t *elapsedTracker) mark() string {
	if t == nil {
		return ""
	}
	at := t.now()
	prev := t.prev
	t.prev = at
	return FormatElapsedNotice(at, prev)
}

// FormatElapsedNotice renders the notice the model reads: the wall clock in
// now's own zone, and the gap when there is a previous request to measure from.
// A since, or a clock that went backwards, states the time alone.
func FormatElapsedNotice(now, since time.Time) string {
	head := elapsedNoticeHead + now.Format(elapsedTimeLayout)
	if since.IsZero() || !now.After(since) {
		return head
	}
	return head + ", " + FormatElapsed(now.Sub(since)) + elapsedNoticeTail
}

// elapsedUnits are the units a gap is rendered in, largest.
var elapsedUnits = []struct {
	size time.Duration
	one  string
	many string
}{
	{24 * time.Hour, "d", "d"},
	{time.Hour, "hr", "hrs"},
	{time.Minute, "min", "mins"},
	{time.Second, "sec", "secs"},
}

// FormatElapsed renders a gap as its largest non- units, e.g. "1d 23hrs".
// Under a reads "<1sec" rather than "0secs", which looks like a stopped clock.
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
		name := u.many
		if n == 1 {
			name = u.one
		}
		parts = append(parts, strconv.FormatInt(n, 10)+name)
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
