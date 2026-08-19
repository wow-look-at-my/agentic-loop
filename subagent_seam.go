package agentic

import "context"

// SubagentReports is the seam an asynchronous sub-agent registry satisfies so
// the loop can wait for in-flight reports without importing that package.
type SubagentReports interface {
	Pending() int
	Delivery(ctx context.Context) (string, error)
}
