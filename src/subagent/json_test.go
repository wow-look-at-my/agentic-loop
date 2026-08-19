package subagent

import (
	"github.com/wow-look-at-my/agentic-loop/src/internal/testkit"
)

var jsonMust = testkit.Must

type jsonObj = testkit.Obj
type jsonArr = testkit.Arr
type scriptProvider = testkit.ScriptProvider
type scriptStep = testkit.ScriptStep
type fakeExec = testkit.FakeExec

var assistantComp = testkit.AssistantComp
