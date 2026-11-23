package arc

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type step func(context.Context, ModelRequest, func(ModelDelta) error) (ModelReply, error)

// scriptedProvider serves one response per request and saves the requests so
// tests can inspect the context seen by each successive turn.
type scriptedProvider struct {
	steps    []step
	requests []ModelRequest
}

// Stream consumes the next response and reports unexpected extra calls.
func (p *scriptedProvider) Stream(ctx context.Context, request ModelRequest, emit func(ModelDelta) error) (ModelReply, error) {
	p.requests = append(p.requests, request)
	if len(p.steps) == 0 {
		return ModelReply{}, errors.New("unexpected provider request")
	}
	next := p.steps[0]
	p.steps = p.steps[1:]
	return next(ctx, request, emit)
}

func response(text, reason string, calls ...ToolCall) step {
	return func(_ context.Context, _ ModelRequest, emit func(ModelDelta) error) (ModelReply, error) {
		if text != "" {
			if err := emit(ModelDelta{Text: text}); err != nil {
				return ModelReply{}, err
			}
		}
		return ModelReply{StopReason: reason, Calls: calls}, nil
	}
}

// testAgent assembles a real agent with a temporary workspace and in-memory history.
func testAgent(t *testing.T, p Provider, tools []Tool) *Agent {
	t.Helper()
	cwd := t.TempDir()
	session, err := CreateSession(cwd, "")
	if err != nil {
		t.Fatal(err)
	}
	registry, err := NewRegistry(tools)
	if err != nil {
		t.Fatal(err)
	}
	return &Agent{Provider: p, Session: session, Tools: registry, Environment: ToolEnv{CWD: cwd}, MaxTurns: 20,
		Policy: Confirm, SystemPrompt: func() string { return "test system" }}
}

func TestAgentOrderedToolsAndFollowUp(t *testing.T) {
	write := ToolCall{"a", "write", Object{"path": "note", "content": "Saved"}}
	read := ToolCall{"b", "read", Object{"path": "note"}}
	provider := &scriptedProvider{steps: []step{response("", "tool_use", write, read), response("ready", "stop"), response("answered", "stop")}}
	agent := testAgent(t, provider, BuiltinTools())
	var events []string
	status, err := agent.Run(t.Context(), "create a note", func(event Event) error {
		events = append(events, event.Type)
		// Queue both kinds of input during a tool batch to exercise their
		// different insertion points.
		if event.Type == "tool_execution_start" && event.Data["tool_call"].(ToolCall).ID == "a" {
			agent.Steer("next round")
			agent.FollowUp("question afterward")
		}
		return nil
	})
	if err != nil || status != "complete" {
		t.Fatalf("%s: %v", status, err)
	}
	round := provider.requests[1].Messages
	if got := round[len(round)-3:]; got[0].ToolCallID == nil || *got[0].ToolCallID != "a" ||
		got[1].ToolCallID == nil || *got[1].ToolCallID != "b" || got[1].Content != "1: Saved" || got[2].Content != "next round" {
		t.Fatalf("unexpected second round: %#v", got)
	}
	round = provider.requests[2].Messages
	if round[len(round)-1].Content != "question afterward" || events[0] != "agent_start" || events[len(events)-1] != "agent_end" {
		t.Fatalf("follow-up or lifecycle missing: %#v %v", round, events)
	}
}

func TestAgentGatesEffectsAndDoesNotRunMalformedCalls(t *testing.T) {
	makeWrite := func(id string) ToolCall { return ToolCall{id, "write", Object{"path": id, "content": "test"}} }
	tests := []struct {
		name, stop, policy, authorization string
		calls                             []ToolCall
		approve                           bool
		wantsFile                         bool
	}{
		{"headless shell", "tool_use", Confirm, "confirmation_required", []ToolCall{{"bash", "bash", Object{"command": "touch bash"}}}, false, false},
		{"approved shell", "tool_use", Confirm, "allowed", []ToolCall{{"bash", "bash", Object{"command": "touch bash"}}}, true, true},
		{"invalid schema", "tool_use", Confirm, "blocked", []ToolCall{{"bad", "write", Object{"path": "bad", "content": 3}}}, false, false},
		{"incomplete response", "length", Autonomous, "blocked", []ToolCall{makeWrite("length")}, false, false},
		{"duplicate ID", "tool_use", Autonomous, "", []ToolCall{makeWrite("duplicate"), makeWrite("duplicate")}, false, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			provider := &scriptedProvider{steps: []step{response("", test.stop, test.calls...), response("done", "stop")}}
			agent := testAgent(t, provider, BuiltinTools())
			agent.Policy = test.policy
			if test.approve {
				agent.Approve = func(_ context.Context, request Authorization) (bool, error) {
					if request.CWD != agent.Environment.CWD || request.Effects[0] != "process" {
						t.Fatalf("approval: %+v", request)
					}
					return true, nil
				}
			}
			var auth string
			status, err := agent.Run(t.Context(), "execute", func(event Event) error {
				if event.Type == "tool_execution_authorization" {
					auth = event.Data["status"].(string)
				}
				return nil
			})
			if (test.authorization == "") != (status == "error") || (status == "error") != (err != nil) || auth != test.authorization {
				t.Fatalf("status=%s err=%v authorization=%s", status, err, auth)
			}
			_, err = os.Stat(filepath.Join(agent.Environment.CWD, test.calls[0].ID))
			if (err == nil) != test.wantsFile {
				t.Fatalf("side effect: %v", err)
			}
		})
	}
}

func TestAgentSavesCompletedToolBeforeObserverError(t *testing.T) {
	call := ToolCall{"a", "write", Object{"path": "note", "content": "saved"}}
	agent := testAgent(t, &scriptedProvider{steps: []step{response("", "tool_use", call)}}, BuiltinTools())
	uiErr := errors.New("renderer exited")
	_, err := agent.Run(t.Context(), "write", func(event Event) error {
		// Fail after the file write to exercise the persistence-before-delivery contract.
		if event.Type == "tool_execution_end" {
			return uiErr
		}
		return nil
	})
	if !errors.Is(err, uiErr) {
		t.Fatal(err)
	}
	messages := agent.Session.Messages()
	if last := messages[len(messages)-1]; last.Role != "tool" || last.IsError || !strings.Contains(last.Content, "Wrote") {
		t.Fatalf("result lost after render error: %#v", last)
	}
}

func TestAgentCancellationPairsPendingCallsAndClosesTurn(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	calls := []ToolCall{{"first", "bash", Object{"command": "sleep 3"}}, {"second", "write", Object{"path": "bad", "content": "bad"}}}
	agent := testAgent(t, &scriptedProvider{steps: []step{response("", "tool_use", calls...)}}, BuiltinTools())
	agent.Approve = func(ctx context.Context, _ Authorization) (bool, error) {
		cancel()
		return false, ctx.Err()
	}
	var events []string
	status, _ := agent.Run(ctx, "cancel this", func(event Event) error { events = append(events, event.Type); return nil })
	if status != "aborted" || !reflect.DeepEqual(events[len(events)-2:], []string{"turn_end", "agent_end"}) {
		t.Fatalf("unexpected cancellation: %s %v", status, events)
	}
	messages := agent.Session.Messages()
	for i, call := range calls {
		if result := messages[2+i]; !result.IsError || *result.ToolCallID != call.ID || !strings.Contains(result.Content, "Interrupted") {
			t.Fatalf("unpaired interrupted result: %+v", result)
		}
	}
}

func TestAgentActiveContextBudgetAndCheckpoint(t *testing.T) {
	for _, failSummary := range []bool{false, true} {
		t.Run(map[bool]string{false: "checkpoint", true: "fallback"}[failSummary], func(t *testing.T) {
			summary := response("short summary", "stop")
			if failSummary {
				summary = func(context.Context, ModelRequest, func(ModelDelta) error) (ModelReply, error) {
					return ModelReply{}, errors.New("summary offline")
				}
			}
			provider := &scriptedProvider{steps: []step{summary, response("complete", "stop")}}
			agent := testAgent(t, provider, nil)
			agent.ContextBudget, agent.AutoCompact = 1024, true
			old := strings.Repeat("早期内容", 500)
			branch, err := agent.Session.Append(Message{Role: "user", Content: old})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := agent.Session.Append(Message{Role: "user", Content: "recent task"}); err != nil {
				t.Fatal(err)
			}
			status, err := agent.Run(t.Context(), "current task", nil)
			if err != nil || status != "complete" {
				t.Fatalf("%s: %v", status, err)
			}
			context := provider.requests[1].Messages
			if messagesCost(context) > 1024 || context[len(context)-1].Content != "current task" {
				t.Fatal("request did not fit budget")
			}
			if hasSummary := agent.Session.Messages()[0].Role == "summary"; hasSummary == failSummary {
				t.Fatalf("unexpected checkpoint state: %v", hasSummary)
			}
			if err := agent.Session.Branch(branch); err != nil {
				t.Fatal(err)
			}
			if agent.Session.Messages()[0].Content != old {
				t.Fatal("original history changed")
			}
		})
	}
}

func TestFitContextKeepsPairedToolResults(t *testing.T) {
	calls := []ToolCall{{"a", "read", Object{"path": "a"}}, {"b", "read", Object{"path": "b"}}}
	content := strings.Repeat("工具输出", 400) + " exit code 1"
	messages := []Message{{Role: "user", Content: "old task"}, {Role: "user", Content: "current task"},
		{Role: "assistant", ToolCalls: calls}, toolMessage(calls[0], content, false), toolMessage(calls[1], content, false)}
	fitted, err := fitContext(messages, 500)
	if err != nil {
		t.Fatal(err)
	}
	if len(fitted) != 4 || messagesCost(fitted) > 500 || *fitted[2].ToolCallID != "a" || *fitted[3].ToolCallID != "b" ||
		!strings.Contains(fitted[3].Content, "exit code 1") || messages[4].Content != content {
		t.Fatalf("unexpected projection (%d tokens)", messagesCost(fitted))
	}
}
