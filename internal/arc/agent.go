package arc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

type Hooks struct {
	BeforeTool func(context.Context, ToolCall, ToolEnv) error
	AfterTool  func(context.Context, ToolCall, ToolResult) (ToolResult, error)
}

// Agent runs one task at a time. Steering and follow-ups may arrive while it runs.
type Agent struct {
	Provider                Provider
	Session                 *Session
	Tools                   *Registry
	Environment             ToolEnv
	SystemPrompt            func() string
	InstructionContext      []Message
	TransformContext        func(context.Context, []Message) ([]Message, error)
	Policy                  string
	Approve                 Approver
	Hooks                   Hooks
	MaxTurns, ContextBudget int
	AutoCompact             bool

	mu                  sync.Mutex
	running             bool
	steering, followups []string
}

func (a *Agent) Steer(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steering = append(a.steering, text)
}

func (a *Agent) FollowUp(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followups = append(a.followups, text)
}

func (a *Agent) takeSteering() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	steering := a.steering
	a.steering = nil
	return steering
}

func (a *Agent) hasNextTurn() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.steering) > 0 {
		return true
	}
	if len(a.followups) == 0 {
		return false
	}
	a.steering = append(a.steering, a.followups[0])
	a.followups = a.followups[1:]
	return true
}

func (a *Agent) begin() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.running {
		return errors.New("Arc already running; use steering or follow-up")
	}
	a.running = true
	return nil
}

func (a *Agent) end() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.running = false
	a.steering, a.followups = nil, nil
}

func (a *Agent) Compact(ctx context.Context, keep int) error {
	if err := a.begin(); err != nil {
		return err
	}
	defer a.end()
	messages, err := compactMessages(ctx, a.Provider, a.Session.Messages(), keep, a.ContextBudget)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return a.Session.Checkpoint(messages)
}

type runState struct {
	agent               *Agent
	observe             Observer
	observerError       error
	turn                int
	pending             []ToolCall
	attemptedCompaction bool
}

func (r *runState) emit(kind string, data Object) error {
	if r.observerError != nil {
		return r.observerError
	}
	if r.observe != nil {
		r.observerError = r.observe(Event{Type: kind, Data: data})
	}
	return r.observerError
}

func (r *runState) append(message Message) error {
	if _, err := r.agent.Session.Append(message); err != nil {
		return err
	}
	return r.emit("message_end", Object{"message": message})
}

// Run keeps each complete result durable before telling the observer. When a
// consumer or tool stops mid-call, deferred cleanup leaves an explicit result.
func (a *Agent) Run(ctx context.Context, prompt string, observe Observer) (status string, err error) {
	if strings.TrimSpace(prompt) == "" {
		return "error", errors.New("prompt cannot be empty")
	}
	if a.MaxTurns < 1 {
		return "error", errors.New("max turns must be positive")
	}
	if err := a.begin(); err != nil {
		return "error", err
	}
	defer a.end()
	run := runState{agent: a, observe: observe}
	defer func() {
		for _, call := range run.pending {
			_, saveErr := a.Session.Append(toolMessage(call,
				"Interrupted. Side effects may have occurred; inspect state before retrying.", true))
			err = errors.Join(err, saveErr)
		}
		if ctx.Err() != nil {
			status = "aborted"
		}
		if status == "" {
			status = "error"
		}
		if run.observerError != nil {
			err = errors.Join(err, run.observerError)
			return
		}
		if run.turn != 0 {
			reason := "error"
			if status == "aborted" {
				reason = "aborted"
			}
			err = errors.Join(err, run.emit("turn_end", Object{"turn": run.turn, "stop_reason": reason}))
		}
		if err != nil && ctx.Err() == nil {
			err = errors.Join(err, run.emit("error", Object{"message": err.Error()}))
		}
		err = errors.Join(err, run.emit("agent_end", Object{"status": status}))
	}()
	if err = run.emit("agent_start", Object{"status": "running"}); err != nil {
		return "error", err
	}
	if err = run.append(Message{Role: "user", Content: prompt}); err != nil {
		return "error", err
	}
	for turn := 1; turn <= a.MaxTurns; turn++ {
		if err := ctx.Err(); err != nil {
			return "aborted", err
		}
		run.turn = turn
		for _, text := range a.takeSteering() {
			if err := run.append(Message{Role: "user", Content: text}); err != nil {
				return "error", err
			}
		}
		if err := run.emit("turn_start", Object{"turn": turn}); err != nil {
			return "error", err
		}
		reply, err := run.modelTurn(ctx)
		if err != nil {
			return "error", err
		}
		for _, call := range reply.Calls {
			recorded, err := run.execute(ctx, call, reply.StopReason)
			if recorded {
				run.pending = run.pending[1:]
			}
			if err != nil {
				return "error", err
			}
		}
		if err := run.emit("turn_end", Object{"turn": turn, "stop_reason": reply.StopReason}); err != nil {
			return "error", err
		}
		run.turn = 0
		switch reply.StopReason {
		case "content_filter":
			return "error", errors.New("model output was blocked by content filtering")
		case "insufficient_system_resource":
			return "error", errors.New("model generation stopped due to insufficient system resources")
		case "aborted":
			return "aborted", errors.New("model generation was aborted by the provider")
		case "length":
			if len(reply.Calls) == 0 {
				return "error", errors.New("model output reached its token limit")
			}
		}
		if len(reply.Calls) > 0 || a.hasNextTurn() {
			continue
		}
		return "complete", nil
	}
	return "max_turns", fmt.Errorf("reached max_turns=%d", a.MaxTurns)
}

func (r *runState) modelContext(ctx context.Context, system string, tools []ToolSpec) ([]Message, error) {
	a := r.agent
	history := a.Session.Messages()
	projection := history
	if a.TransformContext != nil {
		var err error
		projection, err = a.TransformContext(ctx, history)
		if err != nil {
			return nil, err
		}
	}
	messages := projectContext(projection)
	budget := a.ContextBudget
	if budget > 0 {
		available := budget - requestOverhead(system, tools) - messagesCost(a.InstructionContext)
		cost := messagesCost(messages)
		nearLimit := cost >= available-available/10
		if nearLimit && a.AutoCompact && a.TransformContext == nil && !r.attemptedCompaction && countUserTurns(history) > 2 {
			r.attemptedCompaction = true
			if err := r.emit("context_budget", Object{"status": "compacting"}); err != nil {
				return nil, err
			}
			compacted, err := compactMessages(ctx, a.Provider, history, 2, budget)
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			outcome := "compaction_failed"
			if err == nil {
				projected := projectContext(compacted)
				if newCost := messagesCost(projected); newCost < cost {
					if err := a.Session.Checkpoint(compacted); err != nil {
						return nil, err
					}
					messages, cost = projected, newCost
					outcome = "compacted"
				}
			}
			if err := r.emit("context_budget", Object{"status": outcome}); err != nil {
				return nil, err
			}
		}
		if cost > available {
			fitted, err := fitContext(messages, available)
			if err != nil {
				return nil, err
			}
			messages = fitted
			if err := r.emit("context_budget", Object{"status": "trimmed"}); err != nil {
				return nil, err
			}
		} else if nearLimit && !r.attemptedCompaction {
			if err := r.emit("context_budget", Object{"status": "near_limit"}); err != nil {
				return nil, err
			}
		}
	}
	result := make([]Message, 0, len(a.InstructionContext)+len(messages))
	result = append(result, a.InstructionContext...)
	return append(result, messages...), nil
}

func countUserTurns(messages []Message) int {
	count := 0
	for _, message := range messages {
		if message.Role == "user" {
			count++
		}
	}
	return count
}

func (r *runState) modelTurn(ctx context.Context) (ModelReply, error) {
	system := r.agent.SystemPrompt()
	tools := r.agent.Tools.Specs()
	messages, err := r.modelContext(ctx, system, tools)
	if err != nil {
		return ModelReply{}, err
	}
	if err := r.emit("message_start", Object{"role": "assistant"}); err != nil {
		return ModelReply{}, err
	}
	var text, reasoning strings.Builder
	reply, err := r.agent.Provider.Stream(ctx, ModelRequest{System: system, Messages: messages, Tools: tools},
		func(delta ModelDelta) error {
			reasoning.WriteString(delta.Reasoning)
			if delta.Text == "" {
				return nil
			}
			text.WriteString(delta.Text)
			return r.emit("message_update", Object{"delta": delta.Text})
		})
	if err == nil {
		err = validateReply(reply)
	}
	message := Message{Role: "assistant", Content: text.String(), Reasoning: reasoning.String(), StopReason: &reply.StopReason}
	if err != nil {
		reason := "error"
		if ctx.Err() != nil || r.observerError != nil {
			reason = "aborted"
		}
		message.StopReason = &reason
		return ModelReply{}, errors.Join(err, r.append(message))
	}
	message.ToolCalls, message.Usage = reply.Calls, reply.Usage
	if _, err := r.agent.Session.Append(message); err != nil {
		return ModelReply{}, err
	}
	r.pending = reply.Calls
	return reply, r.emit("message_end", Object{"message": message})
}

func validateReply(reply ModelReply) error {
	switch reply.StopReason {
	case "tool_use":
		if len(reply.Calls) == 0 {
			return errors.New("model requested tools without tool calls")
		}
	case "stop":
		if len(reply.Calls) > 0 {
			return errors.New("unexpected tool calls on a stop response")
		}
	case "length", "content_filter", "insufficient_system_resource", "aborted":
	default:
		return errors.New("model did not complete successfully")
	}
	ids := map[string]bool{}
	for _, call := range reply.Calls {
		if call.ID == "" || call.Name == "" || call.Arguments == nil || ids[call.ID] {
			return errors.New("incomplete or duplicate tool call")
		}
		ids[call.ID] = true
	}
	return nil
}

func (r *runState) execute(ctx context.Context, call ToolCall, stop string) (bool, error) {
	a := r.agent
	started := time.Now()
	if err := ctx.Err(); err != nil {
		return false, err
	}
	effects := []string{}
	d := decision{"blocked", "model stopped with " + stop}
	status := "not_executed"
	var result ToolResult
	var tool Tool
	if stop == "tool_use" {
		var err error
		tool, err = a.Tools.validate(call)
		if err != nil {
			d.reason = "invalid proposal: " + err.Error()
		} else {
			effects = tool.Effects(call.Arguments)
			d = authorize(ctx, a.Policy, Authorization{Call: call, CWD: a.Environment.CWD, Effects: effects}, a.Approve)
		}
		status = "blocked"
		result = ToolResult{Content: "Blocked by execution policy: " + d.reason, IsError: true}
	} else {
		result = ToolResult{Content: "Not executed: " + d.reason + ". Inspect the response and reissue complete arguments if appropriate.", IsError: true}
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if err := r.emit("tool_execution_start", Object{"tool_call": call, "effects": effects}); err != nil {
		return false, err
	}
	if err := r.emit("tool_execution_authorization", Object{
		"call_id": call.ID, "status": d.status, "effects": effects, "reason": d.reason,
	}); err != nil {
		return false, err
	}
	if d.status == "allowed" {
		result = r.invoke(ctx, tool, call, started)
		if r.observerError != nil {
			return false, r.observerError
		}
		status = "complete"
		if result.IsError {
			status = "error"
		}
	}
	message := toolMessage(call, result.Content, result.IsError)
	if _, err := a.Session.Append(message); err != nil {
		return false, err
	}
	if err := r.emit("tool_execution_end", Object{
		"call_id": call.ID, "content": result.Content, "is_error": result.IsError,
		"status": status, "duration_ms": time.Since(started).Milliseconds(),
	}); err != nil {
		return true, err
	}
	return true, r.emit("message_end", Object{"message": message})
}

func (r *runState) invoke(ctx context.Context, tool Tool, call ToolCall, started time.Time) ToolResult {
	a := r.agent
	if a.Hooks.BeforeTool != nil {
		proposal := call
		proposal.Arguments = cloneArguments(call.Arguments).(Object)
		if err := a.Hooks.BeforeTool(ctx, proposal, a.Environment); err != nil {
			return ToolResult{Content: "Blocked by hook: " + err.Error(), IsError: true}
		}
	}
	result := tool.Execute(ctx, call.Arguments, a.Environment, func(delta string) error {
		return r.emit("tool_execution_update", Object{
			"call_id": call.ID, "delta": delta, "duration_ms": time.Since(started).Milliseconds(),
		})
	})
	if ctx.Err() != nil && result.IsError {
		return ToolResult{Content: "Interrupted. Side effects may have occurred; inspect state before retrying.", IsError: true}
	}
	if a.Hooks.AfterTool != nil {
		proposal := call
		proposal.Arguments = cloneArguments(call.Arguments).(Object)
		transformed, err := a.Hooks.AfterTool(ctx, proposal, result)
		if err != nil {
			return failed(err)
		}
		return transformed
	}
	return result
}
