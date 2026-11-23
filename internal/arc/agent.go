package arc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Hooks customize tool handling after authorization. The arguments passed to
// each hook are copies of the validated proposal.
type Hooks struct {
	// BeforeTool can stop the authorized call by returning an error.
	BeforeTool func(context.Context, ToolCall, ToolEnv) error
	// AfterTool can replace the result before it is recorded.
	AfterTool func(context.Context, ToolCall, ToolResult) (ToolResult, error)
}

// Agent runs one task at a time against a session. Configure its fields before
// calling Run or Compact. Steer and FollowUp may be called during an active run.
type Agent struct {
	Provider    Provider
	Session     *Session
	Tools       *Registry
	Environment ToolEnv
	// SystemPrompt is called for each model turn so runtime details can change.
	SystemPrompt func() string
	// InstructionContext is the prefix sent with each turn's projected history.
	InstructionContext []Message
	// TransformContext optionally edits a snapshot of history for a request.
	// Providing a transform disables automatic compaction for that run.
	TransformContext func(context.Context, []Message) ([]Message, error)
	Policy           string
	Approve          Approver // Nil keeps calls that require approval from running.
	Hooks            Hooks
	// MaxTurns bounds model turns per run. ContextBudget bounds estimated
	// input tokens per request when positive.
	MaxTurns, ContextBudget int
	AutoCompact             bool // Try a summary once when the turn nears the budget.

	mu                  sync.Mutex // Protects running and both prompt queues.
	running             bool
	steering, followups []string
}

// Steer queues nonblank text for the next model turn, including a turn
// following tool execution.
func (a *Agent) Steer(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steering = append(a.steering, text)
}

// FollowUp queues nonblank text for the point when the current tool loop would
// otherwise finish.
func (a *Agent) FollowUp(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followups = append(a.followups, text)
}

// takeSteering transfers the current queue to the run so new input can be
// enqueued while that turn is in progress.
func (a *Agent) takeSteering() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	steering := a.steering
	a.steering = nil
	return steering
}

// hasNextTurn gives steering priority, then promotes one follow-up into the
// next turn's steering queue.
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

// begin reserves the agent for either a run or a manual compaction.
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

// Compact summarizes older turns and appends a checkpoint while keeping the
// most recent keep user turns. A failed or canceled summary leaves the
// current branch unchanged.
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

// runState tracks the unfinished turn and the tool results still owed to it.
type runState struct {
	agent               *Agent
	observe             Observer
	observerError       error
	turn                int        // Nonzero while a turn still needs a terminal event.
	pending             []ToolCall // In model order, with each call removed after its result is recorded.
	attemptedCompaction bool
}

// emit remembers the first observer failure so later cleanup can stop sending
// events to that observer while still updating the session.
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

// Run processes a prompt through model turns and authorized tools. It returns
// the final task status and any error. A nil observer discards events.
//
// Run records completed tool results before notifying the observer. On
// interruption it records an error for each pending tool proposal.
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
		// The model may have proposed a batch before cancellation. Give every
		// unfinished proposal a result, even if the UI has stopped consuming events.
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
				// A delivery error after persistence still leaves a valid result.
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

// modelContext projects the active branch and budgets its history after
// reserving space for system instructions, configuration, and tool schemas.
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
		// Summarization costs a separate model request. Try it once per run,
		// and use trimming if the result cannot reduce the context.
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

// modelTurn streams one response, records it, and exposes only complete
// proposals to the tool loop.
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
			// Keep reasoning for provider round trips and stream only answer text to the UI.
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
		// Preserve the partial answer for local history and mark it for
		// exclusion from the next provider request.
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

// validateReply checks provider-independent invariants before accepting any
// proposals in a response.
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

// execute validates and authorizes a proposal before invoking its tool. The
// boolean reports whether the corresponding result was recorded, even if
// subsequent event delivery failed.
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
		// Effect functions expect schema-validated arguments.
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

// invoke runs the authorized tool and its hooks, forwarding progress events
// and returning the final result to the caller for persistence.
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
