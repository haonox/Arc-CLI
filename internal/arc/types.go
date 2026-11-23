// Package arc implements the model loop, tool authorization, and session storage.
//
// An [Agent] sends conversation context to a [Provider], validates its proposed
// tool calls, and records the results in a [Session]. Clients receive [Event]
// values in order as the task proceeds.
package arc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// Version is the Arc release version reported by the CLI.
const Version = "0.1.0"

// EventProtocolVersion identifies the schema of serialized runtime events.
const EventProtocolVersion = 1

// Object is a JSON object used for tool arguments, schemas, and event payloads.
type Object = map[string]any

// Usage records token counts reported by the provider for one model response.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ToolCall is a model proposal. Its ID associates it with a tool result.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments Object `json:"arguments"`
}

// ToolSpec describes a tool to the model. Parameters is its JSON Schema.
type ToolSpec struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  Object `json:"parameters"`
}

// Message is a conversation entry stored in a session. Before a model
// request, context projection converts summaries and omits local notes.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls"`
	ToolCallID *string    `json:"tool_call_id"` // Set on a tool result to identify its proposal.
	Name       *string    `json:"name"`
	StopReason *string    `json:"stop_reason"`
	Usage      *Usage     `json:"usage"`
	IsError    bool       `json:"is_error"`
	Reasoning  string     `json:"reasoning_content"` // Provider reasoning, stored separately from the visible answer.
}

// cloneArguments copies the mutable containers in a JSON-decoded value.
func cloneArguments(value any) any {
	switch value := value.(type) {
	case Object:
		result := make(Object, len(value))
		for key, item := range value {
			result[key] = cloneArguments(item)
		}
		return result
	case []any:
		result := make([]any, len(value))
		for i, item := range value {
			result[i] = cloneArguments(item)
		}
		return result
	default:
		return value
	}
}

// cloneMessage detaches a message from maps, slices, and pointer fields owned
// by its caller.
func cloneMessage(message Message) Message {
	if message.ToolCalls != nil {
		calls := make([]ToolCall, len(message.ToolCalls))
		for i, call := range message.ToolCalls {
			call.Arguments = cloneArguments(call.Arguments).(Object)
			calls[i] = call
		}
		message.ToolCalls = calls
	}
	message.Usage = clonePointer(message.Usage)
	message.StopReason = clonePointer(message.StopReason)
	message.Name = clonePointer(message.Name)
	message.ToolCallID = clonePointer(message.ToolCallID)
	return message
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

// MarshalJSON encodes the v1 message shape with an empty array for absent tool
// calls.
func (m Message) MarshalJSON() ([]byte, error) {
	// A distinct type bypasses this method during encoding.
	type wire Message
	if m.ToolCalls == nil {
		m.ToolCalls = []ToolCall{}
	}
	return json.Marshal(wire(m))
}

// validate checks shared message fields before a record enters the session.
func (m Message) validate() error {
	switch m.Role {
	case "user", "assistant", "tool", "summary", "note":
	default:
		return fmt.Errorf("invalid message role %q", m.Role)
	}
	if m.StopReason != nil {
		switch *m.StopReason {
		case "stop", "tool_use", "length", "content_filter", "insufficient_system_resource", "error", "aborted":
		default:
			return fmt.Errorf("invalid stop reason %q", *m.StopReason)
		}
	}
	for _, call := range m.ToolCalls {
		if call.ID == "" || call.Name == "" || call.Arguments == nil {
			return errors.New("invalid tool call")
		}
	}
	if m.Usage != nil && (m.Usage.InputTokens < 0 || m.Usage.OutputTokens < 0) {
		return errors.New("invalid token usage")
	}
	return nil
}

// Event values are observed in order. The protocol envelope remains compatible with v1 consumers.
type Event struct {
	Type string
	Data Object
}

// MarshalJSON flattens event data into the v1 envelope. The event type and
// protocol version take precedence over payload fields with those names.
func (e Event) MarshalJSON() ([]byte, error) {
	data := make(Object, len(e.Data)+2)
	for key, value := range e.Data {
		data[key] = value
	}
	data["type"] = e.Type
	data["protocol_version"] = EventProtocolVersion
	return json.Marshal(data)
}

// Observer consumes ordered events. Returning an error stops event delivery
// and causes the active run to wind down.
type Observer func(Event) error

// ModelRequest supplies the system prompt, projected messages, and available
// tool specifications for a single provider call.
type ModelRequest struct {
	System   string     // System instructions for this request.
	Messages []Message  // Projected history and instruction context.
	Tools    []ToolSpec // Tools the model may propose during this request.
}

// ModelDelta carries newly streamed visible text, reasoning, or both.
type ModelDelta struct {
	Text      string
	Reasoning string
}

// ModelReply describes the end of a stream. Text arrives through the callback,
// and Calls contains the complete tool proposals in execution order.
type ModelReply struct {
	Calls      []ToolCall
	StopReason string
	Usage      *Usage
}

// Provider streams model text and returns tool proposals only after the response is complete.
type Provider interface {
	// Stream calls emit with each delta in order. It returns the final reply
	// after the stream ends or an error if delivery or parsing fails.
	Stream(context.Context, ModelRequest, func(ModelDelta) error) (ModelReply, error)
}

// decodeJSON accepts exactly one JSON value and keeps numbers in untyped
// objects as [json.Number] to preserve their precision and representation.
func decodeJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return errors.New("unexpected trailing JSON data")
	}
	return nil
}

func toolMessage(call ToolCall, content string, failed bool) Message {
	return Message{Role: "tool", Content: content, ToolCallID: &call.ID, Name: &call.Name, IsError: failed}
}
