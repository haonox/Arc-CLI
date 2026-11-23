// Package arc implements Arc's model loop, tools, and durable conversations.
package arc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const Version = "0.1.0"
const EventProtocolVersion = 1

type Object = map[string]any

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments Object `json:"arguments"`
}

type ToolSpec struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  Object `json:"parameters"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls"`
	ToolCallID *string    `json:"tool_call_id"`
	Name       *string    `json:"name"`
	StopReason *string    `json:"stop_reason"`
	Usage      *Usage     `json:"usage"`
	IsError    bool       `json:"is_error"`
	Reasoning  string     `json:"reasoning_content"`
}

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

func (m Message) MarshalJSON() ([]byte, error) {
	type wire Message
	if m.ToolCalls == nil {
		m.ToolCalls = []ToolCall{}
	}
	return json.Marshal(wire(m))
}

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

func (e Event) MarshalJSON() ([]byte, error) {
	data := make(Object, len(e.Data)+2)
	for key, value := range e.Data {
		data[key] = value
	}
	data["type"] = e.Type
	data["protocol_version"] = EventProtocolVersion
	return json.Marshal(data)
}

type Observer func(Event) error

type ModelRequest struct {
	System   string
	Messages []Message
	Tools    []ToolSpec
}

type ModelDelta struct {
	Text      string
	Reasoning string
}

type ModelReply struct {
	Calls      []ToolCall
	StopReason string
	Usage      *Usage
}

// Provider streams model text and returns tool proposals only after the response is complete.
type Provider interface {
	Stream(context.Context, ModelRequest, func(ModelDelta) error) (ModelReply, error)
}

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
