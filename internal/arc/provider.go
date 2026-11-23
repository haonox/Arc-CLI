package arc

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
)

// ProviderOptions configures HTTP timeouts and model request parameters.
type ProviderOptions struct {
	Timeout         time.Duration // Bounds connection setup and waits for response headers or stream activity.
	MaxTokens       int           // A zero value leaves the output limit to the provider.
	Thinking        string        // Empty preserves the default selected by NewProvider.
	ReasoningEffort string        // Empty omits the effort unless another option selects it explicitly.
}

// OpenAIProvider implements [Provider] for streaming Chat Completions endpoints.
type OpenAIProvider struct {
	modelMu      sync.RWMutex
	model        string
	baseURL, key string
	client       *http.Client
	options      ProviderOptions
}

// NewProvider validates the endpoint and request options, then creates a
// direct HTTP client with redirect following disabled.
func NewProvider(config Config, options ProviderOptions) (*OpenAIProvider, error) {
	if strings.TrimSpace(config.Model) == "" || options.Timeout <= 0 {
		return nil, errors.New("model and network timeout must be valid")
	}
	if _, err := validateEndpoint(config.BaseURL, config.apiKey != ""); err != nil {
		return nil, err
	}
	if options.MaxTokens < 0 || options.MaxTokens > 393216 {
		return nil, errors.New("--max-tokens must be between 1 and 393216")
	}
	if !slices.Contains([]string{"", "enabled", "disabled"}, options.Thinking) {
		return nil, errors.New("--thinking must be enabled or disabled")
	}
	// Normalize accepted aliases before choosing the thinking mode for the wire request.
	efforts := map[string]string{
		"": "", "none": "none", "minimal": "low", "low": "low",
		"medium": "high", "high": "high", "xhigh": "high", "max": "max",
	}
	effort, ok := efforts[options.ReasoningEffort]
	if !ok {
		return nil, errors.New("invalid --reasoning-effort")
	}
	switch effort {
	case "none":
		if options.Thinking == "enabled" {
			return nil, errors.New("reasoning effort none conflicts with enabled thinking")
		}
		options.Thinking = "disabled"
		effort = ""
	case "":
		if options.Thinking == "" && (strings.HasPrefix(config.Model, "deepseek-") || strings.HasPrefix(config.BaseURL, DeepSeekBaseURL)) {
			options.Thinking = "disabled"
		}
	default:
		if options.Thinking == "disabled" {
			return nil, errors.New("reasoning effort conflicts with disabled thinking")
		}
		options.Thinking = "enabled"
	}
	options.ReasoningEffort = effort
	transport := &http.Transport{
		DialContext:           (&net.Dialer{Timeout: options.Timeout, KeepAlive: 30 * time.Second}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          10,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   options.Timeout,
		ResponseHeaderTimeout: options.Timeout,
		// Leave Proxy nil so model traffic goes straight to the configured endpoint.
	}
	return &OpenAIProvider{
		model: config.Model, baseURL: strings.TrimRight(config.BaseURL, "/"), key: config.apiKey, options: options,
		client: &http.Client{
			Transport:     transport,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
	}, nil
}

// Model returns the model name used for new requests.
func (p *OpenAIProvider) Model() string {
	p.modelMu.RLock()
	defer p.modelMu.RUnlock()
	return p.model
}

// SetModel changes the model name for subsequent requests. An active request
// has already captured its own name.
func (p *OpenAIProvider) SetModel(model string) {
	p.modelMu.Lock()
	defer p.modelMu.Unlock()
	p.model = model
}

// Endpoint returns the configured API root without a trailing slash.
func (p *OpenAIProvider) Endpoint() string { return p.baseURL }

// Options returns the normalized settings used for requests.
func (p *OpenAIProvider) Options() ProviderOptions { return p.options }

// Close releases idle HTTP connections.
func (p *OpenAIProvider) Close() { p.client.CloseIdleConnections() }

// wireMessages converts projected conversation entries to Chat Completions
// messages. The wire protocol represents tool arguments as JSON strings.
func wireMessages(request ModelRequest) ([]Object, error) {
	result := []Object{{"role": "system", "content": request.System}}
	for _, m := range request.Messages {
		if m.Role != "user" && m.Role != "assistant" && m.Role != "tool" {
			return nil, fmt.Errorf("project %q messages before sending the request", m.Role)
		}
		item := Object{"role": m.Role, "content": m.Content}
		if m.Role == "assistant" && m.Reasoning != "" {
			item["reasoning_content"] = m.Reasoning
		}
		if len(m.ToolCalls) > 0 {
			calls := make([]Object, 0, len(m.ToolCalls))
			for _, call := range m.ToolCalls {
				args, err := json.Marshal(call.Arguments)
				if err != nil {
					return nil, err
				}
				calls = append(calls, Object{"id": call.ID, "type": "function",
					"function": Object{"name": call.Name, "arguments": string(args)}})
			}
			item["tool_calls"] = calls
		}
		if m.Role == "tool" {
			item["tool_call_id"] = m.ToolCallID
		}
		result = append(result, item)
	}
	return result, nil
}

// Stream sends a Chat Completions request and delivers deltas to emit in wire
// order. The returned reply contains proposals only after a valid finish
// reason and stream terminator have been received.
func (p *OpenAIProvider) Stream(ctx context.Context, request ModelRequest, emit func(ModelDelta) error) (ModelReply, error) {
	messages, err := wireMessages(request)
	if err != nil {
		return ModelReply{}, err
	}
	payload := Object{
		"model": p.Model(), "messages": messages, "stream": true,
		"stream_options": Object{"include_usage": true},
	}
	if p.options.Thinking != "" {
		payload["thinking"] = Object{"type": p.options.Thinking}
	}
	if p.options.ReasoningEffort != "" {
		payload["reasoning_effort"] = p.options.ReasoningEffort
	}
	if p.options.MaxTokens > 0 {
		payload["max_tokens"] = p.options.MaxTokens
	}
	if len(request.Tools) > 0 {
		tools := make([]Object, 0, len(request.Tools))
		for _, spec := range request.Tools {
			tools = append(tools, Object{"type": "function", "function": spec})
		}
		payload["tools"] = tools
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return ModelReply{}, err
	}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ModelReply{}, errors.New("invalid model request endpoint")
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")
	if p.key != "" {
		req.Header.Set("Authorization", "Bearer "+p.key)
	}
	res, err := p.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return ModelReply{}, context.Cause(ctx)
		}
		// Transport errors can contain a request URL or other sensitive input.
		return ModelReply{}, errors.New("model transport error; no automatic retry")
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return ModelReply{}, fmt.Errorf("model HTTP %d; check endpoint, model and credentials", res.StatusCode)
	}
	// Use a resettable timer to bound inactivity during streaming, including
	// long responses that continue to make progress.
	timer := time.AfterFunc(p.options.Timeout, func() { cancel(errors.New("model stream timed out")) })
	defer timer.Stop()
	reply, err := parseStream(res.Body, emit, func() { timer.Reset(p.options.Timeout) })
	if ctx.Err() != nil {
		return ModelReply{}, context.Cause(ctx)
	}
	return reply, err
}

// streamChunk is one wire frame. A final usage frame can arrive without a choice.
type streamChunk struct {
	Error json.RawMessage `json:"error"`
	Usage *struct {
		Input  int `json:"prompt_tokens"`
		Output int `json:"completion_tokens"`
	} `json:"usage"`
	Choices []struct {
		Index  int    `json:"index"`
		Finish string `json:"finish_reason"`
		Delta  struct {
			Content   string `json:"content"`
			Refusal   string `json:"refusal"`
			Reasoning string `json:"reasoning_content"`
			Calls     []struct {
				Index    *int   `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

// callFragments accumulates strings that may be split across several frames.
type callFragments struct{ id, name, arguments strings.Builder }

// parseStream assembles tool fragments by index and checks that the entire
// reply is valid before returning any tool proposals.
func parseStream(reader io.Reader, emit func(ModelDelta) error, activity func()) (ModelReply, error) {
	var reply ModelReply
	calls := map[int]*callFragments{}
	var finish string
	err := readSSE(reader, activity, func(data string) error {
		var chunk streamChunk
		if strings.TrimSpace(data) == "null" || json.Unmarshal([]byte(data), &chunk) != nil || chunk.Error != nil {
			return errors.New("malformed model stream; no tools executed")
		}
		if chunk.Usage != nil {
			if chunk.Usage.Input < 0 || chunk.Usage.Output < 0 {
				return errors.New("invalid token usage")
			}
			reply.Usage = &Usage{InputTokens: chunk.Usage.Input, OutputTokens: chunk.Usage.Output}
		}
		for _, choice := range chunk.Choices {
			// Arc requests a single choice. After it finishes, only frames
			// without choices may contribute usage.
			if choice.Index != 0 || finish != "" {
				return errors.New("unexpected choice in model stream")
			}
			delta := ModelDelta{Text: first(choice.Delta.Content, choice.Delta.Refusal), Reasoning: choice.Delta.Reasoning}
			if delta.Text != "" || delta.Reasoning != "" {
				if err := emit(delta); err != nil {
					return err
				}
			}
			for _, fragment := range choice.Delta.Calls {
				if fragment.Index == nil || *fragment.Index < 0 || *fragment.Index > 127 ||
					(fragment.Type != "" && fragment.Type != "function") {
					return errors.New("invalid streamed tool call")
				}
				index := *fragment.Index
				call := calls[index]
				if call == nil {
					call = &callFragments{}
					calls[index] = call
				}
				call.id.WriteString(fragment.ID)
				call.name.WriteString(fragment.Function.Name)
				call.arguments.WriteString(fragment.Function.Arguments)
			}
			finish = choice.Finish
		}
		return nil
	})
	if err != nil {
		return ModelReply{}, err
	}
	switch finish {
	case "tool_calls":
		if len(calls) == 0 {
			return ModelReply{}, errors.New("model requested tools without a tool call")
		}
		reply.StopReason = "tool_use"
	case "stop", "length", "content_filter", "insufficient_system_resource", "aborted":
		if finish == "stop" && len(calls) > 0 {
			return ModelReply{}, errors.New("unexpected tool calls in a stop response")
		}
		reply.StopReason = finish
	default:
		return ModelReply{}, errors.New("incomplete or unsupported model response; no tools executed")
	}
	var indexes []int
	for index := range calls {
		indexes = append(indexes, index)
	}
	// Indices define execution order even if fragments arrived interleaved.
	slices.Sort(indexes)
	seen := map[string]bool{}
	for _, index := range indexes {
		parts := calls[index]
		call := ToolCall{ID: parts.id.String(), Name: parts.name.String()}
		if call.ID == "" || call.Name == "" || seen[call.ID] ||
			decodeJSON([]byte(parts.arguments.String()), &call.Arguments) != nil || call.Arguments == nil {
			return ModelReply{}, errors.New("incomplete, duplicate or invalid tool call; no tools executed")
		}
		seen[call.ID] = true
		reply.Calls = append(reply.Calls, call)
	}
	return reply, nil
}

// readSSE combines data lines into events, bounds their payload size, and
// requires the Chat Completions [DONE] terminator.
func readSSE(reader io.Reader, activity func(), consume func(string) error) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 2_000_010)
	var parts []string
	size, total := 0, 0
	drain := func() (bool, error) {
		if len(parts) == 0 {
			return false, nil
		}
		data := strings.Join(parts, "\n")
		parts, size = nil, 0
		if data == "[DONE]" {
			return true, nil
		}
		total += len(data)
		if total > 4_000_000 {
			return false, errors.New("model stream exceeds 4 MB")
		}
		return false, consume(data)
	}
	for scanner.Scan() {
		// Comments and blank lines can be heartbeats even when they carry no payload.
		activity()
		line := scanner.Text()
		if line == "" {
			if done, err := drain(); done || err != nil {
				return err
			}
		} else if data, ok := strings.CutPrefix(line, "data:"); ok {
			data = strings.TrimPrefix(data, " ")
			size += len(data) + 1
			if size > 2_000_001 {
				return errors.New("SSE event exceeds size limit")
			}
			parts = append(parts, data)
		}
	}
	if scanner.Err() != nil {
		return errors.New("model stream interrupted or invalid")
	}
	// Accept a final terminator even when the peer omits its blank separator.
	if done, err := drain(); done || err != nil {
		return err
	}
	return errors.New("model stream ended without [DONE]; no tools executed")
}
