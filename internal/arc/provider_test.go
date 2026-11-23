package arc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// sse formats payloads as separate server-sent events for the parser fixtures.
func sse(frames ...string) string {
	var stream strings.Builder
	for _, frame := range frames {
		stream.WriteString("data: " + frame + "\n\n")
	}
	return stream.String()
}

func TestProviderStreamsCallsAndUsesDirectTransport(t *testing.T) {
	type exchange struct {
		path, key string
		payload   Object
	}
	requests := make(chan exchange, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload Object
		_ = json.NewDecoder(r.Body).Decode(&payload)
		requests <- exchange{r.URL.Path, r.Header.Get("Authorization"), payload}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, sse(
			`{"choices":[{"delta":{"reasoning_content":"checking","content":"你好"}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"read","arguments":"{\"pa"}},{"index":0,"id":"a","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"th\":\"b\"}"}}]},"finish_reason":"tool_calls"}]}`,
			`{"usage":{"prompt_tokens":12,"completion_tokens":3},"choices":[]}`,
			"[DONE]",
		))
	}))
	defer server.Close()
	t.Setenv("ALL_PROXY", "http://127.0.0.1:1")
	t.Setenv("HTTPS_PROXY", "http://127.0.0.1:1")
	provider, err := NewProvider(Config{BaseURL: server.URL + "/v1", Model: "deepseek-test", apiKey: "synthetic"},
		ProviderOptions{Timeout: time.Second, ReasoningEffort: "medium", MaxTokens: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	if provider.client.Transport.(*http.Transport).Proxy != nil {
		t.Fatal("model requests can use a proxy")
	}
	var deltas []ModelDelta
	reply, err := provider.Stream(t.Context(), ModelRequest{
		System: "system", Messages: []Message{{Role: "user", Content: "user"}}, Tools: []ToolSpec{BuiltinTools()[0].Spec},
	}, func(d ModelDelta) error { deltas = append(deltas, d); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if got := reply.Calls; len(got) != 2 || got[0].ID != "a" || got[1].Arguments["path"] != "b" {
		t.Fatalf("unexpected calls: %#v", got)
	}
	if len(deltas) != 1 || deltas[0].Text != "你好" || deltas[0].Reasoning != "checking" || reply.Usage.InputTokens != 12 {
		t.Fatalf("unexpected reply: %+v, deltas %v", reply, deltas)
	}
	request := <-requests
	if request.key != "Bearer synthetic" || request.path != "/v1/chat/completions" ||
		request.payload["max_tokens"] != float64(128) || request.payload["reasoning_effort"] != "high" ||
		request.payload["thinking"].(Object)["type"] != "enabled" {
		t.Fatalf("unexpected request properties: %#v", request)
	}
}

func TestParseStreamRejectsIncompleteAndMalformedCalls(t *testing.T) {
	tests := map[string][]string{
		"missing done":      {`{"choices":[{"finish_reason":"stop"}]}`},
		"missing finish":    {"[DONE]"},
		"data after stop":   {`{"choices":[{"finish_reason":"stop"}]}`, `{"choices":[{"delta":{"content":"extra"}}]}`, "[DONE]"},
		"invalid arguments": {`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"null"}}]},"finish_reason":"tool_calls"}]}`, "[DONE]"},
		"duplicate IDs":     {`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{}"}},{"index":1,"id":"a","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, "[DONE]"},
		"provider error":    {`{"error":{"message":"bad"}}`, "[DONE]"},
	}
	for name, frames := range tests {
		t.Run(name, func(t *testing.T) {
			reply, err := parseStream(strings.NewReader(sse(frames...)), func(ModelDelta) error { return nil }, func() {})
			if err == nil || len(reply.Calls) != 0 {
				t.Fatalf("expected a rejected response, got %+v %v", reply, err)
			}
		})
	}
}

func TestSSEFramesAndInactivity(t *testing.T) {
	input := ": heartbeat\r\ndata: {\"choices\":\r\ndata: [{\"finish_reason\":\"stop\"}]}\r\n\r\ndata: [DONE]"
	if reply, err := parseStream(strings.NewReader(input), func(ModelDelta) error { return nil }, func() {}); err != nil || reply.StopReason != "stop" {
		t.Fatalf("multiline frame failed: %+v %v", reply, err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()
	provider, err := NewProvider(Config{BaseURL: server.URL, Model: "test"}, ProviderOptions{Timeout: 100 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	if _, err := provider.Stream(ctx, ModelRequest{}, func(ModelDelta) error { return nil }); err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("expected inactivity timeout, got %v", err)
	}
}

func TestProviderRejectsRedirectWithoutSendingCredentials(t *testing.T) {
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/secret", http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	provider, err := NewProvider(Config{BaseURL: redirect.URL, Model: "test", apiKey: "synthetic"}, ProviderOptions{Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	if _, err := provider.Stream(t.Context(), ModelRequest{}, func(ModelDelta) error { return nil }); err == nil || !strings.Contains(err.Error(), "307") {
		t.Fatalf("expected redirect rejection, got %v", err)
	}
}
