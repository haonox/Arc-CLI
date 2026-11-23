package cli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/creack/pty"
)

func testEnvironment(t *testing.T) {
	t.Helper()
	for _, name := range []string{"ARC_MODEL", "ARC_BASE_URL", "ARC_API_KEY", "DEEPSEEK_API_KEY", "NO_COLOR"} {
		t.Setenv(name, "")
	}
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_STATE_HOME", t.TempDir())
}

func textFrame(text string) string {
	data, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": text}}}})
	return string(data)
}

func shellFrame(command string) string {
	args, _ := json.Marshal(map[string]string{"command": command})
	data, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{
		"finish_reason": "tool_calls", "delta": map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "id": "call-shell", "function": map[string]string{"name": "bash", "arguments": string(args)},
		}}},
	}}})
	return string(data)
}

const answerDone = `{"choices":[{"finish_reason":"stop"}]}`
const usageFrame = `{"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":8}}`

type responseFrames struct {
	frames []string
	delays map[int]time.Duration
}

func fakeAPI(t *testing.T, responses ...responseFrames) (*httptest.Server, <-chan map[string]any) {
	t.Helper()
	script := make(chan responseFrames, len(responses))
	for _, response := range responses {
		script <- response
	}
	requests := make(chan map[string]any, len(responses)+5)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			w.WriteHeader(400)
			return
		}
		requests <- payload
		w.Header().Set("Content-Type", "text/event-stream")
		var response responseFrames
		select {
		case response = <-script:
		default:
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		for i, frame := range response.frames {
			select {
			case <-time.After(response.delays[i]):
			case <-r.Context().Done():
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", frame)
			w.(http.Flusher).Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)
	return server, requests
}

func TestCLIJSONAndSessionCommands(t *testing.T) {
	testEnvironment(t)
	server, requests := fakeAPI(t, responseFrames{frames: []string{textFrame("first response"), answerDone}},
		responseFrames{frames: []string{textFrame("resumed"), answerDone}})
	t.Setenv("ARC_BASE_URL", server.URL+"/v1")
	t.Setenv("ARC_MODEL", "offline")
	cwd := t.TempDir()
	run := func(input string, args ...string) (int, string, string) {
		t.Helper()
		var stdout, stderr bytes.Buffer
		code := Run(t.Context(), args, strings.NewReader(input), &stdout, &stderr)
		return code, stdout.String(), stderr.String()
	}
	code, out, errText := run("piped context", "explicit task", "--mode", "json", "--cwd", cwd, "--tools", "none")
	if code != 0 || errText != "" {
		t.Fatalf("JSON launch failed: %d %s", code, errText)
	}
	scanner := bufio.NewScanner(strings.NewReader(out))
	var events []string
	for scanner.Scan() {
		var event map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil || event["protocol_version"] != float64(1) {
			t.Fatalf("bad event %s: %v", scanner.Text(), err)
		}
		events = append(events, event["type"].(string))
	}
	want := []string{"agent_start", "message_end", "turn_start", "message_start", "message_update", "message_end", "turn_end", "agent_end"}
	if !slices.Equal(events, want) {
		t.Fatalf("event protocol: %v", events)
	}
	if got := (<-requests)["messages"].([]any)[1].(map[string]any)["content"]; got != "explicit task\n\npiped context" {
		t.Fatalf("unexpected prompt: %q", got)
	}
	_, list, _ := run("", "sessions", "list", "--cwd", cwd)
	lines := strings.Split(list, "\n")
	if len(lines) < 2 {
		t.Fatalf("missing session: %q", list)
	}
	fields := strings.Fields(lines[1])
	if len(fields) == 0 {
		t.Fatalf("missing session ID: %q", list)
	}
	id := fields[0]
	for _, command := range [][]string{{"rename", id, "工作记录"}, {"archive", id}, {"restore", id}} {
		args := append([]string{"sessions", "--cwd", cwd}, command...)
		if code, _, errText := run("", args...); code != 0 {
			t.Fatalf("%v: %s", args, errText)
		}
	}
	code, out, errText = run("", "--resume", id, "--cwd", cwd, "-p", "next task")
	if code != 0 || out != "resumed\n" || errText != "" {
		t.Fatalf("resume failed: %d %q %q", code, out, errText)
	}
	if prior := (<-requests)["messages"].([]any)[1].(map[string]any)["content"]; prior != "explicit task\n\npiped context" {
		t.Fatalf("prior history missing: %q", prior)
	}
}

func TestCLIHelpAndInvalidOptionsDoNotSendRequests(t *testing.T) {
	testEnvironment(t)
	for _, test := range []struct {
		args []string
		code int
		text string
	}{
		{[]string{"--help"}, 0, "arc sessions"},
		{[]string{"--version"}, 0, "Arc CLI"},
		{[]string{"--context-budget", "12", "-p", "hi"}, 2, "at least 1024"},
		{[]string{"--resume", "123", "--no-session", "-p", "hi"}, 2, "mutually exclusive"},
		{[]string{"-p", "hi"}, 2, "DEEPSEEK_API_KEY"},
	} {
		var out, errText bytes.Buffer
		code := Run(t.Context(), test.args, strings.NewReader(""), &out, &errText)
		if code != test.code || !strings.Contains(out.String()+errText.String(), test.text) {
			t.Fatalf("args %v: %d %q %q", test.args, code, out.String(), errText.String())
		}
	}
}

func TestCLIProcess(t *testing.T) {
	if os.Getenv("ARC_TEST_HELPER") != "1" {
		return
	}
	index := slices.Index(os.Args, "--")
	os.Exit(Run(context.Background(), os.Args[index+1:], os.Stdin, os.Stdout, os.Stderr))
}

type testTTY struct {
	t                  *testing.T
	file               *os.File
	stream             chan string
	transcript         strings.Builder
	dir                string
	answeredColorProbe bool
}

func startTTY(t *testing.T, server *httptest.Server, width uint16) *testTTY {
	t.Helper()
	testEnvironment(t)
	t.Setenv("TERM", "xterm-256color")
	if err := os.Unsetenv("NO_COLOR"); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ARC_TEST_HELPER", "1")
	t.Setenv("ARC_BASE_URL", server.URL)
	t.Setenv("ARC_MODEL", "test-pty")
	dir := t.TempDir()
	command := exec.Command(os.Args[0], "-test.run=^TestCLIProcess$", "--", "--cwd", dir, "--no-session")
	file, err := pty.StartWithSize(command, &pty.Winsize{Rows: 25, Cols: width})
	if err != nil {
		t.Fatal(err)
	}
	tty := &testTTY{t: t, file: file, stream: make(chan string, 128), dir: dir}
	stop := make(chan struct{})
	go func() {
		defer close(tty.stream)
		buffer := make([]byte, 4096)
		for {
			n, err := file.Read(buffer)
			if n > 0 {
				select {
				case tty.stream <- string(buffer[:n]):
				case <-stop:
					return
				}
			}
			if err != nil {
				return
			}
		}
	}()
	done := make(chan struct{})
	go func() { _ = command.Wait(); close(done) }()
	t.Cleanup(func() {
		_, _ = io.WriteString(file, "/quit\r")
		select {
		case <-done:
		case <-time.After(time.Second):
			_ = command.Process.Kill()
			<-done
		}
		close(stop)
		_ = file.Close()
	})
	tty.waitFor("/help commands")
	return tty
}

func (tty *testTTY) send(text string) {
	tty.t.Helper()
	if _, err := io.WriteString(tty.file, text); err != nil {
		tty.t.Fatal(err)
	}
}

func (tty *testTTY) waitFor(text string) {
	tty.t.Helper()
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	for !strings.Contains(tty.transcript.String(), text) {
		select {
		case chunk, ok := <-tty.stream:
			if !ok {
				tty.t.Fatalf("PTY closed before %q\n%q", text, tty.transcript.String())
			}
			tty.transcript.WriteString(chunk)
			if !tty.answeredColorProbe && strings.Contains(tty.transcript.String(), "\x1b]11;?") &&
				strings.Contains(tty.transcript.String(), "\x1b[6n") {
				tty.answeredColorProbe = true
				tty.send("\x1b]11;rgb:0000/0000/0000\x1b\\\x1b[1;1R")
			}
		case <-timer.C:
			tty.t.Fatalf("PTY did not show %q\n%q", text, tty.transcript.String())
		}
	}
}

func TestPTYStreamingChinesePreservesPrefilledInput(t *testing.T) {
	server, _ := fakeAPI(t, responseFrames{
		frames: []string{textFrame("# 流式标题\n"), textFrame("保留"), textFrame("预先输入的命令。"), answerDone, usageFrame},
		delays: map[int]time.Duration{2: 300 * time.Millisecond},
	})
	tty := startTTY(t, server, 45)
	tty.send("Start streaming\r")
	tty.waitFor("流式标题")
	tty.send("/sta")
	tty.waitFor("保留预先输入的命令。")
	tty.send("tus\r")
	tty.waitFor("endpoint")
	tty.waitFor("idle")
	tty.waitFor("42 in")
}

func TestPTYShellApprovalAndCancellation(t *testing.T) {
	server, requests := fakeAPI(t,
		responseFrames{frames: []string{shellFrame("touch refused")}},
		responseFrames{frames: []string{textFrame("Denied request"), answerDone}},
		responseFrames{frames: []string{shellFrame("touch started; (sleep 0.7; touch leaked) & wait")}},
	)
	tty := startTTY(t, server, 110)
	tty.send("Run first command\r")
	tty.waitFor("touch refused")
	tty.send("/deny\r")
	tty.waitFor("Denied request")
	if _, err := os.Stat(filepath.Join(tty.dir, "refused")); !os.IsNotExist(err) {
		t.Fatal("denied command was executed")
	}
	for range 2 {
		<-requests
	}
	tty.send("Run second command\r")
	tty.waitFor("touch started")
	tty.send("/approve\r")
	deadline := time.Now().Add(2 * time.Second)
	for {
		_, err := os.Stat(filepath.Join(tty.dir, "started"))
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("approved Bash did not start")
		}
		time.Sleep(10 * time.Millisecond)
	}
	tty.send("\x03")
	tty.waitFor("aborted")
	tty.send("/status\r")
	tty.waitFor("idle")
	time.Sleep(750 * time.Millisecond)
	if _, err := os.Stat(filepath.Join(tty.dir, "leaked")); !os.IsNotExist(err) {
		t.Fatal("Ctrl-C left a running descendant")
	}
}
