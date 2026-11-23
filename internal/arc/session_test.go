package arc

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLegacySessionCanBranchAndArchive(t *testing.T) {
	cwd, err := Workspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	state := t.TempDir()
	catalog := NewCatalog(cwd, func(key string) string {
		if key == "XDG_STATE_HOME" {
			return state
		}
		return ""
	})
	path := catalog.NewPath()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	// Keep a literal legacy fixture so current encoders cannot mask a reader
	// compatibility regression. Its cursor intentionally splits a tool pair.
	legacy := strings.ReplaceAll(`{"type":"session","version":1,"id":"0123656789abcdef","cwd":"CWD","created_at":"2026-09-18T00:00:00+00:00"}
{"type":"message","id":"user0001","parent":null,"message":{"role":"user","content":"原始问题","tool_calls":[],"tool_call_id":null,"name":null,"stop_reason":null,"usage":null,"is_error":false,"reasoning_content":""}}
{"type":"message","id":"assistant0002","parent":"user0001","message":{"role":"assistant","tool_calls":[{"id":"old-call","name":"write","arguments":{"path":"should-not-exist","content":"never replay"}}],"stop_reason":"tool_use","is_error":false,"reasoning_content":"thinking"}}
{"type":"message","id":"tool0003","parent":"assistant0002","message":{"role":"tool","content":"original result","tool_call_id":"old-call","tool_calls":[],"is_error":false}}
{"type":"cursor","leaf":"assistant0002"}
{"type":"title","title":"早期对话"}`, "CWD", cwd)
	if err := os.WriteFile(path, []byte(legacy), 0600); err != nil {
		t.Fatal(err)
	}
	session, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { session.Close() })
	if _, err := OpenSession(path); err == nil {
		t.Fatal("second writer opened session")
	}
	messages := session.Messages()
	if len(messages) != 3 || !messages[2].IsError || *messages[2].ToolCallID != "old-call" {
		t.Fatalf("old interrupted call not repaired: %#v", messages)
	}
	if _, err := os.Stat(filepath.Join(cwd, "should-not-exist")); !os.IsNotExist(err) {
		t.Fatal("tool was replayed")
	}
	if err := session.Branch("tool000"); err != nil {
		t.Fatal(err)
	}
	if session.Messages()[2].Content != "original result" {
		t.Fatal("old branch lost")
	}
	if err := session.Checkpoint([]Message{{Role: "summary", Content: "summary"}}); err != nil {
		t.Fatal(err)
	}
	if err := session.Rename("以后再看"); err != nil {
		t.Fatal(err)
	}
	if _, err := catalog.Move("0123", false); err == nil {
		t.Fatal("archived an open session")
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	archived, err := catalog.Move("0123", false)
	if err != nil {
		t.Fatal(err)
	}
	list, err := catalog.List(true)
	if err != nil || len(list) != 1 || list[0].Title != "以后再看" || list[0].Path != archived || list[0].Size <= int64(len(legacy)) {
		t.Fatalf("incorrect archive metadata: %+v %v", list, err)
	}
	back, err := catalog.Move("0123", true)
	if err != nil || back != path {
		t.Fatalf("restore: %s %v", back, err)
	}
	reopened, err := OpenSession(back)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if messages := reopened.Messages(); len(messages) != 1 || messages[0].Role != "summary" {
		t.Fatalf("checkpoint was not restored: %#v", messages)
	}
	if err := reopened.Branch("user000"); err != nil {
		t.Fatal(err)
	}
	if reopened.Messages()[0].Content != "原始问题" {
		t.Fatal("old node lost after checkpoint")
	}
}

func TestSessionProtectsCorruptFilesAndSnapshotsArguments(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "session.jsonl")
	session, err := CreateSession(dir, path)
	if err != nil {
		t.Fatal(err)
	}
	args := Object{"path": "original"}
	message := Message{Role: "assistant", ToolCalls: []ToolCall{{"a", "read", args}}}
	if _, err := session.Append(message); err != nil {
		t.Fatal(err)
	}
	args["path"] = "mutated"
	if session.Messages()[0].ToolCalls[0].Arguments["path"] != "original" {
		t.Fatal("history uses mutable arguments")
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := session.Append(message); err == nil {
		t.Fatal("closed store accepted a silent write")
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.WriteString("{\"unfinished\":")
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	if _, err := OpenSession(path); err == nil || !strings.Contains(err.Error(), "file left unchanged") {
		t.Fatalf("expected corrupt tail error, got %v", err)
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("corrupt history changed")
	}
}
