package arc

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestFileToolsSchemaBoundariesAndPatchRollback(t *testing.T) {
	cwd, err := Workspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	registry, err := NewRegistry(BuiltinTools())
	if err != nil {
		t.Fatal(err)
	}
	execute := func(name string, args Object) ToolResult {
		tool, err := registry.validate(ToolCall{"id", name, args})
		if err != nil {
			return failed(err)
		}
		return tool.Execute(t.Context(), args, ToolEnv{CWD: cwd}, func(string) error { return nil })
	}
	note := filepath.Join(cwd, "note")
	writeFixture(t, note, "世界\nold\nold\n")
	got := execute("read", Object{"path": note, "offset": 2, "limit": 1})
	if got.IsError || got.Content != "2: old\n[More lines: use offset=3]" {
		t.Fatalf("read result: %+v", got)
	}
	got = execute("edit", Object{"path": "note", "old_text": "old", "new_text": "new"})
	if !got.IsError || !strings.Contains(got.Content, "matches 2 times") {
		t.Fatalf("ambiguous edit: %+v", got)
	}
	if got = execute("edit", Object{"path": "note", "old_text": "old", "new_text": "new", "replace_all": true}); got.IsError {
		t.Fatal(got)
	}
	if got = execute("read", Object{"path": "note", "extra": "x"}); !got.IsError {
		t.Fatal("schema accepted unknown field")
	}
	outside := filepath.Join(t.TempDir(), "outside")
	writeFixture(t, outside, "protected")
	if err := os.Symlink(outside, filepath.Join(cwd, "link")); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{outside, "link", "../outside"} {
		if got := execute("write", Object{"path": path, "content": "x"}); !got.IsError {
			t.Fatalf("allowed write to %q", path)
		}
	}
	if data, _ := os.ReadFile(outside); string(data) != "protected" {
		t.Fatal("outside file changed")
	}
	// Both add targets are initially absent. Committing the parent as a file
	// makes the child operation fail after earlier changes have been applied.
	got = execute("apply_patch", Object{"operations": []any{
		Object{"type": "update", "path": "note", "edits": []any{Object{"old_text": "世界", "new_text": "changed"}}},
		Object{"type": "add", "path": "collision", "content": "parent"},
		Object{"type": "add", "path": "collision/child", "content": "child"},
	}})
	if !got.IsError || !strings.Contains(got.Content, "patch failed") {
		t.Fatalf("expected commit failure: %+v", got)
	}
	if content, _ := os.ReadFile(note); string(content) != "世界\nnew\nnew\n" {
		t.Fatalf("rollback left %q", content)
	}
	if _, err := os.Stat(filepath.Join(cwd, "collision")); !os.IsNotExist(err) {
		t.Fatal("rollback left newly added file")
	}
	got = execute("apply_patch", Object{"operations": []any{
		Object{"type": "add", "path": "new", "content": "hello"},
		Object{"type": "delete", "path": "note"},
	}})
	if got.IsError || !strings.Contains(got.Content, "Applied 2") {
		t.Fatalf("patch result: %+v", got)
	}
}

func TestBashTimeoutAndCancellationReapDescendants(t *testing.T) {
	for _, mode := range []string{"timeout", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			args := Object{"command": "(sleep 0.25; touch leaked) & printf 'ready\\n'; wait", "timeout": 0.12}
			if mode == "cancel" {
				args["timeout"] = 3
			}
			result := runBash(ctx, args, ToolEnv{CWD: dir, Env: os.Environ()}, func(text string) error {
				if mode == "cancel" && strings.Contains(text, "ready") {
					cancel()
				}
				return nil
			})
			if !result.IsError {
				t.Fatalf("expected interrupted Bash: %+v", result)
			}
			select {
			case <-time.After(350 * time.Millisecond):
			case <-t.Context().Done():
				t.Fatal("test canceled")
			}
			if _, err := os.Stat(filepath.Join(dir, "leaked")); !os.IsNotExist(err) {
				t.Fatal("descendant survived")
			}
		})
	}
}

func TestShellEnvironmentAndEffects(t *testing.T) {
	env := ToolEnvironment([]string{"PATH=/bin", "ARC_API_KEY=fake", "DEEPSEEK_API_KEY=fake", "SERVICE_CREDENTIAL=fake", "APP_TOKEN=allowed", "PASSWD=hidden"},
		[]string{"ARC_API_KEY", "DEEPSEEK_API_KEY", "SERVICE_CREDENTIAL", "APP_TOKEN"}, nil, []string{"SERVICE_CREDENTIAL"})
	if !reflect.DeepEqual(env, []string{"PATH=/bin", "APP_TOKEN=allowed"}) {
		t.Fatalf("unexpected environment names: %v", env)
	}
	if got := bashEffects(Object{"command": "cat ~/.ssh/config"}); !reflect.DeepEqual(got, []string{"process"}) {
		t.Fatal(got)
	}
	if got := bashEffects(Object{"command": "echo test && git fetch"}); !reflect.DeepEqual(got, []string{"process", "external"}) {
		t.Fatal(got)
	}
	result := runBash(t.Context(), Object{"command": "printf '%025000d' 1"}, ToolEnv{CWD: t.TempDir(), Env: env}, func(string) error { return nil })
	if result.IsError || !strings.Contains(result.Content, "[Output truncated at 24000 characters]") {
		t.Fatalf("output was not bounded: error=%v size=%d", result.IsError, len(result.Content))
	}
}
