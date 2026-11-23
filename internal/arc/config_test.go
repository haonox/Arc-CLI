package arc

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// writeFixture creates a private fixture and its parent directory.
func writeFixture(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestConfigBindsCredentialsAndKeepsProjectUntrusted(t *testing.T) {
	cwd, err := Workspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	environment := map[string]string{"XDG_CONFIG_HOME": t.TempDir(), "DEEPSEEK_API_KEY": "synthetic"}
	getenv := func(name string) string { return environment[name] }
	defaults, err := LoadConfig(cwd, "", "", getenv)
	if err != nil || defaults.Model != "deepseek-flash" || defaults.BaseURL != DeepSeekBaseURL || defaults.apiKey != "synthetic" || defaults.Autonomous {
		t.Fatalf("unexpected defaults: model=%s endpoint=%s err=%v", defaults.Model, defaults.BaseURL, err)
	}
	other, err := LoadConfig(cwd, "other", "https://other.invalid/v1", getenv)
	if err != nil || other.apiKey != "" {
		t.Fatalf("credential reused by another endpoint: %v", err)
	}
	writeFixture(t, filepath.Join(cwd, ".arc", "config.toml"), "[provider]\nbase_url = \"https://other.invalid/v1\"\n")
	if _, err := LoadConfig(cwd, "", "", getenv); err == nil || !strings.Contains(err.Error(), "project base_url") {
		t.Fatalf("project changed the endpoint: %v", err)
	}
	writeFixture(t, filepath.Join(cwd, ".arc", "config.toml"), "[runtime]\nautonomous_workspaces = [\"/\"]\n")
	if _, err := LoadConfig(cwd, "", "", getenv); err == nil {
		t.Fatal("project altered policy")
	}
}

func TestNamedAndLegacyProviderConfigurations(t *testing.T) {
	cwd, err := Workspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	environment := map[string]string{"XDG_CONFIG_HOME": t.TempDir(), "ARC_MODEL": "env-model",
		"ARC_BASE_URL": "https://legacy.invalid/v1", "ARC_API_KEY": "old", "SERVICE_CREDENTIAL": "new"}
	getenv := func(name string) string { return environment[name] }
	userPath := filepath.Join(ConfigHome(getenv), "config.toml")
	writeFixture(t, userPath, "base_url = \"https://different.invalid/v1\"\nmodel = \"user-model\"\n")
	config, err := LoadConfig(cwd, "", "", getenv)
	if err != nil || config.Model != "user-model" || config.apiKey != "" {
		t.Fatalf("unexpected legacy override: %v", err)
	}
	writeFixture(t, userPath, "[providers.work]\nbase_url = \"https://work.invalid/v1\"\napi_key_env = \"SERVICE_CREDENTIAL\"\nmodel = \"work-model\"\n[runtime]\ncontext_budget = 4096\nauto_compact = false\nautonomous_workspaces = [\""+cwd+"\"]")
	project := filepath.Join(cwd, ".arc", "config.toml")
	writeFixture(t, project, "[provider]\nname = \"work\"\n")
	config, err = LoadConfig(cwd, "", "", getenv)
	if err != nil || config.Model != "work-model" || config.apiKey != "new" || config.ContextBudget != 4096 ||
		config.AutoCompact || !config.Autonomous || !slices.Contains(config.CredentialNames, "SERVICE_CREDENTIAL") {
		t.Fatalf("named provider or workspace preferences failed: %v", err)
	}
	writeFixture(t, project, "[provider]\napi_key=\"canary\"\n")
	if _, err := LoadConfig(cwd, "", "", getenv); err == nil || strings.Contains(err.Error(), "canary") {
		t.Fatalf("expected redacted error, got %v", err)
	}
}

func TestInstructionLayersAndDotenv(t *testing.T) {
	cwd := t.TempDir()
	getenv := func(string) string { return filepath.Join(cwd, "config") }
	writeFixture(t, filepath.Join(cwd, ".env"), "ARC_MODEL=dotenv\n")
	writeFixture(t, filepath.Join(ConfigHome(getenv), "AGENTS.md"), "my preference")
	writeFixture(t, filepath.Join(cwd, "AGENTS.md"), "project text")
	writeFixture(t, filepath.Join(cwd, "ARC.md"), "internal text")
	messages, err := InstructionContext(cwd, "extra", true, getenv)
	if err != nil || len(messages) != 3 || !strings.Contains(messages[0].Content, "my preference") ||
		!strings.Contains(messages[1].Content, "extra") || !strings.Contains(messages[2].Content, "untrusted") {
		t.Fatalf("wrong instruction layering: %v %v", messages, err)
	}
	messages, err = InstructionContext(cwd, "", false, getenv)
	if err != nil || len(messages) != 1 {
		t.Fatalf("global preferences disabled: %v %v", messages, err)
	}
	if config, err := LoadConfig(cwd, "", "", func(name string) string {
		if name == "XDG_CONFIG_HOME" {
			return ConfigHome(getenv)
		}
		return ""
	}); err != nil || config.Model != "" {
		t.Fatalf("unexpected dotenv configuration: %v", err)
	}
}
