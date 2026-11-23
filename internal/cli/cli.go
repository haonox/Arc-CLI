// Package cli assembles the runtime and adapts it to terminal and JSONL clients.
package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/haonox/Arc-CLI/internal/arc"
	"github.com/charmbracelet/x/term"
)

type options struct {
	mode, model, baseURL, profile, cwd                     string
	session, resume, tools, policy                         string
	thinking, effort, instructions                         string
	printMode, continueSession, noSession, toolsSupplied   bool
	noContext, noColor, noMarkdown, noAutoCompact, version bool
	contextBudget, maxTurns, maxTokens                     int
	contextSupplied, tokensSupplied                        bool
	timeout                                                float64
	allowToolEnv, denyToolEnv                              []string
}

func (o *options) parse(args []string, output io.Writer) ([]string, error) {
	flags := flag.NewFlagSet("arc", flag.ContinueOnError)
	flags.SetOutput(output)
	flags.Usage = func() {
		fmt.Fprintln(output, "Usage: arc [options] [prompt ...]\n       arc sessions [--cwd DIR] {list,rename,archive,restore} ...\n\nArc CLI: terminal-native agent runtime")
		flags.PrintDefaults()
	}
	flags.BoolVar(&o.printMode, "p", false, "run once and exit")
	flags.BoolVar(&o.printMode, "print", false, "run once and exit")
	flags.StringVar(&o.mode, "mode", "text", "output mode: text or json")
	flags.StringVar(&o.model, "model", "", "model name")
	flags.StringVar(&o.baseURL, "base-url", "", "model service API root URL")
	flags.StringVar(&o.profile, "profile", "developer", "agent profile")
	flags.StringVar(&o.cwd, "cwd", "", "workspace directory")
	flags.StringVar(&o.session, "session", "", "open or create a session file")
	flags.StringVar(&o.resume, "resume", "", "resume a saved session by ID prefix")
	flags.BoolVar(&o.continueSession, "c", false, "resume the latest session")
	flags.BoolVar(&o.continueSession, "continue", false, "resume the latest session")
	flags.BoolVar(&o.noSession, "no-session", false, "keep history in memory")
	flags.StringVar(&o.tools, "tools", "", "comma-separated allowlist, or none")
	flags.StringVar(&o.policy, "policy", "", "authorization mode: confirm, autonomous, restricted")
	flags.Func("allow-tool-env", "pass an explicitly named sensitive variable to tools", func(name string) error {
		o.allowToolEnv = append(o.allowToolEnv, name)
		return nil
	})
	flags.Func("deny-tool-env", "remove an additional environment variable from tools", func(name string) error {
		o.denyToolEnv = append(o.denyToolEnv, name)
		return nil
	})
	flags.IntVar(&o.maxTurns, "max-turns", 20, "maximum model turns per task")
	flags.IntVar(&o.contextBudget, "context-budget", 0, "estimated input token budget")
	flags.BoolVar(&o.noAutoCompact, "no-auto-compact", false, "disable automatic summarization")
	flags.StringVar(&o.thinking, "thinking", "", "DeepSeek thinking mode: enabled or disabled")
	flags.StringVar(&o.effort, "reasoning-effort", "", "reasoning effort: none, low, high, max")
	flags.IntVar(&o.maxTokens, "max-tokens", 0, "model completion token limit")
	flags.IntVar(&o.maxTokens, "max-output-tokens", 0, "alias for --max-tokens")
	flags.Float64Var(&o.timeout, "timeout", 60, "model inactivity timeout, in seconds")
	flags.StringVar(&o.instructions, "instructions", "", "user instructions for this run")
	flags.StringVar(&o.instructions, "system", "", "alias for --instructions")
	flags.BoolVar(&o.noContext, "no-context", false, "skip workspace AGENTS.md")
	flags.BoolVar(&o.noColor, "no-color", false, "disable ANSI colors")
	flags.BoolVar(&o.noMarkdown, "no-markdown", false, "show raw Markdown")
	flags.BoolVar(&o.version, "version", false, "show Arc's version")
	if err := parseInterspersed(flags, args); err != nil {
		return nil, err
	}
	flags.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "tools":
			o.toolsSupplied = true
		case "context-budget":
			o.contextSupplied = true
		case "max-tokens", "max-output-tokens":
			o.tokensSupplied = true
		}
	})
	return flags.Args(), nil
}

// Keep prompts after options for flag.FlagSet, including prompts given before a flag.
func parseInterspersed(set *flag.FlagSet, args []string) error {
	var options, positionals []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			positionals = append(positionals, args[i+1:]...)
			break
		}
		if arg == "-" || !strings.HasPrefix(arg, "-") {
			positionals = append(positionals, arg)
			continue
		}
		options = append(options, arg)
		name, _, hasValue := strings.Cut(strings.TrimLeft(arg, "-"), "=")
		f := set.Lookup(name)
		if f == nil || hasValue {
			continue
		}
		isBool, ok := f.Value.(interface{ IsBoolFlag() bool })
		if ok && isBool.IsBoolFlag() {
			continue
		}
		if i+1 < len(args) {
			i++
			options = append(options, args[i])
		}
	}
	return set.Parse(append(append(options, "--"), positionals...))
}

func (o options) validate() error {
	selected := 0
	for _, chosen := range []bool{o.session != "", o.resume != "", o.noSession, o.continueSession} {
		if chosen {
			selected++
		}
	}
	if selected > 1 {
		return errors.New("--session, --resume, --continue and --no-session are mutually exclusive")
	}
	if o.mode != "text" && o.mode != "json" {
		return errors.New("--mode must be text or json")
	}
	if o.profile != "developer" {
		return fmt.Errorf("unknown profile %q", o.profile)
	}
	if !slices.Contains([]string{"", arc.Confirm, arc.Autonomous, arc.Restricted}, o.policy) {
		return errors.New("--policy must be confirm, autonomous or restricted")
	}
	if o.maxTurns < 1 || math.IsNaN(o.timeout) || o.timeout*float64(time.Second) < 1 || o.timeout >= float64(math.MaxInt64)/float64(time.Second) {
		return errors.New("limits must be positive and within range")
	}
	if o.contextSupplied && o.contextBudget < 1024 {
		return errors.New("--context-budget must be at least 1024")
	}
	if o.tokensSupplied && (o.maxTokens < 1 || o.maxTokens > 393216) {
		return errors.New("--max-tokens must be between 1 and 393216")
	}
	return nil
}

func isTerminal(stream any) bool {
	file, ok := stream.(interface{ Fd() uintptr })
	return ok && term.IsTerminal(file.Fd())
}

func Run(ctx context.Context, args []string, in io.Reader, out, stderr io.Writer) int {
	if len(args) > 0 && args[0] == "sessions" {
		code, err := runCatalog(args[1:], out)
		if err != nil {
			fmt.Fprintln(stderr, "arc sessions:", safeText(err.Error()))
		}
		return code
	}
	var opts options
	var usage strings.Builder
	promptArgs, err := opts.parse(args, &usage)
	if errors.Is(err, flag.ErrHelp) {
		fmt.Fprint(out, usage.String())
		return 0
	}
	if err == nil && opts.version {
		fmt.Fprintln(out, "Arc CLI "+arc.Version)
		return 0
	}
	if err == nil {
		err = opts.validate()
	}
	if err != nil {
		writeAppError(opts.mode, err, out, stderr)
		return 2
	}
	code, err := launch(ctx, opts, strings.Join(promptArgs, " "), in, out, stderr)
	if err != nil {
		writeAppError(opts.mode, err, out, stderr)
	}
	return code
}

func writeAppError(mode string, err error, out, stderr io.Writer) {
	if mode == "json" {
		_ = json.NewEncoder(out).Encode(arc.Event{Type: "error", Data: arc.Object{"message": err.Error()}})
	} else {
		fmt.Fprintln(stderr, "arc:", safeText(err.Error()))
	}
}

func openSession(opts options, cwd string) (*arc.Session, string, error) {
	catalog := arc.NewCatalog(cwd, os.Getenv)
	path := opts.session
	var err error
	switch {
	case opts.continueSession:
		path, err = catalog.Latest()
	case opts.resume != "":
		path, err = catalog.Find(opts.resume, false)
	case path != "":
		path, err = filepath.Abs(path)
	}
	if err != nil || path == "" {
		return nil, path, err
	}
	session, err := arc.OpenSession(path)
	if errors.Is(err, os.ErrNotExist) && opts.session != "" {
		return nil, path, nil
	}
	return session, path, err
}

func launch(ctx context.Context, o options, prompt string, in io.Reader, out, stderr io.Writer) (int, error) {
	inputTTY, outputTTY := isTerminal(in), isTerminal(out)
	oneShot := o.printMode || o.mode == "json" || !inputTTY || !outputTTY
	if !inputTTY {
		text, err := io.ReadAll(in)
		if err != nil {
			return 2, err
		}
		prompt = strings.TrimSpace(prompt)
		if piped := strings.TrimSpace(string(text)); piped != "" {
			if prompt != "" {
				prompt += "\n\n"
			}
			prompt += piped
		}
	}
	if oneShot && strings.TrimSpace(prompt) == "" {
		return 2, errors.New("a prompt is required (argument or stdin)")
	}
	cwd, err := arc.Workspace(o.cwd)
	if err != nil {
		return 2, err
	}
	session, path, err := openSession(o, cwd)
	if err != nil {
		return 2, err
	}
	if session != nil {
		defer session.Close()
		saved, err := arc.Workspace(session.Info().CWD)
		if err != nil {
			return 2, errors.New("saved session cwd no longer exists")
		}
		if o.cwd != "" && cwd != saved {
			return 2, errors.New("--cwd differs from saved session cwd; start a new session instead")
		}
		cwd = saved
	}
	config, err := arc.LoadConfig(cwd, o.model, o.baseURL, os.Getenv)
	if err != nil {
		return 2, err
	}
	if config.Model == "" {
		return 2, errors.New("set DEEPSEEK_API_KEY or configure a model with ARC_MODEL or --model")
	}
	if config.BaseURL == arc.DefaultBaseURL && !config.HasCredential() {
		return 2, errors.New("set ARC_API_KEY in the environment for the default OpenAI endpoint")
	}
	provider, err := arc.NewProvider(config, arc.ProviderOptions{
		Timeout: time.Duration(o.timeout * float64(time.Second)), MaxTokens: o.maxTokens, Thinking: o.thinking, ReasoningEffort: o.effort,
	})
	if err != nil {
		return 2, err
	}
	defer provider.Close()
	tools, err := arc.SelectTools(o.tools, o.toolsSupplied)
	if err != nil {
		return 2, err
	}
	instructions, err := arc.InstructionContext(cwd, o.instructions, !o.noContext, os.Getenv)
	if err != nil {
		return 2, err
	}
	catalog := arc.NewCatalog(cwd, os.Getenv)
	if session == nil {
		if !o.noSession && path == "" {
			path = catalog.NewPath()
		}
		session, err = arc.CreateSession(cwd, path)
		if err != nil {
			return 2, err
		}
		defer session.Close()
	}
	policy := o.policy
	if policy == "" {
		policy = arc.Confirm
		if config.Autonomous {
			policy = arc.Autonomous
		}
	}
	budget := config.ContextBudget
	if o.contextSupplied {
		budget = o.contextBudget
	}
	runner := &arc.Agent{
		Provider: provider, Session: session, Tools: tools,
		Environment:        arc.ToolEnv{CWD: cwd, Env: arc.ToolEnvironment(os.Environ(), o.allowToolEnv, o.denyToolEnv, config.CredentialNames)},
		SystemPrompt:       func() string { return arc.SystemPrompt(cwd, o.profile) },
		InstructionContext: instructions, Policy: policy, MaxTurns: o.maxTurns,
		ContextBudget: budget, AutoCompact: config.AutoCompact && !o.noAutoCompact,
	}
	_, noColorEnv := os.LookupEnv("NO_COLOR")
	color := !o.noColor && !noColorEnv && outputTTY && os.Getenv("TERM") != "dumb"
	if oneShot {
		return runOnce(ctx, runner, prompt, o.mode, out, stderr), nil
	}
	history := ""
	if path != "" {
		history = catalog.HistoryPath()
	}
	return runTerminal(ctx, runner, provider, in, out, prompt, history, color, color && !o.noMarkdown)
}

func runOnce(ctx context.Context, agent *arc.Agent, prompt, mode string, out, stderr io.Writer) int {
	encoder := json.NewEncoder(out)
	encoder.SetEscapeHTML(false)
	var textWritten, endedLine bool
	var tool arc.ToolCall
	status, _ := agent.Run(ctx, prompt, func(event arc.Event) error {
		if mode == "json" {
			return encoder.Encode(event)
		}
		switch event.Type {
		case "message_update":
			delta := safeText(event.Data["delta"].(string))
			textWritten, endedLine = true, strings.HasSuffix(delta, "\n")
			_, err := io.WriteString(out, delta)
			return err
		case "tool_execution_start":
			tool = event.Data["tool_call"].(arc.ToolCall)
		case "tool_execution_end":
			_, err := fmt.Fprintf(stderr, "%s · %s · %s\n", singleLine(tool.Name), event.Data["status"], singleLine(plainToolSummary(event, 0)))
			return err
		case "context_budget":
			if label := contextLabel(event.Data["status"].(string)); label != "" {
				_, err := fmt.Fprintln(stderr, "context │ "+label)
				return err
			}
		case "error":
			_, err := fmt.Fprintln(stderr, "error │ "+safeText(event.Data["message"].(string)))
			return err
		}
		return nil
	})
	if mode == "text" && textWritten && !endedLine {
		fmt.Fprintln(out)
	}
	if ctx.Err() != nil {
		return 130
	}
	if status == "complete" {
		return 0
	}
	return 1
}

func runCatalog(args []string, out io.Writer) (int, error) {
	flags := flag.NewFlagSet("arc sessions", flag.ContinueOnError)
	var output strings.Builder
	flags.SetOutput(&output)
	flags.Usage = func() {
		fmt.Fprintln(&output, "Usage: arc sessions [--cwd DIR] list [--archived]\n       arc sessions [--cwd DIR] rename ID TITLE\n       arc sessions [--cwd DIR] {archive,restore} ID")
		flags.PrintDefaults()
	}
	cwd := flags.String("cwd", "", "workspace whose sessions to manage")
	archived := flags.Bool("archived", false, "list archived sessions")
	if err := parseInterspersed(flags, args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprint(out, output.String())
			return 0, nil
		}
		return 2, err
	}
	pos := flags.Args()
	if len(pos) == 0 {
		flags.Usage()
		return 2, errors.New(strings.TrimSpace(output.String()))
	}
	workspace, err := arc.Workspace(*cwd)
	if err != nil {
		return 2, err
	}
	catalog := arc.NewCatalog(workspace, os.Getenv)
	switch pos[0] {
	case "list":
		if len(pos) != 1 {
			return 2, errors.New("usage: arc sessions list [--archived]")
		}
		records, err := catalog.List(*archived)
		if err != nil {
			return 2, err
		}
		if len(records) == 0 {
			text := "No saved sessions."
			if *archived {
				text = "No archived sessions."
			}
			fmt.Fprintln(out, text)
			return 0, nil
		}
		fmt.Fprintln(out, "SESSION ID          LAST MODIFIED      SIZE (KiB)  TITLE")
		for _, record := range records {
			title := record.Title
			if title == "" {
				title = "(untitled)"
			}
			fmt.Fprintln(out, safeText(fmt.Sprintf("%-18.16s  %s  %10.1f  %s", record.ID, record.Modified.Format("2006-01-02 15:04"), float64(record.Size)/1024, title)))
		}
	case "rename":
		if len(pos) < 3 || *archived {
			return 2, errors.New("usage: arc sessions rename ID TITLE")
		}
		path, err := catalog.Find(pos[1], false)
		if err != nil {
			return 2, err
		}
		session, err := arc.OpenSession(path)
		if err != nil {
			return 2, err
		}
		defer session.Close()
		if err := session.Rename(strings.Join(pos[2:], " ")); err != nil {
			return 2, err
		}
		info := session.Info()
		fmt.Fprintln(out, safeText(fmt.Sprintf("Renamed %.16s to %s", info.ID, info.Title)))
	case "archive", "restore":
		if len(pos) != 2 || *archived {
			return 2, fmt.Errorf("usage: arc sessions %s ID", pos[0])
		}
		restore := pos[0] == "restore"
		path, err := catalog.Move(pos[1], restore)
		if err != nil {
			return 2, err
		}
		action := "Archived"
		if restore {
			action = "Restored"
		}
		fmt.Fprintln(out, action, safeText(filepath.Base(path)))
	default:
		return 2, fmt.Errorf("unknown session command %q", pos[0])
	}
	return 0, nil
}
