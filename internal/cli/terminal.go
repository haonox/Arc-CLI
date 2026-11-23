package cli

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
	"github.com/haonox/Arc-CLI/internal/arc"
)

const help = `Enter a task at ›. While Arc is running, text entered at ↳ steers the next model turn.
/help              Show this help
/abort             Cancel the running task
/approve, /deny    Answer the current tool approval request
/steer TEXT        Add instructions before the next model turn
/follow-up TEXT    Queue a task after the current tool loop finishes
/status            Show model, endpoint, session, branch, tools, policy and state
/last-tool [N]     Show the latest full tool result, optionally limited to N lines
/history           Show the current branch as message summaries
/tree              Show the session tree; ● marks the current node
/branch ID         Continue from a node prefix; disk state is retained
/new               Start from empty context while retaining existing branches
/compact [N]       Summarize older messages, keeping N recent user turns (default 2)
/session           Show the session file and title
/rename TITLE      Name the current saved session
/tools             Show enabled tools
/model [NAME]      Show or change the model while idle
/clear             Clear the terminal
/quit              Exit; Ctrl-D also exits and Ctrl-C aborts a running task`

var commands = []string{
	"/abort", "/approve", "/branch", "/clear", "/compact", "/deny", "/follow-up", "/help", "/history", "/last-tool",
	"/model", "/new", "/quit", "/rename", "/session", "/status", "/steer", "/tools", "/tree",
}

// approvalRequest carries a proposal from the agent worker and a channel
// for the UI's decision.
type approvalRequest struct {
	request arc.Authorization
	answer  chan bool
}

// taskFinished reports that a worker has returned from a run or compaction.
type taskFinished struct {
	status     string
	err        error
	compaction bool
}

// spinnerTick identifies the task that scheduled the animation update.
type spinnerTick struct{ task int }

// toolSnapshot retains the last completed tool's call and result for display.
type toolSnapshot struct {
	call  arc.ToolCall
	event arc.Event
}

// terminal is the Bubble Tea model. The event loop owns its mutable state.
// An agent worker sends events and receives approval decisions over channels.
type terminal struct {
	programCtx    context.Context
	agent         *arc.Agent
	provider      *arc.OpenAIProvider
	input         textinput.Model
	historyPath   string
	history       []string
	historyIndex  int // Negative until a prefix search starts.
	historyPrefix string
	initial       string

	width             int
	color, markdown   bool
	inFence           bool
	partial           string // Answer text after the last committed newline.
	activity          string
	activitySince     time.Time
	frame, turn, task int
	pending           *approvalRequest
	tool              arc.ToolCall
	last              *toolSnapshot
	usage             *arc.Usage
	total             arc.Usage

	events         chan tea.Msg
	workerDone     chan struct{} // Closed when the current worker exits.
	cancelTask     context.CancelFunc
	busy, quitting bool
}

// runTerminal runs the interactive UI and waits for an active worker to
// finish before the caller releases its session and provider.
func runTerminal(ctx context.Context, agent *arc.Agent, provider *arc.OpenAIProvider, in io.Reader, out io.Writer,
	prompt, history string, color, markdown bool) (int, error) {
	ctx, cancel := context.WithCancel(ctx)
	ui := &terminal{
		programCtx: ctx, agent: agent, provider: provider, input: textinput.New(), width: 80,
		historyPath: history, historyIndex: -1, initial: prompt, color: color, markdown: markdown,
	}
	if err := ui.loadHistory(); err != nil {
		cancel()
		return 2, err
	}
	// The process entry point already owns OS signal handling.
	program := tea.NewProgram(ui, tea.WithInput(in), tea.WithOutput(out), tea.WithContext(ctx), tea.WithoutSignalHandler())
	_, err := program.Run()
	cancel()
	if ui.cancelTask != nil {
		ui.cancelTask()
	}
	if ui.workerDone != nil {
		<-ui.workerDone
	}
	if errors.Is(err, tea.ErrProgramKilled) && ctx.Err() != nil {
		return 130, nil
	}
	if err != nil {
		return 2, err
	}
	return 0, nil
}

// Init focuses input, prints the banner, and starts an optional initial prompt.
func (t *terminal) Init() tea.Cmd {
	focus := t.input.Focus()
	banner := t.paint("ARC "+arc.Version+" · "+safeText(t.provider.Model())+" · "+safeText(t.agent.Environment.CWD), "36")
	tip := t.paint("› task   ↳ steer   /help commands", "2")
	if strings.TrimSpace(t.initial) != "" {
		return tea.Sequence(tea.Println(banner+"\n"+tip+"\n"), focus, t.start(t.initial, 0))
	}
	return tea.Sequence(tea.Println(banner+"\n"+tip+"\n"), focus)
}

// waitEvent turns one worker message into a Bubble Tea command so waiting
// for it happens outside the UI update loop.
func waitEvent(events <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		return <-events
	}
}

func tick(task int) tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(time.Time) tea.Msg { return spinnerTick{task: task} })
}

// start launches a task worker with a separate cancellation scope. A positive
// keep value selects manual compaction.
func (t *terminal) start(prompt string, keep int) tea.Cmd {
	ctx, cancel := context.WithCancel(t.programCtx)
	t.cancelTask, t.busy = cancel, true
	t.task++
	t.turn, t.frame = 0, 0
	t.partial, t.inFence, t.usage, t.total = "", false, nil, arc.Usage{}
	t.setActivity("thinking")
	if keep > 0 {
		t.setActivity("compacting")
	}
	t.events, t.workerDone = make(chan tea.Msg, 32), make(chan struct{})
	events, done := t.events, t.workerDone
	send := func(message tea.Msg) error {
		// Use the program context so canceling a task can still deliver its
		// final events while the UI is running.
		select {
		case events <- message:
			return nil
		case <-t.programCtx.Done():
			return t.programCtx.Err()
		}
	}
	t.agent.Approve = func(callCtx context.Context, request arc.Authorization) (bool, error) {
		// Buffer the decision so the UI stays responsive if the call is
		// canceled just as the user answers.
		answer := make(chan bool, 1)
		if err := send(approvalRequest{request, answer}); err != nil {
			return false, err
		}
		select {
		case decision := <-answer:
			return decision, nil
		case <-callCtx.Done():
			return false, callCtx.Err()
		}
	}
	go func() {
		defer close(done)
		defer close(events)
		defer cancel()
		if keep > 0 {
			err := t.agent.Compact(ctx, keep)
			_ = send(taskFinished{err: err, compaction: true})
			return
		}
		status, _ := t.agent.Run(ctx, prompt, func(event arc.Event) error { return send(event) })
		_ = send(taskFinished{status: status})
	}()
	return tea.Batch(waitEvent(events), tick(t.task))
}

// Update handles input and worker messages on the UI loop. It returns
// commands for waiting on the worker or writing scrollback lines.
func (t *terminal) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		t.width = msg.Width
		t.input.Width = max(1, msg.Width-4)
		return t, nil
	case spinnerTick:
		// Ignore a timer left over from a task that already finished.
		if t.busy && msg.task == t.task {
			t.frame++
			return t, tick(t.task)
		}
		return t, nil
	case arc.Event:
		line := t.handleEvent(msg)
		if line != "" {
			// Commit this line before consuming the next event.
			return t, tea.Sequence(tea.Println(line), waitEvent(t.events))
		}
		return t, waitEvent(t.events)
	case approvalRequest:
		t.pending = &msg
		t.setActivity("")
		return t, tea.Sequence(tea.Println(t.approvalText(msg.request)), waitEvent(t.events))
	case taskFinished:
		t.busy, t.pending, t.cancelTask, t.workerDone = false, nil, nil, nil
		t.setActivity("")
		if t.quitting {
			return t, tea.Quit
		}
		if msg.compaction {
			if msg.err == nil {
				return t, tea.Println(t.paint("compacted │ original history retained", "32"))
			}
			if errors.Is(msg.err, context.Canceled) {
				return t, tea.Println(t.paint("aborted", "33"))
			}
			return t, t.errorLine(msg.err)
		}
		if msg.status == "complete" {
			return t, tea.Println("")
		}
		if msg.status == "aborted" {
			return t, tea.Println(t.paint("aborted", "33"))
		}
		return t, nil
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			if t.cancelTask != nil {
				t.cancelTask()
				t.pending = nil
				return t, nil
			}
			t.input.Reset()
			return t, tea.Println(t.paint("aborted", "33"))
		case "ctrl+d":
			if t.input.Value() == "" {
				return t, t.quit()
			}
		case "enter":
			return t, t.submit()
		case "up":
			t.searchHistory(-1)
			return t, nil
		case "down":
			t.searchHistory(1)
			return t, nil
		case "tab":
			return t, t.complete()
		default:
			t.historyIndex = -1
		}
	}
	var command tea.Cmd
	t.input, command = t.input.Update(msg)
	return t, command
}

func (t *terminal) setActivity(label string) {
	t.activity, t.activitySince = label, time.Now()
}

var spinnerFrames = []rune("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

// View renders the current input, partial answer, and task status for one
// Bubble Tea redraw.
func (t *terminal) View() string {
	prompt := "› "
	if t.busy {
		prompt = "↳ "
	}
	t.input.Prompt = prompt
	var rows []string
	if t.partial != "" {
		// Keep the uncommitted text in the view, where a redraw cannot overwrite input.
		rows = append(rows, t.paint(ansi.TruncateLeft(t.partial, max(0, ansi.StringWidth(t.partial)-t.width+1), "…"), "2"))
	}
	rows = append(rows, t.input.View())
	var status []string
	if t.activity != "" {
		status = append(status, fmt.Sprintf("%c %s · %.1fs", spinnerFrames[t.frame%len(spinnerFrames)], t.activity, time.Since(t.activitySince).Seconds()))
	}
	if t.pending != nil {
		status = append(status, "waiting for /approve or /deny")
	}
	if t.usage != nil {
		status = append(status, "tokens · "+usageLabel(*t.usage))
		if t.turn > 1 {
			status = append(status, fmt.Sprintf("task %d", t.total.InputTokens+t.total.OutputTokens))
		}
	}
	if len(status) > 0 {
		rows = append(rows, t.paint(ansi.Truncate(strings.Join(status, " │ "), max(1, t.width-1), "…"), "2"))
	}
	return strings.Join(rows, "\n")
}

func usageLabel(usage arc.Usage) string {
	return fmt.Sprintf("%d in · %d out · %d total", usage.InputTokens, usage.OutputTokens, usage.InputTokens+usage.OutputTokens)
}

// handleEvent updates the UI from one runtime event and returns any text
// ready to commit to scrollback.
func (t *terminal) handleEvent(event arc.Event) string {
	switch event.Type {
	case "turn_start":
		t.turn = event.Data["turn"].(int)
		t.setActivity("thinking")
	case "context_budget":
		status := event.Data["status"].(string)
		if status == "compacting" {
			t.setActivity("compacting")
		} else {
			t.setActivity("thinking")
		}
		if label := contextLabel(status); label != "" {
			return t.paint("context │ "+label, "2")
		}
	case "message_start":
		t.partial, t.inFence = "", false
	case "message_update":
		t.partial += safeText(event.Data["delta"].(string))
		if t.activity == "thinking" {
			t.activity = "responding"
		}
		if last := strings.LastIndexByte(t.partial, '\n'); last >= 0 {
			// ANSI and Markdown styling run on complete lines. Keep the
			// remaining bytes in the transient view.
			line := t.formatLines(t.partial[:last])
			t.partial = t.partial[last+1:]
			if line == "" {
				return " "
			}
			return line
		}
	case "message_end":
		message := event.Data["message"].(arc.Message)
		if message.Role == "assistant" {
			t.setActivity("")
			if message.Usage != nil {
				t.usage = message.Usage
				t.total.InputTokens += message.Usage.InputTokens
				t.total.OutputTokens += message.Usage.OutputTokens
			}
			tail := t.formatLines(t.partial)
			t.partial = ""
			return tail
		}
	case "tool_execution_start":
		t.tool = event.Data["tool_call"].(arc.ToolCall)
		t.setActivity(singleLine(t.tool.Name) + " · " + toolTarget(t.tool))
	case "tool_execution_end":
		t.setActivity("")
		t.last = &toolSnapshot{call: t.tool, event: event}
		return t.toolLine(*t.last)
	case "error":
		t.setActivity("")
		return t.paint("error │ "+safeText(event.Data["message"].(string)), "31")
	case "agent_end":
		t.setActivity("")
	}
	return ""
}

// approvalText shows the workspace and the command or deletions that need a
// decision.
func (t *terminal) approvalText(request arc.Authorization) string {
	var lines []string
	lines = append(lines, fmt.Sprintf("Approve %s in %s", request.Call.Name, safeText(request.CWD)))
	switch request.Call.Name {
	case "bash":
		for _, line := range strings.Split(request.Call.Arguments["command"].(string), "\n") {
			lines = append(lines, "  │ "+safeText(line))
		}
	case "apply_patch":
		for _, value := range request.Call.Arguments["operations"].([]any) {
			operation := value.(arc.Object)
			if operation["type"] == "delete" {
				lines = append(lines, fmt.Sprintf("  Delete %q", operation["path"]))
			}
		}
	}
	return strings.Join(append(lines, "Enter /approve or /deny."), "\n")
}

// quit requests immediate exit only when idle. Otherwise taskFinished
// completes the exit after the worker has canceled.
func (t *terminal) quit() tea.Cmd {
	if !t.busy {
		return tea.Quit
	}
	t.quitting = true
	t.pending = nil
	t.cancelTask()
	return nil
}

func (t *terminal) errorLine(err error) tea.Cmd {
	return tea.Println(t.paint("error │ "+safeText(err.Error()), "31"))
}

// submit records an input line and routes it to a command or the agent.
func (t *terminal) submit() tea.Cmd {
	text := strings.TrimSpace(t.input.Value())
	t.input.Reset()
	t.historyIndex = -1
	if text == "" {
		return nil
	}
	if err := t.saveHistory(text); err != nil {
		return t.errorLine(err)
	}
	marker := "› "
	if t.busy {
		marker = "↳ "
	}
	command, argument, _ := strings.Cut(text, " ")
	argument = strings.TrimSpace(argument)
	if command == "/quit" || command == "/exit" {
		return t.quit()
	}
	reply := t.dispatch(command, argument, text)
	return tea.Sequence(tea.Println(marker+safeText(text)), reply)
}

// dispatch handles a slash command or routes ordinary text to a new task
// or the active task's steering queue.
func (t *terminal) dispatch(command, argument, text string) tea.Cmd {
	switch command {
	case "/abort":
		if t.cancelTask != nil {
			t.cancelTask()
			t.pending = nil
		}
	case "/approve", "/deny":
		if argument != "" || t.pending == nil {
			return t.errorLine(errors.New("no pending approval, or invalid approval command"))
		}
		t.pending.answer <- (command == "/approve")
		t.pending = nil
	case "/help":
		return tea.Println(help)
	case "/status":
		return tea.Println(t.status())
	case "/clear":
		return tea.ClearScreen
	case "/history":
		var lines []string
		for _, message := range t.agent.Session.Messages() {
			role := message.Role
			if message.Name != nil {
				role += ":" + *message.Name
			}
			if message.IsError {
				role += " error"
			}
			body := singleLine(message.Content)
			if body == "" && len(message.ToolCalls) > 0 {
				var names []string
				for _, call := range message.ToolCalls {
					names = append(names, call.Name)
				}
				body = "calls " + strings.Join(names, ", ")
			}
			lines = append(lines, fmt.Sprintf("%-14s %s", singleLine(role), ansi.Truncate(singleLine(body), max(16, t.width-18), "…")))
		}
		if len(lines) == 0 {
			lines = append(lines, "(empty)")
		}
		return tea.Println(strings.Join(lines, "\n"))
	case "/tree":
		lines := t.agent.Session.Tree()
		if len(lines) == 0 {
			return tea.Println("(empty)")
		}
		return tea.Println(safeText(strings.Join(lines, "\n")))
	case "/tools":
		names := t.agent.Tools.Names()
		if len(names) == 0 {
			return tea.Println("(none)")
		}
		return tea.Println(strings.Join(names, ", "))
	case "/session":
		info := t.agent.Session.Info()
		if info.Path == "" {
			return tea.Println("(memory)")
		}
		return tea.Println(safeText(strings.TrimSpace(info.Path + "\n" + info.Title)))
	case "/rename":
		if err := t.agent.Session.Rename(argument); err != nil {
			return t.errorLine(err)
		}
		return tea.Println("Session renamed to " + safeText(t.agent.Session.Info().Title))
	case "/last-tool":
		if t.last == nil {
			return tea.Println("No tool result in this run.")
		}
		content := safeText(t.last.event.Data["content"].(string))
		if argument != "" {
			limit, err := strconv.Atoi(argument)
			if err != nil || limit < 1 {
				return t.errorLine(errors.New("line limit must be positive"))
			}
			lines := strings.Split(content, "\n")
			content = strings.Join(lines[:min(len(lines), limit)], "\n")
		}
		return tea.Println(content)
	case "/branch", "/new", "/model", "/compact":
		// These commands change the session cursor or model, or take
		// exclusive ownership of the agent for compaction.
		if t.busy {
			return t.errorLine(errors.New("wait for completion or /abort first"))
		}
		switch command {
		case "/branch", "/new":
			if command == "/new" {
				argument = "root"
			}
			if argument == "" {
				return t.errorLine(errors.New("usage: /branch ID"))
			}
			if err := t.agent.Session.Branch(argument); err != nil {
				return t.errorLine(err)
			}
		case "/model":
			if argument != "" {
				t.provider.SetModel(argument)
			}
			return tea.Println(safeText(t.provider.Model()))
		case "/compact":
			keep := 2
			if argument != "" {
				var err error
				keep, err = strconv.Atoi(argument)
				if err != nil || keep < 1 {
					return t.errorLine(errors.New("keep count must be positive"))
				}
			}
			return t.start("", keep)
		}
	case "/steer", "/follow-up":
		if argument == "" {
			return t.errorLine(errors.New("provide text after the command"))
		}
		if !t.busy {
			return t.start(argument, 0)
		}
		if command == "/steer" {
			t.agent.Steer(argument)
		} else {
			t.agent.FollowUp(argument)
		}
	default:
		if strings.HasPrefix(command, "/") {
			return t.errorLine(errors.New("unknown command; use /help"))
		}
		if !t.busy {
			return t.start(text, 0)
		}
		t.agent.Steer(text)
		return tea.Println(t.paint("queued │ next model turn", "2"))
	}
	return nil
}

// status renders a snapshot of the active configuration and task progress.
func (t *terminal) status() string {
	info := t.agent.Session.Info()
	session := info.Path
	if session == "" {
		session = "(memory)"
	}
	branch := info.Leaf
	if branch == "" {
		branch = "root"
	}
	state := "idle"
	if t.busy {
		state = "running"
	}
	lines := []string{
		"ARC", "────────────────────────────────────────────",
		fmt.Sprintf("model     %s", t.provider.Model()),
		fmt.Sprintf("endpoint  %s", t.provider.Endpoint()),
		fmt.Sprintf("cwd       %s", t.agent.Environment.CWD),
		fmt.Sprintf("session   %s", session),
		fmt.Sprintf("branch    %.8s", branch),
		fmt.Sprintf("messages  %d", len(t.agent.Session.Messages())),
		fmt.Sprintf("tools     %s", strings.Join(t.agent.Tools.Names(), " ")),
		fmt.Sprintf("policy    %s", t.agent.Policy),
		fmt.Sprintf("context   %d estimated input tokens", t.agent.ContextBudget),
		fmt.Sprintf("state     %s", state),
	}
	if info.Title != "" {
		lines = append(lines, "title     "+info.Title)
	}
	if opts := t.provider.Options(); opts.Thinking != "" {
		lines = append(lines, "thinking  "+opts.Thinking, "effort    "+opts.ReasoningEffort)
	}
	if t.usage != nil {
		lines = append(lines, "usage     "+usageLabel(*t.usage))
	}
	if t.busy && t.turn > 0 {
		lines = append(lines, fmt.Sprintf("turn      %d / %d", t.turn, t.agent.MaxTurns))
	}
	if t.pending != nil {
		lines = append(lines, "pending   "+t.pending.request.Call.Name+" approval")
	}
	return safeText(strings.Join(lines, "\n"))
}

// complete fills in a unique slash command or shows all prefix matches.
func (t *terminal) complete() tea.Cmd {
	value := t.input.Value()
	if !strings.HasPrefix(value, "/") || strings.ContainsRune(value, ' ') {
		return nil
	}
	var matches []string
	for _, word := range commands {
		if strings.HasPrefix(word, value) {
			matches = append(matches, word)
		}
	}
	if len(matches) == 1 {
		t.input.SetValue(matches[0] + " ")
		t.input.CursorEnd()
	} else if len(matches) > 1 {
		return tea.Println(strings.Join(matches, "  "))
	}
	return nil
}

// loadHistory reads input lines from the existing history format, in which
// content begins with "+" and metadata begins with "#".
func (t *terminal) loadHistory() error {
	if t.historyPath == "" {
		return nil
	}
	file, err := os.Open(t.historyPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	for {
		line, err := reader.ReadString('\n')
		if text, ok := strings.CutPrefix(line, "+"); ok {
			t.history = append(t.history, strings.TrimRight(text, "\r\n"))
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

// saveHistory appends input to the in-memory list and, when configured, to
// the workspace history file.
func (t *terminal) saveHistory(text string) error {
	if len(t.history) > 0 && t.history[len(t.history)-1] == text {
		return nil
	}
	t.history = append(t.history, text)
	if t.historyPath == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(t.historyPath), 0700); err != nil {
		return err
	}
	file, err := os.OpenFile(t.historyPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(file, "\n# %s\n+%s\n", time.Now().Format(time.RFC3339), strings.ReplaceAll(text, "\n", "\n+"))
	return errors.Join(err, file.Close())
}

// searchHistory walks entries that match the prefix typed before the search
// and restores that prefix when moving past the newest match.
func (t *terminal) searchHistory(direction int) {
	if t.historyIndex < 0 {
		t.historyPrefix = t.input.Value()
		t.historyIndex = len(t.history)
	}
	for cursor := t.historyIndex + direction; cursor >= 0 && cursor < len(t.history); cursor += direction {
		if strings.HasPrefix(t.history[cursor], t.historyPrefix) {
			t.historyIndex = cursor
			t.input.SetValue(t.history[cursor])
			t.input.CursorEnd()
			return
		}
	}
	if direction > 0 {
		t.historyIndex = len(t.history)
		t.input.SetValue(t.historyPrefix)
		t.input.CursorEnd()
	}
}

// safeText escapes control and bidirectional formatting characters before
// display while keeping tabs and newlines for layout.
func safeText(text string) string {
	var result strings.Builder
	for _, r := range text {
		if r == '\t' || r == '\n' || (!unicode.IsControl(r) && !unicode.Is(unicode.Bidi_Control, r)) {
			result.WriteRune(r)
		} else {
			fmt.Fprintf(&result, "\\x%02x", r)
		}
	}
	return result.String()
}

func singleLine(text string) string { return strings.Join(strings.Fields(safeText(text)), " ") }

func (t *terminal) paint(text, code string) string {
	if !t.color || text == "" {
		return text
	}
	return "\x1b[" + code + "m" + text + "\x1b[0m"
}

func contextLabel(status string) string {
	switch status {
	case "compacted":
		return "summarized older turns; original history retained"
	case "compaction_failed":
		return "could not summarize older turns"
	case "trimmed":
		return "trimmed context for this request; session history retained"
	case "near_limit":
		return "near the input budget; use /compact or start a new session"
	default:
		return ""
	}
}

// plainToolSummary selects a short result for the status line. A positive
// width bounds ordinary output in terminal cells. Blocked calls keep their
// full reason for the caller to format.
func plainToolSummary(event arc.Event, width int) string {
	content := strings.TrimSpace(safeText(event.Data["content"].(string)))
	status := event.Data["status"].(string)
	if status == "blocked" {
		return content
	}
	lines := strings.Split(content, "\n")
	// Prefer a failure or test summary over the ordinary process exit footer.
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.HasPrefix(lines[i], "[Exit code: ") && i > 0 && strings.TrimSpace(lines[i-1]) != "" {
			continue
		}
		content = strings.TrimSpace(lines[i])
		if content != "" {
			break
		}
	}
	if content == "" {
		content = "no output"
	}
	if width > 0 {
		content = ansi.Truncate(singleLine(content), width, "…")
	}
	return content
}

// toolTarget describes what a call operates on without printing every argument.
func toolTarget(call arc.ToolCall) string {
	for _, key := range []string{"command", "path"} {
		if value, ok := call.Arguments[key].(string); ok {
			return singleLine(value)
		}
	}
	if operations, ok := call.Arguments["operations"].([]any); ok {
		return fmt.Sprintf("%d file operations", len(operations))
	}
	return singleLine(call.Name)
}

// toolLine formats one tool outcome, prioritizing status and duration on
// narrow terminals.
func (t *terminal) toolLine(snapshot toolSnapshot) string {
	event := snapshot.event
	marker, style := "✓", "32"
	if event.Data["status"] == "blocked" {
		marker, style = "!", "33"
	} else if event.Data["is_error"] == true {
		marker, style = "✗", "31"
	}
	name := t.paint(marker+" "+singleLine(snapshot.call.Name), style)
	suffix := fmt.Sprintf(" · %.1fs", float64(event.Data["duration_ms"].(int64))/1000)
	width := max(0, t.width-1-ansi.StringWidth(name)-ansi.StringWidth(suffix))
	if width < 5 {
		return name + suffix
	}
	target := toolTarget(snapshot.call)
	detail := singleLine(plainToolSummary(event, 0))
	if ansi.StringWidth(target)+ansi.StringWidth(detail)+6 <= width {
		return name + " · " + target + " · " + detail + suffix
	}
	if width < 24 {
		return name + " · " + ansi.Truncate(detail, width-3, "…") + suffix
	}
	targetWidth := min(ansi.StringWidth(target), (width-6)/2)
	return name + " · " + ansi.Truncate(target, targetWidth, "…") + " · " + ansi.Truncate(detail, width-6-targetWidth, "…") + suffix
}

var inlineMarkdown = regexp.MustCompile("`[^`\\n]+`|\\*\\*[^*\\n]+\\*\\*|__[^_\\n]+__|~~[^~\\n]+~~|\\*[^*\\n]+\\*|_[^_\\n]+_|\\[[^]\\n]+\\]\\([^\\n)]+\\)")
var heading = regexp.MustCompile(`^#{1,6}\s+`)

// formatLines renders a subset of Markdown on complete lines and remembers
// fenced-code state across streamed chunks.
func (t *terminal) formatLines(text string) string {
	if !t.markdown || text == "" {
		return text
	}
	lines := strings.Split(text, "\n")
	for index, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			t.inFence = !t.inFence
			lines[index] = t.paint(line, "2")
			continue
		}
		if t.inFence {
			lines[index] = t.paint(line, "36")
			continue
		}
		if heading.MatchString(line) {
			lines[index] = t.paint(heading.ReplaceAllString(line, ""), "1;36")
			continue
		}
		if strings.HasPrefix(line, "> ") {
			lines[index] = t.paint("│ "+strings.TrimPrefix(line, "> "), "2")
			continue
		}
		if slices.Contains([]string{"---", "***", "___"}, trimmed) {
			lines[index] = t.paint("────────────────────", "2")
			continue
		}
		lines[index] = inlineMarkdown.ReplaceAllStringFunc(line, func(token string) string {
			switch {
			case strings.HasPrefix(token, "**") || strings.HasPrefix(token, "__"):
				return t.paint(token[2:len(token)-2], "1")
			case strings.HasPrefix(token, "~~"):
				return t.paint(token[2:len(token)-2], "9")
			case strings.HasPrefix(token, "["):
				label, url, _ := strings.Cut(token[1:len(token)-1], "](")
				return t.paint(label, "4") + " (" + url + ")"
			case strings.HasPrefix(token, "`"):
				return t.paint(token[1:len(token)-1], "36")
			default:
				return t.paint(token[1:len(token)-1], "3")
			}
		})
		lines[index] = strings.ReplaceAll(strings.ReplaceAll(lines[index], "- [ ] ", "☐ "), "- [x] ", "☑ ")
	}
	return strings.Join(lines, "\n")
}
