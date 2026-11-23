package arc

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

// runBash runs an authorized command in its own process group. It streams
// combined output and cleans up the group on exit, cancellation, or timeout.
func runBash(ctx context.Context, args Object, env ToolEnv, emit func(string) error) ToolResult {
	ctx, cancel := context.WithTimeout(ctx, time.Duration(number(args, "timeout", 120)*float64(time.Second)))
	defer cancel()
	if err := ctx.Err(); err != nil {
		return failed(err)
	}
	readPipe, writePipe, err := os.Pipe()
	if err != nil {
		return failed(err)
	}
	defer readPipe.Close()
	command := exec.Command("bash", "-c", args["command"].(string))
	command.Dir = env.CWD
	command.Env = env.Env
	if command.Env == nil {
		// A nil Cmd.Env would inherit the parent process's unfiltered environment.
		command.Env = []string{}
	}
	command.Stdout, command.Stderr = writePipe, writePipe
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := command.Start(); err != nil {
		writePipe.Close()
		return failed(err)
	}
	writePipe.Close()
	wait := make(chan error, 1)
	go func() { wait <- command.Wait() }()
	output := make(chan string, 1)
	go func() {
		output <- collectOutput(readPipe, func(delta string) {
			if ctx.Err() == nil && emit(delta) != nil {
				cancel()
			}
		})
	}()
	// Kill the entire group even when Bash exits first. Background children
	// can outlive their parent and still hold the output pipe.
	select {
	case err = <-wait:
		_ = syscall.Kill(-command.Process.Pid, syscall.SIGKILL)
	case <-ctx.Done():
		_ = syscall.Kill(-command.Process.Pid, syscall.SIGKILL)
		err = <-wait
	}
	// A process that explicitly detached from the group may still hold the pipe.
	// Give normal process-group cleanup time to drain without waiting indefinitely.
	var text string
	select {
	case text = <-output:
	case <-time.After(250 * time.Millisecond):
		readPipe.Close()
		text = <-output
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return ToolResult{Content: text + "\n[Command timed out; process group killed]", IsError: true}
	}
	if err := ctx.Err(); err != nil {
		return failed(err)
	}
	return ToolResult{Content: fmt.Sprintf("%s\n[Exit code: %d]", text, command.ProcessState.ExitCode()), IsError: err != nil}
}

// collectOutput streams rune-aligned chunks and retains enough output for the
// final tool result. It keeps draining the pipe after reaching the limit.
func collectOutput(file *os.File, emit func(string)) string {
	reader := bufio.NewReader(file)
	var output, delta strings.Builder
	kept := 0
	truncated := false
	flush := func() {
		if delta.Len() != 0 {
			emit(delta.String())
			delta.Reset()
		}
	}
	for {
		r, _, err := reader.ReadRune()
		if err != nil {
			break
		}
		if kept == maxOutput {
			truncated = true
			// Draining prevents a verbose process from blocking on a full pipe.
			_, _ = io.Copy(io.Discard, reader)
			break
		}
		kept++
		output.WriteRune(r)
		delta.WriteRune(r)
		if reader.Buffered() == 0 || delta.Len() >= 4096 {
			flush()
		}
	}
	flush()
	if truncated {
		output.WriteString("\n[Output truncated at 24000 characters]")
	}
	return output.String()
}
