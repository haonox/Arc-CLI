// Command arc runs the Arc terminal assistant.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/haonox/Arc-CLI/internal/cli"
)

// main connects process signals and standard streams to the CLI.
func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := cli.Run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr)
	// Release the signal subscription explicitly because os.Exit skips deferred calls.
	stop()
	os.Exit(code)
}
