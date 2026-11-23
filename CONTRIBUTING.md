# Contributing to Arc CLI

Use Go 1.25 or later and run `gofmt` on changed files. Prefer small packages, explicit data flow, and standard-library implementations. Use `apply_patch` for source and documentation edits. Keep CLI text in English and keep credentials out of logs, tests, and configuration files committed to Git.

The entry point lives in `cmd/arc`. `internal/cli` handles application assembly and the terminal. `internal/arc` owns provider requests, model context, tool execution, and session persistence.

```bash
go test ./...
go test -race ./...
go vet ./...
go build ./cmd/arc
```

Preserve tool-call/result pairing, cancellation, event ordering, and the ability to read existing session files. Exercise the normal CLI path with a local mock Provider. Add a PTY scenario when changing input editing or live output. Describe Bash authorization separately from operating-system permissions.
