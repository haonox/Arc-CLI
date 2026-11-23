package arc

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unicode/utf8"
)

// relativePath resolves existing symlinks, including aliases to the same patch target. Root
// also confines the eventual operations if a path is replaced after this check.
func relativePath(root *os.Root, value string) (string, error) {
	path := expandHome(value)
	if !filepath.IsAbs(path) {
		path = filepath.Join(root.Name(), path)
	}
	relative := func(path string) (string, error) {
		result, err := filepath.Rel(root.Name(), path)
		if err != nil || !filepath.IsLocal(result) {
			return "", errors.New("path is outside the workspace")
		}
		return result, nil
	}
	if _, err := relative(path); err != nil {
		return "", err
	}
	probe := path
	var missing []string
	// Resolve the deepest existing ancestor so new nested files can still
	// be checked against the workspace and compared for aliasing.
	for {
		resolved, err := filepath.EvalSymlinks(probe)
		if err == nil {
			for i := len(missing) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, missing[i])
			}
			return relative(resolved)
		}
		if !errors.Is(err, os.ErrNotExist) || probe == root.Name() {
			return "", err
		}
		// Reject dangling symlinks so creating a file cannot replace the link itself.
		if info, statErr := os.Lstat(probe); statErr == nil && info.Mode()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("dangling symlink: %s", probe)
		}
		missing = append(missing, filepath.Base(probe))
		probe = filepath.Dir(probe)
	}
}

// readText returns a bounded UTF-8 file and its permissions.
func readText(root *os.Root, path string) (string, os.FileMode, error) {
	// Avoid blocking on a FIFO before verifying the opened file is regular.
	file, err := root.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return "", 0, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", 0, err
	}
	if !info.Mode().IsRegular() {
		return "", 0, errors.New("target must be a regular file")
	}
	data, err := io.ReadAll(io.LimitReader(file, maxFileBytes+1))
	if err != nil {
		return "", 0, err
	}
	if len(data) > maxFileBytes {
		return "", 0, errors.New("file exceeds 1 MiB; inspect a smaller portion")
	}
	if !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
		return "", 0, errors.New("file must be UTF-8 text")
	}
	return string(data), info.Mode().Perm(), nil
}

// checkText enforces the byte limit and text requirements for file writes.
func checkText(content string) error {
	if len(content) > maxFileBytes {
		return errors.New("file exceeds 1 MiB")
	}
	if !utf8.ValidString(content) || strings.IndexByte(content, 0) >= 0 {
		return errors.New("file must be UTF-8 text")
	}
	return nil
}

// replaceFile stages complete content in the target directory. A rename
// publishes an overwrite, and createOnly publishes via a non-clobbering link.
func replaceFile(root *os.Root, path, content string, perm os.FileMode, createOnly bool) error {
	if err := checkText(content); err != nil {
		return err
	}
	parent := filepath.Dir(path)
	if err := root.MkdirAll(parent, 0755); err != nil {
		return err
	}
	directory, err := root.OpenRoot(parent)
	if err != nil {
		return err
	}
	defer directory.Close()
	name := filepath.Base(path)
	temp := ".arc-tmp-" + sessionID()
	// Keep staged contents private until the final permissions are applied.
	file, err := directory.OpenFile(temp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer directory.Remove(temp)
	_, err = io.WriteString(file, content)
	if err == nil {
		err = file.Chmod(perm)
	}
	if err == nil {
		err = file.Sync()
	}
	if err := errors.Join(err, file.Close()); err != nil {
		return err
	}
	if createOnly {
		return directory.Link(temp, name)
	}
	return directory.Rename(temp, name)
}

// readFile returns a numbered window of normalized text and a continuation
// offset when more lines remain.
func readFile(root *os.Root, args Object) (string, error) {
	path, err := relativePath(root, args["path"].(string))
	if err != nil {
		return "", err
	}
	text, _, err := readText(root, path)
	if err != nil {
		return "", err
	}
	text = strings.ReplaceAll(strings.ReplaceAll(text, "\r\n", "\n"), "\r", "\n")
	lines := strings.Split(strings.TrimSuffix(text, "\n"), "\n")
	if text == "" {
		lines = nil
	}
	offset := number(args, "offset", 1)
	// Check the offset before converting it to a Go index.
	if offset > float64(len(lines)) {
		return "[Empty file or offset beyond end]", nil
	}
	start := int(offset) - 1
	end := min(start+int(number(args, "limit", 200)), len(lines))
	var result strings.Builder
	for i := start; i < end; i++ {
		fmt.Fprintf(&result, "%d: %s\n", i+1, lines[i])
	}
	if end < len(lines) {
		fmt.Fprintf(&result, "[More lines: use offset=%d]", end+1)
	}
	content := strings.TrimSuffix(result.String(), "\n")
	if runes := []rune(content); len(runes) > maxOutput {
		content = string(runes[:maxOutput]) + "\n[Output truncated at 24000 characters]"
	}
	return content, nil
}

// writeFile creates a private file or replaces an existing regular file
// while preserving its permissions.
func writeFile(root *os.Root, args Object) (string, error) {
	path, err := relativePath(root, args["path"].(string))
	if err != nil {
		return "", err
	}
	mode := os.FileMode(0600)
	info, err := root.Stat(path)
	if err == nil {
		if !info.Mode().IsRegular() {
			return "", errors.New("target must be a regular file")
		}
		mode = info.Mode().Perm()
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	if err := replaceFile(root, path, args["content"].(string), mode, false); err != nil {
		return "", err
	}
	return "Wrote " + filepath.Join(root.Name(), path), nil
}

// replaceText performs an exact substitution, requiring a unique match
// unless all matches were requested.
func replaceText(text, old, updated string, all bool) (string, error) {
	matches := strings.Count(text, old)
	switch {
	case matches == 0:
		return "", errors.New("old_text not found; read the file again")
	case matches > 1 && !all:
		return "", fmt.Errorf("old_text matches %d times; supply more context or replace_all=true", matches)
	}
	replacements := 1
	if all {
		replacements = matches
	}
	// Check the expanded size before allocating the replacement string.
	size := int64(len(text)) + int64(replacements)*int64(len(updated)-len(old))
	if size > maxFileBytes {
		return "", errors.New("file exceeds 1 MiB after replacement")
	}
	if all {
		return strings.ReplaceAll(text, old, updated), nil
	}
	return strings.Replace(text, old, updated, 1), nil
}

// editFile commits a validated text substitution with the original permissions.
func editFile(root *os.Root, args Object) (string, error) {
	path, err := relativePath(root, args["path"].(string))
	if err != nil {
		return "", err
	}
	text, mode, err := readText(root, path)
	if err != nil {
		return "", err
	}
	all, _ := args["replace_all"].(bool)
	updated, err := replaceText(text, args["old_text"].(string), args["new_text"].(string), all)
	if err != nil {
		return "", err
	}
	if err := replaceFile(root, path, updated, mode, false); err != nil {
		return "", err
	}
	return "Edited " + filepath.Join(root.Name(), path), nil
}

// patchChange holds the snapshots needed to commit and roll back one operation.
type patchChange struct {
	action, path  string
	before, after string
	mode          os.FileMode
	edits         int
}

// preparePatch resolves targets and computes replacement content without
// modifying any file.
func preparePatch(root *os.Root, operations []any) ([]patchChange, error) {
	seen := make(map[string]bool, len(operations))
	changes := make([]patchChange, 0, len(operations))
	for _, value := range operations {
		op := value.(Object)
		path, err := relativePath(root, op["path"].(string))
		if err != nil {
			return nil, err
		}
		if seen[path] {
			return nil, fmt.Errorf("duplicate patch path: %s", path)
		}
		seen[path] = true
		change := patchChange{action: op["type"].(string), path: path, mode: 0600}
		if change.action == "add" {
			if _, err := root.Lstat(path); !errors.Is(err, os.ErrNotExist) {
				if err != nil {
					return nil, err
				}
				return nil, fmt.Errorf("cannot add existing path: %s", path)
			}
			change.after = op["content"].(string)
		} else {
			change.before, change.mode, err = readText(root, path)
			if err != nil {
				return nil, err
			}
			change.after = change.before
			if change.action == "update" {
				for i, value := range op["edits"].([]any) {
					edit := value.(Object)
					change.after, err = replaceText(change.after, edit["old_text"].(string), edit["new_text"].(string), false)
					if err != nil {
						return nil, fmt.Errorf("update %s edit %d: %w", path, i+1, err)
					}
					if err := checkText(change.after); err != nil {
						return nil, err
					}
					change.edits++
				}
				if change.after == change.before {
					return nil, fmt.Errorf("update does not change file: %s", path)
				}
			}
		}
		if err := checkText(change.after); err != nil {
			return nil, err
		}
		changes = append(changes, change)
	}
	return changes, nil
}

// checkPatchState detects content changes between preflight and commit or
// between commit and rollback.
func checkPatchState(root *os.Root, path, expected string) error {
	current, _, err := readText(root, path)
	if err != nil {
		return err
	}
	if current != expected {
		return fmt.Errorf("%s changed after preflight", path)
	}
	return nil
}

func (c patchChange) commit(root *os.Root) error {
	if c.action == "add" {
		return replaceFile(root, c.path, c.after, c.mode, true)
	}
	if err := checkPatchState(root, c.path, c.before); err != nil {
		return err
	}
	if c.action == "delete" {
		return root.Remove(c.path)
	}
	return replaceFile(root, c.path, c.after, c.mode, false)
}

// rollback undoes one committed operation, checking the current contents to
// avoid overwriting an intervening edit.
func (c patchChange) rollback(root *os.Root) error {
	if c.action == "delete" {
		return replaceFile(root, c.path, c.before, c.mode, true)
	}
	if err := checkPatchState(root, c.path, c.after); err != nil {
		return err
	}
	if c.action == "add" {
		return root.Remove(c.path)
	}
	return replaceFile(root, c.path, c.before, c.mode, false)
}

// applyPatch prepares all changes before committing them in request order.
// A commit failure triggers a best-effort rollback in reverse order.
func applyPatch(root *os.Root, args Object) (string, error) {
	changes, err := preparePatch(root, args["operations"].([]any))
	if err != nil {
		return "", err
	}
	for index, change := range changes {
		if err := change.commit(root); err != nil {
			failure := fmt.Errorf("patch failed: %w", err)
			for previous := index - 1; previous >= 0; previous-- {
				if err := changes[previous].rollback(root); err != nil {
					failure = errors.Join(failure, fmt.Errorf("rollback %s: %w", changes[previous].path, err))
				}
			}
			return "", failure
		}
	}
	var output strings.Builder
	fmt.Fprintf(&output, "Applied %d file operations:", len(changes))
	for _, change := range changes {
		fmt.Fprintf(&output, "\n- %s %s", change.action, filepath.Join(root.Name(), change.path))
		if change.action == "update" {
			fmt.Fprintf(&output, " (%d edits)", change.edits)
		}
	}
	return output.String(), nil
}
