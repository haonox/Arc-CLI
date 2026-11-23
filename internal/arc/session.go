package arc

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode"
	"unicode/utf8"
)

// sessionEntry represents one v1 JSONL record. Message and checkpoint records
// form the tree. Cursor and title records update its metadata.
type sessionEntry struct {
	Type      string    `json:"type"`
	Version   int       `json:"version,omitempty"`
	ID        string    `json:"id,omitempty"`
	CWD       string    `json:"cwd,omitempty"`
	CreatedAt string    `json:"created_at,omitempty"`
	Parent    *string   `json:"parent"`
	Leaf      *string   `json:"leaf,omitempty"`
	Message   *Message  `json:"message,omitempty"`
	Messages  []Message `json:"messages,omitempty"`
	Title     string    `json:"title,omitempty"`
}

// Session holds an append-only conversation tree and a cursor to its active
// branch. A session built by CreateSession or OpenSession protects its state
// with a mutex and keeps an exclusive lock while its file is open.
type Session struct {
	mu            sync.RWMutex
	path, cwd, id string
	file          *os.File
	needsNewline  bool // The previous record ended at EOF without its newline.
	leaf, title   string
	nodes         map[string]sessionEntry
	order         []string // Node IDs in append order, independent of the cursor.
}

// SessionInfo is a snapshot of session metadata. Path is empty for an
// in-memory session, and Leaf is empty when the cursor is at the root.
type SessionInfo struct {
	Path, CWD, ID, Title, Leaf string
}

func sessionID() string {
	var data [16]byte
	rand.Read(data[:])
	return hex.EncodeToString(data[:])
}

// CreateSession starts an empty session for cwd. An empty path selects memory
// storage. Otherwise, the path must be new and remains locked until Close.
func CreateSession(cwd, path string) (*Session, error) {
	s := &Session{cwd: cwd, path: path, id: sessionID(), nodes: map[string]sessionEntry{}}
	if path == "" {
		return s, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_RDWR|os.O_APPEND, 0600)
	if err != nil {
		return nil, err
	}
	s.file = file
	if err := lockSession(file, path); err != nil {
		file.Close()
		return nil, err
	}
	header := Object{"type": "session", "version": 1, "id": s.id, "cwd": cwd, "created_at": time.Now().UTC().Format(time.RFC3339Nano)}
	if err := s.write(header); err != nil {
		file.Close()
		return nil, err
	}
	return s, nil
}

// OpenSession locks and replays an existing v1 session. A corrupt record
// causes an error and leaves the file untouched.
func OpenSession(path string) (*Session, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_APPEND, 0)
	if err != nil {
		return nil, err
	}
	s := &Session{path: path, file: file, nodes: map[string]sessionEntry{}}
	if err := s.load(); err != nil {
		file.Close()
		return nil, err
	}
	return s, nil
}

// lockSession acquires a nonblocking write lock and verifies that the file
// still occupies the requested path.
func lockSession(file *os.File, path string) error {
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return fmt.Errorf("cannot lock session for writing: %w", err)
	}
	opened, err := file.Stat()
	if err != nil {
		return err
	}
	current, err := os.Stat(path)
	if err != nil {
		return err
	}
	// Archive or restore may have moved the file between open and lock.
	if !os.SameFile(opened, current) {
		return errors.New("session moved while opening it; refresh the session list")
	}
	if !opened.Mode().IsRegular() {
		return errors.New("session must be a regular file")
	}
	return nil
}

// load validates and replays a session while acquiring its file lock.
func (s *Session) load() error {
	if err := lockSession(s.file, s.path); err != nil {
		return err
	}
	reader := bufio.NewReader(s.file)
	headerLine, err := reader.ReadBytes('\n')
	if err != nil && err != io.EOF {
		return err
	}
	var header sessionEntry
	if !utf8.Valid(headerLine) || decodeJSON(headerLine, &header) != nil ||
		header.Type != "session" || header.Version != 1 || header.ID == "" || header.CWD == "" {
		return errors.New("invalid or unsupported session header")
	}
	s.id, s.cwd = header.ID, header.CWD
	s.needsNewline = len(headerLine) > 0 && headerLine[len(headerLine)-1] != '\n'
	for number := 2; ; number++ {
		line, err := reader.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return err
		}
		// A valid record at EOF can be recovered even without a final newline.
		if len(line) > 0 {
			var entry sessionEntry
			if !utf8.Valid(line) || decodeJSON(line, &entry) != nil || s.apply(entry) != nil {
				return fmt.Errorf("invalid session line %d; file left unchanged", number)
			}
			s.needsNewline = line[len(line)-1] != '\n'
		}
		if err == io.EOF {
			return nil
		}
	}
}

// write appends and syncs a record before the caller updates in-memory state.
// The caller holds mu once the session is shared.
func (s *Session) write(value any) error {
	if s.file == nil {
		if s.path == "" {
			return nil
		}
		return errors.New("session is closed")
	}
	line, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if s.needsNewline {
		line = append([]byte{'\n'}, line...)
	}
	line = append(line, '\n')
	// Remember the original end so a failed write or sync can roll back
	// a partially appended record.
	position, err := s.file.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	if _, err := s.file.Write(line); err != nil {
		return errors.Join(err, s.file.Truncate(position))
	}
	if err := s.file.Sync(); err != nil {
		return errors.Join(err, s.file.Truncate(position), s.file.Sync())
	}
	s.needsNewline = false
	return nil
}

// apply replays one record into memory. A parent must already be present,
// which keeps the append-order tree acyclic.
func (s *Session) apply(entry sessionEntry) error {
	switch entry.Type {
	case "cursor":
		s.leaf = ""
		if entry.Leaf != nil {
			if _, ok := s.nodes[*entry.Leaf]; !ok {
				return errors.New("unknown cursor")
			}
			s.leaf = *entry.Leaf
		}
		return nil
	case "title":
		if err := validateTitle(entry.Title); err != nil {
			return err
		}
		s.title = entry.Title
		return nil
	case "message":
		if entry.Message == nil {
			return errors.New("missing message")
		}
		if err := entry.Message.validate(); err != nil {
			return err
		}
	case "checkpoint":
		if entry.Messages == nil {
			return errors.New("missing checkpoint messages")
		}
		for _, message := range entry.Messages {
			if err := message.validate(); err != nil {
				return err
			}
		}
	default:
		return errors.New("unknown session entry")
	}
	if _, ok := s.nodes[entry.ID]; ok || entry.ID == "" {
		return errors.New("invalid or duplicate session node")
	}
	if entry.Parent != nil {
		if _, ok := s.nodes[*entry.Parent]; !ok {
			return errors.New("unknown parent")
		}
	}
	s.nodes[entry.ID] = entry
	s.order = append(s.order, entry.ID)
	s.leaf = entry.ID
	return nil
}

// append attaches a validated node to the active cursor and persists it
// before changing the active branch.
func (s *Session) append(entry sessionEntry) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry.ID = sessionID()
	if s.leaf != "" {
		parent := s.leaf
		entry.Parent = &parent
	}
	var value any = entry
	if entry.Type == "checkpoint" {
		// The wire format requires messages even for an empty checkpoint.
		value = Object{"type": "checkpoint", "id": entry.ID, "parent": entry.Parent, "messages": entry.Messages}
	}
	if err := s.write(value); err != nil {
		return "", err
	}
	return entry.ID, s.apply(entry)
}

// Append copies and stores a message on the active branch, returning its
// session node ID.
func (s *Session) Append(message Message) (string, error) {
	if err := message.validate(); err != nil {
		return "", err
	}
	detached := cloneMessage(message)
	return s.append(sessionEntry{Type: "message", Message: &detached})
}

// Checkpoint appends a replacement view of the active history. Earlier
// nodes remain available for branching.
func (s *Session) Checkpoint(messages []Message) error {
	detached := make([]Message, len(messages))
	for i, message := range messages {
		if err := message.validate(); err != nil {
			return err
		}
		detached[i] = cloneMessage(message)
	}
	_, err := s.append(sessionEntry{Type: "checkpoint", Messages: detached})
	return err
}

// Branch moves the cursor to the node with a unique ID prefix. The special
// prefix "root" starts the next append from an empty conversation.
func (s *Session) Branch(prefix string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var leaf *string
	if prefix != "root" {
		for _, id := range s.order {
			if strings.HasPrefix(id, prefix) {
				if leaf != nil {
					return errors.New("entry prefix must match exactly one node")
				}
				leaf = &id
			}
		}
		if prefix == "" || leaf == nil {
			return errors.New("entry prefix must match exactly one node")
		}
	}
	if err := s.write(Object{"type": "cursor", "leaf": leaf}); err != nil {
		return err
	}
	return s.apply(sessionEntry{Type: "cursor", Leaf: leaf})
}

func validateTitle(title string) error {
	if strings.TrimSpace(title) == "" || utf8.RuneCountInString(title) > 120 || strings.ContainsFunc(title, func(r rune) bool {
		return !unicode.IsPrint(r)
	}) {
		return errors.New("session title must be 1 to 120 printable characters")
	}
	return nil
}

// Rename records a printable title for a saved session.
func (s *Session) Rename(title string) error {
	title = strings.TrimSpace(title)
	if err := validateTitle(title); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.path == "" {
		return errors.New("naming requires a saved session")
	}
	if err := s.write(Object{"type": "title", "title": title}); err != nil {
		return err
	}
	s.title = title
	return nil
}

// Info returns a snapshot of the current session metadata.
func (s *Session) Info() SessionInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return SessionInfo{Path: s.path, CWD: s.cwd, ID: s.id, Title: s.title, Leaf: s.leaf}
}

// Messages returns a detached, chronological view of the active branch.
// An interrupted tool call is represented by a synthesized error result in
// the returned view.
func (s *Session) Messages() []Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var messages []Message
	// Follow parent links from the leaf. The nearest checkpoint already
	// contains the context to use for its entire prefix.
	for id := s.leaf; id != ""; {
		entry := s.nodes[id]
		if entry.Type == "checkpoint" {
			for i := len(entry.Messages) - 1; i >= 0; i-- {
				messages = append(messages, cloneMessage(entry.Messages[i]))
			}
			break
		}
		messages = append(messages, cloneMessage(*entry.Message))
		id = ""
		if entry.Parent != nil {
			id = *entry.Parent
		}
	}
	slices.Reverse(messages)
	return repairTools(messages)
}

// repairTools pairs each tool proposal with a result, fills missing results
// with an interruption message, and drops results outside the active batch.
func repairTools(messages []Message) []Message {
	result := make([]Message, 0, len(messages))
	var pending []ToolCall
	present := map[string]bool{}
	flush := func() {
		for _, call := range pending {
			if present[call.ID] {
				result = append(result, toolMessage(call, "Interrupted; inspect side effects before retrying.", true))
			}
		}
		clear(present)
	}
	for _, message := range messages {
		if message.Role == "tool" {
			if message.ToolCallID == nil || !present[*message.ToolCallID] {
				continue
			}
			delete(present, *message.ToolCallID)
		} else {
			flush()
			pending = message.ToolCalls
			for _, call := range pending {
				present[call.ID] = true
			}
		}
		result = append(result, message)
	}
	flush()
	return result
}

// Tree returns a terminal-friendly rendering of every branch with the
// active cursor marked.
func (s *Session) Tree() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Index children by parent so rendering preserves the order of siblings.
	children := map[string][]string{}
	for _, id := range s.order {
		node, parent := s.nodes[id], ""
		if node.Parent != nil {
			parent = *node.Parent
		}
		children[parent] = append(children[parent], id)
	}
	var lines []string
	var visit func(string, string, string)
	visit = func(id, indent, connector string) {
		entry := s.nodes[id]
		label := "checkpoint compressed context"
		if entry.Message != nil {
			message := entry.Message
			content := strings.Join(strings.Fields(message.Content), " ")
			if content == "" && len(message.ToolCalls) > 0 {
				var names []string
				for _, call := range message.ToolCalls {
					names = append(names, call.Name)
				}
				content = "calls " + strings.Join(names, ", ")
			}
			runes := []rune(content)
			label = fmt.Sprintf("%-10s %s", message.Role, string(runes[:min(64, len(runes))]))
		}
		mark, current := "○", ""
		if id == s.leaf {
			mark, current = "●", "  ← current"
		}
		lines = append(lines, strings.TrimRight(fmt.Sprintf("%s%s%s %.8s  %s%s", indent, connector, mark, id, label, current), " "))
		switch connector {
		case "├─":
			indent += "│  "
		case "└─":
			indent += "   "
		}
		for i, child := range children[id] {
			branch := "├─"
			if i == len(children[id])-1 {
				branch = "└─"
			}
			visit(child, indent, branch)
		}
	}
	roots := children[""]
	for i, root := range roots {
		connector := ""
		if len(roots) > 1 {
			connector = "├─"
			if i == len(roots)-1 {
				connector = "└─"
			}
		}
		visit(root, "", connector)
	}
	return lines
}

// Close releases the backing file and its lock. It is safe to call twice.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return nil
	}
	err := s.file.Close()
	s.file = nil
	return err
}

// SessionCatalog locates saved sessions and input history for one workspace.
type SessionCatalog struct {
	directory string
}

// NewCatalog derives the workspace's state directory from its canonical path
// and an absolute XDG_STATE_HOME, with a fallback to the user's local state.
func NewCatalog(cwd string, getenv func(string) string) SessionCatalog {
	state := getenv("XDG_STATE_HOME")
	if !filepath.IsAbs(state) {
		home, _ := os.UserHomeDir()
		state = filepath.Join(home, ".local", "state")
	}
	hash := sha256.Sum256([]byte(cwd))
	return SessionCatalog{directory: filepath.Join(state, "arc", "workspaces", hex.EncodeToString(hash[:]))}
}

// HistoryPath returns the shared input history path for the workspace.
func (c SessionCatalog) HistoryPath() string { return filepath.Join(c.directory, "input-history") }

// NewPath returns a path for a new session without creating the file.
func (c SessionCatalog) NewPath() string {
	name := time.Now().UTC().Format("20060102T150405") + "-" + sessionID()[:8] + ".jsonl"
	return filepath.Join(c.directory, "sessions", name)
}

// SavedSession is the file metadata and most recent title shown by the catalog.
type SavedSession struct {
	ID, Path, Title string
	Size            int64
	Modified        time.Time
}

// list reads metadata from one catalog directory, most recently modified first.
func (c SessionCatalog) list(archived bool, prefix string, titles bool) ([]SavedSession, error) {
	directory := "sessions"
	if archived {
		directory = "archive"
	}
	base := filepath.Join(c.directory, directory)
	files, err := os.ReadDir(base)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var records []SavedSession
	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".jsonl") {
			continue
		}
		info, err := file.Info()
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			continue
		}
		path := filepath.Join(base, file.Name())
		id, title, err := catalogMetadata(path, prefix, titles)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		if id != "" {
			records = append(records, SavedSession{ID: id, Path: path, Title: title, Size: info.Size(), Modified: info.ModTime()})
		}
	}
	slices.SortFunc(records, func(a, b SavedSession) int {
		if cmp := b.Modified.Compare(a.Modified); cmp != 0 {
			return cmp
		}
		return strings.Compare(a.ID, b.ID)
	})
	return records, nil
}

// catalogMetadata reads session IDs without replaying the tree. Only listing
// with titles needs to scan beyond the header.
func catalogMetadata(path, prefix string, titleWanted bool) (string, string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", "", err
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	line, err := reader.ReadBytes('\n')
	if err != nil && err != io.EOF {
		return "", "", err
	}
	var header sessionEntry
	if decodeJSON(line, &header) != nil || header.Type != "session" || header.Version != 1 ||
		!strings.HasPrefix(header.ID, prefix) {
		return "", "", nil
	}
	if !titleWanted {
		return header.ID, "", nil
	}
	title := ""
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return "", "", err
		}
		// Skip decoding ordinary message records while scanning for titles.
		if bytes.Contains(line, []byte("\"title\"")) {
			var entry struct{ Type, Title string }
			if json.Unmarshal(line, &entry) == nil && entry.Type == "title" && validateTitle(entry.Title) == nil {
				title = entry.Title
			}
		}
		if err == io.EOF {
			return header.ID, title, nil
		}
	}
}

// List returns active or archived sessions in reverse modification order.
func (c SessionCatalog) List(archived bool) ([]SavedSession, error) {
	return c.list(archived, "", true)
}

// Latest returns the path of the most recently modified active session.
func (c SessionCatalog) Latest() (string, error) {
	results, err := c.list(false, "", false)
	if err != nil {
		return "", err
	}
	if len(results) == 0 {
		return "", errors.New("no saved sessions in this working directory")
	}
	return results[0].Path, nil
}

// Find returns the path of the unique session matching a nonempty ID prefix
// in the active or archived directory.
func (c SessionCatalog) Find(prefix string, archived bool) (string, error) {
	results, err := c.list(archived, prefix, false)
	if err != nil {
		return "", err
	}
	if prefix == "" || len(results) != 1 {
		return "", errors.New("session ID prefix must match exactly one saved session")
	}
	return results[0].Path, nil
}

// Move archives a saved session or restores it when restore is true. It holds
// the session's exclusive lock through the rename and returns the new path.
func (c SessionCatalog) Move(prefix string, restore bool) (string, error) {
	source, err := c.Find(prefix, restore)
	if err != nil {
		return "", err
	}
	session, err := OpenSession(source)
	if err != nil {
		return "", err
	}
	defer session.Close()
	directory := "archive"
	if restore {
		directory = "sessions"
	}
	dir := filepath.Join(c.directory, directory)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	destination := filepath.Join(dir, filepath.Base(source))
	if _, err := os.Lstat(destination); err == nil {
		return "", errors.New("a session already exists at the destination")
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	if err := os.Rename(source, destination); err != nil {
		return "", err
	}
	return destination, nil
}
