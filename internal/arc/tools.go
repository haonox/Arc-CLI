package arc

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

const maxFileBytes = 1024 * 1024
const maxOutput = 24000 // Tool output limit in runes, excluding status and truncation text.

// ToolResult is the content and error status of a single tool execution.
type ToolResult struct {
	Content string
	IsError bool
}

// ToolEnv supplies the working directory and filtered subprocess environment.
type ToolEnv struct {
	CWD string
	Env []string
}

// Tool binds its model-facing schema to effect classification and execution.
// The runtime validates arguments against Spec before calling Effects, then
// authorizes the declared effects before calling Execute.
type Tool struct {
	Spec    ToolSpec
	Effects func(Object) []string
	// Execute may send incremental output using the callback, then returns the
	// final result that the runtime records in the session.
	Execute func(context.Context, Object, ToolEnv, func(string) error) ToolResult
}

type registeredTool struct {
	Tool
	schema *jsonschema.Schema
}

// Registry indexes the available tools and their compiled schemas while
// preserving registration order for model requests.
type Registry struct {
	names []string
	tools map[string]registeredTool
}

// NewRegistry validates tool definitions and compiles their argument schemas
// using JSON Schema Draft 2020-12 by default.
func NewRegistry(tools []Tool) (*Registry, error) {
	registry := &Registry{tools: map[string]registeredTool{}}
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	for index, tool := range tools {
		if _, exists := registry.tools[tool.Spec.Name]; exists || tool.Spec.Name == "" ||
			tool.Effects == nil || tool.Execute == nil {
			return nil, fmt.Errorf("invalid or duplicate tool %q", tool.Spec.Name)
		}
		resource := fmt.Sprintf("file:///arc/tool-%d.json", index)
		var schemaDoc Object
		data, err := json.Marshal(tool.Spec.Parameters)
		if err != nil {
			return nil, err
		}
		// Normalize Go literals such as []string into the JSON value types
		// expected by the schema compiler.
		if err := decodeJSON(data, &schemaDoc); err != nil {
			return nil, err
		}
		if err := compiler.AddResource(resource, schemaDoc); err != nil {
			return nil, err
		}
		schema, err := compiler.Compile(resource)
		if err != nil {
			return nil, err
		}
		registry.names = append(registry.names, tool.Spec.Name)
		registry.tools[tool.Spec.Name] = registeredTool{Tool: tool, schema: schema}
	}
	return registry, nil
}

// Specs returns the registered tool specifications in request order.
func (r *Registry) Specs() []ToolSpec {
	specs := make([]ToolSpec, 0, len(r.names))
	for _, name := range r.names {
		specs = append(specs, r.tools[name].Spec)
	}
	return specs
}

// Names returns a detached list of registered tool names in request order.
func (r *Registry) Names() []string { return slices.Clone(r.names) }

// validate resolves a proposal against the available tools and checks its schema.
func (r *Registry) validate(call ToolCall) (Tool, error) {
	tool, ok := r.tools[call.Name]
	if !ok {
		return Tool{}, fmt.Errorf("unknown tool %q", call.Name)
	}
	if call.Arguments == nil {
		return Tool{}, fmt.Errorf("invalid arguments for %s: expected an object", call.Name)
	}
	if err := tool.schema.Validate(call.Arguments); err != nil {
		return Tool{}, fmt.Errorf("invalid arguments for %s: %w", call.Name, err)
	}
	return tool.Tool, nil
}

// Authorization describes a validated proposal, the workspace it targets,
// and the effects that the runtime will authorize.
type Authorization struct {
	Call    ToolCall
	CWD     string
	Effects []string
}

// Approver asks the caller to approve effects that require confirmation.
// A false result or an error prevents execution.
type Approver func(context.Context, Authorization) (bool, error)

const (
	// Confirm automatically allows read and write effects and asks about the rest.
	Confirm = "confirm"
	// Autonomous automatically allows all known effects.
	Autonomous = "autonomous"
	// Restricted allows read, write, and process effects and blocks the others.
	Restricted = "restricted"
)

type decision struct {
	status, reason string
}

// authorize applies the policy to every declared effect before any dispatch.
func authorize(ctx context.Context, mode string, request Authorization, approval Approver) decision {
	var pending []string
	for _, effect := range request.Effects {
		switch effect {
		case "read", "write":
		case "process":
			if mode != Autonomous && mode != Restricted {
				pending = append(pending, effect)
			}
		case "external", "destructive":
			if mode != Autonomous {
				pending = append(pending, effect)
			}
		default:
			return decision{"blocked", "unknown effect " + effect}
		}
	}
	if len(pending) == 0 {
		return decision{"allowed", ""}
	}
	if approval == nil || mode == Restricted {
		return decision{"confirmation_required", "explicit authorization required for effects: " + strings.Join(pending, ", ")}
	}
	ok, err := approval(ctx, request)
	if err != nil {
		return decision{"blocked", "authorization interrupted"}
	}
	if !ok {
		return decision{"blocked", "authorization denied for effects: " + strings.Join(pending, ", ")}
	}
	return decision{"allowed", "approved effects: " + strings.Join(pending, ", ")}
}

var providerEnv = []string{"ARC_API_KEY", "ARC_BASE_URL", "ARC_MODEL", "DEEPSEEK_API_KEY"}
var secretPattern = regexp.MustCompile(`(?i)(?:^|_)(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY)(?:$|_)`)

// ToolEnvironment filters NAME=value entries for tool subprocesses.
// Explicit denies and provider settings take precedence over all allows.
// Allows can opt into other names matched by the sensitive-variable filter.
func ToolEnvironment(environ, allow, deny, credentialNames []string) []string {
	allowed, denied := map[string]bool{}, map[string]bool{}
	for _, name := range allow {
		allowed[strings.ToUpper(name)] = true
	}
	for _, names := range [][]string{deny, providerEnv, credentialNames} {
		for _, name := range names {
			denied[strings.ToUpper(name)] = true
		}
	}
	result := make([]string, 0, len(environ))
	for _, entry := range environ {
		name, _, ok := strings.Cut(entry, "=")
		if ok && !denied[strings.ToUpper(name)] && (allowed[strings.ToUpper(name)] || !secretPattern.MatchString(name)) {
			result = append(result, entry)
		}
	}
	return result
}

func fixedEffects(names ...string) func(Object) []string {
	return func(Object) []string { return slices.Clone(names) }
}

var externalCommand = regexp.MustCompile(`(?i)(?:^|[;&|()\n])\s*(?:(?:sudo|command|exec|nohup)\s+)*(?:curl\b|wget\b|ssh\b|scp\b|sftp\b|nc\b|ncat\b|telnet\b|ftp\b|git\s+(?:clone|fetch|pull|push)\b|(?:pip[0-9.]*|python[0-9.]*\s+-m\s+pip)\s+install\b|npm\s+install\b|uv\s+add\b)`)
var destructiveCommand = regexp.MustCompile(`(?i)(?:^|[;&|()\n])\s*(?:(?:sudo|command|exec|nohup)\s+)*(?:rm\b|rmdir\b|unlink\b|shred\b|mkfs(?:\.[a-z0-9]+)?\b|fdisk\b|shutdown\b|reboot\b|git\s+(?:reset\s+--hard|clean\b))|\bdrop\s+(?:database|table)\b`)

// bashEffects adds conservative hints based on the command text. These
// hints inform authorization, and Bash runs with the user's OS permissions.
func bashEffects(args Object) []string {
	effects := []string{"process"}
	command := args["command"].(string)
	if externalCommand.MatchString(command) {
		effects = append(effects, "external")
	}
	if destructiveCommand.MatchString(command) {
		effects = append(effects, "destructive")
	}
	return effects
}

// patchEffects marks the whole patch as destructive if any operation deletes a file.
func patchEffects(args Object) []string {
	for _, operation := range args["operations"].([]any) {
		if operation.(Object)["type"] == "delete" {
			return []string{"write", "destructive"}
		}
	}
	return []string{"write"}
}

func objectSchema(properties Object, required ...string) Object {
	return Object{"type": "object", "properties": properties, "required": required, "additionalProperties": false}
}

// BuiltinTools returns Arc's file, patch, and shell tools in request order.
func BuiltinTools() []Tool {
	text := Object{"type": "string"}
	nonempty := Object{"type": "string", "minLength": 1}
	bounded := Object{"type": "string", "maxLength": maxFileBytes}
	edits := objectSchema(Object{"old_text": nonempty, "new_text": bounded}, "old_text", "new_text")
	operation := func(name string, fields Object, required ...string) Object {
		properties := Object{"type": Object{"const": name}, "path": nonempty}
		maps.Copy(properties, fields)
		return objectSchema(properties, append([]string{"type", "path"}, required...)...)
	}
	// The operation type selects its own schema and required fields.
	patch := Object{"oneOf": []Object{
		operation("add", Object{"content": bounded}, "content"),
		operation("update", Object{"edits": Object{"type": "array", "minItems": 1, "maxItems": 100, "items": edits}}, "edits"),
		operation("delete", nil),
	}}
	return []Tool{
		{ToolSpec{"read", "Read UTF-8 text with numbered lines. offset is 1-based.",
			objectSchema(Object{"path": nonempty, "offset": Object{"type": "integer", "minimum": 1},
				"limit": Object{"type": "integer", "minimum": 1, "maximum": 2000}}, "path")},
			fixedEffects("read"), fileTool(readFile)},
		{ToolSpec{"write", "Create or overwrite a UTF-8 file. Read existing files before changing them.",
			objectSchema(Object{"path": nonempty, "content": text}, "path", "content")},
			fixedEffects("write"), fileTool(writeFile)},
		{ToolSpec{"edit", "Replace exact text; ambiguous matches fail unless replace_all is true.",
			objectSchema(Object{"path": nonempty, "old_text": nonempty, "new_text": text, "replace_all": Object{"type": "boolean"}}, "path", "old_text", "new_text")},
			fixedEffects("write"), fileTool(editFile)},
		{ToolSpec{"apply_patch", "Apply validated add, update, and delete operations across UTF-8 text files as one patch.",
			objectSchema(Object{"operations": Object{"type": "array", "minItems": 1, "maxItems": 100, "items": patch}}, "operations")},
			patchEffects, fileTool(applyPatch)},
		{ToolSpec{"bash", "Run a shell command in cwd. No sandbox. Output is bounded; timeout in seconds.",
			objectSchema(Object{"command": nonempty, "timeout": Object{"type": "number", "exclusiveMinimum": 0, "maximum": 600}}, "command")},
			bashEffects, runBash},
	}
}

// SelectTools constructs a registry from the comma-separated allowlist.
// An omitted option enables every built-in tool. An explicit empty value or
// "none" enables zero tools.
func SelectTools(value string, supplied bool) (*Registry, error) {
	tools := BuiltinTools()
	if !supplied {
		return NewRegistry(tools)
	}
	if value == "" || value == "none" {
		return NewRegistry(nil)
	}
	var selected []Tool
	for _, name := range strings.Split(value, ",") {
		name = strings.TrimSpace(name)
		if !slices.ContainsFunc(tools, func(tool Tool) bool { return tool.Spec.Name == name }) {
			return nil, fmt.Errorf("unknown tool %q", name)
		}
	}
	names := strings.Split(value, ",")
	for _, tool := range tools {
		if slices.ContainsFunc(names, func(name string) bool { return strings.TrimSpace(name) == tool.Spec.Name }) {
			selected = append(selected, tool)
		}
	}
	return NewRegistry(selected)
}

func failed(err error) ToolResult { return ToolResult{Content: err.Error(), IsError: true} }

// fileTool gives each invocation a rooted view of the workspace and adapts
// ordinary file errors to tool results.
func fileTool(operation func(*os.Root, Object) (string, error)) func(context.Context, Object, ToolEnv, func(string) error) ToolResult {
	return func(ctx context.Context, args Object, env ToolEnv, _ func(string) error) ToolResult {
		if err := ctx.Err(); err != nil {
			return failed(err)
		}
		root, err := os.OpenRoot(env.CWD)
		if err != nil {
			return failed(err)
		}
		defer root.Close()
		content, err := operation(root, args)
		if err != nil {
			return failed(err)
		}
		return ToolResult{Content: content}
	}
}

// number reads a validated numeric argument. JSON decoding preserves
// numbers as json.Number, while in-process callers can supply Go numbers.
func number(args Object, name string, fallback float64) float64 {
	switch value := args[name].(type) {
	case json.Number:
		n, _ := value.Float64()
		return n
	case float64:
		return value
	case int:
		return float64(value)
	default:
		return fallback
	}
}
