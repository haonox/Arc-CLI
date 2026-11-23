# Arc CLI

Arc 是用 Go 编写的终端 Agent Runtime。它支持流式回答、可分支会话、显式上下文预算，以及经过校验和授权的工具调用。

## 构建与启动

需要 Go 1.25 或更高版本，支持 Linux 和 macOS。使用 shell 工具时，系统需要安装 Bash。

```bash
go build -o bin/arc ./cmd/arc
./bin/arc --version
./bin/arc
```

也可以用 `go install ./cmd/arc` 安装到 Go 的可执行文件目录（默认 `$(go env GOPATH)/bin`）。

## 连接模型

已有 `DEEPSEEK_API_KEY` 的终端可以直接运行 Arc。默认使用官方 DeepSeek 端点和 `deepseek-flash`，用 `--model NAME` 可以指定模型。模型请求由 Go 的 `net/http` 直连，发送 JSON 并读取 SSE。

其他兼容服务可以设置 `ARC_MODEL`、`ARC_BASE_URL`、`ARC_API_KEY`。`ARC_BASE_URL` 填写 API 根路径，Arc 会追加 `/chat/completions`。多服务配置写入 `$XDG_CONFIG_HOME/arc/config.toml`，通常是 `~/.config/arc/config.toml`。

```toml
[provider]
name = "work"

[providers.work]
base_url = "https://api.example.com/v1"
model = "your-model"
api_key_env = "MY_WORK_API_KEY"

[runtime]
context_budget = 32768
auto_compact = true
autonomous_workspaces = ["~/Projects/my-trusted-project"]
```

TOML 只保存凭据变量名，实际 Key 从启动环境读取。凭据绑定到端点，`--base-url` 切到其他地址时清除原凭据。项目里的 `.arc/config.toml` 可以用 `[provider].name` 选择已有 Provider，或设置 `[provider].model`。如果项目的 `base_url` 和用户选择的地址不同，Arc 会停止启动。自主授权只能通过用户配置或命令行开启。

用户偏好读取自 `$XDG_CONFIG_HOME/arc/AGENTS.md`，项目约定读取当前工作目录的 `AGENTS.md`。`--no-context` 跳过项目文件，`--instructions TEXT` 附加本次运行的用户指令。

## 日常使用

```bash
# 交互式任务
arc

# 单次执行、管道与程序化输出
arc --no-session --tools read -p "阅读 README.md 并概述这个项目"
printf '解释会话分支' | arc --no-session
arc --mode json --no-session "简要回答"

# 会话恢复与管理
arc --continue
arc sessions list
arc --resume ID
arc sessions rename ID "工作记录"
arc sessions archive ID
arc sessions list --archived
arc sessions restore ID
```

`arc --cwd DIR` 指定工作区，`arc --session PATH` 打开或创建指定会话文件，`arc --no-session` 使用内存会话。`arc --help` 显示完整选项。

| 交互命令 | 行为 |
| --- | --- |
| `/help`、`/status`、`/tools` | 查看帮助、运行状态和启用的工具 |
| `/approve`、`/deny` | 决定是否执行当前申请的工具调用 |
| `/abort` 或 Ctrl-C | 取消任务并清理运行中的 Bash 进程组 |
| `/steer TEXT` | 在下一次模型请求前加入指令，运行中直接输入文字也可 |
| `/follow-up TEXT` | 等当前工具循环结束后继续处理 |
| `/last-tool [N]` | 查看上一条完整工具结果或前 N 行 |
| `/history`、`/tree` | 查看当前对话与分支树 |
| `/branch ID`、`/new` | 切换对话节点或从根节点继续，已经发生的文件操作仍然有效 |
| `/compact [N]` | 总结早期对话，保留最近 N 个用户轮次 |
| `/session`、`/rename TITLE` | 查看会话路径并命名 |
| `/model [NAME]` | 查看当前模型，或在空闲时切换模型 |
| `/clear`、`/quit` | 清屏或退出，空输入时 Ctrl-D 也可退出 |

交互模式保留终端滚动记录和输入框，支持斜杠命令补全、上下键历史搜索、工具摘要和常见 Markdown 样式。`--no-markdown` 显示原文，`--no-color` 或 `NO_COLOR` 关闭颜色。

## 工具和会话

- 默认工具为 `read`、`write`、`edit`、`apply_patch`、`bash`。`--tools read` 限定为读取，`--tools none` 关闭工具。
- 默认 `confirm` 模式允许工作区内的常规文件操作。每次 Bash 调用与结构化删除都需要确认，非交互运行会拦截这类请求。`--policy autonomous` 显式授权所有已声明的 effect。`--policy restricted` 按声明的 effect 允许 `read`、`write`、`process`。
- 文件工具使用 `os.Root` 限制工作区范围。Bash 在批准后以当前用户权限运行，shell 文本分类用于授权判断。Provider 凭据始终从工具环境移除。
- 默认上下文预算约为 32,768 个估算输入 token。接近上限时可以自动摘要，仍超预算时缩减发送给模型的旧轮次或工具输出。`--context-budget` 调整预算，`--no-auto-compact` 关闭自动摘要。摘要会调用模型。
- 会话和输入历史存放在 `$XDG_STATE_HOME/arc/workspaces/`（默认 `~/.local/state/arc/workspaces/`）。Go 版可直接继续此前保存的 v1 JSONL 会话。归档移动整个文件，保留所有分支和历史。会话文件可能包含工具读到的内容。
- `--mode json` 输出 v1 JSONL 事件。退出码 `0` 表示完成，`1` 表示模型或任务错误，`2` 表示 CLI 或配置错误，`130` 表示中断。

## 阅读代码

1. [cmd/arc/main.go](cmd/arc/main.go) 和 [internal/cli/cli.go](internal/cli/cli.go) 负责信号、参数解析和运行时组装。
2. [internal/arc/agent.go](internal/arc/agent.go) 是模型、授权与工具执行的主循环。
3. [internal/arc/provider.go](internal/arc/provider.go) 实现 HTTP/SSE 适配，[internal/arc/context.go](internal/arc/context.go) 管理模型上下文。
4. [internal/arc/session.go](internal/arc/session.go) 实现 JSONL、会话树和恢复逻辑。
5. [internal/arc/tools.go](internal/arc/tools.go)、[internal/arc/files.go](internal/arc/files.go)、[internal/arc/bash.go](internal/arc/bash.go) 实现工具校验和执行，[internal/cli/terminal.go](internal/cli/terminal.go) 消费运行时事件。

更完整的调用顺序见[架构说明](doc/architecture-design.md)，新增工具见[工具开发指南](doc/tool-development-guide.md)。

```bash
go test ./...
go test -race ./...
go vet ./...
```

测试使用本地模拟模型、临时工作区和伪终端。依赖记录在 `go.mod` 和 `go.sum`。来源与许可见 [NOTICE](NOTICE) 和 [LICENSE](LICENSE)。
