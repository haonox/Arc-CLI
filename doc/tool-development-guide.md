# 在 Go 版 Arc 添加工具

工具在 [internal/arc/tools.go](../internal/arc/tools.go) 注册。每个 `Tool` 包含 `ToolSpec`、`Effects` 和 `Execute`。`ToolSpec` 描述名字、用途和参数 Schema，`Effects` 根据已经校验过的参数声明影响，`Execute` 实现具体操作并返回 `ToolResult`。

以 `apply_patch` 为例，Schema 用 `oneOf` 区分 `add`、`update` 和 `delete`，拒绝未知字段。普通新增和修改声明 `write`，包含删除的补丁声明 `write` 和 `destructive`。Runtime 在调用工具前处理授权。

## 执行边界

- 需要读写文件时，沿用 `fileTool` 获取工作区 `os.Root`，用 `relativePath` 计算受限路径。
- 有多个步骤时先验证全部参数和当前状态，再提交变更。`apply_patch` 在提交失败后按逆序回滚已经完成的步骤，并报告回滚错误。
- 运行时间较长的工具接受 `context.Context`，并通过回调发送有界的进度输出。
- 在 `Execute` 里返回带错误标记的 `ToolResult`，由 Runtime 保存为配对消息。工具内部无需写会话文件或直接输出终端文本。

## 验证改动

为参数契约、正常路径、错误路径和 effect 分类各保留一个针对性测试。需要授权的路径应通过 `Agent.Run` 测试，验证拒绝时没有副作用。会落盘的测试使用 `t.TempDir()`。

```bash
go test ./internal/arc
go test -race ./...
```
