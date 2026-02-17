---
name: solveit
description: Read, edit, and execute SolveIt dialogs (notebooks) via CLI. Add code/note/prompt cells, run them, and inspect outputs.
metadata: {"openclaw":{"emoji":"📓","requires":{"bins":["uv"],"env":["SOLVEIT_TOKEN"]}}}
---

# SolveIt Dialog Tools

Manipulate **SolveIt dialogs** (similar to Jupyter notebooks) via CLI. A dialog contains **messages** (cells) of type `code`, `note` (markdown), `prompt` (AI prompt/response), or `raw`.

Every command takes a **dialog URL** as its first argument and returns **JSON to stdout**.

## Invocation

```bash
uv run {baseDir}/solveit_tools.py <command> [args...]
```

## Commands

### read_dialog — Read all messages

```bash
uv run {baseDir}/solveit_tools.py read_dialog "<url>"
```

Returns:
```json
{
  "url": "...",
  "name": "my-project/experiment",
  "mode": "learning",
  "messages": [
    {"id": "_ea386899", "type": "note",   "index": 0, "content": "# Title"},
    {"id": "_f70704ce", "type": "code",   "index": 1, "content": "1+1", "output": "2"},
    {"id": "_b3796dd8", "type": "prompt", "index": 2, "content": "explain this", "output": "Sure..."}
  ]
}
```

Use `id` values to reference messages in other commands. The `output` field is omitted when empty.

### add_message — Add a new message

```bash
# Append a code cell at the end (default)
uv run {baseDir}/solveit_tools.py add_message "<url>" "print('hello')"

# Add a note before a specific message
uv run {baseDir}/solveit_tools.py add_message "<url>" "# Section 2" --type note --placement before --ref-id "_f70704ce"

# Add a prompt after a specific message
uv run {baseDir}/solveit_tools.py add_message "<url>" "explain the error above" --type prompt --placement after --ref-id "_f70704ce"
```

| Flag | Values | Default | Notes |
|------|--------|---------|-------|
| `--type` | `code`, `note`, `prompt`, `raw` | `code` | |
| `--placement` | `at_end`, `at_start`, `after`, `before` | `at_end` | |
| `--ref-id` | message ID | — | Required when placement is `after` or `before` |

Returns the new message: `{"id": "_abc123", "type": "code", "content": "print('hello')"}`.

### exec_message — Execute a single message

```bash
uv run {baseDir}/solveit_tools.py exec_message "<url>" "_f70704ce"
uv run {baseDir}/solveit_tools.py exec_message "<url>" "_f70704ce" --timeout 120
```

Executes the message and waits for the result (default timeout: 60s). Returns the message with its output:

```json
{"id": "_f70704ce", "type": "code", "content": "1+1", "output": "2"}
```

### update_message — Change content or type

```bash
uv run {baseDir}/solveit_tools.py update_message "<url>" "_f70704ce" --content "2+2"
uv run {baseDir}/solveit_tools.py update_message "<url>" "_f70704ce" --type note
```

Provide `--content`, `--type`, or both. Returns the updated message.

### delete_message — Remove a message

```bash
uv run {baseDir}/solveit_tools.py delete_message "<url>" "_ea386899"
```

Returns `{"id": "_ea386899", "deleted": true}`.

### run_dialog — Execute the entire dialog

```bash
uv run {baseDir}/solveit_tools.py run_dialog "<url>"
uv run {baseDir}/solveit_tools.py run_dialog "<url>" --timeout 300
```

Runs all cells front-to-back using SolveIt's intelligent defaults: code cells are executed, prompt cells and cells with existing output are skipped. Default timeout: 120s.

Returns a summary:
```json
{"executed": 3, "skipped": 5, "errors": 0}
```

Use `read_dialog` afterward to inspect the full results.

## Typical workflow

1. **Read** the dialog to understand its current state
2. **Add** or **update** messages as needed
3. **Execute** individual messages with `exec_message`, or run everything with `run_dialog`
4. **Read** again to inspect outputs

## Message types

| Type | Content | Output | Execution |
|------|---------|--------|-----------|
| `code` | Python source | Execution result | Runs in SolveIt kernel |
| `note` | Markdown text | — (never has output) | Not executable |
| `prompt` | Natural language prompt | AI-generated response | Sends to SolveIt AI |
| `raw` | Raw text/HTML | — | Not executable |

## Rules

- **Always quote the URL** in shell commands — it contains special characters (`?`, `&`, `%`).
- Message IDs look like `_ea386899` (underscore + 8 hex chars). Always use the exact ID from `read_dialog`.
- After `run_dialog`, check results with `read_dialog` — the summary only shows counts.
- Errors from the tool are JSON on stderr: `{"error": "..."}`.
