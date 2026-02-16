---
name: solveit-notebook
description: Create, read, and open SolveIt notebooks (.ipynb) using local wrappers.
---

Use this skill when the user asks to prepare a notebook, open something in SolveIt,
or summarize an existing `.ipynb`.

## Wrapper discovery

Run this once to locate the workspace root containing this skill:

```bash
ROOT="$HOME/.openclaw/workspace"
if [[ ! -x "$ROOT/skills/solveit-notebook/bin/create_ipynb" ]]; then ROOT="/home/algal/ws-sparky"; fi
if [[ ! -x "$ROOT/skills/solveit-notebook/bin/create_ipynb" ]]; then ROOT="/home/algal/gits/sparky/openclaw_configs/workspace"; fi
```

All commands below assume those wrappers exist under:
`$ROOT/skills/solveit-notebook/bin/`

## Commands

### 1) Create notebook

```bash
"$ROOT/skills/solveit-notebook/bin/create_ipynb" /absolute/or/relative/path/demo.ipynb \
  --title "Notebook title" \
  --summary "What this notebook demonstrates" \
  --code "print('hello from solveit notebook')"
```

Alternative (convert existing markdown/python source with jupytext):

```bash
"$ROOT/skills/solveit-notebook/bin/create_ipynb" /path/out.ipynb --from /path/source.md
```

### 2) Read notebook

```bash
"$ROOT/skills/solveit-notebook/bin/read_ipynb" /path/demo.ipynb
```

This returns a compact JSON summary of cells and metadata.

### 3) Open in SolveIt

```bash
"$ROOT/skills/solveit-notebook/bin/open_solveit" /path/demo.ipynb
```

This wraps `opensolveit`. It returns JSON including:
- `url` when detected (even if browser auto-open failed)
- `ok` status
- command output for debugging

## Typical flow for "prepare and open a SolveIt notebook"

1. Create notebook with `create_ipynb`.
2. Optionally inspect with `read_ipynb`.
3. Open it with `open_solveit`.
4. Report success and include the URL if available.

## Notes

- `create_ipynb` writes valid notebook JSON (`nbformat: 4`).
- If target exists, pass `--overwrite` explicitly.
- Prefer concise demo notebooks (1 markdown + 1 code cell) unless asked for more.
