---
name: peekaboo
description: Capture and automate macOS UI with Peekaboo on the "arrow" node.
---

# Peekaboo (via arrow)

Peekaboo runs on the macOS node **"arrow"** (Alexis's Mac). You are on Linux
and cannot run it locally.

## Capturing the active window (most common use)

This is what you want when someone says "look at my screen," "what am I working
on," or "help me with this":

1. Capture and encode in one step via `system.run` on node "arrow":
   ```
   peekaboo image --mode frontmost --path /tmp/peekaboo-capture.png && base64 -i /tmp/peekaboo-capture.png
   ```
2. The base64-encoded image data is returned in the command output. Decode and
   analyze the image, then respond.

Prefer `--mode frontmost` over `--mode screen` — it captures only the focused
window, which is smaller and more relevant than the full widescreen display.

All commands use only `system.run` on node "arrow" (provided by the native Mac
app). No other node commands are needed.

## Full CLI reference

Peekaboo is a full macOS UI automation CLI: capture/inspect screens, target UI
elements, drive input, and manage apps/windows/menus. Commands share a snapshot
cache and support `--json`/`-j` for scripting.

### Core commands

- `image`: capture screenshots (screen/window/menu bar regions)
- `see`: annotated UI maps, snapshot IDs, optional analysis
- `list`: apps, windows, screens, menubar, permissions
- `permissions`: check Screen Recording/Accessibility status
- `app`: launch/quit/relaunch/hide/unhide/switch/list apps
- `window`: close/minimize/maximize/move/resize/focus/list
- `menu`: click/list application menus + menu extras

### Interaction commands

- `click`: target by ID/query/coords with smart waits
- `type`: text + control keys (`--clear`, delays)
- `hotkey`: modifier combos like `cmd,shift,t`
- `press`: special-key sequences with repeats
- `scroll`: directional scrolling (targeted + smooth)
- `drag`: drag & drop across elements/coords/Dock
- `paste`: set clipboard -> paste -> restore

### Common capture parameters

- Output: `--path`, `--format png|jpg`, `--retina`
- Targeting: `--mode screen|window|frontmost`, `--screen-index`,
  `--window-title`, `--window-id`
- Analysis: `--analyze "prompt"`, `--annotate`

### Common targeting parameters

- App/window: `--app`, `--pid`, `--window-title`, `--window-id`, `--window-index`
- Snapshot targeting: `--snapshot` (ID from `see`; defaults to latest)
- Element/coords: `--on`/`--id` (element ID), `--coords x,y`

### Examples

Capture active window:
```bash
peekaboo image --mode frontmost --path /tmp/capture.png
```

Capture a specific app's window:
```bash
peekaboo image --app Safari --mode window --path /tmp/safari.png
```

Annotated UI map (for subsequent click targeting):
```bash
peekaboo see --app Safari --annotate --path /tmp/see.png
```

See -> click -> type flow:
```bash
peekaboo see --app Safari --annotate --path /tmp/see.png
peekaboo click --on B3 --app Safari
peekaboo type "search query" --app Safari --return
```

Open a URL:
```bash
open "https://example.com"
```

All commands run via `system.run` on node "arrow". To retrieve image results,
append `&& base64 -i <path>` to the capture command.
