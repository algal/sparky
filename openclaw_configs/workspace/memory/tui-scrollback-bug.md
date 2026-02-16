# TUI Scrollback Overwrite Bug — Investigation Notes

## Summary

When tool call output is displayed and then collapses (shrinks) in the OpenClaw TUI, text above the tool output gets visually overwritten/erased. This is a rendering bug in pi-mono's `packages/tui/src/tui.ts`.

## How to reproduce

### Prerequisites
- pi-mono clone at `/home/algal/gits/pi-mono` (currently v0.52.10)
- OpenClaw running with `PI_DEBUG_REDRAW=1` env var
- The TUI is the one at pts/37 started Feb 12 (or any `openclaw` TUI session)

### Steps
1. Start an OpenClaw TUI session: `PI_DEBUG_REDRAW=1 openclaw`
2. Have a conversation that generates tool calls with visible output (e.g., ask the agent to run shell commands, read files, etc.)
3. When a tool call completes and its output collapses (the expanded tool output shrinks back down), text that was visible **above** the tool output area gets overwritten/erased
4. The overwrite is visual — scrollback may look correct, but on-screen content is garbled

### Alternate trigger
- Any scenario where the TUI content shrinks by a moderate amount (5-25 lines) while the viewport is scrolled well past the top
- The bug does NOT trigger if the shrink is large enough to push `firstChanged` above `previousContentViewportTop` (that case correctly falls through to `fullRender`)

## Debug log

The debug log is at `~/.pi/agent/pi-debug.log` (enabled by `PI_DEBUG_REDRAW=1`).

There is also `PI_TUI_DEBUG=1` which dumps full render state (newLines, previousLines, escape buffer) to `/tmp/tui/render-*.log` — much more verbose but gives complete picture.

### Bug signature in debug log

Look for `diffRender` entries where `newLines < prevLines` by a moderate amount (5-30 lines) and `firstChanged` is **above** `newLines.length` but **below** `viewportTop`:

```
prevLines=557 newLines=536 ... firstChanged=529 viewportTop=500
```

After the shrink, `prevContentVPTop` drops (e.g., 500→479) while `viewportTop` stays at 500. Subsequent diff renders have inconsistent viewport state.

## Root cause analysis

In `tui.ts` render method (~line 1006):

```typescript
const previousContentViewportTop = Math.max(0, this.previousLines.length - height);
if (firstChanged < previousContentViewportTop) {
    fullRender(true);  // This is the safety catch
    return;
}
```

When content shrinks from 557→536 with height=57:
- `previousContentViewportTop = 557 - 57 = 500`
- `firstChanged = 529` (where changes start due to deleted lines)
- `529 > 500` → check passes, falls through to **diffRender** instead of fullRender

The diffRender then tries to update lines 529-556 using cursor math based on `viewportTop=500`, but:
1. New content only has 536 lines, so lines 536-556 don't exist in new content
2. The code tries to move the cursor relative to `hardwareCursorRow=555` which is now past the end
3. Lines near the viewport bottom get overwritten by the cursor repositioning

### The fix should probably be:
When content shrinks AND `firstChanged` falls within the "deleted zone" (between newLines.length and previousLines.length), trigger `fullRender` instead of diffRender. Something like:

```typescript
if (firstChanged < previousContentViewportTop || 
    (newLines.length < this.previousLines.length && firstChanged >= newLines.length)) {
    fullRender(true);
    return;
}
```

Or more conservatively: any time content shrinks by more than a few lines, just do a fullRender.

## Related

- GitHub issue #954: viewport cursor overwrite bug (originally filed by Alexis)
- Commit a6f9c3c: original fix by badlogic
- PR #962: Alexis's regression test (closed by maintainer)
- The existing regression test passes, but the bug still manifests in practice — the test covers a different code path than what happens with real tool call collapse
- Standalone repro script at `/home/algal/gits/pi-mono/packages/tui/test/viewport-overwrite-repro.ts` — does NOT reliably trigger the bug because it uses `ProcessTerminal` which has different viewport behavior than the real TUI

## Key files

- `/home/algal/gits/pi-mono/packages/tui/src/tui.ts` — the render method (~line 895-1145)
- `~/.pi/agent/pi-debug.log` — debug render log (when PI_DEBUG_REDRAW=1)
- `/tmp/tui/render-*.log` — verbose render dumps (when PI_TUI_DEBUG=1)

## State as of 2026-02-13

- We confirmed the bug still exists in v0.52.10
- Debug log captures show the exact transitions
- Have not yet attempted a fix — next step would be to modify the render method in tui.ts and test
- Alexis has a local branch of pi-mono for working on improved compaction (separate from this bug)
