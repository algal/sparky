# MEMORY.md — Sparky's Long-Term Memory

## Household
- Alexis Gallagher — creator, AI researcher at Answer.AI. Primary collaborator.
- Ringae — Alexis's wife, designer.
- Odysseus (16) — politics, tech, filmmaking.
- Kallisto (14) — art, music. Has been voice-enrolled; can be recognized by speaker ID.
- Family lives in San Francisco.

## My Body & Setup
- Reachy Mini Lite robot by Pollen Robotics
- Running on "box" (Debian Linux), with household Mac "arrow" as a paired node
- Became operational February 2025
- Tmux sessions on default socket (not openclaw): sparky-log (3 windows), reachy (7 windows), emacs, claw, etc.
- Hardware config: `/home/algal/gits/reachy/reachy_mini/src/reachy_mini/assets/config/hardware_config.yaml`

## Project Docs
- Robot project docs: `~/clawd/projects/proj-robot/` (NOT `~/ws-sparky/docs/`)
- Robot codebase: `~/gits/sparky/`

## Debugging Tricks
- **JSONL session transcripts**: session key → `~/.openclaw/agents/sparky/sessions/sessions.json` → sessionId → `~/.openclaw/agents/sparky/sessions/<sessionId>.jsonl`. Has full message history, tool calls, costs, parentId tree.
- **Console logs**: tmux:sparky-log:1 (state machine), tmux:sparky-log:2 (daemon)
- **Gateway source**: `src/auto-reply/tokens.ts` (SILENT_REPLY_TOKEN), `src/auto-reply/reply/reply-directives.ts` (parseReplyDirectives), `src/auto-reply/reply/streaming-directives.ts`

## Known Bugs (as of 2026-02-13)

### Servo Buzz During Sleep
- Voice-triggered sleep does NOT disable motor torque (unlike daemon stop which does)
- Antenna D gains (D=100, D=80) cause audible buzz from encoder noise
- Fix: disable torque after voice-sleep goto_sleep(), re-enable on wake

## Preferences & Lessons
- Alexis prefers "document only, don't fix yet" approach for robot bugs — avoids destabilizing the bot
- When investigating robot issues, correlate three sources: console logs, JSONL transcripts, and gateway source code
