---
name: gcal-ro
description: Read-only Google Calendar access for OpenClaw via gogcli (today/tomorrow/next days/search).
---

Use this skill when the user asks:
- “What’s on my calendar today/tomorrow?”
- “What are my next meetings?”
- “When is <thing>?” / “Do I have <thing> on my calendar?”

## READ-ONLY policy

This skill is read-only.
- DO NOT create/update/delete events.
- DO NOT respond to invitations.
- Only use the wrapper commands below.

## How to call Calendar

Use the `exec` tool to run these commands:

- Today’s events:
  - `/home/algal/ws-sparky/skills/gcal-ro/bin/gcal-ro today 15`

- Tomorrow’s events:
  - `/home/algal/ws-sparky/skills/gcal-ro/bin/gcal-ro tomorrow 15`

- Next N days (default 7):
  - `/home/algal/ws-sparky/skills/gcal-ro/bin/gcal-ro days 7 30`

- Search (free text) within next 30 days:
  - `/home/algal/ws-sparky/skills/gcal-ro/bin/gcal-ro search "<query>" 30 20`

## Required environment (non-interactive)

The OpenClaw service should have:
- `GOG_KEYRING_BACKEND=file`
- `GOG_KEYRING_PASSWORD=...`
- `GOG_ACCOUNT=you@gmail.com`

Optional:
- `GCAL_ID=primary` (or a specific calendar ID)

## Output handling

The wrapper returns JSON from `gog ... --json`.
Summarize for the user:
- chronological list
- include start time (local), title, and location (if present)
- if there are no events, say so
