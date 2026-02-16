---
name: gmail-ro
description: Read-only Gmail access for OpenClaw (unread + search) via gogcli.
---

Use this skill when the user asks things like:
- “Any unread emails?”
- “Search Gmail for <thing>”
- “Do I have anything from <person/company> recently?”

## How to call Gmail (READ-ONLY)

Run the local wrapper script with the `exec` tool:

- Unread inbox (demo-friendly):
  - `/home/algal/.openclaw/workspace/skills/gmail-ro/bin/gmail-ro unread 10`

- Search:
  - `/home/algal/.openclaw/workspace/skills/gmail-ro/bin/gmail-ro search "<gmail query>" 10`

The wrapper enforces read-only operations (no send/modify/delete).

## Required environment

The OpenClaw *service process* must have:
- `GOG_KEYRING_BACKEND=file`
- `GOG_KEYRING_PASSWORD=...` (so gogcli doesn’t prompt)
- `GOG_ACCOUNT=you@gmail.com` (so gog knows which account to use non-interactively)

## Output handling

The wrapper returns JSON from `gog ... --json`.
Summarize for the user in plain language:
- For unread: list up to ~10 items: from / subject / age.
- For search: show a short list of best matches and offer to refine the query.

Never reveal secrets (tokens/passwords). Never attempt to send email.
