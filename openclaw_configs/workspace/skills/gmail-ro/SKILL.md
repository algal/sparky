---
name: gmail-ro
description: “List unread emails and search Gmail messages in read-only mode via gogcli. Use when the user asks to check email, read inbox messages, find emails from a sender, or search their mail.”
---

Use this skill when the user asks things like:
- “Any unread emails?” / “Check my inbox”
- “Search Gmail for <thing>” / “Find emails about <topic>”
- “Do I have anything from <person/company> recently?”

## How to call Gmail (READ-ONLY)

Run the local wrapper script with the `exec` tool:

- Unread inbox (demo-friendly):
  - `/home/algal/.openclaw/workspace/skills/gmail-ro/bin/gmail-ro unread 10`

- Search:
  - `/home/algal/.openclaw/workspace/skills/gmail-ro/bin/gmail-ro search “<gmail query>” 10`

The wrapper enforces read-only operations (no send/modify/delete).

## Error handling

- If the command returns a non-zero exit code or empty output, check that the OpenClaw service process is running with the required environment variables.
- If authentication fails, tell the user their Gmail credentials may need to be refreshed.
- Distinguish between “no results found” (valid empty response) and an error (non-zero exit or stderr output).

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
