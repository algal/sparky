# OpenClaw Setup for Sparky

This document covers the OpenClaw-specific setup required to run Sparky.
It supplements the top-level `README.md`.

## 1. Required Components

Sparky's OpenClaw path expects:

1. OpenClaw Gateway running and reachable at the configured WebSocket URL.
2. Reachy Mini daemon running.
3. Sparky running with `openclaw.provider: "openclaw"`.
4. A valid Gateway auth token.
5. A valid OpenClaw `session_key` for the target agent/session.

Optional (used in this project for multi-host demos):

- OpenClaw.app on macOS (screen/tool integrations)
- [`openclaw-emacs-tools`](https://github.com/algal/openclaw-emacs-tools) plugin (emacs buffer read/write)
- Peekaboo for targeted screenshots

## 2. Configuration

### Sparky config (`config.yaml`)

```yaml
openclaw:
  provider: "openclaw"
  gateway_ws_url: "ws://127.0.0.1:18789"
  gateway_token_env: "OPENCLAW_GATEWAY_TOKEN"
  session_key: "agent:sparky:reachy"
  timeout_ms: 60000
```

`provider` must be `"openclaw"` or Sparky falls back to Anthropic direct mode.

### Gateway auth

Set the token via environment variable:

```bash
export OPENCLAW_GATEWAY_TOKEN='...'
```

Or via `~/.openclaw/openclaw.json` at `gateway.auth.token`.

## 3. Adding Nemotron as LLM Provider

Sparky's agent model is configurable. To use NVIDIA Nemotron-Super-49B-v1.5
(via OpenRouter), add a provider and model entry to `~/.openclaw/openclaw.json`:

```json
"models": {
  "mode": "merge",
  "providers": {
    "openrouter": {
      "baseUrl": "https://openrouter.ai/api/v1",
      "api": "openai-completions",
      "models": [
        {
          "id": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
          "name": "Nemotron Super 49B v1.5",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

Set `OPENROUTER_API_KEY` in your environment (OpenClaw resolves API keys
by provider name convention).

Then assign the model to the sparky agent, either in the agent definition:

```json
{
  "id": "sparky",
  "model": "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5"
}
```

Or add it as an aliased option in `agents.defaults.models`:

```json
"openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5": {
  "alias": "nemo"
}
```

This lets you switch the agent's brain between providers by changing one
line. The voice pipeline, tools, and personality are all independent of
the model choice.

## 4. Agent Definition

The sparky agent must be registered in `~/.openclaw/openclaw.json` under
`agents.list`:

```json
{
  "id": "sparky",
  "name": "sparky",
  "workspace": "~/ws-sparky",
  "model": "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5",
  "identity": {
    "name": "Sparky",
    "theme": "AI mind in a Reachy Mini Lite robot body",
    "emoji": "⚡"
  },
  "tools": {
    "alsoAllow": ["emacs-tools"]
  },
  "heartbeat": {
    "every": "6h"
  }
}
```

`model` selects the LLM provider (see section 3). `workspace` points to
the agent's workspace directory. `tools.alsoAllow` enables optional
plugins like `emacs-tools`. The `identity` fields are surfaced in
OpenClaw's UI and multi-agent contexts.

The Gateway also needs to allow the node commands Sparky registers. In
`~/.openclaw/openclaw.json` under `gateway.nodes`:

```json
"nodes": {
  "allowCommands": [
    "camera.snap",
    "camera.list",
    "action.perform",
    "action.list",
    "system.run"
  ]
}
```

## 5. Agent Workspace

A reference copy of Sparky's OpenClaw agent workspace is included at
`openclaw_configs/workspace/`. This is the agent's full definition:

- `SOUL.md` — personality, communication style, household context
- `INTERESTS.md` — rotating interests that shape spontaneous conversation
- `TOOLS.md` — capability descriptions
- `IDENTITY.md`, `AGENTS.md`, `USER.md` — agent and user identity
- `HEARTBEAT.md` — periodic awareness check instructions
- `skills/` — agent skills (rotate-interests, peekaboo, gcal-ro, gmail-ro, solveit-notebook)
- `memory/`, `MEMORY.md` — accumulated session memories

To use, set the workspace path for the `sparky` agent in
`~/.openclaw/openclaw.json`:

```json
"workspace": "~/ws-sparky"
```

Then copy the reference workspace:

```bash
cp -r openclaw_configs/workspace/* ~/ws-sparky/
```

Customize `SOUL.md` (household details), `USER.md`, and `MEMORY.md` for
your own setup. The `skills/` and `INTERESTS.md` can be used as-is.

## 6. Bring-Up and Verification

Start the Reachy daemon, then Sparky:

```bash
uv run --frozen reachy-mini-daemon
python -u main.py
```

When healthy, logs should show gateway connected, node client started,
and normal speech loop activity.

Smoke tests (run from `~/gits/sparky` with `.venv` active):

```bash
PYTHONPATH=. .venv/bin/python tools/gateway_smoke_test.py
PYTHONPATH=. .venv/bin/python tools/provider_smoke_test.py
PYTHONPATH=. .venv/bin/python tools/node_camera_smoke_test.py
```

## 7. Node Registration

When `openclaw.provider` is active, Sparky registers as an OpenClaw node
exposing commands: `camera.snap`, `camera.list`, `action.perform`,
`action.list`. The Gateway also routes standard node commands like
`system.run` (provided by the macOS OpenClaw.app node) when allowed in
the Gateway config (see section 4).

The `action.perform` payload uses `action` as the key:

```json
{"action": "stretch"}
{"action": "nod", "params": {"nods": 3}}
```

Available actions: `stretch`, `yawn`, `nod`, `shake`, `look_around`,
`antenna_wiggle`.

Node identity is persisted at `~/.openclaw/reachy-node-identity.json`
to stay stable across restarts.

## 8. Scheduled Jobs

Sparky's personality includes rotating interests (`INTERESTS.md`). Once
per week, a cron job triggers the `rotate-interests` skill in an isolated
session. The skill fetches two random Wikipedia articles, generates a third
candidate from the agent's own memory and experience, picks the most
interesting of the three, and rotates one interest out.

To create the job:

```bash
openclaw cron add \
  --agent sparky \
  --name "Rotate Sparky interests" \
  --cron "0 9 * * 1" \
  --tz "America/Los_Angeles" \
  --session isolated \
  --message "Run the rotate-interests skill." \
  --timeout-seconds 120 \
  --no-deliver
```

The skill definition lives in the workspace at `skills/rotate-interests/SKILL.md`.
The cron job is the only piece of this mechanism outside the workspace.

## 9. SolveIt Notebook Skill

The workspace also includes `skills/solveit-notebook/` for `.ipynb` workflows:

- `create_ipynb` — create a valid notebook JSON file, or convert from markdown/python via `jupytext`
- `read_ipynb` — summarize notebook cells in compact JSON
- `open_solveit` — open notebooks with `opensolveit` and return the resolved URL when available

This skill is intended for prompts like: “prepare and open a SolveIt notebook showing how this works.”

## 10. Troubleshooting

- **Gateway auth failures**: verify `OPENCLAW_GATEWAY_TOKEN` or
  `~/.openclaw/openclaw.json` token, and that `gateway_ws_url` is correct.
- **Wrong memory/session**: verify `openclaw.session_key` points to the
  intended agent/session.
- **Node actions not working**: check logs for node registration and
  `node.invoke.request` events.
- **Camera tool fails**: ensure camera worker is running and returning frames.
- **SolveIt open/create fails**: verify `jupytext` and `opensolveit` are in PATH; run the wrappers in `skills/solveit-notebook/bin/` directly to inspect JSON error output.
