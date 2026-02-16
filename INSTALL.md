# Installing and Running Sparky

For OpenClaw-specific setup, also read `openclaw_configs/README_openclaw.md`.

## Known-Good Hardware and OS

- Reachy Mini Lite (USB-connected)
- 2x NVIDIA RTX 3090
- USB microphone (tested with Shure MV5)
- Debian 13 (trixie), NVIDIA Driver 550.163.01, CUDA 12.4

## System Packages

```bash
sudo apt install -y \
  portaudio19-dev \
  espeak-ng \
  libwebrtc-audio-processing-dev \
  pkg-config \
  g++ \
  build-essential \
  python3-dev
```

Optional: `ffmpeg libgl1 libglib2.0-0 alsa-utils` (useful for troubleshooting).

## Python Environment

```bash
uv venv .venv --python 3.12.11
uv pip install --python .venv/bin/python --no-deps -r requirements.txt
```

`requirements.txt` is a fully pinned snapshot; `--no-deps` is intentional.

## Model Assets

Silero VAD and the wake word model are included in the repo under `models/`.
Parakeet STT (~3 GB) downloads automatically on first run, or you can
prefetch it:

```bash
PYTHONPATH=. .venv/bin/python tools/fetch_models.py --download-parakeet
```

Speaker enrollments (`models/speaker_enrollments.json`) are user-local and
not committed. See `docs/SPEAKER_ENROLLMENT.md` for the enrollment workflow.

## Reachy Mini Daemon

Install the Reachy Mini daemon from Pollen Robotics:
<https://github.com/pollen-robotics/reachy_mini>

Known-good baseline: `reachy_mini` v1.3.0 (commit `d5428dbd`). You may
want to tune PID values to reduce antenna tremors — see the Pollen Discord
for community-tested settings.

Start the daemon before Sparky:

```bash
uv run --frozen reachy-mini-daemon
```

## OpenClaw

OpenClaw Gateway must be running and reachable. Configure it in `config.yaml`:

```yaml
openclaw:
  provider: "openclaw"
  gateway_ws_url: "ws://127.0.0.1:18789"
  gateway_token_env: "OPENCLAW_GATEWAY_TOKEN"
  session_key: "agent:sparky:reachy"
  timeout_ms: 60000
```

The gateway token is read from `OPENCLAW_GATEWAY_TOKEN` or from
`~/.openclaw/openclaw.json` (`gateway.auth.token`).

See `openclaw_configs/README_openclaw.md` for the agent workspace,
node commands, and verification steps.

## Run

```bash
python -u main.py
```

On a healthy start you should see: Reachy initialized, Gateway connected,
STT engine ready, node client started, and the wake word / listening loop
active.

## Troubleshooting

- **Gateway won't connect**: check `gateway_ws_url` and token source.
- **Wrong session/memory**: check `openclaw.session_key`.
- **Node actions fail**: confirm the node client registered in logs; payload key is `action`, not `name`.
- **Camera tool fails**: confirm camera worker is active (`camera.enabled: true` in config).
