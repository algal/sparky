# Attribution

## What Sparky does

Sparky is a voice-interactive robot personality built around the Reachy
Mini Lite. The core of the project is the integration work connecting
a local speech pipeline to an LLM agent (via OpenClaw) and a physical
robot, with the kind of real-time responsiveness that makes the robot
feel present rather than scripted.

The major subsystems, all original to this project:

- **OpenClaw integration** — WebSocket gateway client with streaming
  chat, abort/resend for barge-in, stale event filtering by run ID,
  and Ed25519-authenticated node registration exposing camera and
  action commands to the agent
- **Personality system** — OpenClaw agent workspace (SOUL.md, INTERESTS.md)
  with a weekly interest rotation skill using Wikipedia randomness
- **Spontaneous behavior** — presence-gated spontaneous speech (driven
  by the agent's rotating interests) and weighted gesture selection
  across various recorded emotions
- **Local TTS** — Kokoro-82M synthesis with direct ALSA playback,
  head wobbler integration, and echo-guard to prevent self-triggering
- **Acoustic echo cancellation** — WebRTC APM/AEC3 via a C++ shim
  and ctypes bridge, so the robot doesn't respond to its own voice
- **Barge-in** — RMS-gated activation suppression during playback,
  coordinated with AEC and chat abort for clean mid-sentence interruption
- **Speaker identification** — GE2E embeddings via resemblyzer, with
  a speaker enrollment pipeline and per-utterance identification
- **Wake word** — custom openWakeWord model ("wake up sparky") with
  cooldown, trained on synthetic + recorded samples
- **Face tracking** — SCRFD-based face detection driving head orientation,
  with smooth tracking and orient-on-speech at VAD activation

These subsystems make up roughly 60% of the sparky source by
line count.

## Upstream sources

The robot's movement and animation layer comes from Pollen Robotics'
[reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app) (Apache 2.0) — the movement manager,
speech tapper, head wobbler, and camera worker. Lightly modified, with
provenance headers. About 16% of the current source.

Andrew Morgan's [reachy-glados-example](https://github.com/amorgan101010/reachy-glados-example) (MIT), provided the initial
voice loop scaffold — state machine, STT wrapper, and streaming
middleware pattern. These files have been extensively modified or in
several cases removed. About 25% of the current source traces to this
starting point.


## License summary

| Source                       | License    |
|------------------------------|------------|
| reachy-glados-example        | MIT        |
| reachy_mini_conversation_app | Apache 2.0 |
| This project                 | MIT        |
