# Sparky Architecture

This document describes how Sparky works at runtime: the voice loop, the
embodiment system, the OpenClaw integration, and the audio interaction
design. For installation, see `INSTALL.md`. For OpenClaw-specific setup,
see `openclaw_configs/README_openclaw.md`.

## 1. System Overview

Sparky runs as a single Python process on Linux, alongside the Reachy
Mini daemon, and connects to an OpenClaw Gateway over WebSocket.

Core runtime split:

1. **Reachy daemon**: low-level robot hardware and media backend (motors, camera, speaker).
2. **Sparky app** (`main.py` + state machine): voice loop, behavior logic, middleware pipeline.
3. **OpenClaw Gateway**: LLM orchestration, tool execution, session memory, agent definition.
4. **Optional macOS companion surfaces**: OpenClaw.app, emacs tools, Peekaboo screen capture.

## 2. Main Data and Control Flow

```
                          ┌─────────────────────────────────────────────┐
                          │           OpenClaw Gateway (WS)             │
                          │  LLM · tools · session memory · agent def   │
                          └──────┬──────────────────────────┬───────────┘
                       chat.send │                          │ event:chat deltas
                     (transcript)│                          │ (streamed text)
                                 │                          │
  ┌──────────┐    ┌──────────┐   │   ┌──────────────────┐   │   ┌──────────┐
  │   Mic    │───>│VAD+AEC   │───┼──>│  State Machine   │<──┘──>│  TTS     │───> Speaker
  │ (Shure)  │    │(Silero/  │   │   │                  │       │ (Kokoro) │
  └──────────┘    │ WebRTC)  │   │   │  SLEEP ↔ INTER-  │       └────┬─────┘
                  └────┬─────┘   │   │  ACTIVE ↔ PROC-  │            │
                       │         │   │  ESSING          │            │ wav bytes
              ┌────────┴───┐     │   └──────┬───────────┘            │
              │  STT       │     │          │                        │
              │ (Parakeet) │─────┘          │ state signals          │
              │  +Speaker  │          ┌──────┴───────────┐    ┌──────┴──────┐
              │   ID       │          │ Movement Manager │    │Head Wobbler │
              └────────────┘          │ (100 Hz loop)    │    │(speech-sync │
                                      │                  │    │ sway/roll)  │
  ┌──────────┐    ┌──────────┐        │ Composes:        │    └─────────────┘
  │  Camera  │───>│  SCRFD   │──────> │ · breathing idle │
  │ (Reachy) │    │ (face    │ face   │ · listening frz  │
  └──────────┘    │  detect) │ offsets│ · thinking escl  │
                  └──────────┘        │ · face tracking  │
                                      │ · head wobble    │
  ┌──────────┐                        │ · action queue   │
  │Wake Word │─── gates ────────────> │                  │──────> Reachy Daemon
  │(openWake │    SLEEP↔INTERACTIVE   └──────────────────┘        (motors/LEDs)
  │ Word)    │
  └──────────┘

  Node client (second WS)  ──────── registers camera.snap, action.perform
                                     with OpenClaw Gateway
```

The voice loop runs through these stages:

1. Mic audio captured locally via ALSA.
2. VAD detects speech segments; AEC subtracts the robot's own voice.
3. STT transcribes speech (Parakeet on GPU, ~21ms for 3s audio).
4. Speaker ID tags the transcript with the recognized speaker name.
5. Transcript sent to OpenClaw via `chat.send` with a voice-mode prefix.
6. OpenClaw streams response deltas, filtered by `runId`.
7. Sentence-buffered output is spoken via Kokoro TTS (~71ms synthesis latency).
8. Movement and orientation layers run continuously in parallel.

Wake/sleep gating and interruption (barge-in) are handled by the state machine.

## 3. State Machine

The state machine has three states:

- **SLEEP**: Motors in sleep posture, wake word detector active. Transitions to INTERACTIVE on wake word ("wake up sparky") or programmatic wake. A cooldown timer prevents immediate re-wake after sleep.
- **INTERACTIVE**: Breathing idle active, VAD listening for speech. Transitions to PROCESSING on speech capture, to SLEEP on sleep phrase ("go to sleep") or interactive timeout (default: 4 hours).
- **PROCESSING**: LLM request in flight, thinking animation active, TTS playing response. Transitions back to INTERACTIVE when the response completes or is interrupted by barge-in.

## 4. Audio Interaction Design

The audio path runs entirely on-device. There is no browser in the loop
and no cloud dependency for the core voice pipeline.

### On-device audio

Mic input is captured directly from ALSA (`hw:6,0` for the Shure MV5).
Speaker output goes directly to the Reachy's onboard speaker (`hw:5,0`)
via SoundDevice. This is in contrast to other published Reachy projects,
which route audio through a browser (getting WebRTC AEC for free) or
through cloud STT/TTS services.

### Acoustic echo cancellation (AEC)

A robot with a speaker and microphone in the same body will hear its own
voice. Without echo cancellation, the VAD triggers on the robot's own
speech, creating a feedback loop (the robot transcribes its own reply as
user input) or false self-interruption.

Sparky uses WebRTC Audio Processing Module (APM) with AEC3, accessed via
a C++ shim (`apm_shim.cpp`, ~176 lines) wrapped in ctypes. The TTS
playback provides the far-end reference signal. Post-AEC residual RMS is
low enough that the robot's own voice does not trigger barge-in.

### Barge-in

When the user speaks during robot speech, barge-in proceeds as follows:

1. VAD activates on user speech.
2. Post-AEC RMS is checked against a threshold (default 0.15) to distinguish real speech from residual echo or motor noise.
3. If the threshold is exceeded, the in-flight LLM response is cancelled via `chat.abort` using the current `runId`.
4. TTS playback is cancelled.
5. The new utterance is captured, transcribed, and sent as the next turn.

Stale WebSocket events from the aborted turn are filtered by `runId` so
they cannot leak into subsequent turns.

### Echo guard

As a secondary defense, recently spoken TTS text is compared against
incoming transcripts using Jaccard similarity. If the transcript closely
matches what the robot just said, it is discarded as speaker-to-mic
leakage.

## 5. OpenClaw Integration

Sparky maintains two concurrent WebSocket connections to the Gateway.

### Chat client (operator/backend role)

Handles the voice loop: `chat.send` with transcript text, consumption
of streamed `event:chat` deltas, and `chat.abort` for barge-in
cancellation.

Every `chat.send` receives an acknowledgment containing a `runId`. All
subsequent events are filtered by this `runId`. Events from prior
(aborted or completed) turns are silently dropped. This is the
mechanism that keeps barge-in clean: after an abort, any trailing events
from the cancelled run are discarded, and the new turn's events are
consumed correctly.

The transcript is wrapped in a voice-mode prefix that instructs the
agent to respond conversationally (brief, no markdown, plain speech)
and includes the identified speaker name when available.

Token auth is loaded from `OPENCLAW_GATEWAY_TOKEN` or
`~/.openclaw/openclaw.json`.

### Node client (node role)

Registers the robot as a device with the Gateway, exposing hardware
capabilities to the agent.

The node connection uses a persistent Ed25519 device identity stored at
`~/.openclaw/reachy-node-identity.json` (auto-generated on first run).
The `device.id` is the SHA256 hex of the raw public key. The handshake
includes a signed payload for authentication.

The node registers as "Reachy Mini Lite" with these commands:

| Command          | Function                                                    |
|------------------|-------------------------------------------------------------|
| `camera.snap`    | Grab frame from CameraWorker, return base64 JPEG            |
| `camera.list`    | Return camera descriptor                                    |
| `action.perform` | Queue a physical action (nod, stretch, look_around, etc.)   |
| `action.list`    | List available actions                                      |

This means the agent can decide on its own to look through the robot's
camera, or to make the robot nod or gesture, using standard OpenClaw
tool invocation. No special prompting is required.

## 6. Movement Composition

The MovementManager runs a 100Hz control loop that composes multiple
offset layers into a single motor command sent to the Reachy daemon.

Layers (evaluated every tick, summed):

1. **Breathing idle**: Subtle periodic head movement. Always active when awake.
2. **Listening freeze**: Head movement dampened during speech capture, so the robot holds still while listening.
3. **Thinking escalation**: Three-stage animation during PROCESSING (subtle 0-2s, medium 2-5s, full 5s+) with smoothstep blending between stages. Masks LLM latency — silence is never dead air.
4. **Face tracking offsets**: From SCRFD face detection via CameraWorker. Default mode is orient-on-speech: face position is snapshot at VAD activation (speech onset) and held, rather than continuously tracked.
5. **Head wobble**: Synchronized with TTS audio output. The SpeechTapper analyzes WAV PCM to generate per-hop sway and roll offsets, creating natural head movement while speaking.
6. **Action queue**: Expressive actions (nod, stretch, yawn, look_around, antenna_wiggle) queued by the agent via `action.perform` or by the spontaneous gesture system.

All layers are thread-safe and generation-tracked to handle cancellation
cleanly (e.g., when barge-in interrupts mid-speech, the head wobble
offsets reset immediately).

## 7. Spontaneous Behavior

Two independent systems create autonomous behavior when the robot is
awake but idle.

### Spontaneous gestures

A weighted random selection from recorded emotion animations (curious,
thoughtful, serenity, happy, surprised, etc.), triggered at configurable
intervals (default 15-30 minutes). Dramatic gestures are weighted lower
than subtle ones, so the robot's idle behavior feels organic rather than
performative.

### Spontaneous speech

At longer intervals (default 45-60 minutes), if the CameraWorker detects
a face in the room, the robot initiates conversation. Prompts draw from
the agent's rotating interests (defined in `INTERESTS.md` in the OpenClaw
workspace) and from contextual awareness (time of day, recent activity).

The interest rotation is managed by an OpenClaw cron job that runs weekly,
invoking the `rotate-interests` skill. The skill fetches random Wikipedia
articles, generates a candidate from the agent's own memory, selects the
most interesting option, and swaps one interest out. This gives the robot
conversational topics that evolve gradually over time.

## 8. Speaker Identification

Sparky uses a GE2E speaker encoder (resemblyzer, 256-dim embeddings) to
identify enrolled speakers from each utterance.

The pipeline:

1. WAV audio from VAD capture is converted to 16kHz mono float.
2. The encoder produces a 256-dimensional embedding (~3.9ms on GPU).
3. Cosine similarity is computed against enrolled speaker embeddings.
4. If similarity exceeds the threshold (default 0.7), the speaker name is included in the voice-mode prefix sent to the agent.

Enrollment is done offline via `tools/enroll_speakers.py` from ~30-second
WAV samples. Enrolled embeddings are stored in
`models/speaker_enrollments.json`.

The agent receives the speaker label on every turn and can personalize
responses, greet by name, and maintain distinct conversational context
per household member.

## 9. Inference Stack

All perception models run locally on GPU:

| Component      | Model                  | Latency     | GPU   |
|----------------|------------------------|-------------|-------|
| STT            | Parakeet-TDT-0.6b-v2   | ~21ms / 3s  | GPU 1 |
| TTS            | Kokoro-82M             | ~71ms       | CPU   |
| Face detection | SCRFD-500M             | ~3.9ms      | GPU 1 |
| Speaker ID     | GE2E VoiceEncoder      | ~3.9ms      | GPU   |
| VAD            | Silero VAD v5          | <1ms        | CPU   |
| Wake word      | openWakeWord (custom)  | <1ms        | CPU   |

The LLM runs via OpenClaw Gateway, which is configured to use an
external provider (Nemotron-Super-49B-v1.5 via API, swappable to other
providers by config).

## 10. Core Code Map

Entry point:

- `main.py`

Runtime orchestration:

- `sparky_mvp/core/state_machine.py`

OpenClaw clients and provider:

- `sparky_mvp/core/openclaw_gateway_client.py`
- `sparky_mvp/core/openclaw_node_client.py`
- `sparky_mvp/core/middlewares/openclaw_provider.py`

Embodiment and actions:

- `sparky_mvp/robot/camera_worker.py`
- `sparky_mvp/robot/moves.py`
- `sparky_mvp/robot/gentle_actions.py`
- `sparky_mvp/robot/thinking_move.py`

Speech and audio:

- `sparky_mvp/core/vad_capture.py`
- `sparky_mvp/core/stt_engine.py`
- `sparky_mvp/core/speaker_id.py`
- `sparky_mvp/core/middlewares/direct_tts.py`
- `sparky_mvp/core/aec.py`

Spontaneous behavior:

- `sparky_mvp/core/spontaneous_speech.py`
- `sparky_mvp/core/spontaneous_gestures.py`

## 11. Operational Notes

1. Bring-up order: Reachy daemon first, then Sparky.
2. Session key in `config.yaml` determines memory/context routing in OpenClaw.
3. All GPU models share GPU 1; GPU 0 is available for heavier workloads.
4. The install path uses a pinned `requirements.txt` with `--no-deps` to ensure reproducibility.
