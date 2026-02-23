# Sparky

[![Intro video](https://img.youtube.com/vi/DQi3WcYyCK8/maxresdefault.jpg)](https://youtu.be/DQi3WcYyCK8?si=GeK_WCQvkyvC_PSP)

Sparky is a living and useful agent, inhabiting a Reachy Mini Lite body,
orchestrated by OpenClaw for personality, skills, and multi-host span.

To see Sparky in action, check out his [Hello Sparky video](https://youtu.be/DQi3WcYyCK8?si=wHfwFMxoM4w6kW9q), or read [my blog articles](https://alexisgallagher.com).

Sparky is powered by NVIDIA hardware and AI models. Through an API
endpoint, Sparky can use Nemotron-Super-49B-v1.5 LLM for reasoning,
language and tool-calling. Locally, Sparky uses an NVIDIA RTX 3090 to
run speech-to-text (STT) with Nemotron Parakeet, text-to-speech with
Kokoro (or MagPie TTS), voice recognition with GE2E VoiceEncoder, as
well as SCRFD face detection, and wake-word detection.

Sparky has been designed to be **useful** and **alive**.

To be **alive**, Sparky starts with expressive gestures and sounds
from the Reachy Mini app codebase, but adds various elements:

- **Intelligent conversation**. Backed by frontier-strength text AI, to
  support richer discussion informed by world knowledge

- **Nuanced personality**. Designed by prompts informed by theater
  character-construction methods, and implemented via OpenClaw's agent
  personality injection and memory systems (also cf. "Genuine People
  Personalities, by Sirius Cybernetics Corp.)

- **Independent interests**. Does not just respond, but initiates
  conversation on an organically evolving mix of favorite topics,
  which evolves slowly over time in response to experience.

- **Independent actions**. Spontaneous gestures, non-uniformly
  randomized at a human timescale, for organic surprise.

- **Social awareness**. Sparky knows the identity, interests, and
  preferences of other household members, and recognizes them via
  voice recognition.

- **Relevance Detection**. Sparky is designed to be always-on, like a
  living being, to have awareness of conversation in his environment,
  but to respond to conversation only when he is addressed and
  relevant, like a real being.

To be **useful**, Sparky relies on OpenClaw and various tools/skills
to provide close integration with productivity and knowledge
workflows, relying on the following elements:

- **Strong AI**. Backed by a frontier-class AI, Sparky has strong
  capabilities to help with coding, writing, and other kinds of
  knowledge work. (Despite his unassuming form factor, Sparky is not
  prompted to handicap his underlying capabilities.)

- **Personal productivity data**. Access to calendar and email, so
  Sparky can help the user with queries about communication and
  events, and understand the context himself.

- **Shared workspace**. Sparky can see the user's mac's screen, and
  can see and manipulate the user's tmux windows and text editing
  buffers. So the user can easily direct Sparky to pay attention to what
  they are working on, and can ask Sparky to show the user material
  which Sparky has been focusing on.

- **Multi-channel, multi-host**. Via OpenClaw orchestration, Sparky
  gets a multi-node gateway architecture which can allow Sparky
  directly to access multiple macs or other hosts. Also, Sparky
  inherits multiple other communication channels (webchat, mobile
  messaging, telephony).

- **Tool/skill ecosystem**. Also via OpenClaw, Sparky gets a
  burgeoning ecosystem of typesafe tools and markdown-based skills.

## What It Demonstrates

- Real-time voice interaction: wake flow, voice-activity detection (VAD) capture, local STT, streamed responses, local TTS

- Physical embodiment: head orientation, gentle action primitives, spontaneous gestures/speech, barge-in behavior

- OpenClaw-native orchestration: agent memory, tool use, multi-host workflows, and node command integration

- Robot node capabilities exposed to OpenClaw: camera snapshots and physical action execution

## Main Technical Points

- End-to-end OpenClaw speech loop integrated into a robot runtime
- Reliable OpenClaw node registration path with persistent node identity
- Tool-call bridge for body actions (`stretch`, `nod`, `shake`, `look_around`, `antenna_wiggle`, etc.)
- Local GPU-oriented inference stack for key perception tasks (Parakeet STT, SCRFD face tracking)

## NVIDIA-Powered Pieces

- NVIDIA Nemotron-Super-49B-v1.5 AI for brains, configured as an OpenClaw AI provider
- Running locally on NVIDIA RTX 3090:
  - NeMo Parakeet, for fast local STT
  - Kokoro TTS, for fast local TTS
  - Local SCRFD face tracking on GPU

## Repository Guide

- Installation and runtime steps: `INSTALL.md`
- OpenClaw-specific setup: `openclaw_configs/README_openclaw.md`
- Architecture notes: `ARCHITECTURE.md`
- Speaker enrollment guide (optional voice ID): `docs/SPEAKER_ENROLLMENT.md`

## Quick Start

1. Follow `INSTALL.md`.
2. Start Reachy daemon.
3. Run `python -u main.py`.
