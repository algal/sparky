#!/usr/bin/env python3
"""Smoke test for Parakeet STT engine.

Records from the microphone using VAD, transcribes with both Parakeet and
Faster-Whisper, and prints a side-by-side comparison.

Usage:
    PYTHONPATH=. .venv/bin/python tools/parakeet_smoke_test.py

Env vars:
    MIC_CONTAINS=shure    # substring match for mic device name
    MIC_INDEX=3            # or explicit device index
"""

import os
import sys
import time

# Ensure CUDA libs are found (must happen before onnxruntime import)
from sparky_mvp.core.stt_engine import _ensure_cuda_ld_path
_ensure_cuda_ld_path()

from sparky_mvp.core.stt_engine import STTEngine
from sparky_mvp.core.vad_capture import VADSpeechCapture


def main():
    print("=" * 60)
    print("Parakeet STT Smoke Test")
    print("=" * 60)

    # Load both engines
    print("\n[1/3] Loading Parakeet on GPU 1...")
    t0 = time.monotonic()
    parakeet = STTEngine(
        engine="parakeet",
        parakeet_gpu_device=1,
        min_speech_confidence=-1.0,
        max_no_speech_prob=0.8,
        min_compression_ratio=0.1,
        max_compression_ratio=50.0,
        min_text_length=3,
    )
    print(f"    Parakeet ready in {time.monotonic()-t0:.1f}s")

    print("\n[2/3] Loading Faster-Whisper (base, CPU)...")
    t0 = time.monotonic()
    whisper = STTEngine(
        engine="faster_whisper",
        model_size="base",
        device="cpu",
        min_speech_confidence=-1.0,
        max_no_speech_prob=0.8,
        min_compression_ratio=0.1,
        max_compression_ratio=50.0,
        min_text_length=3,
    )
    print(f"    Faster-Whisper ready in {time.monotonic()-t0:.1f}s")

    # Set up VAD
    print("\n[3/3] Setting up microphone + VAD...")
    mic_contains = os.environ.get("MIC_CONTAINS")
    mic_index = os.environ.get("MIC_INDEX")
    if mic_index is not None:
        mic_index = int(mic_index)

    vad = VADSpeechCapture(
        vad_model_path="models/silero_vad_v5.onnx",
        mic_device_index=mic_index,
        mic_device_name_contains=mic_contains,
    )

    print("\n" + "=" * 60)
    print("Ready! Speak into the microphone. Press Ctrl+C to exit.")
    print("Each utterance will be transcribed by both engines.")
    print("=" * 60 + "\n")

    try:
        round_num = 0
        while True:
            print("Listening...")
            wav_bytes = vad.capture_speech()
            if wav_bytes is None:
                continue

            round_num += 1
            duration_ms = len(wav_bytes) / (16000 * 2) * 1000  # rough estimate
            print(f"\n--- Round {round_num} ({duration_ms:.0f}ms audio) ---")

            # Parakeet
            t0 = time.monotonic()
            p_result = parakeet.transcribe(wav_bytes)
            p_time = (time.monotonic() - t0) * 1000

            # Whisper
            t0 = time.monotonic()
            w_result = whisper.transcribe(wav_bytes)
            w_time = (time.monotonic() - t0) * 1000

            print(f"  Parakeet  ({p_time:6.0f}ms): \"{p_result.text}\"")
            print(f"  Whisper   ({w_time:6.0f}ms): \"{w_result.text}\"")
            print(f"  Speedup: {w_time/p_time:.1f}x faster" if p_time > 0 else "")
            print()

    except KeyboardInterrupt:
        print("\n\nDone!")


if __name__ == "__main__":
    main()
