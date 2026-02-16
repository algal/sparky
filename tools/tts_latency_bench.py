#!/usr/bin/env python3
"""Benchmark warm-start latency for all TTS engines.

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/tts_latency_bench.py
"""

import os
import sys
import time

import numpy as np

SENTENCES = [
    "Hello, I am your robot assistant.",
    "The weather today is sunny with a high of seventy two degrees.",
    "I think the most interesting thing about that question is how many different ways you could approach it.",
]


def bench_kokoro():
    """Benchmark Kokoro TTS (requires espeak-ng)."""
    print("\n=== Kokoro TTS ===")
    try:
        from kokoro import KPipeline
    except ImportError:
        print("  SKIP: kokoro not installed")
        return None

    print("  Initializing (cold start) ...")
    t0 = time.time()
    pipeline = KPipeline(lang_code="a")
    cold_init = time.time() - t0
    print(f"  Cold init: {cold_init:.2f}s")

    # Warm-up call
    print("  Warm-up call ...")
    for _gs, _ps, _audio in pipeline("Warm up.", voice="af_heart", speed=1.0):
        pass

    results = []
    for sentence in SENTENCES:
        t0 = time.time()
        chunks = []
        for _gs, _ps, audio in pipeline(sentence, voice="af_heart", speed=1.0):
            if audio is not None:
                chunks.append(audio)
        elapsed = time.time() - t0
        if chunks:
            full = np.concatenate(chunks)
            duration = len(full) / 24000
        else:
            duration = 0
        results.append((elapsed, duration))
        print(f"  '{sentence[:50]}...' -> {elapsed:.3f}s synth, {duration:.2f}s audio")

    avg_latency = np.mean([r[0] for r in results])
    avg_rtf = np.mean([r[0] / r[1] for r in results if r[1] > 0])
    print(f"  AVG warm latency: {avg_latency:.3f}s, RTF: {avg_rtf:.2f}")
    return results


def bench_orpheus():
    """Benchmark Orpheus TTS (requires GPU + vLLM)."""
    print("\n=== Orpheus TTS ===")

    # Use custom sync engine (orpheus-speech's async wrapper is broken with vLLM >= 0.12)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from sparky_mvp.core.orpheus_engine import OrpheusEngine
    except ImportError:
        print("  SKIP: orpheus_engine not available")
        return None

    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    model_name = os.environ.get("ORPHEUS_MODEL", "canopylabs/orpheus-3b-0.1-ft")
    print(f"  Model: {model_name}")
    print("  Initializing (cold start — model download + GPU load) ...")
    t0 = time.time()
    try:
        engine = OrpheusEngine(
            model_name=model_name,
            gpu_memory_utilization=0.35,
            max_model_len=2048,
        )
    except (OSError, Exception) as e:
        err = str(e)
        if "gated" in err.lower() or "403" in err:
            print(f"  SKIP: Model is gated. Request access at:")
            print(f"         https://huggingface.co/canopylabs/orpheus-3b-0.1-ft")
            return None
        raise
    cold_init = time.time() - t0
    print(f"  Cold init: {cold_init:.2f}s")
    print(f"  Available voices: {OrpheusEngine.AVAILABLE_VOICES}")

    voice = "tara"
    print(f"  Using voice: {voice}")

    # Warm-up call
    print("  Warm-up call ...")
    pcm = engine.synthesize("Warm up.", voice=voice)
    print(f"  Warm-up produced {len(pcm)} bytes ({len(pcm)/48000:.2f}s audio)")

    results = []
    for sentence in SENTENCES:
        t0 = time.time()
        pcm = engine.synthesize(text=sentence, voice=voice)
        elapsed = time.time() - t0
        duration = len(pcm) / (24000 * 2)  # 16-bit = 2 bytes per sample
        results.append((elapsed, duration))
        print(f"  '{sentence[:50]}...' -> {elapsed:.3f}s synth, {duration:.2f}s audio")

    avg_latency = np.mean([r[0] for r in results])
    avg_rtf = np.mean([r[0] / r[1] for r in results if r[1] > 0])
    print(f"  AVG warm latency: {avg_latency:.3f}s, RTF: {avg_rtf:.2f}")
    return results


def bench_openai():
    """Benchmark OpenAI TTS (requires OPENAI_API_KEY)."""
    print("\n=== OpenAI TTS ===")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  SKIP: OPENAI_API_KEY not set")
        return None

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Warm-up
    print("  Warm-up call ...")
    client.audio.speech.create(
        model="tts-1", voice="alloy", input="Warm up.",
        response_format="wav", speed=1.0, timeout=30.0,
    ).read()

    results = []
    for sentence in SENTENCES:
        t0 = time.time()
        resp = client.audio.speech.create(
            model="tts-1", voice="alloy", input=sentence,
            response_format="wav", speed=1.0, timeout=30.0,
        )
        wav_bytes = resp.read()
        elapsed = time.time() - t0
        # Estimate duration from WAV size (16-bit, 24kHz mono after header)
        duration = max(0, (len(wav_bytes) - 44)) / (24000 * 2)
        results.append((elapsed, duration))
        print(f"  '{sentence[:50]}...' -> {elapsed:.3f}s synth, {duration:.2f}s audio")

    avg_latency = np.mean([r[0] for r in results])
    print(f"  AVG warm latency: {avg_latency:.3f}s")
    return results


def main():
    print("TTS Latency Benchmark")
    print("=" * 60)

    kokoro_results = bench_kokoro()
    orpheus_results = bench_orpheus()
    openai_results = bench_openai()

    print("\n" + "=" * 60)
    print("SUMMARY (warm-start, avg of 3 sentences)")
    print("=" * 60)
    if kokoro_results:
        avg = np.mean([r[0] for r in kokoro_results])
        rtf = np.mean([r[0] / r[1] for r in kokoro_results if r[1] > 0])
        print(f"  Kokoro:  {avg:.3f}s avg latency, RTF {rtf:.2f}")
    if orpheus_results:
        avg = np.mean([r[0] for r in orpheus_results])
        rtf = np.mean([r[0] / r[1] for r in orpheus_results if r[1] > 0])
        print(f"  Orpheus: {avg:.3f}s avg latency, RTF {rtf:.2f}")
    if openai_results:
        avg = np.mean([r[0] for r in openai_results])
        print(f"  OpenAI:  {avg:.3f}s avg latency (network-dependent)")


if __name__ == "__main__":
    main()
