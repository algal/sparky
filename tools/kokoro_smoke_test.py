#!/usr/bin/env python3
"""Smoke test for Kokoro TTS — synthesize a sentence and play it."""

import sys
import time

import numpy as np


def main():
    text = " ".join(sys.argv[1:]) or "Hello! I am your robot assistant. How can I help you today?"
    voice = "af_heart"

    print(f"Text:  {text}")
    print(f"Voice: {voice}")
    print("Initializing Kokoro TTS ...")

    t0 = time.time()
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")
    print(f"Init:  {time.time() - t0:.2f}s")

    print("Synthesizing ...")
    t0 = time.time()
    chunks = []
    for gs, ps, audio in pipeline(text, voice=voice, speed=1.0):
        if audio is not None:
            chunks.append(audio)
            print(f"  chunk: {len(audio)} samples ({len(audio)/24000:.2f}s)")

    if not chunks:
        print("ERROR: No audio produced")
        return 1

    full_audio = np.concatenate(chunks)
    elapsed = time.time() - t0
    duration = len(full_audio) / 24000
    print(f"Synth: {elapsed:.3f}s for {duration:.2f}s audio (RTF={elapsed/duration:.2f})")

    # Write WAV
    import io
    import wave

    pcm = (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm.tobytes())
    wav_bytes = buf.getvalue()

    outpath = "/tmp/kokoro_smoke_test.wav"
    with open(outpath, "wb") as f:
        f.write(wav_bytes)
    print(f"Wrote: {outpath} ({len(wav_bytes)} bytes)")

    # Try to play
    try:
        import sounddevice as sd

        print("Playing ...")
        sd.play(full_audio, samplerate=24000)
        sd.wait()
        print("Done.")
    except Exception as e:
        print(f"Could not play (install sounddevice): {e}")
        print(f"Listen manually: aplay {outpath}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
