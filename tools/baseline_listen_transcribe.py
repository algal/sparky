#!/usr/bin/env python3
"""
Baseline physical test: Shure mic → VAD endpointing → Faster-Whisper STT.

This intentionally avoids robot ownership (daemon/testbench can keep running).

Usage:
  PYTHONPATH=. .venv/bin/python tools/baseline_listen_transcribe.py

Env knobs:
  MIC_CONTAINS="shure"     # substring match against PyAudio device name (case-insensitive)
  MIC_INDEX="3"            # explicit PyAudio input device index (overrides MIC_CONTAINS)
  MODEL_PATH="models/silero_vad_v5.onnx"
  VAD_THRESHOLD="0.5"
  VAD_DEBUG_EVERY="0"      # set to e.g. 25 to log audio RMS/peak every ~0.8s
  FIXED_RECORD_SECONDS="0" # if >0, bypass VAD and transcribe fixed-length clips
  WHISPER_MODEL="base"     # tiny|base|small|...
  WHISPER_DEVICE="cpu"     # cpu|cuda
  WHISPER_NO_SPEECH_THRESHOLD="1.0"  # debug: set higher to force segments even in noisy audio
  DUMP_WAV_DIR="/tmp/..."  # optional: dumps each captured utterance
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
import io
import wave
from pathlib import Path

import pyaudio

from sparky_mvp.core.vad_capture import VADSpeechCapture
from sparky_mvp.core.resampling_stream import ResamplingParams, ResamplingPyAudioStream
from sparky_mvp.core.stt_engine import STTEngine


def _iter_input_devices(pa: pyaudio.PyAudio):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            yield i, info


def _pick_input_device_index(pa: pyaudio.PyAudio) -> int:
    mic_index = os.getenv("MIC_INDEX")
    if mic_index:
        return int(mic_index)

    needle = (os.getenv("MIC_CONTAINS") or "shure").lower()
    matches: list[int] = []
    for idx, info in _iter_input_devices(pa):
        name = str(info.get("name", "")).lower()
        if needle in name:
            matches.append(idx)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise SystemExit(f"Multiple input devices match MIC_CONTAINS={needle!r}: {matches}. Set MIC_INDEX.")

    # Fallback to default input device if present
    try:
        default = pa.get_default_input_device_info()
        return int(default["index"])
    except Exception:
        pass

    available = [(idx, info.get("name", "")) for idx, info in _iter_input_devices(pa)]
    raise SystemExit(f"No input devices found matching MIC_CONTAINS={needle!r}. Available: {available}")


def main() -> int:
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("baseline_listen_transcribe")

    model_path = Path(os.getenv("MODEL_PATH") or "models/silero_vad_v5.onnx")
    if not model_path.exists():
        raise SystemExit(
            f"Missing VAD model at {model_path}. Download it before running.\n"
            f"Expected: models/silero_vad_v5.onnx",
        )

    whisper_model = os.getenv("WHISPER_MODEL") or "base"
    whisper_device = os.getenv("WHISPER_DEVICE") or "cpu"
    whisper_no_speech_threshold = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD") or "1.0")
    vad_threshold = float(os.getenv("VAD_THRESHOLD") or "0.5")
    dump_wav_dir = os.getenv("DUMP_WAV_DIR")
    fixed_record_seconds = float(os.getenv("FIXED_RECORD_SECONDS") or "0")

    pa = pyaudio.PyAudio()
    try:
        device_index = _pick_input_device_index(pa)
        dev = pa.get_device_info_by_index(device_index)
        mic_name = dev.get("name")
        print(f"[mic] index={device_index} name={mic_name!r}")
        logger.info("[mic] index=%s name=%r", device_index, mic_name)

        input_rate = int(float(dev.get("defaultSampleRate") or 44100))
        target_rate = 16000
        max_in_ch = int(dev.get("maxInputChannels") or 1)
        channels = 2 if max_in_ch >= 2 else 1
        logger.info("[mic] input_rate=%s channels=%s", input_rate, channels)
        chunk_size = 1280  # 80ms read size; VAD will read 512-sample frames internally
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=input_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
        )

        resampled_stream = ResamplingPyAudioStream(
            stream,
            ResamplingParams(input_rate_hz=input_rate, output_rate_hz=target_rate, channels=channels),
        )

        def record_fixed_wav(seconds: float) -> bytes:
            n_frames = int(target_rate * seconds)
            buf = bytearray()
            remaining = n_frames
            while remaining > 0:
                take = min(512, remaining)
                buf.extend(resampled_stream.read(take, exception_on_overflow=False))
                remaining -= take

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(target_rate)
                wf.writeframes(bytes(buf))
            wav_buffer.seek(0)
            return wav_buffer.read()

        vad = VADSpeechCapture(
            audio_stream=resampled_stream,
            sample_rate=target_rate,
            vad_model_path=str(model_path),
            vad_threshold=vad_threshold,
        )

        stt = STTEngine(
            model_size=whisper_model,
            device=whisper_device,
            language="en",
            min_speech_confidence=-1.0,
            max_no_speech_prob=0.8,
            min_compression_ratio=0.1,
            max_compression_ratio=50.0,
            min_text_length=3,
            no_speech_threshold=whisper_no_speech_threshold,
        )

        import asyncio

        async def run_loop() -> None:
            print("[loop] Speak normally; pause to end an utterance. Ctrl+C to stop.")
            utterance_n = 0
            while True:
                if fixed_record_seconds > 0:
                    await asyncio.sleep(0.05)
                    audio_wav = await asyncio.get_event_loop().run_in_executor(None, record_fixed_wav, fixed_record_seconds)
                else:
                    audio_wav = await vad.capture_speech()
                if not audio_wav:
                    continue

                utterance_n += 1

                # Quick signal check: how loud is the captured utterance?
                try:
                    with wave.open(io.BytesIO(audio_wav), "rb") as wf:
                        frames = wf.readframes(wf.getnframes())
                        import numpy as np

                        x = np.frombuffer(frames, dtype=np.int16)
                        peak = int(np.max(np.abs(x))) if len(x) else 0
                        rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2))) / 32768.0 if len(x) else 0.0
                        dur_s = (wf.getnframes() / float(wf.getframerate())) if wf.getframerate() else 0.0
                    print(f"[audio] dur={dur_s:.2f}s rms={rms:.4f} peak={peak}")
                except Exception as e:
                    print(f"[audio] (failed to inspect wav) {e!r}")

                if dump_wav_dir:
                    out_dir = Path(dump_wav_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"utt_{utterance_n:03d}.wav"
                    out_path.write_bytes(audio_wav)
                    print(f"[dump] wrote {out_path}")
                t0 = time.time()
                result = stt.transcribe(audio_wav)
                dt = time.time() - t0

                if result.is_speech:
                    print(f"[{utterance_n}] {result.text}  (stt={dt:.2f}s conf={result.confidence:.2f})")
                else:
                    print(
                        f"[{utterance_n}] (noise) text={result.text!r} stt={dt:.2f}s "
                        f"avg_logprob={result.avg_logprob:.2f} no_speech={result.no_speech_prob:.2f} "
                        f"compression={result.compression_ratio:.2f}",
                    )

        asyncio.run(run_loop())

    except KeyboardInterrupt:
        print("\n[exit] stopping...")
        return 0
    finally:
        try:
            pa.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
