"""Tests for DirectTTSMiddleware — WAV conversion and engine dispatch."""

import io
import struct
import wave

import numpy as np
import pytest

from sparky_mvp.core.middlewares.direct_tts import (
    DirectTTSMiddleware,
    _numpy_to_wav,
    _pcm_to_wav,
)


class TestNumpyToWav:
    def test_produces_valid_wav(self):
        audio = np.zeros(2400, dtype=np.float32)  # 0.1s of silence at 24kHz
        wav_bytes = _numpy_to_wav(audio, sample_rate=24000)
        assert wav_bytes[:4] == b"RIFF"
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            assert wf.getnframes() == 2400

    def test_clips_to_range(self):
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        wav_bytes = _numpy_to_wav(audio)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(3)
        samples = struct.unpack("<3h", frames)
        assert samples[0] == 32767   # clipped from 2.0
        assert samples[1] == -32767  # clipped from -2.0
        assert 16000 < samples[2] < 17000  # ~0.5 * 32767

    def test_empty_array(self):
        audio = np.array([], dtype=np.float32)
        wav_bytes = _numpy_to_wav(audio)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnframes() == 0


class TestPcmToWav:
    def test_wraps_pcm_correctly(self):
        pcm = b"\x00\x00" * 4800  # 0.2s of silence at 24kHz
        wav_bytes = _pcm_to_wav(pcm, sample_rate=24000, sample_width=2)
        assert wav_bytes[:4] == b"RIFF"
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            assert wf.getnframes() == 4800


class TestEngineDispatch:
    def test_constructor_stores_engine(self):
        mm = object()
        mid = DirectTTSMiddleware(media_manager=mm, tts_engine="kokoro")
        assert mid.tts_engine == "kokoro"
        assert mid.kokoro_voice == "af_heart"
        assert mid.kokoro_lang == "a"

    def test_constructor_stores_orpheus(self):
        mm = object()
        mid = DirectTTSMiddleware(
            media_manager=mm,
            tts_engine="orpheus",
            orpheus_voice="leo",
        )
        assert mid.tts_engine == "orpheus"
        assert mid.orpheus_voice == "leo"
        assert mid.orpheus_model_name == "canopylabs/orpheus-3b-0.1-ft"

    def test_constructor_stores_riva(self):
        mm = object()
        mid = DirectTTSMiddleware(
            media_manager=mm,
            tts_engine="riva",
            riva_url="http://127.0.0.1:9000/v1/audio/synthesize",
            riva_model="magpie-tts-multilingual",
            riva_voice="Magpie-Multilingual.EN-US.Jason",
            riva_language_code="en-US",
            riva_sample_rate_hz=24000,
        )
        assert mid.tts_engine == "riva"
        assert mid.riva_model == "magpie-tts-multilingual"
        assert mid.riva_voice == "Magpie-Multilingual.EN-US.Jason"
        assert mid.riva_language_code == "en-US"
        assert mid.riva_sample_rate_hz == 24000

    @pytest.mark.asyncio
    async def test_unsupported_engine_raises(self):
        mm = object()
        mid = DirectTTSMiddleware(media_manager=mm, tts_engine="invalid")
        with pytest.raises(RuntimeError, match="Unsupported tts_engine"):
            await mid._synthesize_wav("hello")

    @pytest.mark.asyncio
    async def test_openai_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mm = object()
        mid = DirectTTSMiddleware(media_manager=mm, tts_engine="openai")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await mid._synthesize_wav("hello")

    @pytest.mark.asyncio
    async def test_riva_wav_response(self, monkeypatch):
        mm = object()
        mid = DirectTTSMiddleware(
            media_manager=mm,
            tts_engine="riva",
            riva_url="http://127.0.0.1:9000/v1/audio/synthesize",
        )

        wav_bytes = _pcm_to_wav(b"\x00\x00" * 100, sample_rate=24000, sample_width=2)

        class _Resp:
            status_code = 200
            headers = {"content-type": "audio/wav"}
            text = ""
            content = wav_bytes

            @staticmethod
            def raise_for_status():
                return None

        def _fake_post(*_args, **_kwargs):
            return _Resp()

        import requests
        monkeypatch.setattr(requests, "post", _fake_post)
        out = await mid._synthesize_wav("hello")
        assert out.startswith(b"RIFF")
