"""
Smoke tests for WebRtcAecmEchoCanceller (WebRTC AECM via ctypes).

These tests are intentionally conservative: they verify basic wiring, passthrough
behavior, and that echo cancellation tends to reduce energy in a simple
echo-only scenario. If the system WebRTC APM library isn't present, tests skip.
"""

import threading

import numpy as np
import pytest


def _try_make_aecm(sample_rate: int = 16000, ms_in_soundcard_buf: int = 0):
    try:
        from sparky_mvp.core.webrtc_aecm import WebRtcAecmEchoCanceller
    except Exception as e:
        pytest.skip(f"WebRtcAecmEchoCanceller unavailable: {e}")

    try:
        return WebRtcAecmEchoCanceller(sample_rate=sample_rate, ms_in_soundcard_buf=ms_in_soundcard_buf)
    except OSError as e:
        pytest.skip(f"WebRTC AECM shared library not available: {e}")
    except Exception as e:
        pytest.skip(f"Could not initialize WebRTC AECM: {e}")


class TestWebRtcAecmBasics:
    def test_construction(self):
        aec = _try_make_aecm()
        assert aec.sample_rate == 16000
        assert aec.frame_size == 160
        assert aec.ms_in_soundcard_buf == 0

    def test_passthrough_no_far_end(self):
        aec = _try_make_aecm()
        mic = (np.arange(160, dtype=np.int16) - 80).tobytes()
        out = aec.process_mic_chunk(mic)
        assert out == mic

    def test_clear(self):
        aec = _try_make_aecm()
        aec.feed_speaker_pcm(b"\x01\x00" * 160)
        assert aec.stats["speaker_buf_bytes"] > 0
        aec.clear()
        assert aec.stats["speaker_buf_bytes"] == 0


class TestWebRtcAecmEchoCancellation:
    def test_with_far_end_changes_output(self):
        aec = _try_make_aecm()

        rng = np.random.default_rng(0)
        far = (rng.normal(0, 1.0, 160 * 20) * 12000).astype(np.int16)  # 200ms
        mic = far.copy()  # pure echo
        aec.feed_speaker_pcm(far.tobytes())

        out = aec.process_mic_chunk(mic.tobytes())
        out_arr = np.frombuffer(out, dtype=np.int16)

        # With far-end reference present, AECM should do *something* (not a no-op).
        assert np.any(out_arr != mic)

    def test_echo_tends_to_reduce_energy_after_warmup(self):
        aec = _try_make_aecm()

        rng = np.random.default_rng(1)
        far = (rng.normal(0, 1.0, 160 * 60) * 12000).astype(np.int16)  # 600ms
        mic = far.copy()  # pure echo
        mic_bytes = mic.tobytes()

        aec.feed_speaker_pcm(far.tobytes())

        frame_bytes = 160 * 2
        warmup_frames = 10  # 100ms
        in_energy = 0.0
        out_energy = 0.0

        for i in range(0, len(mic_bytes), frame_bytes):
            mic_frame = mic_bytes[i : i + frame_bytes]
            out_frame = aec.process_mic_chunk(mic_frame)

            frame_idx = i // frame_bytes
            if frame_idx < warmup_frames:
                continue

            mic_arr = np.frombuffer(mic_frame, dtype=np.int16).astype(np.float64)
            out_arr = np.frombuffer(out_frame, dtype=np.int16).astype(np.float64)
            in_energy += float(np.sum(mic_arr * mic_arr))
            out_energy += float(np.sum(out_arr * out_arr))

        # Be tolerant: cancellation strength depends on platform/build, but
        # energy should generally go down for pure echo after warmup.
        assert out_energy < in_energy


class TestWebRtcAecmThreadSafety:
    def test_concurrent_feed_and_process(self):
        aec = _try_make_aecm()
        errors = []

        def feed():
            try:
                for _ in range(200):
                    aec.feed_speaker_pcm(b"\x00" * 320)
            except Exception as e:
                errors.append(e)

        def process():
            try:
                for _ in range(200):
                    aec.process_mic_chunk(b"\x00" * 320)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=feed)
        t2 = threading.Thread(target=process)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []
