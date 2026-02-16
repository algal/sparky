"""
Tests for WebRtcApmEchoCanceller (WebRTC APM / full AEC3 via C++ shim + ctypes).

These tests verify:
- The shim .so loads
- Create/destroy lifecycle
- Processing silent frames
- Speaker reference + mic processing produces correct-length output
- clear() doesn't crash
- Echo is actually reduced in a synthetic echo scenario
"""

import threading

import numpy as np
import pytest


def _try_make_apm(sample_rate: int = 16000):
    """Create an APM echo canceller, skipping if the shim isn't available."""
    try:
        from sparky_mvp.core.webrtc_apm import WebRtcApmEchoCanceller
    except Exception as e:
        pytest.skip(f"WebRtcApmEchoCanceller unavailable: {e}")

    try:
        return WebRtcApmEchoCanceller(sample_rate=sample_rate)
    except OSError as e:
        pytest.skip(f"APM shim shared library not available: {e}")
    except Exception as e:
        pytest.skip(f"Could not initialize WebRTC APM: {e}")


class TestApmShimLoads:
    def test_shim_loads(self):
        """Verify the .so can be loaded via ctypes."""
        try:
            from sparky_mvp.core.webrtc_apm import _load_apm_shim
            lib = _load_apm_shim()
            assert lib is not None
        except OSError as e:
            pytest.skip(f"APM shim not available: {e}")


class TestApmBasics:
    def test_construction(self):
        aec = _try_make_apm()
        assert aec.sample_rate == 16000
        assert aec.frame_size == 160

    def test_construction_invalid_rate(self):
        from sparky_mvp.core.webrtc_apm import WebRtcApmEchoCanceller
        with pytest.raises(ValueError, match="8/16/32/48"):
            WebRtcApmEchoCanceller(sample_rate=44100)

    def test_destroy(self):
        aec = _try_make_apm()
        aec.close()
        # Double close should not crash.
        aec.close()

    def test_process_silent_frame(self):
        aec = _try_make_apm()
        silent = b"\x00" * (160 * 2)  # 10ms of silence at 16kHz
        out = aec.process_mic_chunk(silent)
        assert len(out) == len(silent)

    def test_passthrough_no_far_end(self):
        """Without speaker reference, mic should pass through (possibly filtered)."""
        aec = _try_make_apm()
        # Use a simple pattern — note the APM may apply high-pass filter
        # and noise suppression, so output won't be identical to input,
        # but it should have the same length.
        mic = (np.arange(160, dtype=np.int16) - 80).tobytes()
        out = aec.process_mic_chunk(mic)
        assert len(out) == len(mic)

    def test_clear(self):
        aec = _try_make_apm()
        aec.feed_speaker_pcm(b"\x01\x00" * 160)
        assert aec.stats["speaker_buf_bytes"] > 0
        aec.clear()
        assert aec.stats["speaker_buf_bytes"] == 0

    def test_stats(self):
        aec = _try_make_apm()
        s = aec.stats
        assert s["impl"] == "webrtc_apm"
        assert s["frames_processed"] == 0
        assert s["frames_with_ref"] == 0
        assert s["errors"] == 0

    def test_multi_frame_processing(self):
        """Process multiple frames at once (e.g. 50ms = 5 frames)."""
        aec = _try_make_apm()
        n_frames = 5
        mic = b"\x00" * (160 * 2 * n_frames)
        out = aec.process_mic_chunk(mic)
        assert len(out) == len(mic)
        assert aec.stats["frames_processed"] == n_frames


class TestApmEchoCancellation:
    def test_with_far_end_changes_output(self):
        """When speaker reference is present, APM should modify the mic signal."""
        aec = _try_make_apm()

        rng = np.random.default_rng(0)
        far = (rng.normal(0, 1.0, 160 * 20) * 12000).astype(np.int16)  # 200ms
        mic = far.copy()  # pure echo
        aec.feed_speaker_pcm(far.tobytes())

        out = aec.process_mic_chunk(mic.tobytes())
        out_arr = np.frombuffer(out, dtype=np.int16)

        # With far-end reference present, APM should do *something* (not a no-op).
        assert np.any(out_arr != mic)

    def test_echo_tends_to_reduce_energy_after_warmup(self):
        """
        Synthetic echo test: feed identical sine wave as both speaker and mic.
        After warmup, APM should reduce echo energy.
        """
        aec = _try_make_apm()

        # Generate a 1kHz sine wave — 2 seconds of audio (200 frames of 10ms).
        duration_frames = 200
        t = np.arange(160 * duration_frames) / 16000.0
        sine = (np.sin(2 * np.pi * 1000 * t) * 10000).astype(np.int16)

        # Feed all speaker reference up front.
        aec.feed_speaker_pcm(sine.tobytes())

        # Use the same signal as mic (pure echo scenario).
        mic_bytes = sine.tobytes()
        frame_bytes = 160 * 2
        warmup_frames = 50  # 500ms warmup for AEC3 to converge

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

        # After warmup, output energy should be lower than input energy.
        # AEC3 should provide substantial cancellation on a pure echo scenario.
        assert out_energy < in_energy, (
            f"Expected echo reduction: in_energy={in_energy:.0f}, out_energy={out_energy:.0f}"
        )

    def test_echo_reduction_is_substantial(self):
        """
        Verify the cancellation is not just marginal — should reduce energy
        significantly (at least 50% = 3dB) in a pure echo scenario after warmup.
        """
        aec = _try_make_apm()

        # Generate a broadband signal (white noise) — harder for AEC but more realistic.
        rng = np.random.default_rng(42)
        duration_frames = 300  # 3 seconds
        signal = (rng.normal(0, 1.0, 160 * duration_frames) * 8000).astype(np.int16)

        aec.feed_speaker_pcm(signal.tobytes())
        mic_bytes = signal.tobytes()
        frame_bytes = 160 * 2
        warmup_frames = 100  # 1 second warmup

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

        # Should reduce by at least 50% (3dB).
        ratio = out_energy / max(in_energy, 1e-10)
        assert ratio < 0.5, (
            f"Expected substantial reduction (ratio<0.5), got ratio={ratio:.4f} "
            f"(in={in_energy:.0f}, out={out_energy:.0f})"
        )


class TestApmThreadSafety:
    def test_concurrent_feed_and_process(self):
        aec = _try_make_apm()
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
