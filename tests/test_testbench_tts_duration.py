import struct


def _make_streaming_wav_bytes(*, sample_rate: int, channels: int, bits_per_sample: int, data_bytes: bytes) -> bytes:
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align

    riff = b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
    fmt = (
        b"fmt "
        + struct.pack("<I", 16)
        + struct.pack(
            "<HHIIHH",
            1,  # PCM
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
    )
    data = b"data" + struct.pack("<I", 0xFFFFFFFF) + data_bytes
    return riff + fmt + data


def test_wav_duration_handles_streaming_placeholder_sizes() -> None:
    from sparky_mvp.core.middlewares.testbench_tts import _wav_duration_seconds

    sr = 24000
    ch = 1
    bps = 16
    duration_s = 0.5
    data_len = int(sr * ch * (bps // 8) * duration_s)
    wav_bytes = _make_streaming_wav_bytes(
        sample_rate=sr,
        channels=ch,
        bits_per_sample=bps,
        data_bytes=b"\x00" * data_len,
    )

    got = _wav_duration_seconds(wav_bytes)
    assert got is not None
    assert abs(got - duration_s) < 1e-3

