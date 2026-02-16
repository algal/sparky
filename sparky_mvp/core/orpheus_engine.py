"""
Custom Orpheus TTS engine using vLLM sync API.

The orpheus-speech package's OrpheusModel uses AsyncLLMEngine with
asyncio.run() in a thread, which is broken with vLLM >= 0.12.
This module uses vLLM's synchronous LLM class instead.

Output: 24kHz PCM int16 audio bytes via SNAC neural codec.
"""

import logging
import os
import queue
import threading

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# SNAC codec (lazy-loaded on first use)
_snac_model = None
_snac_device = None


def _ensure_snac(device: str = "cuda"):
    """Load the SNAC neural audio codec (24kHz)."""
    global _snac_model, _snac_device
    if _snac_model is not None:
        return
    from snac import SNAC

    logger.info("Loading SNAC codec ...")
    _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    _snac_device = device
    logger.info("SNAC codec ready on %s", device)


def _decode_snac_tokens(token_ids: list[int]) -> bytes:
    """Decode Orpheus token IDs into PCM int16 audio bytes via SNAC."""
    _ensure_snac()
    device = _snac_device

    # Group tokens into frames of 7
    num_frames = len(token_ids) // 7
    if num_frames == 0:
        return b""

    codes_0 = []
    codes_1 = []
    codes_2 = []

    for j in range(num_frames):
        i = 7 * j
        codes_0.append(token_ids[i])
        codes_1.extend([token_ids[i + 1], token_ids[i + 4]])
        codes_2.extend([token_ids[i + 2], token_ids[i + 3],
                        token_ids[i + 5], token_ids[i + 6]])

    codes = [
        torch.tensor([codes_0], device=device, dtype=torch.int32),
        torch.tensor([codes_1], device=device, dtype=torch.int32),
        torch.tensor([codes_2], device=device, dtype=torch.int32),
    ]

    # Validate token ranges
    for c in codes:
        if torch.any(c < 0) or torch.any(c > 4096):
            return b""

    assert _snac_model is not None
    with torch.inference_mode():
        audio_hat = _snac_model.decode(codes)

    audio_np = audio_hat.detach().cpu().numpy().squeeze()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


class OrpheusEngine:
    """Synchronous Orpheus TTS engine using vLLM LLM class."""

    AVAILABLE_VOICES = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]

    def __init__(
        self,
        model_name: str = "canopylabs/orpheus-3b-0.1-ft",
        dtype=torch.bfloat16,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 2048,
        gpu_device: int = 1,
    ):
        logger.info("Initializing Orpheus engine: %s (GPU %d) ...", model_name, gpu_device)

        # vLLM spawns a subprocess for the engine core. On Linux the default
        # multiprocessing method is "fork", which inherits the parent's CUDA
        # context. If Parakeet/SCRFD already initialized CUDA in the parent,
        # the forked child gets a corrupted context → cudaErrorInitializationError.
        # Force "spawn" so the child starts with a clean CUDA state.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # vLLM selects GPU via CUDA_VISIBLE_DEVICES. Set it before creating
        # the LLM so the spawned child process inherits the right GPU.
        # Existing ONNX sessions (Parakeet, SCRFD) are unaffected — they
        # already have their sessions bound via provider_options device_id.
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.llm = LLM(
                model=model_name,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        finally:
            # Restore original CUDA_VISIBLE_DEVICES
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
        _ensure_snac(device=f"cuda:{gpu_device}")
        logger.info("Orpheus engine ready")

    def synthesize(
        self,
        text: str,
        voice: str = "tara",
        temperature: float = 0.6,
        top_p: float = 0.8,
        max_tokens: int = 1200,
        repetition_penalty: float = 1.3,
    ) -> bytes:
        """Synthesize text to PCM int16 audio bytes (24kHz mono)."""
        prompt_string = self._format_prompt(text, voice)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[49158],
            repetition_penalty=repetition_penalty,
        )

        outputs = self.llm.generate([prompt_string], sampling_params)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("Orpheus generated no output")

        # Extract token IDs from the output
        generated_ids = outputs[0].outputs[0].token_ids
        audio_token_ids = self._extract_audio_tokens(generated_ids)

        if not audio_token_ids:
            raise RuntimeError("Orpheus generated no audio tokens")

        pcm_bytes = _decode_snac_tokens(audio_token_ids)
        if not pcm_bytes:
            raise RuntimeError("SNAC decoding produced no audio")

        return pcm_bytes

    def _format_prompt(self, text: str, voice: str) -> str:
        """Format prompt with voice and special tokens."""
        adapted = f"{voice}: {text}"
        prompt_tokens = self.tokenizer(adapted, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        return self.tokenizer.decode(all_ids[0])

    def _extract_audio_tokens(self, token_ids: list[int]) -> list[int]:
        """Convert raw LLM token IDs to SNAC codec indices."""
        audio_tokens = []
        for i, tid in enumerate(token_ids):
            # Orpheus audio tokens are custom_token_N where N >= 10
            # The codec index = token_id - 10 - (position_in_frame * 4096)
            frame_pos = len(audio_tokens) % 7
            codec_id = tid - 128266 - (frame_pos * 4096)
            if 0 <= codec_id <= 4096:
                audio_tokens.append(codec_id)
        return audio_tokens
