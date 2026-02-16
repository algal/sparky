#!/usr/bin/env python3
"""Model bootstrap/check helper for Sparky.

Usage examples:
  PYTHONPATH=. .venv/bin/python tools/fetch_models.py
  PYTHONPATH=. .venv/bin/python tools/fetch_models.py --download-parakeet

What this does:
  - Checks required small local models are present.
  - Optionally pre-downloads Parakeet model assets (~3 GB) from Hugging Face.

Notes:
  - Required small models are expected to be in git:
      models/silero_vad_v5.onnx
      models/wake_up_sparky.onnx
  - Speaker enrollments are intentionally local/private and not required to run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REQUIRED_SMALL_MODELS = [
    Path("models/silero_vad_v5.onnx"),
    Path("models/wake_up_sparky.onnx"),
]

PARAKEET_DIR = Path("models/parakeet-tdt-0.6b-v2")
PARAKEET_SENTINELS = [
    PARAKEET_DIR / "encoder-model.onnx.data",
    PARAKEET_DIR / "encoder-model.int8.onnx",
    PARAKEET_DIR / "decoder_joint-model.int8.onnx",
    PARAKEET_DIR / "vocab.txt",
]

PARAKEET_REPO_ID = "istupakov/parakeet-tdt-0.6b-v2"


def _fmt_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(num_bytes)
    for unit in units:
        if val < 1024.0 or unit == units[-1]:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{num_bytes} B"


def _file_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.is_file():
        return True, _fmt_size(path.stat().st_size)
    return True, "present"


def _check_small_models() -> list[Path]:
    print("[check] Required small models")
    missing: list[Path] = []
    for p in REQUIRED_SMALL_MODELS:
        ok, meta = _file_status(p)
        tag = "OK" if ok else "MISSING"
        print(f"  - {p}: {tag} ({meta})")
        if not ok:
            missing.append(p)
    return missing


def _check_parakeet() -> bool:
    print("[check] Optional Parakeet model bundle")
    present = True
    total = 0
    for p in PARAKEET_SENTINELS:
        ok, meta = _file_status(p)
        tag = "OK" if ok else "MISSING"
        print(f"  - {p}: {tag} ({meta})")
        if ok and p.is_file():
            total += p.stat().st_size
        present = present and ok
    if present:
        print(f"  -> Parakeet appears present (checked subset total: {_fmt_size(total)})")
    else:
        print("  -> Parakeet not fully present; first STT init can auto-download it.")
    return present


def _download_parakeet() -> bool:
    print("[action] Downloading Parakeet model bundle (~3 GB) ...")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"ERROR: huggingface_hub import failed: {e}")
        print("Install dependencies first, then retry.")
        return False

    os.makedirs(PARAKEET_DIR, exist_ok=True)
    try:
        snapshot_download(repo_id=PARAKEET_REPO_ID, local_dir=str(PARAKEET_DIR))
    except Exception as e:
        print(f"ERROR: Parakeet download failed: {e}")
        return False

    print("[action] Parakeet download completed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sparky model bootstrap/check helper")
    parser.add_argument(
        "--download-parakeet",
        action="store_true",
        help="Download the optional Parakeet model bundle (~3 GB) now",
    )
    args = parser.parse_args()

    missing_small = _check_small_models()
    parakeet_present = _check_parakeet()

    if args.download_parakeet and not parakeet_present:
        ok = _download_parakeet()
        if not ok:
            return 1
        parakeet_present = _check_parakeet()

    if missing_small:
        print("\nFAIL: required local models are missing.")
        print("Expected these files to be present in the repo:")
        for p in missing_small:
            print(f"  - {p}")
        return 1

    print("\nOK: required local models are present.")
    if not parakeet_present:
        print("INFO: Parakeet is optional to prefetch; first STT init can download it automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
