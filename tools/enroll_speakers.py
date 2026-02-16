#!/usr/bin/env python3
"""
Enroll speakers for voice identification.

Usage:
    PYTHONPATH=. .venv/bin/python tools/enroll_speakers.py \
        --name alexis --wav /home/algal/family_voice_samples/enroll_alexis.wav \
        --name kallisto --wav /home/algal/family_voice_samples/enroll_kallistollisto.wav \
        --output models/speaker_enrollments.json

Each --name/--wav pair enrolls one speaker.  The output JSON maps names to
256-dimensional embedding vectors.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Enroll speakers for voice ID")
    parser.add_argument(
        "--name", action="append", required=True,
        help="Speaker name (repeat for each speaker)",
    )
    parser.add_argument(
        "--wav", action="append", required=True,
        help="WAV file path (repeat for each speaker, must match --name order)",
    )
    parser.add_argument(
        "--output", default="models/speaker_enrollments.json",
        help="Output JSON path (default: models/speaker_enrollments.json)",
    )
    args = parser.parse_args()

    if len(args.name) != len(args.wav):
        print("Error: number of --name and --wav arguments must match", file=sys.stderr)
        sys.exit(1)

    from sparky_mvp.core.speaker_id import enroll_from_wav

    enrollments = {}
    for name, wav_path in zip(args.name, args.wav):
        if not Path(wav_path).exists():
            print(f"Error: WAV file not found: {wav_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Enrolling {name} from {wav_path}...")
        embedding = enroll_from_wav(wav_path)
        enrollments[name] = embedding
        print(f"  → {len(embedding)}-dim embedding extracted")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(enrollments, f)
    print(f"\nSaved {len(enrollments)} enrollment(s) to {output_path}")


if __name__ == "__main__":
    main()
