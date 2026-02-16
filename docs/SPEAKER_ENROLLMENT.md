# Speaker Enrollment (Optional Voice ID)

Sparky runs without speaker enrollment. If no enrollment file is present, the app still works normally and just skips speaker identification.

This guide explains how to create a local `models/speaker_enrollments.json` for personalized voice identification.

## 1. What This Produces

- Output file: `models/speaker_enrollments.json`
- Contents: speaker-name to embedding mapping (256-d vectors)
- Use at runtime: loaded automatically if present

## 2. Recording Recommendations

- Use a quiet room.
- Record one WAV per person, ideally 10-20 seconds of natural speech.
- Keep mic setup consistent with real usage.
- Avoid strong background music/TV/noise.

The enrollment code can resample audio, but cleaner inputs improve matching quality.

## 3. Enroll One Speaker

```bash
PYTHONPATH=. .venv/bin/python tools/enroll_speakers.py \
  --name alexis \
  --wav /absolute/path/to/alexis_enroll.wav \
  --output models/speaker_enrollments.json
```

## 4. Enroll Multiple Speakers

```bash
PYTHONPATH=. .venv/bin/python tools/enroll_speakers.py \
  --name alexis   --wav /absolute/path/to/alexis_enroll.wav \
  --name kallisto --wav /absolute/path/to/kallisto_enroll.wav \
  --output models/speaker_enrollments.json
```

The `--name` and `--wav` arguments are positional pairs by order.

## 5. Quick Validation

Check that the JSON exists and has names:

```bash
test -f models/speaker_enrollments.json && echo "enrollments file present"
```

If `jq` is available:

```bash
jq 'keys' models/speaker_enrollments.json
```

Python fallback:

```bash
PYTHONPATH=. .venv/bin/python -c "import json;print(list(json.load(open('models/speaker_enrollments.json')).keys()))"
```

On startup, expected log pattern includes:

- `Speaker identification ready (<N> enrolled)` when file is present
- `Speaker ID disabled (...)` when file is absent

## 6. Common Failure Modes

1. `WAV file not found`:
- Use absolute paths for `--wav`.

2. Low/unstable identification confidence:
- Re-record with cleaner audio and longer speech.
- Keep enrollment mic path similar to runtime mic path.

3. Runtime shows speaker ID disabled:
- Confirm `models/speaker_enrollments.json` exists.
- Confirm `config.yaml` `speaker_id.enrollments_path` points to that file.

## 7. Privacy and Repo Policy

- This file encodes biometric voice embeddings.
- Treat it as sensitive and local-only.
- Do not commit personal enrollments to public git history.
