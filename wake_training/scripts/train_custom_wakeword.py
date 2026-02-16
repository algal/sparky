from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path
import shutil

import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "wakeword"


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
        chunk = r.read(1024 * 1024)
        while chunk:
            f.write(chunk)
            chunk = r.read(1024 * 1024)


def _download_with_curl(url: str, dst: Path) -> None:
    """
    Large-file-friendly download with resume support.
    Falls back to `_download` if `curl` isn't available.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    if shutil.which("curl"):
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(dst), url],
            check=True,
        )
        return
    _download(url, dst)


def _ensure_openwakeword_feature_models(openwakeword_dir: Path) -> None:
    """
    `openWakeWord/openwakeword/utils.AudioFeatures` expects these by default:
      - resources/models/melspectrogram.onnx
      - resources/models/embedding_model.onnx

    The URLs are defined in `openWakeWord/openwakeword/__init__.py` (FEATURE_MODELS).
    """
    init_py = openwakeword_dir / "openwakeword" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")

    m = re.search(r"FEATURE_MODELS\s*=\s*\{[\s\S]*?\n\}", text)
    if not m:
        raise RuntimeError(f"Could not find FEATURE_MODELS in {init_py}")

    # Minimal parse: look for the two .tflite download_url entries and derive .onnx URLs.
    tflite_urls = re.findall(r"\"download_url\"\s*:\s*\"(https?://[^\"]+\.tflite)\"", m.group(0))
    if len(tflite_urls) < 2:
        raise RuntimeError(f"Could not extract feature model download_url(s) from {init_py}")

    resources_dir = openwakeword_dir / "openwakeword" / "resources" / "models"
    for url in tflite_urls:
        name = url.split("/")[-1]
        _download(url, resources_dir / name)
        _download(url.replace(".tflite", ".onnx"), resources_dir / name.replace(".tflite", ".onnx"))


def _ensure_piper_sample_generator(repo_root: Path, repo_url: str, voice_model_url: str, voice_model_name: str) -> Path:
    psg_dir = repo_root / "piper-sample-generator-oww"
    if not psg_dir.exists():
        subprocess.run(
            ["git", "clone", repo_url, str(psg_dir)],
            check=True,
        )

    model_path = psg_dir / "models" / voice_model_name
    _download(voice_model_url, model_path)
    return psg_dir


def _write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_train_py(repo_root: Path, config_path: Path, *args: str) -> None:
    train_py = repo_root / "openWakeWord" / "openwakeword" / "train.py"
    cmd = [sys.executable, str(train_py), "--training_config", str(config_path), *args]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _make_fp_validation_from_negative_test(model_dir: Path, fp_out: Path) -> None:
    neg_test = model_dir / "negative_features_test.npy"
    if not neg_test.exists():
        raise FileNotFoundError(f"Expected file not found: {neg_test}")

    x = np.load(neg_test)
    if x.ndim != 3 or x.shape[-1] != 96:
        raise ValueError(f"Unexpected negative_features_test.npy shape: {x.shape}")

    # Flatten examples into a single timeline of frames: (frames, 96)
    fp = x.reshape(-1, x.shape[-1]).astype(np.float32)
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(fp_out, fp)


def _ensure_acav100m_features(repo_root: Path, dst_rel: str = "data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy") -> Path:
    url = (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
        "resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    )
    dst = repo_root / dst_rel
    _download_with_curl(url, dst)
    return dst


def _ensure_hf_validation_set(repo_root: Path, dst_rel: str = "data/validation_set_features.npy") -> Path:
    url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
    dst = repo_root / dst_rel
    _download(url, dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="train_custom_wakeword",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            End-to-end wrapper around `openWakeWord/openwakeword/train.py` that:
              1) Ensures TTS + openWakeWord feature models are present
              2) Writes a training YAML config
              3) Runs: generate_clips -> augment_clips -> train_model

            This is optimized for *quick iteration* (no 17GB negative feature download by default).
            """
        ).strip(),
    )
    parser.add_argument(
        "--phrase",
        action="append",
        required=True,
        help="Wake word / phrase to train; repeat flag to include multiple phrases",
    )
    parser.add_argument("--model-name", default="", help="Output model name (default: derived from --phrase)")
    parser.add_argument("--output-dir", default="my_custom_model", help="Output directory for artifacts")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of positive training samples to generate")
    parser.add_argument("--n-samples-val", type=int, default=200, help="Number of positive validation samples")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--layer-size", type=int, default=32, help="Model layer size")
    parser.add_argument("--model-type", choices=["dnn", "rnn"], default="dnn", help="Model type")
    parser.add_argument(
        "--piper-repo",
        default="https://github.com/dscripka/piper-sample-generator",
        help="Git repo URL for piper-sample-generator (openWakeWord expects dscripka's fork).",
    )
    parser.add_argument(
        "--voice-model-url",
        default="https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt",
        help="URL for the Piper voice model checkpoint",
    )
    parser.add_argument(
        "--voice-model-name",
        default="en-us-libritts-high.pt",
        help="Filename to save the Piper voice model as (under piper-sample-generator/models/)",
    )
    parser.add_argument(
        "--config-out",
        default="configs/my_model.yaml",
        help="Where to write the generated training config YAML",
    )
    parser.add_argument(
        "--negative-features",
        choices=["none", "acav100m"],
        default="none",
        help="Extra negative feature dataset to use for training (default: none).",
    )
    parser.add_argument(
        "--fp-validation",
        choices=["hf", "generated"],
        default="hf",
        help="False-positive validation set source (default: hf validation_set_features.npy).",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Create background directory but do not generate background noise clips",
    )
    parser.add_argument(
        "--convert-to-tflite",
        action="store_true",
        help="Also convert trained model ONNX -> TFLite (requires TensorFlow + onnx-tf; not installed by default).",
    )

    args = parser.parse_args()

    repo_root = _repo_root()
    openwakeword_dir = repo_root / "openWakeWord"

    phrases = [p.strip() for p in (args.phrase or []) if p.strip()]
    if not phrases:
        raise SystemExit("--phrase must be provided at least once")

    model_name = args.model_name.strip() or _slugify(phrases[0])
    output_dir = Path(args.output_dir)
    config_path = Path(args.config_out)
    steps = int(args.steps)
    if steps < 100:
        raise SystemExit("--steps must be >= 100 (openWakeWord auto-train scheduling assumes non-trivial steps)")

    # 1) Ensure local dependencies / assets
    _ensure_openwakeword_feature_models(openwakeword_dir=openwakeword_dir)
    _ensure_piper_sample_generator(
        repo_root,
        repo_url=args.piper_repo,
        voice_model_url=args.voice_model_url,
        voice_model_name=args.voice_model_name,
    )

    rir_dir = repo_root / "mit_rirs"
    rir_dir.mkdir(parents=True, exist_ok=True)

    bg_dir = repo_root / "background_clips"
    bg_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_background:
        # If empty, create a few synthetic noise clips for augmentation.
        if not any(bg_dir.glob("*.wav")):
            import scipy.io.wavfile

            sr = 16000
            seconds = 10
            n = sr * seconds
            for i in range(5):
                x = (np.random.uniform(-1, 1, n) * 0.2 * 32767).astype(np.int16)
                scipy.io.wavfile.write(bg_dir / f"noise_{i:02d}.wav", sr, x)

    # 2) Write training config
    # Start from the upstream template to avoid missing keys.
    template_path = openwakeword_dir / "examples" / "custom_model.yml"
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    if args.fp_validation == "hf":
        fp_val_path = Path("data/validation_set_features.npy").as_posix()
    else:
        fp_val_path = (Path(args.output_dir) / model_name / "fp_validation_features.npy").as_posix()

    template["model_name"] = model_name
    template["target_phrase"] = phrases
    template["n_samples"] = args.n_samples
    template["n_samples_val"] = args.n_samples_val
    template["steps"] = steps
    template["model_type"] = args.model_type
    template["layer_size"] = args.layer_size
    template["piper_sample_generator_path"] = "./piper-sample-generator-oww"
    template["output_dir"] = output_dir.as_posix()
    template["rir_paths"] = ["./mit_rirs"]
    template["background_paths"] = ["./background_clips"]
    template["background_paths_duplication_rate"] = [1]
    template["false_positive_validation_data_path"] = fp_val_path

    if args.negative_features == "acav100m":
        template["feature_data_files"] = {
            "ACAV100M_2000_hrs": "data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        }
        template["batch_n_per_class"] = {
            "ACAV100M_2000_hrs": 1024,
            "adversarial_negative": 64,
            "positive": 64,
        }
    else:
        # Quick-iteration default: no 17GB negative feature download.
        template["feature_data_files"] = {}
        template["batch_n_per_class"] = {
            "adversarial_negative": 128,
            "positive": 128,
        }

    _write_yaml(repo_root / config_path, template)

    # 3) Run the training pipeline
    _run_train_py(repo_root, repo_root / config_path, "--generate_clips")
    _run_train_py(repo_root, repo_root / config_path, "--augment_clips")

    model_dir = repo_root / args.output_dir / model_name
    if args.fp_validation == "hf":
        _ensure_hf_validation_set(repo_root)
    else:
        # Create an FP validation feature stream from the generated negative test set.
        fp_out = model_dir / "fp_validation_features.npy"
        if not fp_out.exists():
            _make_fp_validation_from_negative_test(model_dir=model_dir, fp_out=fp_out)

    if args.negative_features == "acav100m":
        _ensure_acav100m_features(repo_root)

    train_args = ["--train_model"]
    if args.convert_to_tflite:
        train_args.append("--convert_to_tflite")
    _run_train_py(repo_root, repo_root / config_path, *train_args)

    # openWakeWord currently exports to `output_dir/<model_name>.onnx` (not inside `<output_dir>/<model_name>/`).
    root_onnx = repo_root / args.output_dir / f"{model_name}.onnx"
    subdir_onnx = model_dir / f"{model_name}.onnx"
    if root_onnx.exists() and not subdir_onnx.exists():
        shutil.copy2(root_onnx, subdir_onnx)

    root_tflite = repo_root / args.output_dir / f"{model_name}.tflite"
    subdir_tflite = model_dir / f"{model_name}.tflite"
    if args.convert_to_tflite and root_tflite.exists() and not subdir_tflite.exists():
        shutil.copy2(root_tflite, subdir_tflite)

    print()
    print("Artifacts:")
    if subdir_onnx.exists():
        print(f"  {subdir_onnx}")
    elif root_onnx.exists():
        print(f"  {root_onnx}")
    if args.convert_to_tflite:
        if subdir_tflite.exists():
            print(f"  {subdir_tflite}")
        elif root_tflite.exists():
            print(f"  {root_tflite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
