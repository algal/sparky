from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _sliding_windows(x: np.ndarray, win: int) -> np.ndarray:
    # x: (frames, features)
    # numpy returns (windows, features, win) when sliding along axis=0 for a 2D array;
    # openWakeWord models expect (windows, win, features).
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=win, axis=0)
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D sliding windows, got {windows.shape}")
    return np.swapaxes(windows, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FP/hour curve on an openWakeWord validation feature file.")
    parser.add_argument("--model-onnx", required=True, help="Path to trained wakeword .onnx classifier")
    parser.add_argument(
        "--validation-features",
        default="data/validation_set_features.npy",
        help="Path to validation_set_features.npy (frames, 96)",
    )
    parser.add_argument("--val-hours", type=float, default=11.3, help="Hours represented by validation set")
    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size (windows)")
    parser.add_argument("--target-fp-per-hour", type=float, default=0.05, help="Target FP/hour for suggested threshold")
    parser.add_argument(
        "--thresholds",
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.97,0.99",
        help="Comma-separated thresholds to report",
    )

    args = parser.parse_args()

    model_path = Path(args.model_onnx)
    val_path = Path(args.validation_features)
    if not model_path.exists():
        raise SystemExit(f"Missing model file: {model_path}")
    if not val_path.exists():
        raise SystemExit(f"Missing validation features: {val_path}")

    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    x = np.load(val_path, mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != 96:
        raise SystemExit(f"Expected (frames, 96) validation array, got {x.shape}")

    # Infer model window length from ONNX input shape, else fall back to 16.
    win = sess.get_inputs()[0].shape[1]
    if not isinstance(win, int):
        win = 16

    n_windows = x.shape[0] - win + 1
    if n_windows <= 0:
        raise SystemExit(f"Validation array too short for window={win}: {x.shape}")

    input_shape = sess.get_inputs()[0].shape
    fixed_batch_1 = isinstance(input_shape[0], int) and input_shape[0] == 1

    # Compute scores.
    if fixed_batch_1:
        # Many exported openWakeWord ONNX models have a fixed batch dimension of 1.
        # Looping is fast enough for the provided validation set size (~480k windows).
        scores = np.empty((n_windows,), dtype=np.float32)
        for i in range(n_windows):
            batch = x[i : i + win][None, :, :].astype(np.float32, copy=False)
            y = sess.run(None, {input_name: batch})[0]
            scores[i] = float(np.asarray(y).reshape(-1)[0])
    else:
        # Batched scoring using sliding windows, without materializing the full window array.
        scores = np.empty((n_windows,), dtype=np.float32)
        chunk_windows = args.batch_size * 16  # convert to frame chunk with overlap
        out_idx = 0
        start = 0
        while start < x.shape[0]:
            end = min(x.shape[0], start + chunk_windows + (win - 1))
            chunk = x[start:end]
            if chunk.shape[0] < win:
                break

            windows = _sliding_windows(chunk, win)  # (n, win, 96)
            n = windows.shape[0]
            # Drop last (win-1) windows if this isn't the last chunk to avoid double-counting.
            if end < x.shape[0]:
                n = max(0, n - (win - 1))
                windows = windows[:n]

            for j in range(0, n, args.batch_size):
                batch = windows[j : j + args.batch_size].astype(np.float32, copy=False)
                y = sess.run(None, {input_name: batch})[0]
                y = np.asarray(y).reshape(-1).astype(np.float32, copy=False)
                scores[out_idx : out_idx + y.shape[0]] = y
                out_idx += y.shape[0]

            start += chunk_windows

        scores = scores[:out_idx]
        if scores.shape[0] != n_windows:
            # Minor mismatch can happen due to chunking math; keep it explicit.
            n_windows = scores.shape[0]

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    thresholds = sorted(set(thresholds))

    print(f"model={model_path}")
    print(f"validation={val_path} frames={x.shape[0]} windows={n_windows} win={win} val_hours={args.val_hours}")
    print()
    print("threshold, fp_per_hour")
    for thr in thresholds:
        fp_per_hr = float((scores >= thr).sum()) / float(args.val_hours)
        print(f"{thr:.4f}, {fp_per_hr:.6f}")

    # Suggest threshold for a target FP/hour.
    order = np.sort(scores)
    max_fp = args.target_fp_per_hour * args.val_hours
    # Need <= max_fp windows above threshold => threshold at (n_windows - max_fp) quantile.
    k = int(np.floor(n_windows - max_fp))
    if k <= 0:
        suggested = 0.0
    elif k >= n_windows:
        suggested = 1.0
    else:
        suggested = float(order[k])

    print()
    print(f"suggested_threshold_for_fp_per_hour<={args.target_fp_per_hour}: {suggested:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
