from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


TRIAL_SEC = 16.0
IMAGINE_SEC = 2.0
WINDOW_SEC = 0.2
STEP_SEC = 0.05

BANDS_HZ: list[tuple[float, float]] = [
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
    (32.0, 40.0),
]

DIRECTION_TO_ID = {"front": 0, "back": 1, "left": 2, "right": 3}
ID_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_ID.items()}

KEY_TO_DIRECTION = {
    "w": "front",
    "up": "front",
    "s": "back",
    "down": "back",
    "a": "left",
    "left": "left",
    "d": "right",
    "right": "right",
}

JP_LABEL_TO_DIRECTION = {
    "\u524d": "front",
    "\u5f8c": "back",
    "\u5de6": "left",
    "\u53f3": "right",
}

EEG_COORDS = {
    "AF7": (-0.60, 0.90),
    "Fp1": (-0.35, 1.00),
    "Fp2": (0.35, 1.00),
    "AF8": (0.60, 0.90),
    "F3": (-0.50, 0.45),
    "F4": (0.50, 0.45),
    "P3": (-0.50, -0.35),
    "P4": (0.50, -0.35),
    "P7": (-0.90, -0.25),
    "O1": (-0.35, -0.90),
    "O2": (0.35, -0.90),
    "P8": (0.90, -0.25),
}

PHASE_PAIRS = [
    ("AF7", "Fp1"),
    ("AF8", "Fp2"),
    ("Fp1", "F3"),
    ("Fp2", "F4"),
    ("F3", "P3"),
    ("F4", "P4"),
    ("P3", "O1"),
    ("P4", "O2"),
    ("P7", "P3"),
    ("P8", "P4"),
    ("P7", "O1"),
    ("P8", "O2"),
    ("AF7", "F3"),
    ("AF8", "F4"),
    ("Fp1", "Fp2"),
    ("F3", "F4"),
    ("P3", "P4"),
    ("O1", "O2"),
]


@dataclass
class Segment:
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class AlignmentResult:
    offset: int
    slope: float
    intercept: float
    rmse_sec: float
    extra_pulses_before: list[int]
    extra_pulses_after: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Photodiode-log alignment + IMAGINE EEG extraction + 4-class comparison."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("map/rawdata/RecordingS17R000_CSV"),
        help="Directory with wert_eeg.csv, wert_photodiode.csv, info.json",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Session log CSV path. If omitted, the latest map/logs/session_*.csv is used.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("map/logs"),
        help="Used when --log-csv is omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("map/analysis_results"),
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--no-torch", action="store_true", help="Skip CNN/Transformer even if torch is available.")
    return parser.parse_args()


def read_csv_with_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise RuntimeError(f"Failed to read CSV: {path}") from last_error


def find_latest_log(logs_dir: Path) -> Path:
    candidates = sorted(logs_dir.glob("session_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No session log files under: {logs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def vector_to_direction(dx: int, dy: int) -> str | None:
    if dx == 0 and dy == -1:
        return "front"
    if dx == 0 and dy == 1:
        return "back"
    if dx == -1 and dy == 0:
        return "left"
    if dx == 1 and dy == 0:
        return "right"
    return None


def detect_segments(binary_mask: np.ndarray) -> list[Segment]:
    if binary_mask.ndim != 1:
        raise ValueError("binary_mask must be 1-D")
    edges = np.diff(binary_mask.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if binary_mask[0]:
        starts = np.r_[0, starts]
    if binary_mask[-1]:
        ends = np.r_[ends, binary_mask.size]
    return [Segment(int(s), int(e)) for s, e in zip(starts, ends)]


def longest_periodic_chain(
    starts: np.ndarray,
    expected_period: int,
    tolerance: int,
) -> tuple[int, int]:
    if starts.size == 0:
        raise ValueError("No pulse starts given.")
    if starts.size == 1:
        return 0, 1

    best_start = 0
    best_len = 1
    chain_start = 0

    for i in range(1, starts.size):
        gap = starts[i] - starts[i - 1]
        if abs(gap - expected_period) <= tolerance:
            continue

        chain_len = i - chain_start
        if chain_len > best_len:
            best_start = chain_start
            best_len = chain_len
        chain_start = i

    chain_len = starts.size - chain_start
    if chain_len > best_len:
        best_start = chain_start
        best_len = chain_len

    return best_start, best_start + best_len


def detect_periodic_photodiode_pulses(
    pd_signal: np.ndarray,
    fs: float,
    trial_sec: float = TRIAL_SEC,
) -> dict[str, Any]:
    low = float(np.quantile(pd_signal, 0.20))
    high = float(np.quantile(pd_signal, 0.99))
    threshold = low + 0.5 * (high - low)
    mask = pd_signal > threshold
    all_segments = detect_segments(mask)

    min_dur = int(round(0.08 * fs))
    max_dur = int(round(0.50 * fs))
    short_segments = [seg for seg in all_segments if min_dur <= seg.duration <= max_dur]
    long_segments = [seg for seg in all_segments if seg.duration > max_dur]
    if not short_segments:
        raise RuntimeError("No short photodiode pulses found.")

    starts = np.array([seg.start for seg in short_segments], dtype=np.int64)
    expected_period = int(round(trial_sec * fs))
    tolerance = int(round(expected_period * 0.15))
    i0, i1 = longest_periodic_chain(starts, expected_period=expected_period, tolerance=tolerance)
    periodic_segments = short_segments[i0:i1]
    if len(periodic_segments) < 2:
        raise RuntimeError("Periodic photodiode chain is too short.")

    return {
        "threshold": threshold,
        "min_dur_samples": min_dur,
        "max_dur_samples": max_dur,
        "all_segments": all_segments,
        "short_segments": short_segments,
        "long_segments": long_segments,
        "periodic_segments": periodic_segments,
        "expected_period_samples": expected_period,
        "period_tolerance_samples": tolerance,
    }


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return float(slope), float(intercept), rmse, pred


def align_pulses_to_log(
    pulse_starts: np.ndarray,
    log_imagine_times: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, AlignmentResult]:
    n_pulse = pulse_starts.size
    n_log = log_imagine_times.size
    if n_pulse < n_log:
        raise RuntimeError(
            f"Not enough pulses ({n_pulse}) for log IMAGINE starts ({n_log})."
        )

    if n_pulse == n_log:
        candidate_offsets = [0]
    elif n_pulse == n_log + 1:
        candidate_offsets = [0, 1]
    else:
        candidate_offsets = list(range(0, n_pulse - n_log + 1))

    best: AlignmentResult | None = None
    best_starts: np.ndarray | None = None

    for offset in candidate_offsets:
        selected = pulse_starts[offset : offset + n_log]
        x = selected.astype(np.float64) / fs
        slope, intercept, rmse, _ = fit_affine(x, log_imagine_times)
        candidate = AlignmentResult(
            offset=offset,
            slope=slope,
            intercept=intercept,
            rmse_sec=rmse,
            extra_pulses_before=pulse_starts[:offset].astype(int).tolist(),
            extra_pulses_after=pulse_starts[offset + n_log :].astype(int).tolist(),
        )
        if best is None or candidate.rmse_sec < best.rmse_sec:
            best = candidate
            best_starts = selected

    assert best is not None and best_starts is not None
    return best_starts, best


def load_sampling_rate(raw_dir: Path) -> float:
    info_path = raw_dir / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    signals = {x["id"]: x for x in info["signals"]}
    eeg_fs = float(signals["wert_eeg"]["sampling_rate"])
    pd_fs = float(signals["wert_photodiode"]["sampling_rate"])
    if not math.isclose(eeg_fs, pd_fs):
        raise RuntimeError(f"Sampling mismatch: eeg={eeg_fs}, photodiode={pd_fs}")
    return eeg_fs


def load_trial_directions(log_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = log_df.sort_values("t").groupby("trial_idx", sort=True)
    for trial_idx_raw, trial_df in grouped:
        trial_idx = int(trial_idx_raw)
        trial_df = trial_df.sort_values("t")

        move_rows = trial_df[(trial_df["event"] == "PHASE_START") & (trial_df["phase"] == "MOVE")]
        move_dx = int(round(float(move_rows["move_dx"].iloc[0]))) if not move_rows.empty else 0
        move_dy = int(round(float(move_rows["move_dy"].iloc[0]))) if not move_rows.empty else 0
        direction = vector_to_direction(move_dx, move_dy)

        manual_keys = (
            trial_df.loc[trial_df["phase"] == "MANUAL", "manual_key"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        manual_key = next((k for k in manual_keys if k), "")
        if direction is None and manual_key:
            direction = KEY_TO_DIRECTION.get(manual_key)

        if direction is None:
            labels = (
                trial_df["label"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            for label in labels:
                direction = JP_LABEL_TO_DIRECTION.get(label)
                if direction is not None:
                    break

        class_id = DIRECTION_TO_ID.get(direction, -1)
        rows.append(
            {
                "trial_idx": trial_idx,
                "move_dx": move_dx,
                "move_dy": move_dy,
                "manual_key": manual_key,
                "direction": direction if direction is not None else "",
                "class_id": class_id,
            }
        )

    return pd.DataFrame(rows).sort_values("trial_idx").reset_index(drop=True)


def build_channel_groups(channel_names: list[str]) -> dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(channel_names)}
    left_names = ["AF7", "Fp1", "F3", "P3", "P7", "O1"]
    right_names = ["AF8", "Fp2", "F4", "P4", "P8", "O2"]
    front_names = ["AF7", "Fp1", "Fp2", "AF8", "F3", "F4"]
    post_names = ["P3", "P4", "P7", "O1", "O2", "P8"]

    def ids(names: list[str]) -> np.ndarray:
        present = [idx[n] for n in names if n in idx]
        if not present:
            raise RuntimeError(f"Missing channel group in data: {names}")
        return np.array(present, dtype=np.int64)

    return {
        "left": ids(left_names),
        "right": ids(right_names),
        "front": ids(front_names),
        "posterior": ids(post_names),
    }


def build_phase_pairs(channel_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {name: i for i, name in enumerate(channel_names)}
    pairs: list[tuple[int, int]] = []
    vectors: list[tuple[float, float]] = []
    for a, b in PHASE_PAIRS:
        if a not in idx or b not in idx:
            continue
        if a not in EEG_COORDS or b not in EEG_COORDS:
            continue
        x0, y0 = EEG_COORDS[a]
        x1, y1 = EEG_COORDS[b]
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist <= 1e-9:
            continue
        pairs.append((idx[a], idx[b]))
        vectors.append((dx / (dist * dist), dy / (dist * dist)))

    if not pairs:
        raise RuntimeError("No valid phase-gradient pairs could be created.")

    return np.array(pairs, dtype=np.int64), np.array(vectors, dtype=np.float64)


def circular_mean_phase_diff(phase_diff: np.ndarray) -> float:
    c = np.exp(1j * phase_diff)
    return float(np.angle(np.mean(c)))


def extract_feature_tensor(
    epochs: np.ndarray,
    channel_names: list[str],
    fs: float,
    bands_hz: list[tuple[float, float]] = BANDS_HZ,
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    n_trials, _n_ch, n_samples = epochs.shape
    if n_trials == 0:
        raise RuntimeError("No epochs available for feature extraction.")

    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if win <= 1 or step <= 0:
        raise RuntimeError("Window or step size is invalid.")
    starts = np.arange(0, n_samples - win + 1, step, dtype=np.int64)
    if starts.size == 0:
        raise RuntimeError("No sliding windows can be created.")

    groups = build_channel_groups(channel_names)
    pair_idx, pair_vec = build_phase_pairs(channel_names)
    eps = 1e-9

    sos_bank = []
    nyq = fs * 0.5
    for lo, hi in bands_hz:
        if lo <= 0 or hi >= nyq:
            raise RuntimeError(f"Band out of range: ({lo}, {hi}) Hz with Nyquist={nyq}")
        sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        sos_bank.append(sos)

    features = np.zeros((n_trials, starts.size, len(bands_hz), 4), dtype=np.float32)

    for ti in range(n_trials):
        x = epochs[ti]  # [C, T]
        for bi, sos in enumerate(sos_bank):
            filtered = sosfiltfilt(sos, x, axis=1)
            analytic = hilbert(filtered, axis=1)
            power = np.abs(analytic) ** 2
            phase = np.angle(analytic)

            for wi, st in enumerate(starts):
                ed = int(st + win)
                p = power[:, st:ed]

                left_p = float(np.mean(p[groups["left"]]))
                right_p = float(np.mean(p[groups["right"]]))
                front_p = float(np.mean(p[groups["front"]]))
                post_p = float(np.mean(p[groups["posterior"]]))

                lr_ratio = math.log((left_p + eps) / (right_p + eps))
                fp_ratio = math.log((front_p + eps) / (post_p + eps))

                gx = 0.0
                gy = 0.0
                for (i, j), (vx, vy) in zip(pair_idx, pair_vec):
                    diff = phase[j, st:ed] - phase[i, st:ed]
                    mean_diff = circular_mean_phase_diff(diff)
                    gx += mean_diff * vx
                    gy += mean_diff * vy
                gx /= pair_idx.shape[0]
                gy /= pair_idx.shape[0]

                features[ti, wi, bi, 0] = lr_ratio
                features[ti, wi, bi, 1] = fp_ratio
                features[ti, wi, bi, 2] = gx
                features[ti, wi, bi, 3] = gy

    return features, starts


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_sklearn_model(
    model_name: str,
    estimator: Any,
    x: np.ndarray,
    y: np.ndarray,
    splitter: StratifiedKFold,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for fold_idx, (tr, te) in enumerate(splitter.split(x, y), start=1):
        est = clone(estimator)
        est.fit(x[tr], y[tr])
        pred = est.predict(x[te])

        fold_metrics = classification_metrics(y[te], pred)
        fold_rows.append({"fold": fold_idx, **fold_metrics})
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())

    y_true_np = np.array(y_true_all, dtype=np.int64)
    y_pred_np = np.array(y_pred_all, dtype=np.int64)
    overall = classification_metrics(y_true_np, y_pred_np)
    cm = confusion_matrix(
        y_true_np,
        y_pred_np,
        labels=[0, 1, 2, 3],
    )

    return {
        "model": model_name,
        "overall": overall,
        "folds": fold_rows,
        "confusion_matrix": cm.tolist(),
    }


if HAS_TORCH:

    class TemporalCNN(nn.Module):
        def __init__(self, in_features: int, num_classes: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, F]
            z = x.transpose(1, 2)  # [B, F, T]
            z = self.net(z).squeeze(-1)
            return self.fc(z)


    class TemporalTransformer(nn.Module):
        def __init__(
            self,
            in_features: int,
            seq_len: int,
            num_classes: int,
            d_model: int = 64,
        ) -> None:
            super().__init__()
            self.in_proj = nn.Linear(in_features, d_model)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=128,
                dropout=0.20,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.norm = nn.LayerNorm(d_model)
            self.fc = nn.Linear(d_model, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, F]
            z = self.in_proj(x) + self.pos[:, : x.size(1)]
            z = self.encoder(z)
            z = self.norm(z.mean(dim=1))
            return self.fc(z)


def standardize_sequence(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # feature-wise statistics over train set only
    tr2d = train_x.reshape(-1, train_x.shape[-1])
    mean = tr2d.mean(axis=0, keepdims=True)
    std = tr2d.std(axis=0, keepdims=True) + 1e-6
    train_z = (train_x - mean) / std
    test_z = (test_x - mean) / std
    return train_z.astype(np.float32), test_z.astype(np.float32)


def train_torch_one_fold(
    model: "nn.Module",
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    if not HAS_TORCH:
        raise RuntimeError("torch is not available.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cpu")
    model = model.to(device)

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    class_counts = np.bincount(y_train.astype(np.int64), minlength=4).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights /= class_weights.mean()
    weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ds = TensorDataset(x_train_t, y_train_t)
    loader = DataLoader(ds, batch_size=min(16, len(ds)), shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = 15
    wait = 0
    epochs_ran = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())

        epochs_ran = epoch + 1
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x_val_t)
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)

    return pred, epochs_ran


def evaluate_torch_model(
    model_name: str,
    model_builder: Callable[[int, int, int], "nn.Module"],
    x_seq: np.ndarray,
    y: np.ndarray,
    splitter: StratifiedKFold,
    max_epochs: int,
    seed: int,
) -> dict[str, Any]:
    if not HAS_TORCH:
        raise RuntimeError("torch is not available.")

    folds: list[dict[str, Any]] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    seq_len = x_seq.shape[1]
    feat_dim = x_seq.shape[2]

    for fold_idx, (tr, te) in enumerate(splitter.split(x_seq, y), start=1):
        x_tr, x_te = standardize_sequence(x_seq[tr], x_seq[te])
        y_tr = y[tr]
        y_te = y[te]

        model = model_builder(feat_dim, seq_len, 4)
        pred, epochs_ran = train_torch_one_fold(
            model=model,
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_te,
            y_val=y_te,
            max_epochs=max_epochs,
            seed=seed + fold_idx,
        )

        fold_metrics = classification_metrics(y_te, pred)
        folds.append({"fold": fold_idx, "epochs": epochs_ran, **fold_metrics})
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(pred.tolist())

    y_true_np = np.array(y_true_all, dtype=np.int64)
    y_pred_np = np.array(y_pred_all, dtype=np.int64)
    overall = classification_metrics(y_true_np, y_pred_np)
    cm = confusion_matrix(
        y_true_np,
        y_pred_np,
        labels=[0, 1, 2, 3],
    )

    return {
        "model": model_name,
        "overall": overall,
        "folds": folds,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_csv = args.log_csv if args.log_csv is not None else find_latest_log(args.logs_dir)

    fs = load_sampling_rate(raw_dir)
    eeg_df = read_csv_with_fallback(raw_dir / "wert_eeg.csv")
    pd_df = read_csv_with_fallback(raw_dir / "wert_photodiode.csv")
    log_df = read_csv_with_fallback(log_csv)

    eeg = eeg_df.to_numpy(dtype=np.float64).T  # [C, T]
    pd_signal = pd_df.iloc[:, 0].to_numpy(dtype=np.float64)
    channel_names = [str(col).split("-", 1)[-1].strip() for col in eeg_df.columns.tolist()]

    if eeg.shape[1] != pd_signal.size:
        raise RuntimeError(
            f"Sample mismatch: eeg={eeg.shape[1]}, photodiode={pd_signal.size}"
        )

    pd_detection = detect_periodic_photodiode_pulses(pd_signal, fs=fs, trial_sec=TRIAL_SEC)
    periodic_pulses = np.array([seg.start for seg in pd_detection["periodic_segments"]], dtype=np.int64)

    imagine_rows = (
        log_df[(log_df["event"] == "PHASE_START") & (log_df["phase"] == "IMAGINE")]
        .sort_values("trial_idx")
        .copy()
    )
    imagine_rows["trial_idx"] = imagine_rows["trial_idx"].astype(int)
    trial_ids = imagine_rows["trial_idx"].to_numpy(dtype=np.int64)
    log_imagine_times = imagine_rows["t"].to_numpy(dtype=np.float64)

    aligned_pulses, align_result = align_pulses_to_log(
        pulse_starts=periodic_pulses,
        log_imagine_times=log_imagine_times,
        fs=fs,
    )

    label_df = load_trial_directions(log_df)
    label_map = dict(zip(label_df["trial_idx"].astype(int), label_df["class_id"].astype(int)))
    direction_map = dict(zip(label_df["trial_idx"].astype(int), label_df["direction"].astype(str)))
    key_map = dict(zip(label_df["trial_idx"].astype(int), label_df["manual_key"].astype(str)))
    move_dx_map = dict(zip(label_df["trial_idx"].astype(int), label_df["move_dx"].astype(int)))
    move_dy_map = dict(zip(label_df["trial_idx"].astype(int), label_df["move_dy"].astype(int)))

    epoch_len = int(round(IMAGINE_SEC * fs))
    all_epochs: list[np.ndarray] = []
    all_class_ids: list[int] = []
    all_directions: list[str] = []
    all_trial_ids: list[int] = []
    all_pulse_samples: list[int] = []

    alignment_rows: list[dict[str, Any]] = []
    for trial_idx, pulse_sample, log_t in zip(trial_ids, aligned_pulses, log_imagine_times):
        start = int(pulse_sample)
        end = start + epoch_len
        if end > eeg.shape[1]:
            continue

        class_id = int(label_map.get(int(trial_idx), -1))
        direction = direction_map.get(int(trial_idx), "")
        all_epochs.append(eeg[:, start:end].astype(np.float32))
        all_class_ids.append(class_id)
        all_directions.append(direction)
        all_trial_ids.append(int(trial_idx))
        all_pulse_samples.append(start)

        pulse_sec = start / fs
        predicted_log_t = align_result.slope * pulse_sec + align_result.intercept
        alignment_rows.append(
            {
                "trial_idx": int(trial_idx),
                "log_imagine_t_sec": float(log_t),
                "pulse_sample": start,
                "pulse_t_sec": pulse_sec,
                "predicted_log_t_sec": float(predicted_log_t),
                "align_error_sec": float(predicted_log_t - log_t),
                "class_id": class_id,
                "direction": direction,
                "manual_key": key_map.get(int(trial_idx), ""),
                "move_dx": int(move_dx_map.get(int(trial_idx), 0)),
                "move_dy": int(move_dy_map.get(int(trial_idx), 0)),
            }
        )

    if not all_epochs:
        raise RuntimeError("No epochs extracted.")

    epochs = np.stack(all_epochs, axis=0)  # [N, C, T]
    y_all = np.array(all_class_ids, dtype=np.int64)
    trial_all = np.array(all_trial_ids, dtype=np.int64)
    pulse_all = np.array(all_pulse_samples, dtype=np.int64)
    direction_all = np.array(all_directions, dtype=object)

    valid_mask = y_all >= 0
    x_valid = epochs[valid_mask]
    y_valid = y_all[valid_mask]
    trial_valid = trial_all[valid_mask]
    pulse_valid = pulse_all[valid_mask]
    direction_valid = direction_all[valid_mask]

    class_counts = np.bincount(y_valid, minlength=4)
    min_class_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
    if min_class_count < 2:
        raise RuntimeError(f"Too few samples in at least one class: {class_counts.tolist()}")
    n_splits = min(5, min_class_count)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    feature_tensor, window_starts = extract_feature_tensor(
        x_valid,
        channel_names=channel_names,
        fs=fs,
        bands_hz=BANDS_HZ,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
    )  # [N, W, B, 4]

    x_flat = feature_tensor.reshape(feature_tensor.shape[0], -1)
    x_seq = feature_tensor.reshape(feature_tensor.shape[0], feature_tensor.shape[1], -1)

    sklearn_models: list[tuple[str, Any]] = [
        (
            "L1_LogisticRegression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            solver="saga",
                            l1_ratio=1.0,
                            C=1.0,
                            max_iter=6000,
                            random_state=args.seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "SVM_RBF",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(C=2.0, kernel="rbf", gamma="scale", random_state=args.seed)),
                ]
            ),
        ),
        (
            "MLP",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=1e-3,
                            max_iter=1200,
                            random_state=args.seed,
                        ),
                    ),
                ]
            ),
        ),
    ]

    model_results: list[dict[str, Any]] = []
    for name, est in sklearn_models:
        model_results.append(evaluate_sklearn_model(name, est, x_flat, y_valid, splitter))

    if HAS_TORCH and not args.no_torch:
        model_results.append(
            evaluate_torch_model(
                model_name="CNN",
                model_builder=lambda feat_dim, _seq_len, n_cls: TemporalCNN(feat_dim, n_cls),
                x_seq=x_seq,
                y=y_valid,
                splitter=splitter,
                max_epochs=args.max_epochs,
                seed=args.seed,
            )
        )
        model_results.append(
            evaluate_torch_model(
                model_name="Transformer",
                model_builder=lambda feat_dim, seq_len, n_cls: TemporalTransformer(feat_dim, seq_len, n_cls),
                x_seq=x_seq,
                y=y_valid,
                splitter=splitter,
                max_epochs=args.max_epochs,
                seed=args.seed,
            )
        )

    # Save outputs
    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(out_dir / "trial_alignment.csv", index=False, encoding="utf-8")

    pulse_rows: list[dict[str, Any]] = []
    aligned_pulse_set = {int(x) for x in aligned_pulses.tolist()}
    for i, seg in enumerate(pd_detection["periodic_segments"]):
        matched = int(seg.start) in aligned_pulse_set
        pulse_rows.append(
            {
                "pulse_idx_in_periodic_chain": i,
                "pulse_sample_start": int(seg.start),
                "pulse_sample_end": int(seg.end),
                "pulse_duration_samples": int(seg.duration),
                "pulse_start_sec": float(seg.start / fs),
                "matched_to_saved_log_trial": matched,
            }
        )
    pd.DataFrame(pulse_rows).to_csv(out_dir / "photodiode_pulses.csv", index=False, encoding="utf-8")

    np.savez_compressed(
        out_dir / "imagine_epochs_4class.npz",
        eeg_epochs=x_valid.astype(np.float32),
        labels=y_valid.astype(np.int64),
        trial_idx=trial_valid.astype(np.int64),
        pulse_sample=pulse_valid.astype(np.int64),
        fs=np.array([fs], dtype=np.float64),
        channel_names=np.array(channel_names, dtype=object),
        direction=np.array(direction_valid, dtype=object),
    )

    np.savez_compressed(
        out_dir / "imagine_features_4class.npz",
        feature_tensor=feature_tensor.astype(np.float32),  # [N, W, B, 4]
        window_start_samples=window_starts.astype(np.int64),
        bands_hz=np.array(BANDS_HZ, dtype=np.float32),
        labels=y_valid.astype(np.int64),
        trial_idx=trial_valid.astype(np.int64),
        pulse_sample=pulse_valid.astype(np.int64),
    )

    summary_rows: list[dict[str, Any]] = []
    for res in model_results:
        summary_rows.append(
            {
                "model": res["model"],
                "accuracy": res["overall"]["accuracy"],
                "balanced_accuracy": res["overall"]["balanced_accuracy"],
                "f1_macro": res["overall"]["f1_macro"],
            }
        )
    model_summary_df = pd.DataFrame(summary_rows).sort_values(
        "balanced_accuracy", ascending=False
    )
    model_summary_df.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8")

    details = {
        "args": {
            "raw_dir": str(raw_dir),
            "log_csv": str(log_csv),
            "out_dir": str(out_dir),
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "no_torch": bool(args.no_torch),
        },
        "sampling_rate_hz": fs,
        "photodiode": {
            "threshold": float(pd_detection["threshold"]),
            "min_dur_samples": int(pd_detection["min_dur_samples"]),
            "max_dur_samples": int(pd_detection["max_dur_samples"]),
            "all_segments": len(pd_detection["all_segments"]),
            "short_segments": len(pd_detection["short_segments"]),
            "long_segments": len(pd_detection["long_segments"]),
            "long_segment_ranges": [
                {
                    "start_sample": int(seg.start),
                    "end_sample": int(seg.end),
                    "duration_samples": int(seg.duration),
                    "start_sec": float(seg.start / fs),
                    "end_sec": float(seg.end / fs),
                }
                for seg in pd_detection["long_segments"]
            ],
            "periodic_segments": len(pd_detection["periodic_segments"]),
            "expected_period_samples": int(pd_detection["expected_period_samples"]),
            "period_tolerance_samples": int(pd_detection["period_tolerance_samples"]),
            "first_task_pulse_sample": int(pd_detection["periodic_segments"][0].start),
            "last_task_pulse_sample": int(pd_detection["periodic_segments"][-1].start),
            "first_task_pulse_sec": float(pd_detection["periodic_segments"][0].start / fs),
            "last_task_pulse_sec": float(pd_detection["periodic_segments"][-1].start / fs),
        },
        "alignment": asdict(align_result),
        "counts": {
            "logged_trials": int(trial_ids.size),
            "aligned_trials": int(len(alignment_rows)),
            "epochs_4class": int(x_valid.shape[0]),
            "class_counts": {ID_TO_DIRECTION[i]: int(class_counts[i]) for i in range(4)},
        },
        "model_results": model_results,
    }
    with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print(f"log_csv={log_csv}")
    print(f"fs={fs} Hz")
    print(f"periodic_pulses={len(pd_detection['periodic_segments'])}, logged_trials={trial_ids.size}")
    print(f"alignment_rmse={align_result.rmse_sec:.6f} s, alignment_offset={align_result.offset}")
    print(f"epochs_4class={x_valid.shape[0]}, class_counts={class_counts.tolist()}")
    print("model_comparison:")
    print(model_summary_df.to_string(index=False))
    print(f"saved_to={out_dir}")


if __name__ == "__main__":
    main()
