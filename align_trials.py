from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imagery_core import (
    TRIAL_SEC,
    align_pulses_to_log,
    detect_periodic_photodiode_pulses,
    find_latest_log,
    load_sampling_rate,
    load_trial_directions,
    read_csv_with_fallback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align photodiode pulses with trial logs.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("map/rawdata/RecordingS17R000_CSV"),
        help="Directory with wert_photodiode.csv and info.json",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Session log CSV path. If omitted, latest session_*.csv is used.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("map/logs"),
        help="Directory searched when --log-csv is omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("map/analysis_results"),
        help="Output directory.",
    )
    return parser.parse_args()


def run_alignment(
    raw_dir: Path,
    out_dir: Path,
    log_csv: Path | None = None,
    logs_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_csv is None:
        if logs_dir is None:
            raise ValueError("logs_dir is required when log_csv is None.")
        log_csv = find_latest_log(logs_dir)

    fs = load_sampling_rate(raw_dir)
    pd_df = read_csv_with_fallback(raw_dir / "wert_photodiode.csv")
    log_df = read_csv_with_fallback(log_csv)
    pd_signal = pd_df.iloc[:, 0].to_numpy(dtype=np.float64)

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

    alignment_rows: list[dict[str, Any]] = []
    for trial_idx, pulse_sample, log_t in zip(trial_ids, aligned_pulses, log_imagine_times):
        pulse_sec = pulse_sample / fs
        predicted_log_t = align_result.slope * pulse_sec + align_result.intercept
        alignment_rows.append(
            {
                "trial_idx": int(trial_idx),
                "log_imagine_t_sec": float(log_t),
                "pulse_sample": int(pulse_sample),
                "pulse_t_sec": float(pulse_sec),
                "predicted_log_t_sec": float(predicted_log_t),
                "align_error_sec": float(predicted_log_t - log_t),
                "class_id": int(label_map.get(int(trial_idx), -1)),
                "direction": direction_map.get(int(trial_idx), ""),
                "manual_key": key_map.get(int(trial_idx), ""),
                "move_dx": int(move_dx_map.get(int(trial_idx), 0)),
                "move_dy": int(move_dy_map.get(int(trial_idx), 0)),
            }
        )

    trial_alignment_path = out_dir / "trial_alignment.csv"
    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(trial_alignment_path, index=False, encoding="utf-8")

    pulse_rows: list[dict[str, Any]] = []
    aligned_pulse_set = {int(x) for x in aligned_pulses.tolist()}
    for i, seg in enumerate(pd_detection["periodic_segments"]):
        pulse_rows.append(
            {
                "pulse_idx_in_periodic_chain": i,
                "pulse_sample_start": int(seg.start),
                "pulse_sample_end": int(seg.end),
                "pulse_duration_samples": int(seg.duration),
                "pulse_start_sec": float(seg.start / fs),
                "matched_to_saved_log_trial": int(seg.start) in aligned_pulse_set,
            }
        )
    photodiode_pulses_path = out_dir / "photodiode_pulses.csv"
    pd.DataFrame(pulse_rows).to_csv(photodiode_pulses_path, index=False, encoding="utf-8")

    summary = {
        "raw_dir": str(raw_dir),
        "log_csv": str(log_csv),
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
            "first_task_pulse_sample": int(pd_detection["periodic_segments"][0].start),
            "last_task_pulse_sample": int(pd_detection["periodic_segments"][-1].start),
            "first_task_pulse_sec": float(pd_detection["periodic_segments"][0].start / fs),
            "last_task_pulse_sec": float(pd_detection["periodic_segments"][-1].start / fs),
        },
        "alignment": asdict(align_result),
        "counts": {
            "logged_trials": int(trial_ids.size),
            "aligned_trials": int(alignment_df.shape[0]),
            "class_id_counts": alignment_df["class_id"].value_counts().sort_index().to_dict(),
        },
    }

    alignment_summary_path = out_dir / "alignment_summary.json"
    with alignment_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"log_csv={log_csv}")
    print(f"fs={fs} Hz")
    print(f"periodic_pulses={len(pd_detection['periodic_segments'])}, logged_trials={trial_ids.size}")
    print(f"alignment_rmse={align_result.rmse_sec:.6f} s, alignment_offset={align_result.offset}")
    print(f"saved={trial_alignment_path}")
    print(f"saved={photodiode_pulses_path}")
    print(f"saved={alignment_summary_path}")

    return {
        "log_csv": log_csv,
        "trial_alignment_csv": trial_alignment_path,
        "photodiode_pulses_csv": photodiode_pulses_path,
        "alignment_summary_json": alignment_summary_path,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    run_alignment(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        log_csv=args.log_csv,
        logs_dir=args.logs_dir,
    )


if __name__ == "__main__":
    main()
