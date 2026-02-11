from __future__ import annotations

import argparse
import json
from pathlib import Path

from align_trials import run_alignment
from extract_imagine_epochs import run_epoch_extraction
from extract_imagery_features import run_feature_extraction
from train_imagery_models import run_model_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full imagery pipeline (alignment -> epoch extraction -> feature extraction -> training)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("map/rawdata/RecordingS17R000_CSV"),
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("map/logs"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("map/analysis_results"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--no-torch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment = run_alignment(
        raw_dir=args.raw_dir,
        out_dir=out_dir,
        log_csv=args.log_csv,
        logs_dir=args.logs_dir,
    )

    epochs = run_epoch_extraction(
        raw_dir=args.raw_dir,
        alignment_csv=alignment["trial_alignment_csv"],
        out_dir=out_dir,
    )

    features = run_feature_extraction(
        epochs_npz=epochs["epochs_npz"],
        out_dir=out_dir,
    )

    training = run_model_training(
        features_npz=features["features_npz"],
        epochs_npz=epochs["epochs_npz"],
        out_dir=out_dir,
        seed=args.seed,
        max_epochs=args.max_epochs,
        no_torch=args.no_torch,
    )

    best_model = ""
    best_input = ""
    best_architecture = ""
    if training["model_results"]:
        best = max(
            training["model_results"],
            key=lambda x: x["overall"]["balanced_accuracy"],
        )
        best_model = best["model"]
        best_input = best.get("input_name", "")
        best_architecture = best.get("architecture", "")

    analysis_summary = {
        "pipeline": "full",
        "steps": {
            "alignment": {
                "trial_alignment_csv": str(alignment["trial_alignment_csv"]),
                "photodiode_pulses_csv": str(alignment["photodiode_pulses_csv"]),
                "alignment_summary_json": str(alignment["alignment_summary_json"]),
            },
            "epoch_extraction": {
                "epochs_npz": str(epochs["epochs_npz"]),
                "manifest_csv": str(epochs["manifest_csv"]),
                "summary_json": str(epochs["summary_json"]),
            },
            "feature_extraction": {
                "features_npz": str(features["features_npz"]),
                "summary_json": str(features["summary_json"]),
            },
            "training": {
                "input_search_csv": str(training["input_search_csv"]),
                "model_comparison_csv": str(training["model_comparison_csv"]),
                "model_details_json": str(training["model_details_json"]),
            },
        },
        "key_metrics": {
            "aligned_trials": alignment["summary"]["counts"]["aligned_trials"],
            "epochs_4class": epochs["summary"]["valid_epochs"],
            "best_model": best_model,
            "best_architecture": best_architecture,
            "best_input": best_input,
        },
    }

    summary_path = out_dir / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, ensure_ascii=False, indent=2)

    print(f"saved={summary_path}")


if __name__ == "__main__":
    main()
