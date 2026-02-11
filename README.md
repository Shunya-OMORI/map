# Map BCI Analysis

## Scripts (split by responsibility)

1. `map/align_trials.py`  
   Photodiode pulse detection and log alignment.
2. `map/extract_imagine_epochs.py`  
   2-second IMAGINE EEG epoch extraction.
3. `map/extract_imagery_features.py`  
   Hilbert-based preprocessing and feature extraction.
4. `map/train_imagery_models.py`  
   4-class model training and evaluation.

## Full pipeline (convenience)

```bash
python map/analyze_imagery.py \
  --raw-dir map/rawdata/RecordingS17R000_CSV \
  --log-csv map/logs/session_20260210_184604.csv \
  --out-dir map/analysis_results
```

## Run step-by-step

```bash
python map/align_trials.py \
  --raw-dir map/rawdata/RecordingS17R000_CSV \
  --log-csv map/logs/session_20260210_184604.csv \
  --out-dir map/analysis_results

python map/extract_imagine_epochs.py \
  --raw-dir map/rawdata/RecordingS17R000_CSV \
  --alignment-csv map/analysis_results/trial_alignment.csv \
  --out-dir map/analysis_results

python map/extract_imagery_features.py \
  --epochs-npz map/analysis_results/imagine_epochs_4class.npz \
  --out-dir map/analysis_results

python map/train_imagery_models.py \
  --features-npz map/analysis_results/imagine_features_4class.npz \
  --epochs-npz map/analysis_results/imagine_epochs_4class.npz \
  --out-dir map/analysis_results
```

## Main outputs

- `trial_alignment.csv`
- `photodiode_pulses.csv`
- `alignment_summary.json`
- `imagine_epochs_4class.npz`
- `epoch_manifest.csv`
- `imagine_features_4class.npz`
- `model_comparison.csv`
- `input_search_results.csv` (input candidate search per model)
- `input_search_results.csv` (input + architecture candidate search per model)
- `model_details.json`
- `analysis_summary.json` (full pipeline summary)
