# Experiments

This directory contains evaluation scripts for validating anomaly-score behavior,
triage separation, and ranking stability.

## Setup

Install the plotting dependencies from `requirements.txt`, then run the scripts
from the project root so they can import the application modules.

## Alignment Experiment

Measures whether anomaly scores increase with larger deviations from rolling
baselines and whether `low` / `medium` / `high` triage groups separate on
supporting statistics.

```bash
python -m experiments.score_alignment \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  -- min-history-days 3
  --model-artifact-path artifacts/crime_model.joblib
```

Outputs:

- `evaluation_frame.csv`
- `triage_summary.csv`
- `metrics.json`
- `score_vs_abs_count_delta.png`
- `score_vs_abs_count_zscore.png`
- `triage_vs_abs_count_zscore.png`

## Stability Experiment

Measures whether anomaly rankings remain similar across different random seeds
and lookback windows.

```bash
python -m experiments.stability_analysis \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  -- min-history-days 3
```

Outputs:

- `seed_stability.csv`
- `lookback_stability.csv`
- `seed_spearman_heatmap.png`
- `lookback_spearman_heatmap.png`
- `summary.json`

## Notes

- Both scripts use the same feature-building path as training/scoring so the
  analysis reflects the real workflow.
- The alignment script uses an existing saved artifact.
- The stability script retrains temporary models in memory for comparison and
  does not overwrite your saved production artifact.
