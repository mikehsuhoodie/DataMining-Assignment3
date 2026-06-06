# NYCU Data Mining Assignment 3

This project is for the Kaggle competition `nycu-data-mining-assignment-3`, a Human Activity Recognition task using accelerometer summary data.

## Task Summary

- Each CSV file is one 5-minute activity window.
- Each file has exactly 300 rows, one row per second.
- Training files contain a constant `label` column.
- Test files do not contain labels and must be predicted one label per file.
- Kaggle evaluates submissions with macro F1.

## Data Layout

Expected data layout after download:

```text
data/
|-- train/train/User_001 ... User_060
|-- test/test/User_061 ... User_100
`-- sample_submission.csv
```

Important identifiers:

- User group comes from the parent folder name, for example `User_001`.
- Submission ID comes from each test CSV's `file_id`.
- Submission output must contain columns exactly `Id,Label`.

## Modeling Direction

The main approach is a tabular time-series feature pipeline:

1. Load one CSV as one supervised example.
2. Extract window-level statistical, temporal, magnitude, and FFT features.
3. Validate with user-grouped cross-validation.
4. Train strong tabular models such as ExtraTrees, RandomForest, LightGBM, or XGBoost.
5. Use class/sample weighting because labels are highly imbalanced.
6. Optionally compare against a small sequence model if time allows.

Random row or random file splitting should not be used because test users are unseen users.

## Planned Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Download data:

```bash
bash download_data.sh
```

Final training and prediction commands:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python -u -m src.train --feature-set full_correlation --models lightgbm --n-splits 5 --output-dir outputs_lgbm_full_correlation
MPLCONFIGDIR=/tmp .venv/bin/python -u -m src.predict --output-dir outputs_lgbm_full_correlation
```

Feature ablation commands:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python -u -m src.train --feature-set full_correlation --models lightgbm --n-splits 5 --output-dir outputs_lgbm_full_correlation
MPLCONFIGDIR=/tmp .venv/bin/python -u -m src.train --feature-set full_peak --models lightgbm --n-splits 5 --output-dir outputs_lgbm_full_peak
MPLCONFIGDIR=/tmp .venv/bin/python -u -m src.train --feature-set full_rolling --models lightgbm --n-splits 5 --output-dir outputs_lgbm_full_rolling
```

## Expected Outputs

```text
outputs/
|-- validation_results.csv
|-- confusion_matrix.csv
`-- submission.csv
```

`outputs/submission.csv` should match `data/sample_submission.csv` row count and ID order.

The current Kaggle-facing output is `outputs_lgbm_full_correlation/submission.csv`.
