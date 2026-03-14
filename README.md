# Ottawa-CrimeLens-ML

## Project Structure

OTTAWA-CRIMELENS-ML/
│
├── app/
├── train.py
├── score.py
├── Dockerfile
└── requirements.txt 

### Main Entry-Points - `train.py` and `score.py`

These two files act as the main entry points of this project, depending on the desired service.

`train.py`

This file is meant to act as an offline service that will be run locally. It will train and produce the model required for the scoring portion of this ML service.

This script should:

1. Build historical features
2. Train model
3. Save model artifact
4. Possibly log metrics

Run with:

```Bash
python -m train --start-date 2020-01-01 --end-date 2026-02-28 --model-version crime-anomaly-v1
```

`score.py`

This file contains the daily inference workflow that will act as the production scoring job. This is what will be deployed to AWS (ECS or Step Function) to be run daily.

It should:

1. Build daily features
2. Load trained model
3. Compute scores
4. Assign triage labels
5. Persist results to Postgres

### `app/` (Core ML Logic Layer)

This folder contains all reusable, modular logic, acting as the **ML domain layer**.

```Bash
python score.py --target-date 2026-03-13 --model-artifact-path artifacts/crime_model.joblib
```

## Formatting and Linting

```Bash
ruff format
```

```Bash
ruff check --fix
```