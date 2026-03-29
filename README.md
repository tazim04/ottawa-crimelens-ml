# Ottawa-CrimeLens-ML

Ottawa-CrimeLens-ML is a machine learning service for training and serving a crime anomaly detection model using historical crime data and engineered temporal and geographic features.

The trained model identifies anomalies within Ottawa crime spatial grid cells and assigns triage labels (`low`, `medium`, `high`) based on severity. These scores are pushed to the Ottawa CrimeLens database, and stored within the table `crime_anomaly_scores`.

Latest model version: `crime-anomaly-v1.1.4`

## Project Structure

```text
ottawa-crimelens-ml/
|-- app/
|   |-- features/
|   |-- model/
|   |-- config.py
|   `-- db.py
|-- artifacts/
|-- tests/
|-- train.py
|-- score.py
|-- Dockerfile
`-- requirements.txt
```

## Main Entry Points

### `train.py`

This file acts as the offline training service that runs locally. It trains the model used by the scoring portion of the ML service.

This script should:

1. Build historical features
2. Train the model
3. Save the model artifact
4. Optionally log metrics

Run with:

```bash
python -m train --start-date 2020-01-01 --end-date 2026-02-28 --model-version crime-anomaly-v1 --min-history-days 3
```

### `score.py`

This file contains the daily inference workflow that acts as the production scoring job. It is intended to be deployed to AWS, such as ECS or Step Functions, and run daily.

This script should:

1. Build daily features
2. Load the trained model
3. Compute scores
4. Assign triage labels
5. Persist results to Postgres

Run with:

```bash
python score.py --min-history-days 3
```

`MODEL_ARTIFACT_PATH` supports either a local filesystem path or an `s3://bucket/key.joblib` URI. S3 access uses the standard AWS credential chain from the runtime environment.

Model artifact location is env-only for both training and scoring:

1. Environment variable: `MODEL_ARTIFACT_PATH`
2. Default when unset: `artifacts/crime_model.joblib`

## `app/`

This folder contains reusable, modular logic and serves as the core ML domain layer.

## Formatting and Linting

Format the codebase with:

```bash
ruff format
```

Run lint checks with:

```bash
ruff check --fix
```
