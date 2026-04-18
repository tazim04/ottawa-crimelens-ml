# Ottawa-CrimeLens-ML

Ottawa-CrimeLens-ML is a machine learning service for training and serving a crime anomaly detection model using historical crime data and engineered temporal and geographic features.

The trained model identifies anomalies within Ottawa crime spatial grid cells and assigns triage labels (`low`, `medium`, `high`) based on severity. These scores are pushed to the Ottawa CrimeLens database, and stored within the table `crime_anomaly_scores`.

## Project Structure

```text
ottawa-crimelens-ml/
|-- app/
|   |-- features/
|   |-- model/
|   |-- config.py
|   `-- db.py
|-- artifacts/
|-- experiments/
|-- tests/
|-- train.py
|-- score.py
|-- Dockerfile
`-- requirements.txt
```

## Main Entry Points

### `train.py`

This file acts as the entrypoint to the training service. It is automated on AWS and runs on a weekly cron schedule to retrain the model used by the scoring portion of the ML service.

This script should:

1. Build historical features
2. Train the model
3. Save and push the model artifact to S3 (or locally if specified)
4. Optionally log metrics

Run with:

```bash
python -m train --start-date 2022-01-01 --model-version crime-anomaly-v1 --min-history-days 3
```

### `score.py`

This file contains the daily inference workflow that acts as the production scoring job. It is deployed on AWS and pulls the latest trained `.joblib` model artifact from S3 (or locally) during scoring.

This script should:

1. Build daily features
2. Pull and load the trained model artifact
3. Compute scores
4. Assign triage labels
5. Persist results to Postgres

Run with:

```bash
python score.py --min-history-days 3
```

When `--target-date` or `--end-date` is omitted, the service resolves "today" using `APP_TIMEZONE`. The default is `America/Toronto`, which keeps AWS runs aligned with Ottawa local dates even if the container itself is running in UTC.

`MODEL_ARTIFACT_PATH` supports either a local filesystem path or an `s3://bucket/key.joblib` URI. In production, training pushes the latest model artifact to S3 and scoring reads that same artifact back from S3. S3 access uses the standard AWS credential chain from the runtime environment.

Model artifact location is env-only for both training and scoring:

1. Environment variable: `MODEL_ARTIFACT_PATH`
2. Default when unset: `artifacts/crime_model.joblib`

## `app/`

This folder contains reusable, modular logic and serves as the core ML domain layer.

## `experiments/`

This folder contains exploratory evaluation scripts and analysis outputs used to validate model behavior, score alignment, and ranking stability. It has its own README with experiment-specific usage details.

## Formatting and Linting

Format the codebase with:

```bash
ruff format
```

Run lint checks with:

```bash
ruff check --fix
```
