# Ottawa-CrimeLens-ML

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
python score.py --target-date 2026-03-13 --model-artifact-path artifacts/crime_model.joblib
```

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
