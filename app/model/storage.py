from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

import joblib

if TYPE_CHECKING:
    from app.model.model import ModelArtifact

ArtifactLocation = str | Path
MODEL_ARTIFACT_PATH_ENV_VAR = "MODEL_ARTIFACT_PATH"


class ModelArtifactStorage(Protocol):
    """Storage backend for persisted model artifacts."""

    def save_artifact(
        self,
        artifact: ModelArtifact,
        location: ArtifactLocation,
    ) -> ArtifactLocation: ...

    def load_artifact(self, location: ArtifactLocation) -> ModelArtifact: ...


class LocalModelArtifactStorage:
    """Persist model artifacts on the local filesystem."""

    def save_artifact(
        self,
        artifact: ModelArtifact,
        location: ArtifactLocation,
    ) -> Path:
        resolved_path = Path(location)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(asdict(artifact), resolved_path)
        return resolved_path

    def load_artifact(self, location: ArtifactLocation) -> ModelArtifact:
        resolved_path = Path(location)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{resolved_path}'. "
                "Train a model first or provide a valid --model-artifact-path / MODEL_ARTIFACT_PATH."
            )

        payload = joblib.load(resolved_path)
        from app.model.model import ModelArtifact

        return ModelArtifact(**payload)


class S3ModelArtifactStorage:
    """Persist model artifacts in S3 using standard AWS runtime credentials."""

    def __init__(self, client: object | None = None) -> None:
        self._client = client

    def save_artifact(
        self,
        artifact: ModelArtifact,
        location: ArtifactLocation,
    ) -> str:
        parsed = parse_s3_uri(location)
        buffer = BytesIO()
        joblib.dump(asdict(artifact), buffer)
        buffer.seek(0)

        try:
            self.client.put_object(
                Bucket=parsed.bucket,
                Key=parsed.key,
                Body=buffer.getvalue(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to save model artifact to '{parsed.uri}'. "
                "Ensure AWS credentials, region, and S3 permissions are configured in the runtime environment."
            ) from exc
        return parsed.uri

    def load_artifact(self, location: ArtifactLocation) -> ModelArtifact:
        parsed = parse_s3_uri(location)
        try:
            response = self.client.get_object(Bucket=parsed.bucket, Key=parsed.key)
        except Exception as exc:
            if is_missing_s3_object_error(exc):
                raise FileNotFoundError(
                    f"Model artifact not found at '{parsed.uri}'. "
                    "Train a model first or provide a valid --model-artifact-path / MODEL_ARTIFACT_PATH."
                ) from exc
            raise RuntimeError(
                f"Unable to load model artifact from '{parsed.uri}'. "
                "Ensure AWS credentials, region, and S3 access are configured in the runtime environment."
            ) from exc

        payload = joblib.load(BytesIO(response["Body"].read()))
        from app.model.model import ModelArtifact

        return ModelArtifact(**payload)

    @property
    def client(self) -> object:
        if self._client is None:
            self._client = create_s3_client()
        return self._client


class ResolvedS3Location:
    """Parsed representation of an S3 artifact URI."""

    def __init__(self, bucket: str, key: str) -> None:
        self.bucket = bucket
        self.key = key

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(location: ArtifactLocation) -> ResolvedS3Location:
    """Parse and validate an S3 artifact URI."""
    candidate = str(location)
    parsed = urlparse(candidate)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key:
        raise ValueError(
            "Invalid S3 model artifact URI. Expected format: s3://bucket/key.joblib"
        )
    return ResolvedS3Location(bucket=bucket, key=key)


def normalize_artifact_location(location: ArtifactLocation) -> ArtifactLocation:
    """Normalize an artifact location to ``Path`` for local paths or ``str`` for S3 URIs."""
    if isinstance(location, Path):
        return location
    if is_s3_location(location):
        return str(location)
    return Path(location)


def resolve_artifact_location(
    location: ArtifactLocation | None,
    *,
    default_location: ArtifactLocation,
    env_var: str = MODEL_ARTIFACT_PATH_ENV_VAR,
) -> ArtifactLocation:
    """Resolve artifact location precedence: explicit arg, then env var, then default."""
    if location is not None:
        return normalize_artifact_location(location)

    env_value = os.getenv(env_var)
    if env_value:
        return normalize_artifact_location(env_value)

    return normalize_artifact_location(default_location)


def is_s3_location(location: ArtifactLocation) -> bool:
    """Return whether the model artifact location should use the S3 backend."""
    return str(location).startswith("s3://")


def resolve_model_artifact_storage(
    location: ArtifactLocation,
    *,
    s3_client: object | None = None,
) -> ModelArtifactStorage:
    """Select the storage backend for a model artifact location."""
    if is_s3_location(location):
        return S3ModelArtifactStorage(client=s3_client)
    return LocalModelArtifactStorage()


def create_s3_client() -> object:
    """Create an S3 client using the standard AWS credential chain."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "S3 model artifact support requires the 'boto3' package. "
            "Install project dependencies with S3 support before using an s3:// model artifact path."
        ) from exc

    try:
        return boto3.client("s3")
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize the S3 client for model artifact storage. "
            "Ensure AWS credentials and region are configured in the runtime environment."
        ) from exc


def is_missing_s3_object_error(exc: Exception) -> bool:
    """Return whether an S3 error indicates a missing object or bucket."""
    if isinstance(exc, KeyError):
        return True

    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False

    error = response.get("Error", {})
    code = str(error.get("Code", ""))
    return code in {"404", "NoSuchBucket", "NoSuchKey", "NotFound"}
