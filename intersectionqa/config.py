"""Configuration primitives for deterministic smoke generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from intersectionqa.enums import TaskType
from intersectionqa.hashing import sha256_json
from intersectionqa.schema import LabelPolicy


class SmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_synthetic_fixtures: bool = True
    include_cadevolve_if_available: bool = True
    geometry_limit: int = 100
    object_validation_limit: int = 25
    object_validation_timeout_seconds: float = 5.0
    use_object_validation_cache: bool = True
    object_validation_cache_dir: Path = Path(".cache/intersectionqa/objects")
    task_types: list[TaskType] = Field(
        default_factory=lambda: [
            TaskType.BINARY_INTERFERENCE,
            TaskType.RELATION_CLASSIFICATION,
            TaskType.VOLUME_BUCKET,
        ]
    )


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "IntersectionQA"
    dataset_version: str = "v0.1"
    seed: int = 1234
    output_dir: Path = Path("data/intersectionqa_v0_1")
    cadevolve_archive: Path | None = None
    label_policy: LabelPolicy = Field(default_factory=LabelPolicy)
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)
    license: str = "cc-by-4.0"

    @property
    def config_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load_config(path: Path | None) -> DatasetConfig:
    if path is None:
        return DatasetConfig()
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    return DatasetConfig.model_validate(data)
