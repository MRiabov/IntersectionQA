"""Configuration primitives for deterministic smoke generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from intersectionqa.hashing import sha256_json
from intersectionqa.schema import LabelPolicy


class SmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_synthetic_fixtures: bool = True
    include_cadevolve_if_available: bool = True
    geometry_limit: int = 100
    task_types: list[str] = Field(
        default_factory=lambda: [
            "binary_interference",
            "relation_classification",
            "volume_bucket",
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
