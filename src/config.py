"""Central configuration for the Hiver support-agent data pipeline.

Reads `configs/config.yaml` (or a path given via the CONFIG_PATH environment
variable), overlays environment-variable overrides for the handful of
settings that are commonly changed per-machine (raw data location, output
directories, split seed), and returns a single validated `Config` object.

Nothing in this module talks to an LLM or knows about intents — Phase 1 only
needs paths, split ratios, and analysis thresholds. Later phases should add
their settings here rather than growing a second config file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = "configs/config.yaml"


class PathsConfig(BaseModel):
    """All paths are relative by default and resolved against the current
    working directory (or an explicit base_dir) at use time — never baked in
    as absolute paths, so the same config works on any machine or CI runner.
    """

    raw_csv: str = "data/raw/twcs.csv"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"


class SplitConfig(BaseModel):
    train: float = 0.7
    dev: float = 0.15
    test: float = 0.15
    seed: int = 42

    @field_validator("train", "dev", "test")
    @classmethod
    def _fraction_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("split fractions must be strictly between 0 and 1")
        return v

    def validate_sums_to_one(self) -> None:
        total = self.train + self.dev + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split.train + split.dev + split.test must sum to 1.0, got {total}"
            )


class AnalysisConfig(BaseModel):
    # A brand needs at least this many conversations AND this many usable
    # resolution pairs before it is even considered for recommendation —
    # below this, any golden-eval-set built from it would be too thin to
    # trust. Tune per corpus size, not per brand name.
    min_conversations_for_recommendation: int = 20
    short_message_max_words: int = 3
    top_brands_to_report: int = 15


class Config(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    def resolve_paths(self, base_dir: Optional[str] = None) -> "Config":
        """Return a copy with every path in `paths` resolved against
        `base_dir` (defaults to the current working directory). An
        already-absolute path is left untouched.
        """
        base = Path(base_dir) if base_dir else Path.cwd()

        def _resolve(p: str) -> str:
            path = Path(p)
            return str(path if path.is_absolute() else (base / path))

        cfg = self.model_copy(deep=True)
        cfg.paths.raw_csv = _resolve(cfg.paths.raw_csv)
        cfg.paths.processed_dir = _resolve(cfg.paths.processed_dir)
        cfg.paths.splits_dir = _resolve(cfg.paths.splits_dir)
        return cfg


# env var -> (section, key). Only the settings someone would plausibly want
# to override without editing the checked-in yaml.
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "RAW_CSV_PATH": ("paths", "raw_csv"),
    "PROCESSED_DIR": ("paths", "processed_dir"),
    "SPLITS_DIR": ("paths", "splits_dir"),
    "SPLIT_SEED": ("split", "seed"),
}


def _apply_env_overrides(raw: dict) -> dict:
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        val = os.environ.get(env_var)
        if val is None:
            continue
        raw.setdefault(section, {})
        raw[section][key] = int(val) if key == "seed" else val
    return raw


def load_config(path: Optional[str] = None) -> Config:
    """Load config.yaml (if present) plus environment overrides.

    Falls back to all built-in defaults when no yaml file exists at the
    resolved path, so tests and ad-hoc scripts never require one on disk.
    """
    config_path = Path(path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    raw: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    raw = _apply_env_overrides(raw)
    cfg = Config(**raw)
    cfg.split.validate_sums_to_one()
    return cfg
