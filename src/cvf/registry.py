"""File-based model registry: versioned bundles plus a LATEST pointer."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import sklearn

from cvf.config import PROJECT_ROOT, Settings
from cvf.models.gbm import GBMModel, Task


@dataclass
class ModelBundle:
    version: str
    models: dict[Task, GBMModel]
    metadata: dict[str, Any]


def _registry_dir(settings: Settings) -> Path:
    return settings.artifacts_dir / "models"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_bundle(models: dict[Task, GBMModel], metadata: dict[str, Any],
                settings: Settings) -> ModelBundle:
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = _registry_dir(settings) / version
    target.mkdir(parents=True, exist_ok=False)
    for task, model in models.items():
        joblib.dump(model, target / f"{task}.joblib")
    metadata = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "sklearn": sklearn.__version__,
        **metadata,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str),
                                          encoding="utf-8")
    (_registry_dir(settings) / "LATEST").write_text(version, encoding="utf-8")
    return ModelBundle(version, models, metadata)


def load_bundle(settings: Settings, version: str | None = None) -> ModelBundle:
    root = _registry_dir(settings)
    if version is None:
        pointer = root / "LATEST"
        if not pointer.exists():
            raise FileNotFoundError(f"No trained model in {root}; run `cvf train` first")
        version = pointer.read_text(encoding="utf-8").strip()
    path = root / version
    models = {task: joblib.load(path / f"{task}.joblib") for task in ("propensity", "revenue")}
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return ModelBundle(version, models, metadata)
