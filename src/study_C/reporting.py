"""Run folder, metadata and machine-readable tables.

Output layout (``reports/study_C/runs/<run_id>/``)::

    config.toml, resolved_config.json, run_metadata.json, run.log
    data/      data_profile.json, label_counts_by_step.csv
    splits/    split_manifest.csv, split_summary.csv, split_audit.json
    models/<model>__<feature_set>__seed<k>/
               predictions_val.csv, predictions_test.csv, metrics.json,
               per_step_test.csv, history.csv (torch only), model.pt (torch only)
    summary/   metrics_all.csv (one row per model/feature set/seed),
               metrics_summary.csv (mean ± std over seeds), per_step_summary.csv
    figures/

"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from . import PREPROCESSING_VERSION
from .config import PROJECT_ROOT, StudyCConfig
from .data import CLASSES_FILE, EDGES_FILE, FEATURES_FILE

PACKAGES = ["numpy", "pandas", "scikit-learn", "torch", "torch_geometric", "matplotlib"]
# Columns that identify an experiment: never averaged in summarise_metrics.
KEY_COLUMNS = ["model", "feature_set", "split"]


def run_directory(config: StudyCConfig) -> Path:
    """``output_root/runs/<run_id>``; empty run_id -> UTC timestamp. Creates the folder."""
    run_id = config.run.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def model_directory(run_dir: Path, model: str, feature_set: str, seed: int) -> Path:
    """``models/<model>__<feature_set>__seed<k>`` (created)."""
    path = run_dir / "models" / f"{model}__{feature_set}__seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: dict[str, Any], path: Path) -> None:
    """Pretty JSON, UTF-8, parent folders created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """CSV without the pandas index, parent folders created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def _versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def _device() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_cuda": None, "cuda_available": False, "gpu": None}
    available = torch.cuda.is_available()
    return {"torch_cuda": torch.version.cuda, "cuda_available": available,
            "gpu": torch.cuda.get_device_name(0) if available else None}


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def run_metadata(config: StudyCConfig, data_dir: Path) -> dict[str, Any]:
    """Provenance: timestamp, git commit, Python/torch/CUDA/GPU, package versions, input files."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": config.seed,
        "device_setting": config.run.device,
        "git": _git_state(),
        "versions": _versions(),
        "device": _device(),
        "input_files": [_file_fingerprint(data_dir / name) for name in (FEATURES_FILE, CLASSES_FILE, EDGES_FILE)],
    }


def summarise_metrics(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all rows, mean ± std over seeds grouped by model, feature_set and split)."""
    # The RF has a single seed: its std will be NaN, and that is correct (n_seeds makes it explicit).
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.select_dtypes("number").columns if c not in KEY_COLUMNS + ["seed"]]
    grouped = frame.groupby(KEY_COLUMNS, sort=True)
    summary = grouped[numeric].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_seeds", grouped["seed"].nunique())
    return frame, summary.reset_index()
