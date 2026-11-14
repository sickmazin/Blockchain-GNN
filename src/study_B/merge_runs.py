"""Merge the models of several Study B runs into one new run folder and redraw its figures.

Valid only because every run scores the same saved evaluation pairs: the merge stops if any
samples/pairs_*.csv differs. Source runs are never modified. Edit the constants below and run
this file (no CLI, same as main.py).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script: make ``src.study_B`` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.study_B import data, figures, pipeline, splits  # noqa: E402
from src.study_B.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.study_B.features import RECENT_WINDOWS  # noqa: E402

BASE = "COMPLETO_2"                  # copied whole: splits, samples, configs, every model
EXTRA = ["COMPLETO_2_mixed"]         # their models are added (predictions + metric rows)
OUT = "COMPLETO_2_all"
METRIC_FILES = ["metrics.csv", "per_window.csv", "by_source_degree.csv", "training_curves.csv"]


def merge(base: Path, extras: list[Path], out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"{out} exists: choose another OUT or delete it first")
    for extra in extras:
        for pairs in (base / "samples").glob("pairs_*.csv"):
            if pairs.read_bytes() != (extra / "samples" / pairs.name).read_bytes():
                raise ValueError(f"{extra.name}: {pairs.name} differs from {base.name}, runs not comparable")
    shutil.copytree(base, out)
    for extra in extras:
        for config in extra.glob("config*.toml"):
            shutil.copyfile(config, out / f"{config.stem}_from_{extra.name}.toml")
        for pred in (extra / "predictions").glob("*.csv"):
            shutil.copyfile(pred, out / "predictions" / pred.name)
        for file in METRIC_FILES:
            if (extra / "metrics" / file).is_file():
                merged = pd.concat([pd.read_csv(out / "metrics" / file), pd.read_csv(extra / "metrics" / file)],
                                   ignore_index=True)
                merged.to_csv(out / "metrics" / file, index=False)


def recent_pools(config, windows: list[int]) -> dict[int, np.ndarray]:
    """Same pool as samples.build_target: nodes that received in [w - RECENT_WINDOWS, w)."""
    events, _, _ = data.load_events(config.data.token, config.paths.data_dir, config.paths.cache_dir)
    _, events["window_index"] = splits.make_windows(events, config.data.freq)
    return {w: np.unique(events.loc[events["window_index"].between(w - RECENT_WINDOWS, w - 1), "dst"].to_numpy())
            for w in windows}


def main() -> Path:
    config = load_config(DEFAULT_CONFIG)
    runs = config.paths.output_root / "runs"
    out = runs / OUT
    merge(runs / BASE, [runs / e for e in EXTRA], out)
    targets = pd.read_csv(out / "splits" / "targets.csv")
    pools = recent_pools(config, targets.loc[targets["phase"] == "test", "window_index"].tolist())
    pipeline._save(pipeline._by_dst_recent(out / "predictions", pools, config.evaluation.hits_k),
                   out / "metrics" / "by_dst_recent.csv")
    figures.make_all(out, config.figures, n_neg=config.sampling.n_neg_eval)
    return out


if __name__ == "__main__":
    print(main())
