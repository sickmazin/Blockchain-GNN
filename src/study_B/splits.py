"""Disjoint calendar windows and the chronological phase of every window.

Output: ``splits/windows.csv`` with one row per window:
    window_index, window (start date), start_inclusive, end_exclusive, coverage, partial, phase
where phase is "history" (never a target), "train", "val", "test" or "excluded" (partial window).

"""

from __future__ import annotations

import pandas as pd

from src.study_A.windows import assign_windows, calendar_windows

from .config import SplitConfig

PHASES = ("train", "val", "test")


def make_windows(events: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (window manifest, window_index of every event).
    """
    # min_coverage=1.0: a window not fully covered by the data (first/last week) is "partial".
    manifest = calendar_windows(events["date"], freq, min_coverage=1.0)
    labels = assign_windows(events, manifest)
    window_index = labels.map(manifest.set_index("window")["window_index"]).astype("int64")
    return manifest, window_index


def assign_phases(manifest: pd.DataFrame, split: SplitConfig) -> pd.DataFrame:
    """Add the ``phase`` column to the manifest (rows stay in chronological order)."""
    manifest = manifest.sort_values("window_index", ignore_index=True)
    phase = pd.Series("target", index=manifest.index)
    phase.iloc[:split.min_history_windows] = "history"
    if split.drop_partial_targets:
        phase[(phase == "target") & manifest["partial"]] = "excluded"

    # Chronological 70/15/15 over the target windows only.
    targets = phase.index[phase == "target"]
    n_train = round(split.train_frac * len(targets))
    n_val = round(split.val_frac * len(targets))
    phase[targets[:n_train]] = "train"
    phase[targets[n_train:n_train + n_val]] = "val"
    phase[targets[n_train + n_val:]] = "test"

    counts = phase.value_counts()
    empty = [p for p in PHASES if counts.get(p, 0) == 0]
    if empty:
        raise ValueError(f"no target window in phase(s) {empty}: {len(targets)} target windows "
                         f"after {split.min_history_windows} history and the partial-window filter")
    return manifest.assign(phase=phase)
