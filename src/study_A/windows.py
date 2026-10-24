"""Disjoint calendar windows over deduplicated events.

Windows are half-open ``[start, end)`` calendar intervals (calendar months by
default), so every unique event belongs to exactly one window. TGS snapshot IDs
are not used for windowing because snapshots overlap.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.frequencies import to_offset

FULL_PERIOD = "FULL_PERIOD"


def calendar_windows(dates: pd.Series, freq: str = "MS", min_coverage: float = 0.8) -> pd.DataFrame:
    """Return the window manifest covering the observed date range.

    ``coverage`` is the share of a window's calendar days that fall inside the
    token's observed period ``[first date, last date]``; windows below
    ``min_coverage`` (typically the first and last month) are flagged partial
    so temporal comparisons can mark them instead of silently mixing them in.
    """
    offset = to_offset(freq)
    first, last = dates.min().normalize(), dates.max().normalize()
    observed_end = last + pd.Timedelta(days=1)
    start = first if offset.is_on_offset(first) else offset.rollback(first)
    rows = []
    while start < observed_end:
        end = start + offset
        days = (end - start).days
        covered = (min(end, observed_end) - max(start, first)).days
        rows.append({
            "window": f"{start:%Y-%m-%d}",
            "window_index": len(rows) + 1,
            "start_inclusive": start,
            "end_exclusive": end,
            "calendar_days": days,
            "covered_days": covered,
            "coverage": covered / days,
            "partial": covered / days < min_coverage,
        })
        start = end
    return pd.DataFrame(rows)


def assign_windows(events: pd.DataFrame, manifest: pd.DataFrame) -> pd.Series:
    """Window label of every event (each event falls in exactly one window)."""
    edges = pd.DatetimeIndex(list(manifest["start_inclusive"]) + [manifest["end_exclusive"].iloc[-1]])
    labels = pd.cut(events["date"], bins=edges, right=False, labels=list(manifest["window"]))
    if labels.isna().any():
        raise ValueError(f"{int(labels.isna().sum())} events fall outside the window manifest")
    return labels.astype(str)
