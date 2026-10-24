"""Inequality and concentration of activity across addresses.

Measures over a non-negative per-address quantity (degree or strength):
Gini coefficient, Herfindahl-Hirschman index of shares, shares held by the top
1% / 10% of addresses, and the minimum number of addresses accounting for half
of the total. These describe structural concentration of transfer activity;
they do not identify the economic role of any address.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

QUANTITIES = ["in_degree", "out_degree", "in_strength", "out_strength"]


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0 or x.sum() <= 0:
        return math.nan
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def concentration(values: np.ndarray) -> dict[str, float]:
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    x = x[np.isfinite(x)]
    total = x.sum()
    if len(x) == 0 or total <= 0:
        return {k: math.nan for k in ("gini", "hhi", "top1pct_share", "top10pct_share",
                                      "addresses_for_half", "addresses_for_half_share")}
    shares = x / total
    cumulative = np.cumsum(shares)
    half = int(np.searchsorted(cumulative, 0.5) + 1)
    return {
        "gini": gini(x),
        "hhi": float((shares ** 2).sum()),
        "top1pct_share": float(shares[: max(1, math.ceil(0.01 * len(x)))].sum()),
        "top10pct_share": float(shares[: max(1, math.ceil(0.10 * len(x)))].sum()),
        "addresses_for_half": half,
        "addresses_for_half_share": half / len(x),
    }


def concentration_table(centralities: pd.DataFrame) -> pd.DataFrame:
    """Concentration of every degree/strength quantity of one graph."""
    rows = []
    for quantity in QUANTITIES:
        rows.append({"quantity": quantity, **concentration(centralities[quantity].to_numpy())})
    return pd.DataFrame(rows)


def concentration_curve(values: np.ndarray, points: int = 200) -> pd.DataFrame:
    """Share of the total held by the top fraction of addresses, log-spaced in that fraction.

    This is the complement of the Lorenz curve read from the top; log spacing keeps
    the extreme upper tail resolvable when a handful of addresses dominate.
    """
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    x = x[np.isfinite(x)]
    if len(x) == 0 or x.sum() <= 0:
        return pd.DataFrame(columns=["top_fraction", "share_held", "addresses"])
    counts = np.unique(np.round(np.geomspace(1, len(x), points)).astype(int))
    held = np.cumsum(x)[counts - 1] / x.sum()
    return pd.DataFrame({"top_fraction": counts / len(x), "share_held": held, "addresses": counts})
