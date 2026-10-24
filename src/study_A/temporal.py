"""Comparisons between consecutive calendar windows of one token.

* address turnover: retained, new and departed addresses, Jaccard of address sets;
* tie persistence: share of ordered pairs of window ``t`` observed again in ``t+1``;
* centrality-rank stability: Jaccard overlap of the top-k addresses and Kendall's
  tau-b on the union of both top-k sets (an address absent from a window scores 0
  there, i.e. it is ranked last).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def address_turnover(previous: set[str], current: set[str]) -> dict[str, float]:
    retained = previous & current
    union = previous | current
    return {
        "addresses_previous": len(previous),
        "addresses_current": len(current),
        "retained_addresses": len(retained),
        "new_addresses": len(current - previous),
        "departed_addresses": len(previous - current),
        "address_jaccard": len(retained) / len(union) if union else math.nan,
        "retained_share_of_previous": len(retained) / len(previous) if previous else math.nan,
        "new_share_of_current": len(current - previous) / len(current) if current else math.nan,
    }


def tie_persistence(previous: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    prev_pairs = set(zip(previous["source"], previous["destination"]))
    curr_pairs = set(zip(current["source"], current["destination"]))
    repeated = prev_pairs & curr_pairs
    return {
        "pairs_previous": len(prev_pairs),
        "pairs_current": len(curr_pairs),
        "repeated_pairs": len(repeated),
        "repeated_share_of_previous": len(repeated) / len(prev_pairs) if prev_pairs else math.nan,
        "repeated_share_of_current": len(repeated) / len(curr_pairs) if curr_pairs else math.nan,
    }


def rank_stability(previous: pd.DataFrame, current: pd.DataFrame, measures: list[str],
                   k: int) -> pd.DataFrame:
    rows = []
    for measure in measures:
        prev = previous[["node", measure]].dropna()
        curr = current[["node", measure]].dropna()
        if prev.empty or curr.empty:
            rows.append({"measure": measure, "k": k, "topk_jaccard": math.nan, "kendall_tau": math.nan})
            continue
        top_prev = set(prev.nlargest(k, measure, keep="first")["node"])
        top_curr = set(curr.nlargest(k, measure, keep="first")["node"])
        union = sorted(top_prev | top_curr)
        a = prev.set_index("node")[measure].reindex(union).fillna(0.0).to_numpy()
        b = curr.set_index("node")[measure].reindex(union).fillna(0.0).to_numpy()
        tau = stats.kendalltau(a, b).statistic if len(union) > 2 and np.ptp(a) and np.ptp(b) else math.nan
        rows.append({"measure": measure, "k": k,
                     "topk_jaccard": len(top_prev & top_curr) / len(top_prev | top_curr),
                     "kendall_tau": float(tau)})
    return pd.DataFrame(rows)
