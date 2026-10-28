"""History graph, positives and negatives for every target window.

For a target window ``t``:
    history graph  = unique ordered pairs (u, v) with at least one event in a window < t
    history nodes  = nodes with at least one event in a window < t
    positive       = pair (u, v) whose FIRST event is in t, with u and v both history nodes
    cold-start     = pair first seen in t with u or v NOT a history node (counted, then excluded)
    negative       = (u, v') with the same source u as a positive, v' drawn from a candidate pool, and
                     (u, v') neither a history edge nor a pair of window t (assistant.md §6.2)

Two evaluation negative sets, differing only in the pool v' is drawn from (NEGATIVE_POOLS):
    uniform        every history node (most are long dormant: an easy contrast)
    recent         nodes that RECEIVED at least one transfer in the last ``recent_windows`` windows
                   before t (hard negatives: "active" destinations u did not choose)

Output: one ``TargetWindow`` per target window, plus ``splits/targets.csv`` (one row per window:
n_history_nodes, n_recent_receivers, n_history_edges, n_positives, n_cold_start) and the evaluation
pairs ``samples/pairs_<negatives>_<phase>.csv`` (window_index, src, dst, label, group).

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TargetWindow:
    window_index: int
    phase: str
    history_edges: np.ndarray    # shape [2, E]: unique ordered pairs with first event < t (src row, dst row)
    history_nodes: np.ndarray    # sorted node ids active before t
    positives: np.ndarray        # shape [P, 2]: new pairs in t, both endpoints historical
    window_pairs: np.ndarray     # shape [Q, 2]: ALL pairs with an event in t (to exclude from negatives)
    n_cold_start: int
    recent_receivers: np.ndarray  # sorted nodes that received >= 1 transfer in [t - recent_windows, t)


NEGATIVE_POOLS = {"uniform": "history_nodes", "recent": "recent_receivers"}   # set name -> TargetWindow field


def first_seen(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (pairs, node_first_window).

    pairs:             one row per ordered pair: src, dst, first_window
    node_first_window: for every node id, the first window in which it appears (as src OR dst)
    """
    pairs = (events.groupby(["src", "dst"])["window_index"].min()
             .rename("first_window").reset_index())

    nodes = pd.concat([events[["src", "window_index"]].rename(columns={"src": "node"}),
                       events[["dst", "window_index"]].rename(columns={"dst": "node"})])
    node_first_window = nodes.groupby("node")["window_index"].min()

    return pairs, node_first_window


def build_target(events: pd.DataFrame, pairs: pd.DataFrame, node_first_window: pd.Series,
                 window_index: int, phase: str, recent_windows: int) -> TargetWindow:
    """Assemble the TargetWindow of window ``t = window_index``."""
    history = pairs[pairs["first_window"] < window_index]
    history_edges = np.vstack([history["src"].to_numpy(), history["dst"].to_numpy()]).astype(np.int64)

    history_nodes = np.sort(node_first_window.index[node_first_window < window_index].to_numpy()).astype(np.int64)

    new_pairs = pairs[pairs["first_window"] == window_index]
    known = new_pairs["src"].isin(history_nodes) & new_pairs["dst"].isin(history_nodes)
    positives = new_pairs.loc[known, ["src", "dst"]].to_numpy(dtype=np.int64)
    n_cold_start = int((~known).sum())

    # All pairs active in t (new AND repeated): none of them may be drawn as a negative.
    in_window = events.loc[events["window_index"] == window_index, ["src", "dst"]].drop_duplicates()
    window_pairs = in_window.to_numpy(dtype=np.int64)

    recent = events["window_index"].between(window_index - recent_windows, window_index - 1)
    recent_receivers = np.unique(events.loc[recent, "dst"].to_numpy()).astype(np.int64)

    return TargetWindow(window_index=window_index, phase=phase, history_edges=history_edges,
                        history_nodes=history_nodes, positives=positives, window_pairs=window_pairs,
                        n_cold_start=n_cold_start, recent_receivers=recent_receivers)


MAX_RESAMPLE_ROUNDS = 1000


def sample_negatives(target: TargetWindow, n_neg: int, rng: np.random.Generator,
                     pool: str = "uniform") -> np.ndarray:
    """Return dst ids of shape [P, n_neg]: row i holds the negatives of ``target.positives[i]``,
    drawn uniformly from the ``NEGATIVE_POOLS[pool]`` nodes of the window."""
    candidates = getattr(target, NEGATIVE_POOLS[pool])
    # Encode every ordered pair as one int64 key: src * base + dst (base > every node id).
    base = np.int64(max(target.history_nodes.max(initial=0), target.window_pairs.max(initial=0)) + 1)
    forbidden = np.unique(np.concatenate([
        target.history_edges[0] * base + target.history_edges[1],
        target.window_pairs[:, 0] * base + target.window_pairs[:, 1],
    ]))

    src = target.positives[:, [0]]                      # [P, 1], broadcast over the n_neg columns
    neg = rng.choice(candidates, size=(len(src), n_neg))
    bad = (neg == src) | np.isin(src * base + neg, forbidden)
    for _ in range(MAX_RESAMPLE_ROUNDS):
        if not bad.any():
            return neg
        neg[bad] = rng.choice(candidates, size=int(bad.sum()))
        bad = (neg == src) | np.isin(src * base + neg, forbidden)
    raise RuntimeError(f"window {target.window_index}: {int(bad.sum())} negatives still invalid "
                       f"after {MAX_RESAMPLE_ROUNDS} rounds (a source is linked to almost every {pool} candidate)")


def eval_pairs(target: TargetWindow, negatives: np.ndarray, group_offset: int) -> pd.DataFrame:
    """Long table of a window: window_index, src, dst, label (1/0), group."""
    n_pos, n_neg = negatives.shape
    groups = group_offset + np.arange(n_pos)
    pos = pd.DataFrame({"src": target.positives[:, 0], "dst": target.positives[:, 1],
                        "label": 1, "group": groups})
    neg = pd.DataFrame({"src": np.repeat(target.positives[:, 0], n_neg), "dst": negatives.ravel(),
                        "label": 0, "group": np.repeat(groups, n_neg)})
    table = pd.concat([pos, neg], ignore_index=True)
    table.insert(0, "window_index", target.window_index)
    return table


def target_summary(targets: list[TargetWindow]) -> pd.DataFrame:
    """One row per target window: window_index, phase, n_history_nodes, n_recent_receivers,
    n_history_edges, n_positives, positives_dst_recent_share, n_cold_start, cold_start_share."""
    rows = []
    for t in targets:
        n_new = len(t.positives) + t.n_cold_start
        rows.append({"window_index": t.window_index, "phase": t.phase,
                     "n_history_nodes": len(t.history_nodes), "n_recent_receivers": len(t.recent_receivers),
                     "n_history_edges": t.history_edges.shape[1],
                     # share of true destinations that are themselves in the "recent" pool
                     "positives_dst_recent_share": (np.isin(t.positives[:, 1], t.recent_receivers).mean()
                                                    if len(t.positives) else np.nan),
                     "n_positives": len(t.positives), "n_cold_start": t.n_cold_start,
                     "cold_start_share": t.n_cold_start / n_new if n_new else np.nan})
    return pd.DataFrame(rows)
