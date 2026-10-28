"""Historical-only node features for each target window.

For target window t every feature is computed from events in windows < t only; nodes not yet
seen get zeros (they are never scored: cold-start pairs are excluded).

Features (one column each, assistant.md §6.3):
    out_degree, in_degree         log1p of distinct counterparties in the history graph
    out_count, in_count           log1p of the number of transfers sent / received
    out_volume, in_volume         log1p of the raw amount sent / received
    pagerank                      on the history graph induced by history nodes (directed, unweighted)
    clustering                    local clustering on the undirected projection
    age                           t - first window of the node
    recency                       t - last window in which the node was active
    recent_out, recent_in         log1p of transfers sent / received in the last ``RECENT_WINDOWS`` windows

TUTOR — perché log1p e scaler fit solo sul train?
    Volumi e conteggi hanno code pesantissime (Study A): senza log1p pochi hub schiacciano la scala
    di tutti gli altri. Lo StandardScaler va stimato sulle finestre train e applicato invariato a
    val e test, altrimenti media e varianza "del futuro" entrano nelle feature (assistant.md §4).
"""

from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .samples import TargetWindow

FEATURES = ["out_degree", "in_degree", "out_count", "in_count", "out_volume", "in_volume",
            "pagerank", "clustering", "age", "recency", "recent_out", "recent_in"]
RECENT_WINDOWS = 4


def _per_node(ids: pd.Series, weights: pd.Series, n_nodes: int) -> np.ndarray:
    """Sum ``weights`` per node id into an array of length n_nodes (0 for absent nodes)."""
    return np.bincount(ids.to_numpy(), weights=weights.to_numpy(dtype=np.float64), minlength=n_nodes)


def node_features(events: pd.DataFrame, target: TargetWindow, n_nodes: int) -> np.ndarray:
    """Return float32 array [n_nodes, len(FEATURES)] for the target window."""
    t = target.window_index
    history = events[events["window_index"] < t]
    volume = history["weight"] * history["multiplicity"]
    recent = history[history["window_index"] >= t - RECENT_WINDOWS]

    cols = {
        "out_degree": np.bincount(target.history_edges[0], minlength=n_nodes),
        "in_degree": np.bincount(target.history_edges[1], minlength=n_nodes),
        "out_count": _per_node(history["src"], history["multiplicity"], n_nodes),
        "in_count": _per_node(history["dst"], history["multiplicity"], n_nodes),
        "out_volume": _per_node(history["src"], volume, n_nodes),
        "in_volume": _per_node(history["dst"], volume, n_nodes),
        "recent_out": _per_node(recent["src"], recent["multiplicity"], n_nodes),
        "recent_in": _per_node(recent["dst"], recent["multiplicity"], n_nodes),
    }
    cols = {name: np.log1p(values) for name, values in cols.items()}

    # PageRank and clustering on the history nodes only: nodes that appear after t must not take
    # teleport mass. Without self-loops, history_nodes are exactly the endpoints of history_edges.
    nodes = target.history_nodes
    position = np.full(n_nodes, -1, dtype=np.int64)
    position[nodes] = np.arange(len(nodes))
    g = ig.Graph(n=len(nodes), edges=position[target.history_edges].T.tolist(), directed=True)
    for name, values in (("pagerank", g.pagerank(directed=True)),
                         ("clustering", g.as_undirected().transitivity_local_undirected(mode="zero"))):
        cols[name] = np.zeros(n_nodes)
        cols[name][nodes] = values

    seen = pd.concat([history[["src", "window_index"]].rename(columns={"src": "node"}),
                      history[["dst", "window_index"]].rename(columns={"dst": "node"})])
    span = seen.groupby("node")["window_index"].agg(["min", "max"])
    cols["age"] = np.zeros(n_nodes)
    cols["recency"] = np.zeros(n_nodes)
    cols["age"][span.index] = t - span["min"].to_numpy()
    cols["recency"][span.index] = t - span["max"].to_numpy()

    # Every column is already 0 outside history_nodes: nothing from windows >= t enters.
    return np.column_stack([cols[name] for name in FEATURES]).astype(np.float32)


def fit_scaler(x_by_window: dict[int, np.ndarray], targets: list[TargetWindow]) -> StandardScaler:
    """StandardScaler fit on history-node rows of the TRAIN target windows only."""
    rows = []
    for target in targets:
        if target.phase == 'train':
            rows.append(x_by_window[target.window_index][target.history_nodes])

    stack =  np.vstack(rows)

    scaler = StandardScaler()

    scaler.fit(stack)
    return scaler