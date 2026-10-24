"""Graph construction: pair aggregation, audited filters, weight variants and projections.

Graph definitions (all directed unless stated):

* **frequency**: weight of ``(u, v)`` = number of unique transfers ``u -> v`` in the window;
* **raw_volume**: weight = summed raw token amount (on-chain integer units, not
  decimal-adjusted, so comparable only within a token);
* **log1p_volume**: weight = ``log(1 + raw volume)``, a labelled variant for
  measures sensitive to extreme values.

The unweighted topology of every window is that of the frequency graph (all
transfer pairs). For shortest-path measures tie strength is converted to a
distance ``1 / log(1 + w)``; raw amounts are never used as distances. The
undirected projection sums the weights of reciprocal arcs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd

from .data import ZERO_ADDRESS

VARIANTS = {"frequency": "event_count", "raw_volume": "volume", "log1p_volume": "log1p_volume"}
VARIANT_LABELS = {
    "frequency": "transfer count",
    "raw_volume": "raw token volume",
    "log1p_volume": "log(1 + raw token volume)",
}
DISTANCE_RULE = "distance = 1 / log(1 + w)"


def aggregate_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """One row per ordered address pair with event count, volume and activity span."""
    frame = events.assign(amount=events["weight"] * events["multiplicity"])
    pairs = (frame.groupby(["source", "destination"], sort=True, observed=True)
             .agg(event_count=("multiplicity", "sum"), volume=("amount", "sum"),
                  active_days=("date", "nunique"), first_date=("date", "min"),
                  last_date=("date", "max"))
             .reset_index())
    pairs["log1p_volume"] = np.log1p(pairs["volume"].to_numpy())
    return pairs


def _totals(pairs: pd.DataFrame) -> dict[str, Any]:
    nodes = pd.unique(pd.concat([pairs["source"], pairs["destination"]], ignore_index=True))
    return {"pairs": int(len(pairs)), "events": int(pairs["event_count"].sum()),
            "volume": float(pairs["volume"].sum()), "nodes": int(len(nodes))}


def filter_pairs(pairs: pd.DataFrame, *, exclude_self_loops: bool = True,
                 exclude_zero_address: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the configurable analytical filters; audit totals before, removed and after.

    Filters are applied in order and each removal is counted after the earlier
    filters, so the removed totals never overlap.
    """
    is_loop = pairs["source"].eq(pairs["destination"])
    is_zero = pairs["source"].eq(ZERO_ADDRESS) | pairs["destination"].eq(ZERO_ADDRESS)
    audit: dict[str, Any] = {f"before_{k}": v for k, v in _totals(pairs).items()}
    audit.update({"self_loop_pairs": int(is_loop.sum()),
                  "self_loop_events": int(pairs.loc[is_loop, "event_count"].sum()),
                  "zero_address_pairs": int(is_zero.sum()),
                  "zero_address_events": int(pairs.loc[is_zero, "event_count"].sum()),
                  "zero_address_volume": float(pairs.loc[is_zero, "volume"].sum()),
                  "exclude_self_loops": exclude_self_loops,
                  "exclude_zero_address": exclude_zero_address})
    keep = pd.Series(True, index=pairs.index)
    for name, mask, active in (("self_loops", is_loop, exclude_self_loops),
                               ("zero_address", is_zero, exclude_zero_address)):
        removed = keep & mask if active else pd.Series(False, index=pairs.index)
        audit[f"removed_{name}_pairs"] = int(removed.sum())
        audit[f"removed_{name}_events"] = int(pairs.loc[removed, "event_count"].sum())
        audit[f"removed_{name}_volume"] = float(pairs.loc[removed, "volume"].sum())
        keep &= ~removed
    filtered = pairs.loc[keep].reset_index(drop=True)
    audit.update({f"after_{k}": v for k, v in _totals(filtered).items()})
    return filtered, audit


def variant_edges(pairs: pd.DataFrame, variant: str, *, drop_zero_volume: bool = True
                  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Edge list ``(source, destination, weight)`` of one weight variant, with its audit."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    column = VARIANTS[variant]
    zero = pairs["volume"].le(0) if variant != "frequency" else pd.Series(False, index=pairs.index)
    removed = zero & drop_zero_volume
    kept = pairs.loc[~removed]
    edges = pd.DataFrame({"source": kept["source"].to_numpy(),
                          "destination": kept["destination"].to_numpy(),
                          "weight": kept[column].to_numpy(dtype=float)})
    audit = {"variant": variant, "weight_column": column,
             "zero_volume_pairs": int(zero.sum()),
             "zero_volume_events": int(pairs.loc[zero, "event_count"].sum()),
             "removed_zero_volume_pairs": int(removed.sum()),
             "edges": int(len(edges))}
    return edges, audit


@dataclass
class GraphBundle:
    """A directed weighted graph plus its undirected projection and descriptive metadata."""

    directed: ig.Graph
    undirected: ig.Graph
    variant: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return self.directed.vs["name"]


def _distance(weights: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        distance = np.where(weights > 0, 1.0 / np.log1p(np.clip(weights, 0, None)), np.nan)
    return distance


def build_digraph(edges: pd.DataFrame) -> ig.Graph:
    """Directed igraph with vertex ``name`` and edge ``weight``/``distance`` attributes."""
    codes, names = pd.factorize(pd.concat([edges["source"], edges["destination"]], ignore_index=True),
                                sort=True)
    n_edges = len(edges)
    graph = ig.Graph(n=len(names), edges=np.column_stack([codes[:n_edges], codes[n_edges:]]).tolist(),
                     directed=True)
    graph.vs["name"] = list(names)
    weights = edges["weight"].to_numpy(dtype=float)
    graph.es["weight"] = weights.tolist()
    graph.es["distance"] = _distance(weights).tolist()
    return graph


def project_undirected(graph: ig.Graph) -> ig.Graph:
    """Undirected projection: one edge per unordered pair, weight = sum of both arcs."""
    projection = graph.copy()
    projection.to_undirected(mode="collapse", combine_edges={"weight": "sum"})
    weights = np.asarray(projection.es["weight"], dtype=float) if projection.ecount() else np.array([])
    projection.es["distance"] = _distance(weights).tolist()
    return projection


def build_bundle(edges: pd.DataFrame, variant: str, metadata: dict[str, Any] | None = None) -> GraphBundle:
    directed = build_digraph(edges)
    return GraphBundle(directed, project_undirected(directed), variant, dict(metadata or {}))


def distances_valid(graph: ig.Graph) -> bool:
    if graph.ecount() == 0:
        return False
    distance = np.asarray(graph.es["distance"], dtype=float)
    return bool(np.all(np.isfinite(distance)) and np.all(distance > 0))


def largest_component(graph: ig.Graph, mode: str = "weak") -> np.ndarray:
    """Vertex ids of the largest (weak/strong) component, ties broken by lowest vertex id."""
    if graph.vcount() == 0:
        return np.array([], dtype=int)
    membership = np.asarray(graph.connected_components(mode=mode).membership)
    sizes = np.bincount(membership)
    return np.flatnonzero(membership == int(np.argmax(sizes)))
