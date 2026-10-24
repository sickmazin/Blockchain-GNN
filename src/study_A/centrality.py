"""Node centralities of a directed weighted transfer graph.

All measures are computed per window and weight variant. Choices:

* degree centrality: in/out degree divided by ``n - 1`` (unweighted), plus in/out strength;
* PageRank (directed, weighted, damping configurable) and HITS hub/authority scores;
* eigenvector centrality: defined on the largest strongly connected component (LSCC)
  of the directed graph (left eigenvector = "in", reversed graph = "out"), because
  outside a strongly connected graph it degenerates; plus the undirected
  eigenvector centrality on the projection's largest connected component;
* betweenness: weighted by ``distance = 1/log(1+w)``, normalised by ``(n-1)(n-2)``;
  exact up to ``exact_max_nodes`` nodes, otherwise Brandes-Pich source sampling
  (seeded pivots, rescaled by ``n/k``);
* closeness: *harmonic* closeness (Marchiori-Latora), well defined on disconnected
  directed graphs, incoming and outgoing, weighted by the same distance. Above
  ``exact_max_nodes`` it is computed for a documented candidate set (the top nodes
  by degree, strength and PageRank plus a seeded random sample); other nodes are NaN.

Scale-free measures (PageRank, HITS, eigenvector) use weights divided by their
maximum, which leaves the scores unchanged but avoids overflow for 1e27-scale raw
token amounts. Every measure records a status row; failures are never zero-filled.
"""

from __future__ import annotations

import logging
import random
import time
import warnings
from typing import Any, Callable

import igraph as ig
import numpy as np
import pandas as pd

from .graphs import GraphBundle, distances_valid, largest_component

logger = logging.getLogger("study_A")

MEASURES = [
    "in_degree_centrality", "out_degree_centrality", "in_strength", "out_strength",
    "pagerank", "hub", "authority", "eigenvector_in_lscc", "eigenvector_out_lscc",
    "eigenvector_undirected_lcc", "betweenness", "harmonic_in", "harmonic_out",
]
MEASURE_LABELS = {
    "in_degree_centrality": "In-degree", "out_degree_centrality": "Out-degree",
    "in_strength": "In-strength", "out_strength": "Out-strength", "pagerank": "PageRank",
    "hub": "Hub", "authority": "Authority", "eigenvector_in_lscc": "Eigenvector (in, LSCC)",
    "eigenvector_out_lscc": "Eigenvector (out, LSCC)",
    "eigenvector_undirected_lcc": "Eigenvector (undirected)", "betweenness": "Betweenness",
    "harmonic_in": "Harmonic closeness (in)", "harmonic_out": "Harmonic closeness (out)",
}


def _scaled_weights(graph: ig.Graph) -> list[float]:
    weights = np.asarray(graph.es["weight"], dtype=float)
    top = weights.max(initial=0.0)
    return (weights / top).tolist() if top > 0 else weights.tolist()


def compute_centralities(bundle: GraphBundle, *, seed: int, exact_max_nodes: int,
                         betweenness_pivots: int, harmonic_top_candidates: int,
                         harmonic_random_candidates: int, damping: float
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = bundle.directed
    n = g.vcount()
    table = pd.DataFrame({"node": g.vs["name"],
                          "in_degree": g.indegree(), "out_degree": g.outdegree()})
    for column in MEASURES:
        table[column] = np.nan
    statuses: list[dict[str, Any]] = []

    def record(names: list[str], action: Callable[[], dict[str, Any]], **details: Any) -> None:
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                values = action()
                for name in names:
                    table[name] = np.asarray(values[name], dtype=float)
                status, error = "ok", None
            except Exception as exc:  # Keep failures visible in the status table.
                status, error = "omitted", f"{type(exc).__name__}: {exc}"
                logger.warning("Centrality %s failed: %s", names, error)
        # igraph diagnostics (e.g. many zero HITS scores) are kept with the measure.
        notes = "; ".join(sorted({str(w.message).split(" Location:")[0] for w in caught})) or None
        for name in names:
            statuses.append({"measure": name, "status": status, "error": error, "warnings": notes,
                             "seconds": round(time.perf_counter() - started, 3), **details})

    if n == 0:
        return table, pd.DataFrame(statuses)
    weights = _scaled_weights(g)
    record(["in_degree_centrality", "out_degree_centrality", "in_strength", "out_strength"], lambda: {
        "in_degree_centrality": np.asarray(g.indegree()) / max(n - 1, 1),
        "out_degree_centrality": np.asarray(g.outdegree()) / max(n - 1, 1),
        "in_strength": g.strength(mode="in", weights="weight"),
        "out_strength": g.strength(mode="out", weights="weight"),
    }, method="exact", sample_size=n)
    record(["pagerank"], lambda: {"pagerank": g.pagerank(directed=True, damping=damping,
                                                          weights=weights)},
           method=f"exact, damping={damping}", sample_size=n)
    record(["hub", "authority"], lambda: {"hub": g.hub_score(weights=weights, scale=True),
                                          "authority": g.authority_score(weights=weights, scale=True)},
           method="exact", sample_size=n)

    lscc = largest_component(g, "strong")
    table["in_lscc"] = False
    table.loc[lscc, "in_lscc"] = True

    def eigen_lscc() -> dict[str, Any]:
        result = {"eigenvector_in_lscc": np.full(n, np.nan), "eigenvector_out_lscc": np.full(n, np.nan)}
        if len(lscc) < 3:
            raise ValueError(f"largest strongly connected component has {len(lscc)} nodes")
        sub = g.induced_subgraph(lscc.tolist())
        sub_weights = _scaled_weights(sub)
        result["eigenvector_in_lscc"][lscc] = sub.eigenvector_centrality(directed=True, weights=sub_weights)
        reverse = sub.copy()
        reverse.reverse_edges()
        result["eigenvector_out_lscc"][lscc] = reverse.eigenvector_centrality(
            directed=True, weights=_scaled_weights(reverse))
        return result

    record(["eigenvector_in_lscc", "eigenvector_out_lscc"], eigen_lscc,
           method="exact on LSCC", sample_size=int(len(lscc)))

    u = bundle.undirected
    lcc = largest_component(u, "weak")

    def eigen_undirected() -> dict[str, Any]:
        values = np.full(n, np.nan)
        sub = u.induced_subgraph(lcc.tolist())
        # igraph 1.0 rejects directed=False explicitly; the graph is undirected anyway.
        values[lcc] = sub.eigenvector_centrality(weights=_scaled_weights(sub))
        return {"eigenvector_undirected_lcc": values}

    record(["eigenvector_undirected_lcc"], eigen_undirected,
           method="exact on projection LCC", sample_size=int(len(lcc)))

    if not distances_valid(g):
        for name in ("betweenness", "harmonic_in", "harmonic_out"):
            statuses.append({"measure": name, "status": "omitted", "seconds": 0.0,
                             "error": "non-positive or missing tie strengths: no valid distance",
                             "method": None, "sample_size": 0})
        return table, pd.DataFrame(statuses)

    rng = random.Random(seed)
    if n <= exact_max_nodes:
        pivots, method = None, "exact"
    else:
        pivots = sorted(rng.sample(range(n), min(betweenness_pivots, n)))
        method = f"source sampling, k={len(pivots)}, rescaled n/k"

    def betweenness() -> dict[str, Any]:
        raw = np.asarray(g.betweenness(directed=True, weights="distance", sources=pivots), dtype=float)
        if pivots is not None:
            raw *= n / len(pivots)
        return {"betweenness": raw / ((n - 1) * (n - 2)) if n > 2 else raw}

    record(["betweenness"], betweenness, method=method,
           sample_size=n if pivots is None else len(pivots))

    if n <= exact_max_nodes:
        candidates, method = list(range(n)), "exact"
    else:
        ranked = set()
        for column in ("in_degree", "out_degree", "in_strength", "out_strength", "pagerank"):
            order = table[column].fillna(-np.inf).to_numpy()
            ranked.update(np.argsort(-order, kind="stable")[:harmonic_top_candidates].tolist())
        rest = sorted(set(range(n)) - ranked)
        ranked.update(rng.sample(rest, min(harmonic_random_candidates, len(rest))))
        candidates = sorted(ranked)
        method = (f"candidate set: top {harmonic_top_candidates} by degree/strength/PageRank "
                  f"+ {harmonic_random_candidates} random")

    def harmonic() -> dict[str, Any]:
        result = {"harmonic_in": np.full(n, np.nan), "harmonic_out": np.full(n, np.nan)}
        for mode in ("in", "out"):
            result[f"harmonic_{mode}"][candidates] = g.harmonic_centrality(
                vertices=candidates, mode=mode, weights="distance", normalized=True)
        return result

    record(["harmonic_in", "harmonic_out"], harmonic, method=method, sample_size=len(candidates))
    return table, pd.DataFrame(statuses)


def top_k(table: pd.DataFrame, k: int, measures: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for measure in measures or MEASURES:
        ranked = (table.dropna(subset=[measure])
                  .sort_values([measure, "node"], ascending=[False, True], kind="stable").head(k))
        for rank, (node, score) in enumerate(zip(ranked["node"], ranked[measure]), start=1):
            rows.append({"measure": measure, "rank": rank, "node": node, "score": score})
    return pd.DataFrame(rows)


def centrality_correlations(table: pd.DataFrame, measures: list[str] | None = None) -> pd.DataFrame:
    """Spearman rank correlations between measures, on nodes where both are defined."""
    columns = [m for m in (measures or MEASURES) if table[m].notna().sum() > 2]
    return table[columns].corr(method="spearman", min_periods=3)
