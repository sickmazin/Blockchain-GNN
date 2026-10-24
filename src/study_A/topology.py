"""Unweighted structural descriptors of the directed graph and its undirected projection.

- Directed measures (density, reciprocity, components, bow-tie, directed degree
    assortativity) are computed on the directed frequency graph.

- Clustering,undirected assortativity, path lengths, k-cores and the rich-club coefficient
    are computed on the undirected projection; path lengths always refer to its
    largest connected component (LCC). Exact path statistics are used up to a
    configured size, otherwise a seeded uniform sample of BFS sources is used and
    the diameter is reported as a lower bound.
"""

from __future__ import annotations

import math
import random
from typing import Any

import igraph as ig
import numpy as np

from .graphs import largest_component

BOWTIE_ORDER = ["CORE", "IN", "OUT", "TENDRILS_TUBES", "DISCONNECTED"]


def directed_descriptors(graph: ig.Graph) -> dict[str, Any]:
    n, m = graph.vcount(), graph.ecount()
    indeg = np.asarray(graph.indegree())
    outdeg = np.asarray(graph.outdegree())
    weak = graph.connected_components(mode="weak")
    strong = graph.connected_components(mode="strong")
    largest_weak = max(weak.sizes(), default=0)
    largest_strong = max(strong.sizes(), default=0)
    return {
        "nodes": n,
        "edges": m,
        "density": m / (n * (n - 1)) if n > 1 else math.nan,
        "mean_out_degree": m / n if n else math.nan,
        "max_in_degree": int(indeg.max(initial=0)),
        "max_out_degree": int(outdeg.max(initial=0)),
        # Share of arcs (u, v) whose reverse arc (v, u) is also present.
        "reciprocity": graph.reciprocity(ignore_loops=True, mode="default") if m else math.nan,
        "weak_components": len(weak),
        "largest_wcc_nodes": largest_weak,
        "largest_wcc_share": largest_weak / n if n else math.nan,
        "strong_components": len(strong),
        "largest_scc_nodes": largest_strong,
        "largest_scc_share": largest_strong / n if n else math.nan,
        "sender_only_share": float(np.mean((indeg == 0) & (outdeg > 0))) if n else math.nan,
        "receiver_only_share": float(np.mean((outdeg == 0) & (indeg > 0))) if n else math.nan,
        "sender_and_receiver_share": float(np.mean((outdeg > 0) & (indeg > 0))) if n else math.nan,
    }


def bowtie(graph: ig.Graph) -> tuple[dict[str, Any], np.ndarray]:
    """Bow-tie decomposition around the largest strongly connected component (Broder et al.)."""
    n = graph.vcount()
    labels = np.full(n, "DISCONNECTED", dtype=object)
    if n == 0:
        return {f"bowtie_{k.lower()}_share": math.nan for k in BOWTIE_ORDER}, labels
    core = largest_component(graph, "strong")
    seed_vertex = int(core[0])
    forward = np.asarray(graph.subcomponent(seed_vertex, mode="out"))
    backward = np.asarray(graph.subcomponent(seed_vertex, mode="in"))
    weak = largest_component(graph, "weak")
    labels[weak] = "TENDRILS_TUBES"
    labels[forward] = "OUT"
    labels[backward] = "IN"
    labels[core] = "CORE"
    summary = {f"bowtie_{k.lower()}_share": float(np.mean(labels == k)) for k in BOWTIE_ORDER}
    summary["bowtie_core_nodes"] = int(len(core))
    return summary, labels


def directed_assortativity(graph: ig.Graph) -> dict[str, float]:
    """Pearson degree correlation over arcs u -> v for the four (source, target) degree modes."""
    if graph.ecount() < 2:
        return {f"assortativity_{a}_{b}": math.nan for a in ("out", "in") for b in ("in", "out")}
    edges = np.asarray(graph.get_edgelist())
    degree = {"in": np.asarray(graph.indegree(), dtype=float),
              "out": np.asarray(graph.outdegree(), dtype=float)}
    result = {}
    for a in ("out", "in"):
        for b in ("in", "out"):
            x, y = degree[a][edges[:, 0]], degree[b][edges[:, 1]]
            result[f"assortativity_{a}_{b}"] = (float(np.corrcoef(x, y)[0, 1])
                                                if x.std() > 0 and y.std() > 0 else math.nan)
    return result


def simple_undirected(projection: ig.Graph) -> ig.Graph:
    """Unweighted copy of the projection (topology only), used for nulls and path statistics."""
    return ig.Graph(n=projection.vcount(), edges=projection.get_edgelist(), directed=False)


def path_statistics(graph: ig.Graph, *, exact_max_nodes: int, n_sources: int,
                    seed: int) -> dict[str, Any]:
    """Average shortest path, 90th-percentile distance and diameter on the LCC (unweighted)."""
    lcc = graph.induced_subgraph(largest_component(graph, "weak").tolist())
    n = lcc.vcount()
    if n < 2:
        return {"lcc_nodes": n, "path_method": "trivial", "path_sources": 0,
                "average_path_length": math.nan, "distance_p90": math.nan, "diameter": math.nan}
    if n <= exact_max_nodes:
        hist = lcc.path_length_hist(directed=False)
        lengths = np.array([int(start) for start, _, _ in hist.bins()])
        counts = np.array([count for _, _, count in hist.bins()], dtype=float)
        method, sources = "exact", n
        diameter = float(lengths[counts > 0].max())
    else:
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(n), min(n_sources, n)))
        counts = np.zeros(n, dtype=float)
        for start in range(0, len(chosen), 64):
            block = np.asarray(lcc.distances(source=chosen[start:start + 64]), dtype=float)
            values = block[np.isfinite(block) & (block > 0)].astype(int)
            counts[: values.max() + 1] += np.bincount(values, minlength=values.max() + 1)
        lengths = np.arange(len(counts))
        # Double sweep from the farthest node found raises the diameter lower bound cheaply.
        far = int(np.argmax(np.asarray(lcc.distances(source=[chosen[0]])[0])))
        sweep = max(np.asarray(lcc.distances(source=[far])[0]))
        method, sources = "sampled_sources", len(chosen)
        diameter = float(max(lengths[counts > 0].max(), sweep))
    total = counts.sum()
    mean = float((lengths * counts).sum() / total)
    cdf = np.cumsum(counts) / total
    p90 = float(lengths[np.searchsorted(cdf, 0.9)])
    return {"lcc_nodes": n, "path_method": method, "path_sources": sources,
            "average_path_length": mean, "distance_p90": p90, "diameter": diameter,
            "diameter_is_lower_bound": method != "exact"}


def undirected_descriptors(projection: ig.Graph, *, exact_max_nodes: int, n_sources: int,
                           seed: int) -> tuple[dict[str, Any], np.ndarray]:
    simple = simple_undirected(projection)
    n, m = simple.vcount(), simple.ecount()
    coreness = np.asarray(simple.coreness(), dtype=int) if n else np.array([], dtype=int)
    max_core = int(coreness.max(initial=0))
    summary = {
        "undirected_nodes": n,
        "undirected_edges": m,
        "undirected_density": 2 * m / (n * (n - 1)) if n > 1 else math.nan,
        "transitivity_global": simple.transitivity_undirected() if m else math.nan,
        # Average local clustering with degree<2 nodes counted as 0, and excluding them.
        "avg_clustering_zero": simple.transitivity_avglocal_undirected(mode="zero") if m else math.nan,
        "avg_clustering_deg2": _avg_clustering_deg2(simple),
        "assortativity_undirected": simple.assortativity_degree(directed=False) if m > 1 else math.nan,
        "lcc_share": (len(largest_component(simple, "weak")) / n) if n else math.nan,
        "max_core": max_core,
        "max_core_nodes": int(np.sum(coreness == max_core)) if n else 0,
        **path_statistics(simple, exact_max_nodes=exact_max_nodes, n_sources=n_sources, seed=seed),
    }
    return summary, coreness


def _avg_clustering_deg2(graph: ig.Graph) -> float:
    local = np.asarray(graph.transitivity_local_undirected(mode="nan"), dtype=float)
    local = local[np.isfinite(local)]
    return float(local.mean()) if len(local) else math.nan


def rich_club_ks(degrees: np.ndarray, points: int, min_nodes: int = 5) -> np.ndarray:
    """Log-spaced degree thresholds with at least ``min_nodes`` nodes above them."""
    if len(degrees) == 0 or degrees.max() < 2:
        return np.array([], dtype=int)
    grid = np.unique(np.round(np.logspace(0, math.log10(degrees.max()), points)).astype(int))
    above = np.array([(degrees > k).sum() for k in grid])
    return grid[above >= min_nodes]


def rich_club_curve(graph: ig.Graph, ks: np.ndarray) -> np.ndarray:
    """Unweighted rich-club coefficient phi(k) = 2 E_k / (N_k (N_k - 1)) for degree > k."""
    if len(ks) == 0:
        return np.array([])
    degree = np.asarray(graph.degree())
    edges = np.asarray(graph.get_edgelist())
    edge_min = np.sort(np.minimum(degree[edges[:, 0]], degree[edges[:, 1]]))
    sorted_degree = np.sort(degree)
    n_k = len(degree) - np.searchsorted(sorted_degree, ks, side="right")
    e_k = len(edge_min) - np.searchsorted(edge_min, ks, side="right")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_k > 1, 2 * e_k / (n_k * (n_k - 1.0)), np.nan)
