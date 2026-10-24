"""Community detection on the weighted undirected projection.

Leiden (``leidenalg``, RB-configuration / modularity quality) is the primary
algorithm; Louvain (igraph multilevel) is a comparison, with agreement measured
by the adjusted Rand index (ARI) and normalised mutual information (NMI). Both
optimise weighted modularity with the configured resolution and a fixed seed.
Community IDs are canonical: 0 is the largest community, ties by lowest node id.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.special import comb


def canonical(membership: np.ndarray) -> np.ndarray:
    membership = np.asarray(membership)
    labels, inverse = np.unique(membership, return_inverse=True)
    sizes = np.bincount(inverse)
    first = np.full(len(labels), len(membership))
    np.minimum.at(first, inverse, np.arange(len(membership)))
    order = np.lexsort((first, -sizes))
    remap = np.empty(len(labels), dtype=int)
    remap[order] = np.arange(len(labels))
    return remap[inverse]


def leiden(graph: ig.Graph, *, resolution: float, seed: int) -> np.ndarray:
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=resolution, seed=seed, n_iterations=-1)
    return canonical(np.asarray(partition.membership))


def louvain(graph: ig.Graph, *, resolution: float, seed: int) -> np.ndarray:
    ig.set_random_number_generator(random.Random(seed))
    try:
        clustering = graph.community_multilevel(weights="weight", resolution=resolution)
    finally:
        ig.set_random_number_generator(random)
    return canonical(np.asarray(clustering.membership))


def modularity(graph: ig.Graph, membership: np.ndarray, resolution: float) -> float:
    if graph.ecount() == 0:
        return math.nan
    return float(graph.modularity(membership.tolist(), weights="weight", resolution=resolution,
                                  directed=False))


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    return table


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n < 2:
        return math.nan
    table = _contingency(a, b)
    index = comb(table, 2).sum()
    rows, cols = comb(table.sum(1), 2).sum(), comb(table.sum(0), 2).sum()
    expected = rows * cols / comb(n, 2)
    maximum = 0.5 * (rows + cols)
    return float((index - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """NMI with arithmetic-mean normalisation."""
    table = _contingency(a, b).astype(float)
    n = table.sum()
    if n == 0:
        return math.nan
    pij = table / n
    pi, pj = pij.sum(1), pij.sum(0)
    nz = pij > 0
    mi = float((pij[nz] * np.log(pij[nz] / np.outer(pi, pj)[nz])).sum())
    hi = -float((pi[pi > 0] * np.log(pi[pi > 0])).sum())
    hj = -float((pj[pj > 0] * np.log(pj[pj > 0])).sum())
    denominator = 0.5 * (hi + hj)
    return mi / denominator if denominator > 0 else 1.0


def partition_summary(graph: ig.Graph, membership: np.ndarray, resolution: float) -> dict[str, Any]:
    sizes = np.bincount(membership) if len(membership) else np.array([], dtype=int)
    n = len(membership)
    ordered = np.sort(sizes)[::-1]
    return {
        "communities": int(len(sizes)),
        "modularity": modularity(graph, membership, resolution),
        "largest_community_share": float(ordered[0] / n) if n else math.nan,
        "top10_coverage": float(ordered[:10].sum() / n) if n else math.nan,
        "singleton_communities": int(np.sum(sizes == 1)),
        "communities_ge10": int(np.sum(sizes >= 10)),
        "median_size": float(np.median(sizes)) if len(sizes) else math.nan,
    }


def seed_stability(graph: ig.Graph, *, resolution: float, seeds: list[int]) -> pd.DataFrame:
    """Leiden partitions across seeds: modularity per seed and pairwise ARI/NMI."""
    partitions = {s: leiden(graph, resolution=resolution, seed=s) for s in seeds}
    rows = [{"seed_a": a, "seed_b": b,
             "ari": adjusted_rand(partitions[a], partitions[b]),
             "nmi": normalized_mutual_info(partitions[a], partitions[b]),
             "modularity_a": modularity(graph, partitions[a], resolution),
             "modularity_b": modularity(graph, partitions[b], resolution)}
            for a, b in itertools.combinations(seeds, 2)]
    return pd.DataFrame(rows)


def resolution_sweep(graph: ig.Graph, *, resolutions: list[float], seed: int) -> pd.DataFrame:
    rows = []
    reference = leiden(graph, resolution=1.0, seed=seed)
    for gamma in resolutions:
        membership = leiden(graph, resolution=gamma, seed=seed)
        summary = partition_summary(graph, membership, gamma)
        rows.append({"resolution": gamma, **summary,
                     "modularity_at_gamma_1": modularity(graph, membership, 1.0),
                     "ari_vs_gamma_1": adjusted_rand(membership, reference)})
    return pd.DataFrame(rows)


def persistence(previous: pd.DataFrame, current: pd.DataFrame, *, min_size: int) -> pd.DataFrame:
    """Best-match Jaccard of each community (>= min_size) in the previous window.

    ``jaccard`` uses all addresses; ``jaccard_shared`` restricts both communities to
    addresses active in both windows, separating community change from address turnover.
    """
    shared_nodes = set(previous["node"]) & set(current["node"])
    prev_groups = {c: set(g) for c, g in previous.groupby("community")["node"]}
    curr_groups = {c: set(g) for c, g in current.groupby("community")["node"]}
    node_to_current = dict(zip(current["node"], current["community"]))
    rows = []
    for community, members in prev_groups.items():
        if len(members) < min_size:
            continue
        candidates = {node_to_current[v] for v in members if v in node_to_current}
        best = (None, 0.0, 0.0)
        for other in candidates:
            target = curr_groups[other]
            jaccard = len(members & target) / len(members | target)
            ps, cs = members & shared_nodes, target & shared_nodes
            jaccard_shared = len(ps & cs) / len(ps | cs) if ps | cs else 0.0
            if jaccard > best[1]:
                best = (other, jaccard, jaccard_shared)
        rows.append({"community": community, "size": len(members),
                     "retained_addresses_share": len(members & shared_nodes) / len(members),
                     "best_match": best[0], "jaccard": best[1], "jaccard_shared": best[2]})
    return pd.DataFrame(rows, columns=["community", "size", "retained_addresses_share",
                                       "best_match", "jaccard", "jaccard_shared"])


def quotient(graph: ig.Graph, membership: np.ndarray) -> pd.DataFrame:
    """Total projected weight between every pair of communities (intra-community on the diagonal)."""
    edges = np.asarray(graph.get_edgelist())
    if len(edges) == 0:
        return pd.DataFrame(columns=["community_a", "community_b", "weight", "edges"])
    a, b = membership[edges[:, 0]], membership[edges[:, 1]]
    low, high = np.minimum(a, b), np.maximum(a, b)
    frame = pd.DataFrame({"community_a": low, "community_b": high,
                          "weight": np.asarray(graph.es["weight"], dtype=float)})
    return (frame.groupby(["community_a", "community_b"])
            .agg(weight=("weight", "sum"), edges=("weight", "size")).reset_index())
