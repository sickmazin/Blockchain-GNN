"""Degree-preserving null models for small-world and rich-club assessment.

Each null graph is the undirected projection rewired by degree-preserving
double-edge swaps (igraph ``rewire``, simple graphs only), seeded per sample.
The same clustering, LCC path-length procedure and rich-club thresholds used for
the observed graph are applied to every null graph. A small-world reading
requires *both* a clustering ratio well above 1 and a path-length ratio near 1;
clustering alone is never sufficient.
"""

from __future__ import annotations

import math
import random
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd

from .topology import _avg_clustering_deg2, path_statistics, rich_club_curve, simple_undirected


def _seeded(seed: int) -> None:
    ig.set_random_number_generator(random.Random(seed))


def null_samples(projection: ig.Graph, *, n_null: int, swaps_per_edge: int, seed: int,
                 exact_max_nodes: int, n_sources: int, rich_ks: np.ndarray
                 ) -> tuple[pd.DataFrame, np.ndarray]:
    base = simple_undirected(projection)
    rows, rich = [], []
    try:
        for index in range(n_null):
            sample_seed = seed + index
            _seeded(sample_seed)
            null = base.copy()
            null.rewire(n=swaps_per_edge * null.ecount(), allowed_edge_types="simple")
            paths = path_statistics(null, exact_max_nodes=exact_max_nodes, n_sources=n_sources,
                                    seed=sample_seed)
            rows.append({
                "null_index": index,
                "null_seed": sample_seed,
                "transitivity_global": null.transitivity_undirected(),
                "avg_clustering_zero": null.transitivity_avglocal_undirected(mode="zero"),
                "avg_clustering_deg2": _avg_clustering_deg2(null),
                "lcc_nodes": paths["lcc_nodes"],
                "average_path_length": paths["average_path_length"],
                "path_method": paths["path_method"],
            })
            rich.append(rich_club_curve(null, rich_ks))
    finally:
        ig.set_random_number_generator(random)
    return pd.DataFrame(rows), np.asarray(rich)


def small_world_summary(observed: dict[str, Any], samples: pd.DataFrame) -> dict[str, Any]:
    """Observed/null ratios, z-scores and Humphries' sigma for two clustering definitions."""
    result: dict[str, Any] = {"null_graphs": int(len(samples))}
    for metric in ("transitivity_global", "avg_clustering_zero", "average_path_length"):
        values = samples[metric].astype(float)
        mean, sd = float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else math.nan
        obs = float(observed[metric])
        result[f"{metric}_observed"] = obs
        result[f"{metric}_null_mean"] = mean
        result[f"{metric}_null_sd"] = sd
        result[f"{metric}_ratio"] = obs / mean if mean > 0 else math.nan
        result[f"{metric}_z"] = (obs - mean) / sd if sd and sd > 0 else math.nan
    path_ratio = result["average_path_length_ratio"]
    for metric in ("transitivity_global", "avg_clustering_zero"):
        result[f"sigma_{metric}"] = (result[f"{metric}_ratio"] / path_ratio
                                     if path_ratio and path_ratio > 0 else math.nan)
    return result
