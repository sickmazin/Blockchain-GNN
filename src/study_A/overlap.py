"""Cross-token overlap of addresses and communities between the selected tokens.

Comparisons are always time-aligned:

* **monthly**: for every calendar month observed in both tokens, the address sets of
  the two monthly graphs and their Leiden partitions (transfer-count projection,
  already computed per token) restricted to the shared addresses;
* **common period**: the graphs of both tokens rebuilt on the intersection of their
  observed date ranges, with the same filters, Leiden partition and PageRank.

Partition agreement on shared addresses is measured by ARI/NMI and compared with a
permutation null that shuffles one token's community labels among the shared
addresses (community sizes preserved). Shared addresses are on-chain identifiers
active in both token networks; they are not inferred to be the same real-world
actor, and their role (e.g. a contract) is not identified.
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .communities import adjusted_rand, leiden, normalized_mutual_info
from .config import StudyAConfig
from .graphs import aggregate_pairs, build_bundle, filter_pairs, variant_edges
from .data import load_token
from .windows import FULL_PERIOD

logger = logging.getLogger("study_A")


def _read(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def partition_agreement(a: pd.DataFrame, b: pd.DataFrame, *, permutations: int, seed: int,
                        min_shared: int) -> dict[str, Any]:
    """ARI/NMI of two memberships on their shared nodes, with a label-permutation null."""
    joined = a[["node", "community"]].merge(b[["node", "community"]], on="node", suffixes=("_a", "_b"))
    result: dict[str, Any] = {"shared_in_partitions": int(len(joined))}
    if len(joined) < min_shared:
        result["agreement_note"] = f"fewer than {min_shared} shared addresses"
        return result
    x, y = joined["community_a"].to_numpy(), joined["community_b"].to_numpy()
    rng = np.random.default_rng(seed)
    null = np.array([adjusted_rand(x, rng.permutation(y)) for _ in range(permutations)])
    observed = adjusted_rand(x, y)
    result.update({
        "communities_a_on_shared": int(len(np.unique(x))),
        "communities_b_on_shared": int(len(np.unique(y))),
        "ari": observed, "nmi": normalized_mutual_info(x, y),
        "ari_null_mean": float(null.mean()), "ari_null_q95": float(np.quantile(null, 0.95)),
        "ari_permutation_p": float((1 + np.sum(null >= observed)) / (permutations + 1)),
        "permutations": permutations,
    })
    return result


def _address_sets(pairs: pd.DataFrame) -> set[str]:
    return set(pairs["source"]) | set(pairs["destination"])


def _set_overlap(a: set[str], b: set[str]) -> dict[str, Any]:
    shared = a & b
    return {"addresses_a": len(a), "addresses_b": len(b), "shared_addresses": len(shared),
            "shared_share_of_a": len(shared) / len(a) if a else math.nan,
            "shared_share_of_b": len(shared) / len(b) if b else math.nan,
            "address_jaccard": len(shared) / len(a | b) if a | b else math.nan}


def _common_period_graph(token: str, start: pd.Timestamp, end: pd.Timestamp, config: StudyAConfig
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filtered pairs, Leiden membership and PageRank of one token on [start, end]."""
    events = load_token(token, config.paths.data_dir, config.paths.cache_dir).events
    subset = events.loc[events["date"].between(start, end)]
    pairs, _ = filter_pairs(aggregate_pairs(subset), exclude_self_loops=config.filters.exclude_self_loops,
                            exclude_zero_address=config.filters.exclude_zero_address)
    edges, _ = variant_edges(pairs, "frequency")
    bundle = build_bundle(edges, "frequency")
    membership = pd.DataFrame({"node": bundle.undirected.vs["name"],
                               "community": leiden(bundle.undirected, resolution=config.communities.resolution,
                                                   seed=config.seed)})
    pagerank = pd.DataFrame({"node": bundle.directed.vs["name"],
                             "pagerank": bundle.directed.pagerank(directed=True,
                                                                  damping=config.centrality.pagerank_damping,
                                                                  weights="weight")})
    pagerank["pagerank_percentile"] = pagerank["pagerank"].rank(pct=True, method="average")
    return pairs, membership, pagerank


def cross_token_overlap(run_dir: Path, tokens: list[str], config: StudyAConfig) -> None:
    oc = config.overlap
    out = run_dir / "cross_token"
    out.mkdir(parents=True, exist_ok=True)
    monthly_rows, period_rows, placement_rows, matrix_rows = [], [], [], []
    for a, b in itertools.combinations(tokens, 2):
        pair = f"{a}–{b}"
        logger.info("Cross-token overlap %s", pair)
        windows_a = _read(run_dir / a / "window_manifest.csv")
        windows_b = _read(run_dir / b / "window_manifest.csv")
        common = windows_a.merge(windows_b[["window", "partial"]], on="window", suffixes=("_a", "_b"))
        common = common.loc[common["window"] != FULL_PERIOD]
        for row in common.itertuples():
            pa = _read(run_dir / a / "graphs" / f"{row.window}_pairs.csv.gz", usecols=["source", "destination"])
            pb = _read(run_dir / b / "graphs" / f"{row.window}_pairs.csv.gz", usecols=["source", "destination"])
            if pa.empty or pb.empty:
                continue
            ma = _read(run_dir / a / "communities" / f"{row.window}__frequency__leiden.csv.gz")
            mb = _read(run_dir / b / "communities" / f"{row.window}__frequency__leiden.csv.gz")
            monthly_rows.append({"pair": pair, "token_a": a, "token_b": b, "window": row.window,
                                 "partial": bool(row.partial_a or row.partial_b),
                                 **_set_overlap(_address_sets(pa), _address_sets(pb)),
                                 **partition_agreement(ma, mb, permutations=oc.permutations, seed=config.seed,
                                                       min_shared=oc.min_shared)})

        dates_a = pd.to_datetime(_read(run_dir / a / "daily_activity.csv")["date"])
        dates_b = pd.to_datetime(_read(run_dir / b / "daily_activity.csv")["date"])
        start, end = max(dates_a.min(), dates_b.min()), min(dates_a.max(), dates_b.max())
        if start > end:
            period_rows.append({"pair": pair, "token_a": a, "token_b": b, "note": "no common dates"})
            continue
        pairs_a, members_a, rank_a = _common_period_graph(a, start, end, config)
        pairs_b, members_b, rank_b = _common_period_graph(b, start, end, config)
        set_a, set_b = _address_sets(pairs_a), _address_sets(pairs_b)
        shared = set_a & set_b
        row: dict[str, Any] = {"pair": pair, "token_a": a, "token_b": b, "start": start.date(),
                               "end": end.date(), "days": (end - start).days + 1, **_set_overlap(set_a, set_b),
                               **partition_agreement(members_a, members_b, permutations=oc.permutations,
                                                     seed=config.seed, min_shared=oc.min_shared)}
        for side, token, pairs, ranks in (("a", a, pairs_a, rank_a), ("b", b, pairs_b, rank_b)):
            touches = pairs["source"].isin(shared) | pairs["destination"].isin(shared)
            row[f"transfers_touching_shared_share_{side}"] = (
                float(pairs.loc[touches, "event_count"].sum() / pairs["event_count"].sum()))
            ranked = ranks.assign(shared=ranks["node"].isin(shared))
            row[f"median_pagerank_percentile_shared_{side}"] = float(
                ranked.loc[ranked["shared"], "pagerank_percentile"].median()) if shared else math.nan
            placement_rows.extend(ranked[["node", "pagerank_percentile", "shared"]]
                                  .assign(pair=pair, token=token, side=side).to_dict("records"))
        period_rows.append(row)

        # Community x community counts of shared addresses (the readable form of "community overlap").
        joined = (members_a.merge(members_b, on="node", suffixes=("_a", "_b")))
        if len(joined):
            top_a = joined["community_a"].value_counts().head(oc.matrix_communities).index
            top_b = joined["community_b"].value_counts().head(oc.matrix_communities).index
            counts = (joined.assign(community_a=joined["community_a"].where(joined["community_a"].isin(top_a), -1),
                                    community_b=joined["community_b"].where(joined["community_b"].isin(top_b), -1))
                      .groupby(["community_a", "community_b"]).size().rename("shared_addresses").reset_index())
            sizes_a = members_a["community"].value_counts()
            sizes_b = members_b["community"].value_counts()
            counts["size_a"] = counts["community_a"].map(sizes_a)
            counts["size_b"] = counts["community_b"].map(sizes_b)
            matrix_rows.extend(counts.assign(pair=pair, token_a=a, token_b=b).to_dict("records"))

    pd.DataFrame(monthly_rows).to_csv(out / "overlap_monthly.csv", index=False)
    pd.DataFrame(period_rows).to_csv(out / "overlap_common_period.csv", index=False)
    pd.DataFrame(placement_rows).to_csv(out / "shared_address_placement.csv.gz", index=False)
    pd.DataFrame(matrix_rows).to_csv(out / "community_overlap_matrix.csv", index=False)
