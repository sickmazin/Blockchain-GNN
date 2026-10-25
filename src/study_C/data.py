"""Load the three Elliptic CSV files and validate them before any modelling.

Input (read-only, ``datasets/elliptic_bitcoin_dataset/``):
    elliptic_txs_features.csv  NO header; 167 columns = txId, time_step, 165 features
    elliptic_txs_classes.csv   header ``txId,class``; values to be verified ("1", "2", "unknown")
    elliptic_txs_edgelist.csv  header ``txId1,txId2``; directed edge txId1 -> txId2

Output: an ``EllipticData`` object (three clean DataFrames) and a profile dict
that the pipeline saves as ``data/data_profile.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"
# txId + time_step + 165 features (verified on the local file).
N_FEATURE_FILE_COLUMNS = 167

# Internal label encoding used by ALL other modules:
#   1 = illicit (positive class, the one we want to find)
#   0 = licit
#  -1 = unknown (never used in the loss or in the metrics)
# The mapping from the CSV ("1"/"2"/"unknown") to these values must be CONFIRMED with the counts.
ILLICIT, LICIT, UNKNOWN = 1, 0, -1
LABEL_CODES = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}   # CSV value -> internal code
LABEL_NAMES = {"illicit": ILLICIT, "licit": LICIT, "unknown": UNKNOWN}


@dataclass
class EllipticData:
    """The three raw tables after column naming and label encoding (no rows dropped)."""

    nodes: pd.DataFrame     # one row per transaction: txId, time_step, label, features...
    edges: pd.DataFrame     # columns: src, dst (original txIds, not yet indices)
    feature_columns: list[str]   # names of the 165 feature columns, in file order
    local_columns: list[str]     # sottoinsieme "local"
    aggregated_columns: list[str]  # "aggregated over neighbours" subset


def feature_column_names(n_features: int, n_local: int) -> tuple[list[str],list[str]]:
    """Name the header-less feature columns: ``local_001..`` then ``agg_001..``."""
    if n_local > n_features:
        raise ValueError("Local features greater than number of features")

    local_f = [f"local_{i:03d}" for i in range(1,n_local+1)]
    aggr_f = [f"aggr_{i:03d}" for i in range(1,n_features-n_local+1)]
    return local_f, aggr_f


def load_elliptic(data_dir: Path, n_local: int) -> EllipticData:
    """Read the three files, name the columns, encode labels and join them on txId."""

    #Reading 3 CSV. txId must stay an integer: float32 rounds IDs above 2^24 (230425980 -> 230425984).
    features_df = pd.read_csv(data_dir / FEATURES_FILE, header=None)
    classes_df = pd.read_csv(data_dir / CLASSES_FILE, dtype={"txId": "int64", "class": str})
    edges_df = pd.read_csv(data_dir / EDGES_FILE, dtype="int64")
    if features_df.shape[1] != N_FEATURE_FILE_COLUMNS:
        raise ValueError(f"{FEATURES_FILE}: expected {N_FEATURE_FILE_COLUMNS} columns, "
                         f"found {features_df.shape[1]}")

    #  Rename: col 0 -> "txId", col 1 -> "time_step", others with feature_column_names.
    local_columns, aggr_columns = feature_column_names(features_df.shape[1] - 2, n_local)
    feature_columns = local_columns + aggr_columns
    features_df.columns = ["txId", "time_step"] + feature_columns
    features_df = features_df.astype({"txId": "int64", "time_step": "int64"}
                                     | dict.fromkeys(feature_columns, "float32"))

    # Map the CSV values to the internal codes; an unmapped value becomes NaN and is an error.
    classes_df["label"] = classes_df["class"].map(LABEL_CODES)
    unexpected = classes_df.loc[classes_df["label"].isna(), "class"].unique()
    if len(unexpected):
        raise ValueError(f"{CLASSES_FILE}: unexpected class values {sorted(map(str, unexpected))}")
    classes_df = classes_df[["txId", "label"]].astype({"label": "int8"})

    #Df merge (a node missing from the classes file keeps label NaN: counted in validate_elliptic)
    # .copy() consolidates the one-block-per-column layout left by astype (168 -> 3 blocks):
    # without it, adding a column later (e.g. "phase") raises a PerformanceWarning.
    nodes_df = features_df.merge(classes_df, on="txId", how="left", validate="one_to_one").copy()

    #Rename edge columns
    edges_df = edges_df.rename(columns={"txId1": "src", "txId2": "dst"})

    return EllipticData(nodes_df, edges_df, feature_columns, local_columns, aggr_columns)

def validate_elliptic(data: EllipticData) -> dict[str, Any]:
    """Return a JSON-serialisable profile; raise only on errors that make the data unusable."""
    nodes, edges = data.nodes, data.edges

    # Rows and ID alignment
    n_columns = 2 + len(data.feature_columns)
    if n_columns != N_FEATURE_FILE_COLUMNS:
        raise ValueError(f"expected {N_FEATURE_FILE_COLUMNS} feature-file columns, found {n_columns}")
    n_duplicate_txids = int(nodes["txId"].duplicated().sum())
    if n_duplicate_txids:
        raise ValueError(f"{n_duplicate_txids} duplicated txId in the features file")
    n_nodes_without_class = int(nodes["label"].isna().sum())

    # Labels: illicit must be the minority, otherwise "1" and "2" are swapped in LABEL_CODES
    label_counts = {name: int((nodes["label"] == code).sum()) for name, code in LABEL_NAMES.items()}
    n_labelled = label_counts["illicit"] + label_counts["licit"]
    illicit_share = label_counts["illicit"] / n_labelled if n_labelled else None
    if illicit_share is not None and illicit_share > 0.5:
        raise ValueError(f"illicit share of labelled nodes is {illicit_share:.2%}: class mapping looks inverted")

    # Time
    time_steps = sorted(int(s) for s in nodes["time_step"].unique())
    steps_missing = sorted(set(range(time_steps[0], time_steps[-1] + 1)) - set(time_steps))

    # Features
    feature_values = nodes[data.feature_columns]
    n_missing_feature_values = int(feature_values.isna().sum().sum())
    n_constant_features = int((feature_values.nunique() <= 1).sum())

    # Edges: look up step and label of both endpoints (NaN = endpoint not among the nodes)
    by_tx = nodes.set_index("txId")
    src_step, dst_step = edges["src"].map(by_tx["time_step"]), edges["dst"].map(by_tx["time_step"])
    src_label, dst_label = edges["src"].map(by_tx["label"]), edges["dst"].map(by_tx["label"])
    both_known = src_step.notna() & dst_step.notna()
    n_cross_step_edges = int((both_known & (src_step != dst_step)).sum())

    profile = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_feature_columns": len(data.feature_columns),
        "n_local_columns": len(data.local_columns),
        "n_aggregated_columns": len(data.aggregated_columns),
        "n_duplicate_txids": n_duplicate_txids,
        "n_nodes_without_class": n_nodes_without_class,
        "label_counts": label_counts,
        "n_labelled": int(n_labelled),
        "illicit_share_of_labelled": illicit_share,
        "time_steps": time_steps,
        "steps_missing": steps_missing,
        "n_missing_feature_values": n_missing_feature_values,
        "n_constant_features": n_constant_features,
        "n_self_loops": int((edges["src"] == edges["dst"]).sum()),
        "n_duplicate_edges": int(edges.duplicated().sum()),
        "n_reciprocal_pairs": int(edges.merge(edges, left_on=["src", "dst"], right_on=["dst", "src"]).shape[0] // 2),
        "n_edges_with_unknown_endpoint": int((~both_known).sum()),
        "n_edges_touching_unknown_label": int(((src_label == UNKNOWN) | (dst_label == UNKNOWN)).sum()),
        "n_edges_both_labelled": int((src_label.isin([ILLICIT, LICIT]) & dst_label.isin([ILLICIT, LICIT])).sum()),
        "n_cross_step_edges": n_cross_step_edges,
        "n_edges_backwards_in_time": int((both_known & (src_step > dst_step)).sum()),
    }
    return profile


def label_counts_by_step(nodes: pd.DataFrame) -> pd.DataFrame:
    """Table time_step x {illicit, licit, unknown}: saved as data/label_counts_by_step.csv."""
    df = pd.crosstab(nodes["time_step"], nodes["label"]).rename(columns={
        1: "illicit",
        0: "licit",
        -1: "unknown"
    })
    return df

def _label_homophily(src_label: pd.Series, dst_label: pd.Series) -> dict[str, Any]:
    """Same-class tendency of one set of edges (endpoint labels aligned), labelled-labelled only."""
    labelled = src_label.isin([ILLICIT, LICIT]) & dst_label.isin([ILLICIT, LICIT])
    # undirected view: every labelled edge seen from both endpoints
    own = np.concatenate([src_label[labelled], dst_label[labelled]])
    other = np.concatenate([dst_label[labelled], src_label[labelled]])
    same = own == other
    # expected same-class share if endpoints were paired at random (endpoint label shares kept)
    p_ill = float((own == ILLICIT).mean()) if len(own) else float("nan")
    expected = p_ill ** 2 + (1 - p_ill) ** 2
    observed = float(same.mean()) if len(own) else float("nan")
    return {
        "n_edges": int(len(src_label)),
        "n_edges_both_labelled": int(labelled.sum()),
        "n_edges_with_illicit": int((labelled & ((src_label == ILLICIT) | (dst_label == ILLICIT))).sum()),
        "same_class_share": observed,
        "same_class_share_expected_random": expected,
        # Newman's assortativity on the two classes: 0 = random, 1 = only same-class edges
        "label_assortativity": (observed - expected) / (1 - expected) if expected < 1 else float("nan"),
        "illicit_endpoint_share": p_ill,
        "p_neighbour_illicit_given_illicit": float((other[own == ILLICIT] == ILLICIT).mean())
        if (own == ILLICIT).any() else float("nan"),
        "p_neighbour_illicit_given_licit": float((other[own == LICIT] == ILLICIT).mean())
        if (own == LICIT).any() else float("nan"),
    }


def edge_label_homophily(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Label pairs of edge endpoints and same-class tendency, for all edges and per phase.

    Returns (directed counts table scope x src_label x dst_label, summary per scope). An edge
    belongs to the phase of both its endpoints ("cross" if they differ). The summary uses only
    edges with BOTH endpoints labelled, read as undirected (each edge counted from both ends).
    Statistics per phase let design choices rest on the training phase only.
    """
    by_tx = nodes.set_index("txId")
    names = {ILLICIT: "illicit", LICIT: "licit", UNKNOWN: "unknown"}
    src_label = edges["src"].map(by_tx["label"]).fillna(UNKNOWN).astype(int)
    dst_label = edges["dst"].map(by_tx["label"]).fillna(UNKNOWN).astype(int)
    scopes = {"all": pd.Series(True, index=edges.index)}
    if "phase" in nodes:
        src_phase = edges["src"].map(by_tx["phase"].astype(str))
        dst_phase = edges["dst"].map(by_tx["phase"].astype(str))
        phase = src_phase.where(src_phase == dst_phase, "cross")
        scopes |= {name: phase == name for name in sorted(phase.dropna().unique())}

    tables, summary = [], {}
    for scope, mask in scopes.items():
        s, d = src_label[mask], dst_label[mask]
        counts = pd.crosstab(s.map(names), d.map(names)).stack().rename("n_edges").reset_index()
        tables.append(counts.set_axis(["src_label", "dst_label", "n_edges"], axis=1).assign(scope=scope))
        summary[scope] = _label_homophily(s, d)
    table = pd.concat(tables, ignore_index=True)[["scope", "src_label", "dst_label", "n_edges"]]
    return table, summary


def edge_homophily_by_step(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """One row per time step with the ``_label_homophily`` statistics of the edges inside it.

    An edge belongs to a step only if both endpoints are in it; edges joining two steps (none in
    the verified Elliptic file) or touching an unknown txId are left out and counted in the
    ``n_edges_cross_step`` column of every row, so the exclusion stays visible.
    """
    by_tx = nodes.set_index("txId")
    src_step, dst_step = edges["src"].map(by_tx["time_step"]), edges["dst"].map(by_tx["time_step"])
    inside = src_step.notna() & (src_step == dst_step)
    src_label = edges["src"].map(by_tx["label"]).fillna(UNKNOWN).astype(int)
    dst_label = edges["dst"].map(by_tx["label"]).fillna(UNKNOWN).astype(int)
    steps = src_step[inside].astype(int)
    rows = [{"time_step": step, **_label_homophily(src_label[idx], dst_label[idx])}
            for step, idx in steps.groupby(steps).groups.items()]
    return pd.DataFrame(rows).assign(n_edges_cross_step=int((~inside).sum()))
