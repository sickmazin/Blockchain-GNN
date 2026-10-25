"""Build the PyTorch Geometric graph visible to each phase.

a ``torch_geometric.data.Data`` object?
    It is a container of tensors:
        x          [num_nodes, num_features]  float32  -> the features (row i = node i)
        edge_index [2, num_edges]             int64    -> row 0 = sources, row 1 = destinations,
                                                         given as INDICES 0..num_nodes-1
        y          [num_nodes]                int64    -> label (1, 0, -1)
    plus any attribute you add (boolean masks, txId, ...).
    Elliptic txIds are huge, non-consecutive numbers: they must be remapped to 0..N-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from .data import ILLICIT, LICIT


@dataclass
class PhaseGraphs:
    """One graph per phase; each carries an ``eval_mask`` for its own labelled nodes."""
    train: Data
    val: Data
    test: Data


def build_phase_graph(nodes: pd.DataFrame, edges: pd.DataFrame, x: np.ndarray, max_step: int,
                      eval_phase: str, undirected: bool = True) -> Data:
    """Graph of all nodes with ``time_step <= max_step`` and the edges among them.

    ``x`` is the full feature matrix aligned row-by-row with ``nodes``.
    ``eval_mask`` marks labelled (label 0/1) nodes whose phase == ``eval_phase``.
    """
    if len(x) != len(nodes):
        raise ValueError(f"x has {len(x)} rows but nodes has {len(nodes)}")

    # Visible nodes: nothing from a step after max_step
    keep = nodes["time_step"].to_numpy() <= max_step
    visible = nodes.loc[keep]

    # txId -> consecutive index 0..N-1
    visible_ids = visible["txId"].to_numpy()
    index = pd.Index(visible_ids)

    # Edges with BOTH endpoints visible: an edge towards a future node would leak information
    edges_kept = edges[edges["src"].isin(visible_ids) & edges["dst"].isin(visible_ids)]
    src = index.get_indexer(edges_kept["src"])
    dst = index.get_indexer(edges_kept["dst"])
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("edge endpoint not found among the visible nodes")
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    if undirected:
        # adds the reverse edges, then sorts and removes duplicates
        edge_index = to_undirected(edge_index, num_nodes=len(visible_ids))

    # Labels: NaN (node without class row) -> -1, never labelled
    label = visible["label"].fillna(-1).astype(np.int64).to_numpy()
    phase = visible["phase"].astype(str).to_numpy()
    eval_mask = (phase == eval_phase) & np.isin(label, [ILLICIT, LICIT])

    # torch.tensor copies: pandas 3 may hand back read-only arrays, which torch.from_numpy rejects
    return Data(
        x=torch.from_numpy(np.ascontiguousarray(x[keep], dtype=np.float32)),
        edge_index=edge_index,
        y=torch.tensor(label, dtype=torch.long),
        eval_mask=torch.tensor(eval_mask, dtype=torch.bool),
        tx_id=torch.tensor(visible_ids, dtype=torch.long),
    )


def build_phase_graphs(nodes: pd.DataFrame, edges: pd.DataFrame, x: np.ndarray,
                       train_end: int, val_end: int, undirected: bool = True) -> PhaseGraphs:
    """The three graphs of the chronological protocol."""
    last_step = int(nodes["time_step"].max())
    return PhaseGraphs(
        train=build_phase_graph(nodes, edges, x, max_step=train_end, eval_phase="train", undirected=undirected),
        val=build_phase_graph(nodes, edges, x, max_step=val_end, eval_phase="val", undirected=undirected),
        test=build_phase_graph(nodes, edges, x, max_step=last_step, eval_phase="test", undirected=undirected),
    )


def visible_degree(nodes: pd.DataFrame, edges: pd.DataFrame, max_step: int) -> pd.Series:
    """Undirected degree (in + out) of each node with ``time_step <= max_step``, counting only edges
    whose endpoints are both visible. Indexed by txId; used as a structure-only baseline score."""
    visible_ids = nodes.loc[nodes["time_step"].to_numpy() <= max_step, "txId"]
    kept = edges[edges["src"].isin(visible_ids) & edges["dst"].isin(visible_ids)]
    counts = pd.concat([kept["src"], kept["dst"]]).value_counts()
    return counts.reindex(visible_ids.to_numpy(), fill_value=0).rename("degree")
