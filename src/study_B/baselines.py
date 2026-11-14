"""Non-GNN link scorers: destination popularity, preferential attachment, common neighbours, Node2Vec.

Every scorer returns one float score per requested (src, dst) pair, computed from the past of the
target window only, so all models see exactly the same history.

dst_popularity ignores the source on purpose: with candidates drawn among history nodes, "is v
active right now?" already separates many positives from negatives. It is the floor that shows how
much of a model's score is explained by destination activity alone (assistant.md §6.3).
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch_cluster  # noqa: F401  (registers torch.ops.torch_cluster.random_walk)
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec  # noqa: F401  (used in fit_node2vec)

from .config import Node2VecConfig
from .features import FEATURES
from .samples import TargetWindow


def adjacency(target: TargetWindow, n_nodes: int) -> sp.csr_matrix:
    """Directed 0/1 adjacency of the history graph, shape [n_nodes, n_nodes]."""
    return sp.csr_matrix((np.ones(target.history_edges.shape[1]),(target.history_edges[0], target.history_edges[1])), shape=[n_nodes,n_nodes])

def dst_popularity(x: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Transfers received by dst in the last RECENT_WINDOWS windows (log1p feature: same ranking)."""
    return x[dst, FEATURES.index("recent_in")]


def preferential_attachment(target: TargetWindow, src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    """out_degree(src) * in_degree(dst) on the history graph."""
    out_degree = np.bincount(target.history_edges[0], minlength=n_nodes)
    in_degree = np.bincount(target.history_edges[1], minlength=n_nodes)
    return out_degree[src] * in_degree[dst]

def common_neighbours(target: TargetWindow, src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    """Number of shared neighbours on the undirected projection of the history graph."""
    A =  adjacency(target,n_nodes)
    U =  ((A + A.T) > 0)
    return np.asarray(U[src].multiply(U[dst]).sum(axis=1)).ravel()



_CRT = ctypes.cdll.ucrtbase if sys.platform == "win32" else ctypes.CDLL(None)


def _seeded_biased_walk(rowptr, col, start, walk_length, p, q):
    """torch-cluster's CPU biased walk draws from C ``rand()``, not torch's RNG: seed it from torch
    and use one thread (``rand()`` state is per thread), so the walks follow ``torch.manual_seed``."""
    _CRT.srand(int(torch.randint(2**31 - 1, ()).item()))
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)
    finally:
        torch.set_num_threads(n_threads)


def fit_node2vec(target: TargetWindow, n_nodes: int, cfg: Node2VecConfig, device: torch.device,
                 seed: int, p: float = 1.0, q: float = 1.0) -> np.ndarray:
    """Train ``torch_geometric.nn.Node2Vec`` on the history graph; return embeddings [n_nodes, dim]."""
    torch.manual_seed(seed)

    edge_index = torch.as_tensor(np.concat([target.history_edges, target.history_edges[::-1]], axis=1))

    model = Node2Vec(edge_index, embedding_dim=cfg.dim, walk_length=cfg.walk_length,
                     context_size= cfg.context_size, walks_per_node=cfg.walks_per_node, p=p, q=q,
                     num_negative_samples=1, num_nodes=n_nodes, sparse=True).to(device)
    if (p, q) != (1.0, 1.0):   # pyg-lib 0.7.0 only samples uniform walks; torch-cluster does biased ones
        model.random_walk_fn = _seeded_biased_walk

    # Walks start from history nodes only: later nodes are isolated here and would give 1-node walks.
    loader = DataLoader(torch.as_tensor(target.history_nodes), batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=model.sample)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=cfg.lr)

    model.train()
    for _ in range(cfg.epochs):
        for (pos_rw, neg_rw) in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            
    return model.embedding.weight.detach().cpu().numpy()

def node2vec_score(embeddings: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Dot product of the two node embeddings."""
    return (embeddings[src] * embeddings[dst]).sum(axis=1)
