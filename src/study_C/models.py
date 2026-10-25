"""Model definitions: Random Forest (sklearn), MLP, GraphSAGE and GAT (PyTorch / PyG).

All torch models share one interface so that ``train.py`` can treat them the same way:

    logits = model(x, edge_index)     # shape [num_nodes], one raw score per node

The MLP simply ignores ``edge_index``. Raw scores (logits) become probabilities with
``torch.sigmoid``; the loss takes logits directly (``BCEWithLogitsLoss``) because it is
numerically more stable.

why a single output and not two classes?
    Binary classification: one number per node is enough, the "propensity to be illicit".
    sigmoid(logit) = probability of illicit. PR-AUC and the optimal threshold are computed
    on this probability.

what the GNN layers actually do
    SAGEConv: new_h_i = W1 · h_i + W2 · mean(h_j for j neighbour of i)
              -> combines the node with the mean of its neighbours. With 2 layers it sees neighbours of neighbours.
    GATConv:  as above, but the "mean" is weighted by learned attention coefficients:
              the model learns which neighbours matter most. With several heads it learns several
              weightings in parallel and concatenates them.
    MLP:      h_i only, no neighbours. It is the direct comparison: same network, no graph.
"""

from __future__ import annotations

import torch
import torch_geometric
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv

from .config import GATConfig, NeuralConfig, RandomForestConfig


def make_random_forest(cfg: RandomForestConfig, seed: int) -> RandomForestClassifier:
    """Configured but untrained RandomForestClassifier."""
    return RandomForestClassifier(n_estimators=cfg.n_estimators, max_features=cfg.max_features, class_weight=cfg.class_weight,n_jobs=cfg.n_jobs, random_state=seed)


class MLP(nn.Module):
    """Feature-only baseline: Linear -> ReLU -> Dropout, repeated, then a 1-unit head."""

    def __init__(self, in_dim: int, cfg: NeuralConfig) -> None:
        super().__init__()

        layers = []
        dim = in_dim
        for _ in range(cfg.num_layers):
            layers += [nn.Linear(dim,cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout)]
            dim = cfg.hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden_dim, 1)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.mlp(x)).squeeze(-1)


class GraphSAGE(nn.Module):
    """Primary GNN: stacked SAGEConv layers with mean aggregation, then a 1-unit head."""

    def __init__(self, in_dim: int, cfg: NeuralConfig) -> None:
        super().__init__()

        layers = []
        dim = in_dim
        for _ in range(cfg.num_layers):
            layers+= [
                (SAGEConv(dim,cfg.hidden_dim), 'x, edge_index -> x' ),
                nn.ReLU(),
                nn.Dropout(cfg.dropout)
            ]
            dim = cfg.hidden_dim

        self.sage = torch_geometric.nn.Sequential('x, edge_index', layers)
        self.head = nn.Linear(cfg.hidden_dim,1)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.sage(x,edge_index)).squeeze(-1)


class GAT(nn.Module):
    """Attention GNN: GATConv layers with ``heads`` attention heads (concatenated)."""

    def __init__(self, in_dim: int, cfg: NeuralConfig, gat_cfg: GATConfig) -> None:
        super().__init__()

        activation = nn.ELU if gat_cfg.ELU_activation else nn.ReLU

        layers = []
        dim = in_dim
        for _ in range(cfg.num_layers):
            layers += [
                (GATConv(dim, cfg.hidden_dim // gat_cfg.heads, heads=gat_cfg.heads, dropout=cfg.dropout if gat_cfg.attention_dropout else 0), 'x, edge_index -> x'),
                activation(),
                nn.Dropout(cfg.dropout)
            ]
            dim = cfg.hidden_dim

        self.gat = torch_geometric.nn.Sequential('x, edge_index',layers)
        self.head = nn.Linear(cfg.hidden_dim,1)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.gat(x,edge_index)).squeeze(-1)



def make_torch_model(name: str, in_dim: int, neural: NeuralConfig, gat: GATConfig) -> nn.Module:
    """Factory: "mlp" | "graphsage" | "gat" -> untrained module."""
    match name:
        case "mlp":
            return MLP(in_dim,neural)
        case "graphsage":
            return GraphSAGE(in_dim,neural)
        case "gat":
            return GAT(in_dim,neural,gat)
        case _:
            raise ValueError(f"Torch model not implemented for : {name}")
