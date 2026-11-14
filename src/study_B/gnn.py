"""GraphSAGE / GAT link predictor and its chronological training loop.

Encoder: at every layer a node aggregates messages along BOTH relations of the history graph
    sent_to       u -> v   (edge_index)          v receives from the addresses that paid it
    received_from v -> u   (edge_index.flip(0))  u receives from the addresses it paid
with separate weights per relation, summed:  h' = act(conv_sent(h, E) + conv_recv(h, E.flip(0))).
Decoder: MLP on [z_u || z_v] -> logit. The concatenation is ordered, so score(u,v) != score(v,u).

"""

from __future__ import annotations

import copy
import logging
from math import inf

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch_geometric.nn import GATConv, SAGEConv
from .config import GATConfig, NeuralConfig
import torch.nn.functional as F

logger = logging.getLogger("study_B")

class BiRelationalEncoder(nn.Module):
    """Stack of layers, each = conv on sent_to + conv on received_from (separate weights)."""

    def __init__(self, in_dim: int, cfg: NeuralConfig, kind: str, gat: GATConfig | None = None):
        super().__init__()
        self.sent = nn.ModuleList()
        self.recv = nn.ModuleList()
        d_in = in_dim
        for _ in range(cfg.num_layers):
            if kind == "gat":
                assert cfg.hidden_dim % gat.heads == 0
                self.sent.append(GATConv(d_in,cfg.hidden_dim // gat.heads, heads=gat.heads))
                self.recv.append(GATConv(d_in,cfg.hidden_dim // gat.heads, heads=gat.heads))
            elif kind == "graphsage":
                self.sent.append(SAGEConv(d_in,cfg.hidden_dim))
                self.recv.append(SAGEConv(d_in,cfg.hidden_dim))
            else:
                raise ValueError(kind)
            d_in = cfg.hidden_dim
        self.dropout = cfg.dropout


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (conv_s, conv_r) in enumerate(zip(self.sent,self.recv)):
            h = conv_s(h,edge_index) + conv_r(h,edge_index.flip(0))
            if i< len(self.sent)-1:
                h = F.relu(h)
                h = F.dropout(h, p= self.dropout, training = self.training)
        return h

class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, cfg: NeuralConfig, kind: str, gat: GATConfig | None = None):
        super().__init__()
        self.encoder = BiRelationalEncoder(in_dim, cfg, kind, gat)
        self.decoder = nn.Sequential(
            nn.Linear(2* cfg.hidden_dim,cfg.hidden_dim),
            nn.ReLU(), nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.hidden_dim,1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Logits of the ordered pairs (src[i], dst[i])."""
        z= self.encoder(x, edge_index)
        return self.decoder(torch.cat([z[src], z[dst]], dim=1)).squeeze(-1)


def train_model(model: LinkPredictor, train_windows: list[dict], val_windows: list[dict], cfg: NeuralConfig,
                n_neg_train: int, seed: int, neg_pools: tuple[str, ...] = ("history_nodes",)) -> tuple[LinkPredictor, pd.DataFrame]:
    """Train with early stopping on mean validation AP; return (best model, training curve).

    Every element of ``*_windows`` is a dict already on the device, built in pipeline.py:
        x, edge_index, src, dst, label (val: fixed eval pairs), history_nodes, recent_receivers, positives (train)
    ``neg_pools``: window keys training negatives are drawn from (history_nodes, recent_receivers);
    with several pools each negative picks one of them with equal probability.
    Training curve columns: epoch, train_loss, val_ap.
    """
    torch.manual_seed(seed)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = BCEWithLogitsLoss()

    best_ap, best_state, waited, rows = -inf, None, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        for w in train_windows:
            # Fresh negatives every epoch: same source as each positive, random destination from neg_pools.
            # A rare false negative (a pair already linked) is tolerated here; eval negatives are exact.
            pos_src, pos_dst = w["positives"][:, 0], w["positives"][:, 1]
            neg_src = pos_src.repeat_interleave(n_neg_train)
            if len(neg_pools) == 1:   # no extra RNG draw: keeps single-pool runs identical to earlier ones
                pool = w[neg_pools[0]]
                neg_dst = pool[torch.randint(len(pool), (len(neg_src),), device=neg_src.device)]
            else:
                which = torch.randint(len(neg_pools), (len(neg_src),), device=neg_src.device)
                neg_dst = torch.empty_like(neg_src)
                for i, key in enumerate(neg_pools):
                    mask = which == i
                    pool = w[key]
                    neg_dst[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), device=neg_src.device)]
            src, dst = torch.cat([pos_src, neg_src]), torch.cat([pos_dst, neg_dst])
            labels = torch.cat([torch.ones(len(pos_src)), torch.zeros(len(neg_src))]).to(src.device)

            opt.zero_grad()
            loss = criterion(model(w["x"], w["edge_index"], src, dst), labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_ap = float(np.mean([average_precision_score(w["label"].cpu().numpy(), predict(model, w))
                                for w in val_windows]))
        rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_ap": val_ap})
        if epoch % 10 == 0:
            logger.info("epoch %d  train_loss %.4f  val_ap %.4f", epoch, rows[-1]["train_loss"], val_ap)

        if val_ap > best_ap:
            best_ap, best_state, waited = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)

@torch.no_grad()
def predict(model: LinkPredictor, window: dict) -> np.ndarray:
    """Sigmoid scores of the window's fixed evaluation pairs."""
    model.eval()
    return torch.sigmoid(model(window['x'], window['edge_index'], window['src'], window['dst'])).cpu().numpy()