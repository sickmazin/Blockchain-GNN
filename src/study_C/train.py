"""
Training loops: sklearn Random Forest and full-batch torch models on the GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data

from . import evaluate
from .config import NeuralConfig


@dataclass
class TrainResult:
    """What a training call hands back to the pipeline."""

    model: object
    best_epoch: int | None = None
    history: pd.DataFrame = field(default_factory=pd.DataFrame)   # epoch, train_loss, val_pr_auc

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(device: str) -> torch.device:
    """"auto" -> cuda if available else cpu; otherwise the requested device."""
    match device:
        case "cuda":
            if not torch.cuda.is_available():
                raise ValueError('run.device = "cuda" but CUDA is not available')
            return torch.device("cuda")
        case "cpu":
            return torch.device("cpu")
        case "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case _:
            raise ValueError(f"unknown device {device!r} (expected auto, cuda or cpu)")


def train_random_forest(model, x_train: np.ndarray, y_train: np.ndarray) -> TrainResult:
    """Fit on labelled training rows only."""
    # x_train/y_train must contain ONLY train nodes labelled 0/1 (no UNKNOWN).
    model.fit(x_train,y_train)
    return  TrainResult(model=model)


def train_torch(model: nn.Module, train_graph: Data, val_graph: Data, cfg: NeuralConfig,
                pos_weight: float, device: torch.device) -> TrainResult:
    """Full-batch training with class-weighted BCE and early stopping on validation PR-AUC."""

    # Data.to() moves the graph in place: the caller's graphs stay on the device afterwards.
    model = model.to(device)
    train_graph = train_graph.to(device)
    val_graph = val_graph.to(device)

    # Forward on ALL visible nodes (UNKNOWN included, as context); loss only on labelled train nodes.
    mask = train_graph.eval_mask
    target = train_graph.y[mask].float()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_pr_auc, best_epoch, best_state, waited = -np.inf, None, None, 0
    history = []
    for epoch in range(cfg.max_epochs):
        model.train()   # every epoch: pr_auc_on_graph leaves the model in eval()
        optimizer.zero_grad()
        logits = model(train_graph.x, train_graph.edge_index)
        loss = loss_fn(logits[mask], target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()

        val_pr_auc = evaluate.pr_auc_on_graph(model, val_graph)
        history.append((epoch, loss.item(), val_pr_auc))
        if val_pr_auc > best_pr_auc:
            # clone: state_dict() returns live references that the optimizer keeps changing
            best_pr_auc, best_epoch, waited = val_pr_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, best_epoch=best_epoch,
                       history=pd.DataFrame(history, columns=["epoch", "train_loss", "val_pr_auc"]))
