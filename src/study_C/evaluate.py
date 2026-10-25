"""Predictions, threshold selection on validation, and illicit-class metrics.

why not accuracy? About 90% of the labelled transactions are licit: a model that always says "licit"
has ~90% accuracy and finds no illicit one. The right metrics look at the minority class:
    precision (illicit)  = of the flagged transactions, how many were really illicit
    recall    (illicit)  = of the truly illicit ones, how many were found
    F1                   = harmonic mean of the two
    PR-AUC               = area under the precision-recall curve, threshold-independent
                           (``average_precision_score``); main metric under class
                           imbalance. A random classifier scores ~ the illicit share,
                           NOT 0.5: report it as a reference.
    ROC-AUC              = useful but optimistic with imbalanced classes.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.data import Data


def _frame(tx_ids: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"txId": tx_ids, "y_true": y_true.astype(np.int64), "y_score": y_score.astype(np.float64)})


@torch.no_grad()
def predict_graph(model: torch.nn.Module, graph: Data) -> pd.DataFrame:
    """Probabilities for the ``eval_mask`` nodes of ``graph``: columns txId, y_true, y_score."""
    model.eval()
    logits = model(graph.x, graph.edge_index).view(-1)   # [N] even if the model returns [N, 1]
    prob = torch.sigmoid(logits)
    mask = graph.eval_mask
    return _frame(graph.tx_id[mask].cpu().numpy(),
                  graph.y[mask].cpu().numpy(),
                  prob[mask].cpu().numpy())


def pr_auc_on_graph(model: torch.nn.Module, graph: Data) -> float:
    """Shortcut used by early stopping in ``train.py``."""
    pred = predict_graph(model, graph)
    return float(average_precision_score(pred["y_true"], pred["y_score"]))


def predict_sklearn(model, x: np.ndarray, tx_ids: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Same output format as ``predict_graph`` for the Random Forest."""
    # Column 1 of predict_proba is the illicit class only if classes_ is exactly [0, 1].
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"expected model.classes_ == [0, 1], found {list(model.classes_)}")
    prob = model.predict_proba(x)[:, 1]
    return _frame(np.asarray(tx_ids), np.asarray(y_true), prob)


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1") -> float:
    """Threshold maximising the chosen metric (default illicit F1) on VALIDATION predictions."""
    if metric != "f1":
        raise ValueError(f"unsupported threshold metric {metric!r} (only 'f1')")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)
    # precision/recall have a final point (1, 0) with no threshold: drop it.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return float(thresholds[np.argmax(f1)])


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Illicit precision/recall/F1, PR-AUC, ROC-AUC, supports and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(np.int64)   # same ">=" as precision_recall_curve
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    support_illicit = int((y_true == 1).sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "support_illicit": support_illicit,
        "support_licit": int((y_true == 0).sum()),
        "prevalence": support_illicit / len(y_true),   # PR-AUC of a random classifier
    }


def per_step_metrics(predictions: pd.DataFrame, nodes: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Illicit F1 / precision / recall / support for each test time step."""
    # Steps without illicit transactions (or without alarms): metric undefined -> NaN, not 0.
    merged = predictions.merge(nodes[["txId", "time_step"]], on="txId", how="left", validate="one_to_one")
    if merged["time_step"].isna().any():
        raise ValueError("some predicted txId are not in nodes")
    rows = []
    for step, group in merged.groupby("time_step", sort=True):
        y_true = group["y_true"].to_numpy()
        y_pred = (group["y_score"].to_numpy() >= threshold).astype(np.int64)
        rows.append({
            "time_step": int(step),
            "n": int(len(group)),
            "support_illicit": int((y_true == 1).sum()),
            "n_flagged": int(y_pred.sum()),
            "precision": float(precision_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=np.nan)),
            "recall": float(recall_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=np.nan)),
            "f1": float(f1_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=np.nan)),
        })
    return pd.DataFrame(rows)
