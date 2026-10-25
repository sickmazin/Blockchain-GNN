"""Feature sets, train-only standardisation and train-only class weights.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

from .data import ILLICIT, LICIT, EllipticData


def select_feature_columns(data: EllipticData, feature_set: str) -> list[str]:
    """Return the column names for "local" or "all" (never ``time_step``)."""
    #   "local" -> data.local_columns
    #   "all"   -> data.local_columns + data.aggregated_columns
    #   altro   -> ValueError
    match feature_set:
        case "local":
            return data.local_columns
        case "all":
            return data.local_columns + data.aggregated_columns
        case _:
            raise ValueError("Feature set errato")


def fit_scaler(nodes: pd.DataFrame, columns: list[str], train_mask: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training-phase rows only."""
    scaler = StandardScaler()
    scaler.fit(nodes.loc[train_mask,columns].to_numpy())
    return scaler


def build_matrix(nodes: pd.DataFrame, columns: list[str], scaler: StandardScaler | None) -> np.ndarray:
    """Feature matrix for all nodes (float32), transformed with an already-fitted scaler."""
    # ``scaler.transform``. Return float32: it is the type torch will use on the GPU.
    # The Random Forest gets scaler=None (trees are scale-invariant).
    data = nodes[columns].to_numpy(np.float32)
    if scaler is not None:
        return scaler.transform(data)
    return data


def class_weights(train_labels: np.ndarray) -> dict[int, float]:
    """Balanced weights from TRAINING labels only: w_c = n_total / (n_classes * n_c)."""
    classes = np.array([LICIT, ILLICIT])
    found = sorted(np.unique(train_labels).tolist())
    if found != classes.tolist():
        raise ValueError(f"train labels must be exactly {classes.tolist()} (UNKNOWN excluded), found {found}")
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}