"""Ranking and classification metrics on the saved evaluation pairs.

Input: a predictions table with columns window_index, src, dst, label, group, score
(``group`` = one positive + its n_neg negatives, see samples.py).

    ap        Average Precision = area under the PR curve. PRIMARY metric (assistant.md §6.4).
    roc_auc   ROC-AUC.
    mrr       mean over positives of 1 / rank of the positive inside its group.
    hits@k    share of positives ranked in the top k of their group.

TUTOR — perché AP e non accuracy?
    Con 1 positivo e 50 negativi, "tutto negativo" ha accuracy 98%. AP guarda solo quanto in alto
    finiscono i positivi. Il riferimento casuale di AP è la prevalenza, 1 / (1 + n_neg): va
    riportato accanto ai risultati (lo disegna figures.py).

TUTOR — pareggi
    PA e CN danno molti punteggi uguali (es. CN = 0). Con il rango "ottimista" un modello che dà
    lo stesso punteggio a tutti avrebbe MRR = 1. Usa il rango medio:
        rank = 1 + #(neg > pos) + 0.5 * #(neg == pos)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score




def positive_ranks(pred: pd.DataFrame) -> pd.Series:
    """Rank of the positive inside each group (ties count one half), indexed by group."""
    pos_score = pred.loc[pred["label"] == 1].set_index("group")["score"]
    neg = pred.loc[pred["label"] == 0, ["group", "score"]]
    against = pos_score.reindex(neg["group"]).to_numpy()
    compare = pd.DataFrame({"group": neg["group"].to_numpy(),
                            "above": neg["score"].to_numpy() > against,
                            "tied": neg["score"].to_numpy() == against})
    counts = compare.groupby("group")[["above", "tied"]].sum().reindex(pos_score.index, fill_value=0)
    # rank = 1 + #(neg > pos) + 0.5 * #(neg == pos)
    return (1 + counts["above"] + 0.5 * counts["tied"]).rename("rank")


def metrics(pred: pd.DataFrame, hits_k: list[int]) -> dict[str, float]:
    """ap, roc_auc, mrr, hits@k, n_positives, n_negatives on one predictions table."""
    y, s = pred["label"].to_numpy(), pred["score"].to_numpy()
    ranks = positive_ranks(pred)
    out = {"ap": float(average_precision_score(y, s)),
           "roc_auc": float(roc_auc_score(y, s)),
           "mrr": float((1 / ranks).mean())}
    out |= {f"hits@{k}": float((ranks <= k).mean()) for k in hits_k}
    return out | {"n_positives": int(y.sum()), "n_negatives": int(len(y) - y.sum())}


def per_window(pred: pd.DataFrame, hits_k: list[int]) -> pd.DataFrame:
    """metrics() for every window_index (one row per window)."""
    return pd.DataFrame([{"window_index": w} | metrics(part, hits_k)
                         for w, part in pred.groupby("window_index")])


def by_source_degree(pred: pd.DataFrame, bins: list[float], hits_k: list[int]) -> pd.DataFrame:
    """Error analysis: metrics of the groups whose source falls in each historical out-degree bin.

    ``pred["src_out_degree"]`` = out-degree of the source in the history of the pair's window
    (raw count added in pipeline.py). Output: one row per non-empty bin, plus column ``degree_bin``.
    TUTOR — serve a rispondere a "il modello indovina solo per gli hub?". Se l'AP è alto solo per
    sorgenti con grado alto, il modello sta imparando l'attività, non la struttura.
    """
    # The same group always has the same source, so a bin holds whole groups.
    degree_bin = pd.cut(pred["src_out_degree"], bins, right=False)
    return pd.DataFrame([{"degree_bin": str(b)} | metrics(part, hits_k)
                         for b, part in pred.groupby(degree_bin, observed=True)])
