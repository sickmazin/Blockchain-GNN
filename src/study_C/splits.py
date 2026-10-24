"""Chronological train/validation/test split on the verified Elliptic time steps.

Output: a ``phase`` column ("train" / "val" / "test") on every node, plus a
manifest (``splits/split_manifest.csv``) and a per-phase summary.

TUTOR — perché NON uno split casuale?
    Con uno split casuale, transazioni dello stesso giorno finiscono sia in train sia in test:
    il modello impara pattern "del momento" e il punteggio sul test è gonfiato. Nella realtà un
    classificatore viene addestrato sul passato e usato sul futuro: lo split cronologico misura
    proprio questo. In Elliptic il comportamento illecito CAMBIA nel tempo (dopo ~step 43 un
    grande dark market chiude), quindi questa scelta cambia davvero i risultati.
    AGENTS.md lo rende obbligatorio: "Random splits are prohibited for Elliptic".
"""

from __future__ import annotations

from math import inf
from typing import Any

import pandas as pd

from .config import SplitConfig
from .data import LABEL_NAMES

PHASES = ("train", "val", "test")


def assign_phases(nodes: pd.DataFrame, split: SplitConfig) -> pd.Series:
    """Map each node's time_step to "train" / "val" / "test" (index aligned with ``nodes``)."""
    #   step <= train_end              -> "train"
    #   train_end < step <= val_end    -> "val"
    #   step > val_end                 -> "test"
    df = pd.cut(nodes["time_step"], bins=[-inf, split.train_end, split.val_end, inf], labels=list(PHASES))
    return df


def split_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    """One row per phase: step range, n_nodes, n_licit, n_illicit, n_unknown, illicit share."""
    # Questa tabella va nel report (sezione "Data and methods") e mostra lo sbilanciamento.
    # observed=False: una fase vuota compare comunque con conteggi 0 invece di sparire.
    grouped = nodes.groupby("phase", observed=False)
    summary = pd.DataFrame({
        "step_min": grouped["time_step"].min(),
        "step_max": grouped["time_step"].max(),
        "n_nodes": grouped.size(),
    })
    for name, code in LABEL_NAMES.items():
        summary[f"n_{name}"] = grouped["label"].apply(lambda s, c=code: int((s == c).sum()))
    n_labelled = summary["n_illicit"] + summary["n_licit"]
    summary["illicit_share"] = (summary["n_illicit"] / n_labelled).where(n_labelled > 0)
    return summary.reindex(list(PHASES)).rename_axis("phase").reset_index()


def check_split(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    """Verify the split is usable and leak-free; raise on violations, return audit numbers."""
    summary = split_summary(nodes)

    # 1. Ogni fase deve avere almeno un'illecita e una lecita, altrimenti PR-AUC/F1 non esistono.
    for row in summary.itertuples():
        if not row.n_illicit or not row.n_licit:
            raise ValueError(
                f"phase {row.phase!r} has {row.n_illicit} illicit and {row.n_licit} licit nodes: both classes are required for evaluation")

    # 2. Gli step delle fasi devono essere disgiunti e in ordine cronologico.
    for earlier, later in zip(summary.itertuples(), summary.iloc[1:].itertuples()):
        if not earlier.step_max < later.step_min:
            raise ValueError(
                f"phases {earlier.phase!r} (max step {earlier.step_max}) and {later.phase!r} (min step {later.step_min}) overlap in time")

    # 3. Archi tra fasi diverse: fase di src e di dst per ogni arco, conteggio per coppia.
    phase_of = nodes.set_index("txId")["phase"].astype(str)
    src_phase = edges["src"].map(phase_of).fillna("missing")
    dst_phase = edges["dst"].map(phase_of).fillna("missing")
    pair_counts = (src_phase + "->" + dst_phase).value_counts().sort_index()
    n_cross = int(pair_counts[[pair for pair in pair_counts.index
                               if pair.split("->")[0] != pair.split("->")[1]]].sum())

    return {
        "phase_steps": {row.phase: [int(row.step_min), int(row.step_max)] for row in summary.itertuples()},
        "edge_phase_pairs": {pair: int(n) for pair, n in pair_counts.items()},
        "n_cross_phase_edges": n_cross,
    }


def split_manifest(nodes: pd.DataFrame) -> pd.DataFrame:
    """Columns txId, time_step, label, phase: saved so the split can be reproduced exactly."""
    return nodes[["txId", "time_step", "label", "phase"]].copy()
