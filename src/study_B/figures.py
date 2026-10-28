"""Report figures, drawn only from the CSV files saved by pipeline.py.

Same visual style as Study C (``apply_style`` is reused), so the report's figures read as one set.
Model colors are assigned in a fixed order and never depend on rank; the palette was checked for
colour-vision deficiency, but three hues are below 3:1 contrast on white, so every figure also
identifies models by text (axis labels or legend) and marker shape, never by colour alone.

    fig1_windows            events per window (a); positives and cold-start pairs per window (b);
                            train/val/test shaded -> justifies the task and the split
    fig2_model_comparison   test AP, MRR, Hits@K: one column per metric, one row per negative set,
                            mean ± std over seeds, dashed random-ranker reference -> the main result
    fig3_pr_curves          test precision-recall curves per negative set, prevalence line
    fig4_gain_over_floor    per test window: GNN minus Dst popularity, AP and MRR -> is the real
                            gain over "who is active" stable in time?
    fig5_by_source_degree   test AP and MRR per source out-degree bin, floor + GNNs -> "only hubs?"
    fig6_training_curves    GNN train loss and validation AP per epoch -> appendix
    fig7_hits_at_k          test Hits@K for every K (rank curve of the positive) -> MRR in full
    fig8_negative_difficulty  weeks since the destination was last active: true destinations vs
                            uniform vs recent negatives -> why the uniform protocol is easy

fig4 and fig5 drop preferential attachment, common neighbours and Node2Vec (near random on the
recent set): they compare the GNNs with the floor only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from src.study_C.figures import DOUBLE, MUTED, SECONDARY, apply_style  # noqa: E402

from .config import GNNS, MODELS, FigureConfig  # noqa: E402
from .evaluate import positive_ranks  # noqa: E402

FLOOR = "dst_popularity"

MODEL_LABELS = {"dst_popularity": "Dst popularity", "preferential_attachment": "Pref. attachment",
                "common_neighbours": "Common neighbours", "node2vec": "Node2Vec (p=q=1)",
                "graphsage": "GraphSAGE", "gat": "GAT"}
# Fixed categorical slots, validated with the dataviz palette script (CVD and normal-vision PASS;
# three hues < 3:1 on white -> models are always also named by text and marker shape).
_SLOTS = ["preferential_attachment", "common_neighbours", "node2vec", "graphsage", "gat", "dst_popularity"]
MODEL_COLORS = dict(zip(_SLOTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MODEL_MARKERS = dict(zip(_SLOTS, ["o", "s", "^", "D", "v", "P"]))
NEG_LABELS = {"uniform": "uniform negatives", "recent": "recent-receiver negatives"}
PHASE_SHADE = {"train": "#f4f7fc", "val": "#fdf0e9", "test": "#eaf7f1"}   # very light, behind the data
RING = "#ffffff"   # surface-coloured edge that separates a marker from the line under it


def _save(fig: plt.Figure, name: str, out: Path, cfg: FigureConfig) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for fmt in cfg.formats:
        fig.savefig(out / f"{name}.{fmt}", dpi=cfg.dpi)
    plt.close(fig)


def _shade_phases(ax: plt.Axes, windows: pd.DataFrame) -> None:
    """Light background band for each phase, using the window start dates."""
    for phase, color in PHASE_SHADE.items():
        rows = windows[windows["phase"] == phase]
        if len(rows):
            ax.axvspan(rows["start_inclusive"].min(), rows["end_exclusive"].max(), color=color, lw=0, zorder=0)


def _label_phases(ax: plt.Axes, windows: pd.DataFrame) -> None:
    """Phase names centred just above their band."""
    for phase in PHASE_SHADE:
        rows = windows[windows["phase"] == phase]
        if len(rows):
            start, end = rows["start_inclusive"].min(), rows["end_exclusive"].max()
            ax.text(start + (end - start) / 2, 1.0, phase, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=SECONDARY)


def random_reference(n_neg: int, hits_k: int = 10) -> dict[str, float]:
    """Expected metrics of a random ranker with 1 positive among n_neg + 1 candidates."""
    n = n_neg + 1
    return {"ap": 1 / n, "mrr": float(np.mean(1 / np.arange(1, n + 1))), f"hits@{hits_k}": min(hits_k, n) / n}


def _models(df: pd.DataFrame) -> list[str]:
    """Models present in ``df``, in the fixed MODELS order (never by rank)."""
    return [m for m in MODELS if m in set(df["model"])]


def _neg_sets(df: pd.DataFrame) -> list[str]:
    return [n for n in NEG_LABELS if n in set(df["negatives"])]


def _hits_col(df: pd.DataFrame) -> str:
    """The Hits@K column with the largest K."""
    return max((c for c in df.columns if c.startswith("hits@")), key=lambda c: int(c.split("@")[1]))


def _bin_label(interval: str) -> str:
    """'[5.0, 20.0)' -> '5–19'; '[100.0, inf)' -> '≥100'; '[0.0, 1.0)' -> '0'."""
    lo, hi = (float(v) for v in interval.strip("[)").split(","))
    if np.isinf(hi):
        return f"≥{lo:.0f}"
    return f"{lo:.0f}" if hi - lo == 1 else f"{lo:.0f}–{hi - 1:.0f}"


def fig1_windows(run_dir: Path) -> plt.Figure:
    """(a) n_events per window; (b) n_positives and n_cold_start per target window. Shared x (dates)."""
    windows = pd.read_csv(run_dir / "splits" / "windows.csv", parse_dates=["start_inclusive", "end_exclusive"])
    targets = pd.read_csv(run_dir / "splits" / "targets.csv").merge(
        windows[["window_index", "start_inclusive"]], on="window_index")
    fig, (a, b) = plt.subplots(2, 1, figsize=(DOUBLE, 3.6), sharex=True)
    for ax in (a, b):
        _shade_phases(ax, windows)
        ax.set_yscale("log")
    _label_phases(a, windows)
    a.plot(windows["start_inclusive"], windows["n_events"].clip(lower=1), color=SECONDARY, lw=1.1)
    a.set_ylabel("transfers per week")
    a.set_title("(a) Activity", pad=12)
    for col, label, color, marker in (("n_positives", "new pairs, both ends seen (positives)", "#2a78d6", "o"),
                                      ("n_cold_start", "new pairs with an unseen end (cold-start)", "#eb6834", "s")):
        b.plot(targets["start_inclusive"], targets[col].clip(lower=1), color=color, marker=marker,
               markersize=2.5, lw=1.1, label=label)
    b.set_ylabel("pairs per week")
    b.set_title("(b) Prediction targets")
    b.legend(loc="lower left")
    return fig


def fig2_model_comparison(metrics: pd.DataFrame, n_neg: int) -> plt.Figure:
    """Dot plot: y = model, x = metric value, error bar = std over seeds; one panel per metric."""
    test = metrics[metrics["phase"] == "test"]
    hits = _hits_col(test)
    k = int(hits.split("@")[1])
    cols, titles = ["ap", "mrr", hits], ["AP", "MRR", f"Hits@{k}"]
    sets, models = _neg_sets(test), _models(test)
    agg = test.groupby(["negatives", "model"])[cols].agg(["mean", "std"]).fillna(0.0)
    ref = random_reference(n_neg, k)
    fig, axes = plt.subplots(len(sets), len(cols), figsize=(DOUBLE, 0.3 * len(models) * len(sets) + 1.0),
                             sharey=True, squeeze=False)
    y = np.arange(len(models))[::-1]
    for r, neg in enumerate(sets):
        for c, (col, title) in enumerate(zip(cols, titles)):
            ax = axes[r, c]
            for yi, m in zip(y, models):
                ax.errorbar(agg.loc[(neg, m), (col, "mean")], yi, xerr=agg.loc[(neg, m), (col, "std")],
                            fmt=MODEL_MARKERS[m], color=MODEL_COLORS[m], ms=4.5, elinewidth=1, capsize=2)
            ax.axvline(ref[col], color=MUTED, ls="--", lw=0.8)
            ax.text(ref[col], y.max() + 0.75, " random", color=MUTED, fontsize=6.5, va="center")
            ax.set(xlim=(0, 1), ylim=(-0.6, y.max() + 1.1), title=f"{title} · {NEG_LABELS[neg]}")
            ax.grid(axis="y", visible=False)
        axes[r, 0].set_yticks(y, [MODEL_LABELS[m] for m in models])
    return fig


def _median_seed_file(run_dir: Path, model: str, neg: str, metrics: pd.DataFrame) -> Path:
    """Test prediction file of the model; for neural models the seed with median test AP."""
    rows = metrics[(metrics["model"] == model) & (metrics["negatives"] == neg) & (metrics["phase"] == "test")]
    if rows["seed"].isna().all():
        return run_dir / "predictions" / f"{model}_{neg}_test.csv"
    seed = int(rows.sort_values("ap")["seed"].iloc[(len(rows) - 1) // 2])
    return run_dir / "predictions" / f"{model}_seed{seed}_{neg}_test.csv"


def fig3_pr_curves(run_dir: Path) -> plt.Figure:
    """Test PR curves, one per model (for neural models the seed with median test AP)."""
    metrics = pd.read_csv(run_dir / "metrics" / "metrics.csv")
    sets = _neg_sets(metrics)
    fig, axes = plt.subplots(1, len(sets), figsize=(DOUBLE, 3.0), sharey=True, squeeze=False)
    for ax, neg in zip(axes[0], sets):
        prevalence = 0.0
        for m in _models(metrics):
            pred = pd.read_csv(_median_seed_file(run_dir, m, neg, metrics), usecols=["label", "score"])
            precision, recall, _ = precision_recall_curve(pred["label"], pred["score"])
            ap = average_precision_score(pred["label"], pred["score"])
            ax.plot(recall, precision, drawstyle="steps-post", color=MODEL_COLORS[m], lw=1.1,
                    label=f"{MODEL_LABELS[m]} (AP = {ap:.2f})")
            prevalence = pred["label"].mean()
        ax.axhline(prevalence, color=MUTED, ls="--", lw=0.8)
        ax.text(1.0, prevalence, "random", color=MUTED, fontsize=6.5, ha="right", va="bottom")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="recall", title=NEG_LABELS[neg])
        # below the panel: the curves fill the whole plot area, so an inside legend hides data
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5)
    axes[0, 0].set_ylabel("precision")
    return fig


def fig4_gain_over_floor(per_window: pd.DataFrame, windows: pd.DataFrame) -> plt.Figure | None:
    """Per test window: GNN metric minus Dst popularity metric (mean ± std over seeds), AP and MRR."""
    gnns = [m for m in GNNS if m in set(per_window["model"])]
    if FLOOR not in set(per_window["model"]) or not gnns:
        return None
    key = ["negatives", "window_index"]
    floor = per_window[per_window["model"] == FLOOR].set_index(key)[["ap", "mrr"]]
    gain = per_window[per_window["model"].isin(gnns)].copy()
    for col in ("ap", "mrr"):
        gain[col] = gain[col].to_numpy() - floor.loc[list(zip(gain["negatives"], gain["window_index"])), col].to_numpy()
    agg = gain.groupby(["negatives", "model", "window_index"]).agg(
        ap_mean=("ap", "mean"), ap_std=("ap", "std"), mrr_mean=("mrr", "mean"), mrr_std=("mrr", "std"))
    agg = agg.fillna(0.0).reset_index().merge(windows[["window_index", "start_inclusive"]], on="window_index")
    sets = _neg_sets(gain)
    fig, axes = plt.subplots(len(sets), 2, figsize=(DOUBLE, 1.8 * len(sets) + 0.7), sharex=True, sharey="col",
                             squeeze=False)
    for r, neg in enumerate(sets):
        for c, (col, title) in enumerate((("ap", "AP"), ("mrr", "MRR"))):
            ax = axes[r, c]
            ax.axhline(0, color=MUTED, lw=0.8, ls="--")
            for m in gnns:
                d = agg[(agg["negatives"] == neg) & (agg["model"] == m)].sort_values("start_inclusive")
                mean, std = d[f"{col}_mean"], d[f"{col}_std"]
                ax.fill_between(d["start_inclusive"], mean - std, mean + std, color=MODEL_COLORS[m], alpha=0.2, lw=0)
                ax.plot(d["start_inclusive"], mean, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], markersize=3,
                        mec=RING, mew=0.5, lw=1.1, label=MODEL_LABELS[m])
            ax.set_title(f"{title} gain over Dst popularity · {NEG_LABELS[neg]}")
            if c == 0:
                ax.set_ylabel("GNN − Dst popularity")
    for ax in axes[-1]:
        ax.tick_params(axis="x", labelrotation=30)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(gnns))
    return fig


def fig5_by_source_degree(by_degree: pd.DataFrame) -> plt.Figure:
    """x = source out-degree bin (ordered), y = test AP | MRR, floor + GNNs only (dodged markers)."""
    df = by_degree[by_degree["model"].isin([FLOOR, *GNNS])]
    df = df.groupby(["negatives", "model", "degree_bin"], as_index=False).agg(
        ap=("ap", "mean"), ap_std=("ap", "std"), mrr=("mrr", "mean"), mrr_std=("mrr", "std"),
        n_positives=("n_positives", "first"))
    df["left"] = df["degree_bin"].map(lambda b: float(b.strip("[)").split(",")[0]))
    bins = df.drop_duplicates("degree_bin").sort_values("left")["degree_bin"].tolist()
    n_pos = df.drop_duplicates("degree_bin").set_index("degree_bin")["n_positives"]
    sets, models = _neg_sets(df), _models(df)
    fig, axes = plt.subplots(len(sets), 2, figsize=(DOUBLE, 2.2 * len(sets) + 0.8), sharex=True, sharey=True,
                             squeeze=False)
    offsets = np.linspace(-0.2, 0.2, len(models)) if len(models) > 1 else [0.0]
    for r, neg in enumerate(sets):
        for c, (col, title) in enumerate((("ap", "AP"), ("mrr", "MRR"))):
            ax = axes[r, c]
            for off, m in zip(offsets, models):
                d = df[(df["negatives"] == neg) & (df["model"] == m)].set_index("degree_bin").reindex(bins)
                ax.errorbar(np.arange(len(bins)) + off, d[col], yerr=d[f"{col}_std"].fillna(0),
                            fmt=MODEL_MARKERS[m], color=MODEL_COLORS[m], ms=4.5, elinewidth=0.8, capsize=1.5,
                            label=MODEL_LABELS[m])
            ax.grid(axis="x", visible=False)
            ax.set(ylim=(0, 1), title=f"test {title} · {NEG_LABELS[neg]}")
    for ax in axes[-1]:
        ax.set_xticks(np.arange(len(bins)), [f"{_bin_label(b)}\nn={int(n_pos.get(b, 0))}" for b in bins])
        ax.set_xlabel("source out-degree in history (n = test positives)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(models))
    return fig


def fig6_training_curves(curves: pd.DataFrame) -> plt.Figure:
    """(a) train loss, (b) validation AP per epoch; one thin line per seed, color = model."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.4))
    for m in _models(curves):
        for i, (_, d) in enumerate(curves[curves["model"] == m].groupby("seed")):
            a.plot(d["epoch"], d["train_loss"], color=MODEL_COLORS[m], lw=0.9, alpha=0.85,
                   label=MODEL_LABELS[m] if i == 0 else None)
            b.plot(d["epoch"], d["val_ap"], color=MODEL_COLORS[m], lw=0.9, alpha=0.85)
            best = d.loc[d["val_ap"].idxmax()]
            b.plot(best["epoch"], best["val_ap"], marker=MODEL_MARKERS[m], color=MODEL_COLORS[m], ms=5,
                   mec=RING, mew=0.8)
    a.set(xlabel="epoch", ylabel="training loss (BCE)", title="(a) Training loss")
    b.set(xlabel="epoch", ylabel="validation AP (uniform negatives)",
          title="(b) Validation AP · marker = selected epoch")
    a.legend(loc="upper right")
    return fig


def fig7_hits_at_k(run_dir: Path, n_neg: int) -> plt.Figure:
    """Share of test positives ranked within the top K of their group, for every K (log x).
    Neural models: the curve of all seeds pooled. Dashed: random ranker, K / (n_neg + 1)."""
    metrics = pd.read_csv(run_dir / "metrics" / "metrics.csv")
    sets, models = _neg_sets(metrics), _models(metrics)
    k = np.arange(1, n_neg + 2)
    fig, axes = plt.subplots(1, len(sets), figsize=(DOUBLE, 3.0), sharey=True, squeeze=False)
    for ax, neg in zip(axes[0], sets):
        for m in models:
            files = sorted((run_dir / "predictions").glob(f"{m}_*{neg}_test.csv"))
            ranks = np.concatenate([positive_ranks(pd.read_csv(f, usecols=["label", "group", "score"])).to_numpy()
                                    for f in files])
            ax.plot(k, (ranks[:, None] <= k).mean(axis=0), color=MODEL_COLORS[m], lw=1.2, label=MODEL_LABELS[m])
        ax.plot(k, k / (n_neg + 1), color=MUTED, ls="--", lw=0.8)
        ax.text(10, 10 / (n_neg + 1) - 0.03, "random", color=MUTED, fontsize=6.5, ha="left", va="top")   # below the line: curves sit above it
        ax.set_xscale("log")
        ticks = [t for t in (1, 2, 5, 10, 20, n_neg + 1) if t <= n_neg + 1]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set(xlim=(1, n_neg + 1), ylim=(0, 1.02), xlabel=f"K (of {n_neg + 1} candidates)", title=NEG_LABELS[neg])
    axes[0, 0].set_ylabel("Hits@K (share of test positives)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(models), 6))
    return fig


def fig8_negative_difficulty(run_dir: Path) -> plt.Figure | None:
    """ECDF of the weeks since the destination was last active, test pairs: true destinations vs the
    two negative sets. Needs the ``dst_recency`` column of the pair files (older runs: skipped)."""
    groups = []
    for neg, label, ls in (("uniform", "uniform negatives", "--"), ("recent", "recent-receiver negatives", ":")):
        path = run_dir / "samples" / f"pairs_{neg}_test.csv"
        if not path.is_file():
            continue
        pairs = pd.read_csv(path)
        if "dst_recency" not in pairs:
            return None
        if not groups:
            groups.append(("true destinations (positives)", pairs.loc[pairs["label"] == 1, "dst_recency"], "-"))
        groups.append((label, pairs.loc[pairs["label"] == 0, "dst_recency"], ls))
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 3.1))
    for (label, values, ls), color in zip(groups, ("#2a78d6", "#eb6834", "#1baf7a")):
        x = np.sort(values.to_numpy())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), drawstyle="steps-post", color=color, ls=ls, lw=1.4,
                label=f"{label} (median {np.median(x):.0f} wk)")
    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set(ylim=(0, 1.02), xlabel="weeks since the destination was last active (log)",
           ylabel="share of test pairs (cumulative)", title="How active are the candidate destinations?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)   # the curves span the whole plot
    return fig


def make_all(run_dir: Path, cfg: FigureConfig, n_neg: int) -> None:
    """Draw every figure whose input CSV exists (stale figure files are removed first)."""
    apply_style()
    m = run_dir / "metrics"
    windows = pd.read_csv(run_dir / "splits" / "windows.csv", parse_dates=["start_inclusive", "end_exclusive"])
    out = run_dir / "figures"
    if out.is_dir():
        for old in out.iterdir():
            if old.is_file():
                old.unlink()
    figs = {
        "fig1_windows": fig1_windows(run_dir),
        "fig2_model_comparison": fig2_model_comparison(pd.read_csv(m / "metrics.csv"), n_neg),
        "fig3_pr_curves": fig3_pr_curves(run_dir),
        "fig4_gain_over_floor": fig4_gain_over_floor(pd.read_csv(m / "per_window.csv"), windows),
        "fig5_by_source_degree": fig5_by_source_degree(pd.read_csv(m / "by_source_degree.csv")),
        "fig6_training_curves": (fig6_training_curves(pd.read_csv(m / "training_curves.csv"))
                                 if (m / "training_curves.csv").is_file() else None),
        "fig7_hits_at_k": fig7_hits_at_k(run_dir, n_neg),
        "fig8_negative_difficulty": fig8_negative_difficulty(run_dir),
    }
    for name, fig in figs.items():
        if fig is not None:
            _save(fig, name, out, cfg)
