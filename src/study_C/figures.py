"""Report figures, drawn only from the CSV/JSON files saved by the pipeline.

Each figure is saved in ``figures/`` in every format of ``[figures]`` and gets a row in
``figures/captions.csv`` (and ``captions.md``): title, what is measured, split and the
source files it was drawn from, so every plotted number traces back to a saved artifact.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # windowless backend: saves to file even without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import StudyCConfig  # noqa: E402

logger = logging.getLogger("study_C")

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
FEATURE_COLORS = {"local": "#2a78d6", "all": "#eb6834"}
FEATURE_MARKERS = {"local": "o", "all": "s"}
FEATURE_LABELS = {"local": "local features", "all": "local + aggregated"}
LABEL_COLORS = {"licit": "#1baf7a", "illicit": "#6250d6"}
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f4f7fc", "#5598e7", "#184f95"])
MODEL_LABELS = {"degree_baseline": "Degree only", "random_forest": "Random forest", "mlp": "MLP",
                "graphsage": "GraphSAGE", "gat": "GAT"}
MODEL_ORDER = list(MODEL_LABELS)
BASELINES = {"degree_baseline"}   # reference scores: drawn as lines, never as a compared model
SINGLE, DOUBLE = 3.5, 7.16
MIN_HOMOPHILY_EVIDENCE = 20   # labelled edges with an illicit end below which a step's r is drawn hollow


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titlelocation": "left",
        "axes.titlepad": 4, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": SECONDARY, "text.color": INK, "axes.titlecolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.grid.which": "major", "grid.color": GRID, "grid.linewidth": 0.45, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": True, "mathtext.default": "regular",
    })


class FigureWriter:
    """Saves figures in every configured format and collects their captions."""

    def __init__(self, run_dir: Path, config: StudyCConfig):
        self.run_dir = run_dir
        self.dir = run_dir / "figures"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.formats, self.dpi = config.figures.formats, config.figures.dpi
        self.captions: list[dict[str, Any]] = []

    def save(self, fig: plt.Figure, figure_id: str, *, title: str, caption: str, split: str,
             sources: list[Path]) -> None:
        files = []
        for fmt in self.formats:
            path = self.dir / f"{figure_id}.{fmt}"
            fig.savefig(path, dpi=self.dpi)
            files.append(path.name)
        plt.close(fig)
        self.captions.append({"figure": figure_id, "files": "; ".join(files), "title": title,
                              "caption": caption, "split": split,
                              "source_artifacts": "; ".join(s.relative_to(self.run_dir).as_posix()
                                                            for s in sources)})
        logger.info("Figure %s saved", figure_id)

    def write_captions(self) -> None:
        table = pd.DataFrame(self.captions)
        table.to_csv(self.dir / "captions.csv", index=False)
        lines = ["# Study C figures", ""]
        for row in table.itertuples():
            lines += [f"## {row.figure}: {row.title}", "", row.caption, "",
                      f"- Split: {row.split}", f"- Files: {row.files}", f"- Sources: {row.source_artifacts}", ""]
        (self.dir / "captions.md").write_text("\n".join(lines), encoding="utf-8")


def _split_lines(ax: plt.Axes, train_end: int, val_end: int, label: bool) -> None:
    for boundary in (train_end + 0.5, val_end + 0.5):
        ax.axvline(boundary, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    if label:
        lo, hi = ax.get_xlim()
        for x, text in (((lo + train_end + 0.5) / 2, "train"), ((train_end + val_end + 1) / 2, "val"),
                        ((val_end + 0.5 + hi) / 2, "test")):
            ax.text(x, 1.0, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7, color=SECONDARY)


def _model_names(frame: pd.DataFrame) -> list[str]:
    present = set(frame["model"]) - BASELINES
    return [m for m in MODEL_ORDER if m in present]


def _feature_sets(frame: pd.DataFrame) -> list[str]:
    return [f for f in FEATURE_COLORS if f in set(frame["feature_set"])]


def plot_labels_by_step(label_counts: pd.DataFrame, train_end: int, val_end: int, writer: FigureWriter,
                        source: Path) -> None:
    """Stacked bars of licit/illicit per time step (top) and illicit share of labelled (bottom)."""
    steps = label_counts["time_step"].to_numpy()
    licit, illicit = label_counts["licit"].to_numpy(), label_counts["illicit"].to_numpy()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(DOUBLE, 3.6), sharex=True, height_ratios=[1.6, 1])
    top.bar(steps, licit, width=0.8, color=LABEL_COLORS["licit"], edgecolor=SURFACE, lw=0.6, label="licit")
    top.bar(steps, illicit, width=0.8, bottom=licit, color=LABEL_COLORS["illicit"], edgecolor=SURFACE, lw=0.6,
            label="illicit")
    top.set_ylabel("labelled transactions")
    fig.legend(*top.get_legend_handles_labels(), loc="outside upper right", ncols=2)
    labelled = licit + illicit
    share = np.divide(illicit, labelled, out=np.full(len(steps), np.nan), where=labelled > 0)
    bottom.plot(steps, share, color=LABEL_COLORS["illicit"], marker="o", markersize=2.6,
                markeredgecolor=SURFACE, markeredgewidth=0.4)
    bottom.set_ylabel("illicit share")
    bottom.set_ylim(0, None)
    bottom.set_xlabel("time step")
    bottom.set_xlim(steps.min() - 0.6, steps.max() + 0.6)
    for ax in (top, bottom):
        _split_lines(ax, train_end, val_end, label=ax is top)
    writer.save(fig, "fig01_labels_by_step", title="Labelled transactions per time step",
                caption=("Top: licit and illicit labelled transactions per time step (unknown nodes omitted). "
                         "Bottom: illicit share of labelled transactions. Dashed lines mark the chronological "
                         f"split (train 1-{train_end}, validation {train_end + 1}-{val_end}, test after)."),
                split="all steps", sources=[source])


def plot_metric_comparison(summary: pd.DataFrame, writer: FigureWriter, source: Path) -> None:
    """Grouped bars on TEST: models on x, one bar per feature set, error bars = std over seeds;
    horizontal references = prevalence (random ranking, PR-AUC only) and the degree-only baseline."""
    test = summary[summary["split"] == "test"]
    names, feature_sets = _model_names(test), _feature_sets(test[~test["model"].isin(BASELINES)])
    baseline = test[test["model"] == "degree_baseline"]
    width = 0.8 / len(feature_sets)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), sharey=True)
    for ax, metric, label in zip(axes, ("pr_auc", "f1"), ("PR-AUC (illicit)", "F1 (illicit)")):
        for j, fs in enumerate(feature_sets):
            rows = test[test["feature_set"] == fs].set_index("model").reindex(names)
            x = np.arange(len(names)) + (j - (len(feature_sets) - 1) / 2) * width
            mean, std = rows[f"{metric}_mean"].to_numpy(), rows[f"{metric}_std"].to_numpy()
            ax.bar(x, mean, width=width * 0.92, color=FEATURE_COLORS[fs], edgecolor=SURFACE, lw=0.6,
                   label=FEATURE_LABELS[fs])
            spread = rows["n_seeds"].to_numpy() > 1
            ax.errorbar(x[spread], mean[spread], yerr=std[spread], fmt="none", ecolor=INK, elinewidth=0.7,
                        capsize=1.8)
            for xi, m, s in zip(x, mean, np.nan_to_num(std)):
                if np.isfinite(m):
                    ax.text(xi, m + s + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=6, color=SECONDARY)
        if metric == "pr_auc":
            prevalence = float(test["prevalence_mean"].iloc[0])
            ax.axhline(prevalence, color=MUTED, lw=0.8, ls=(0, (3, 2)), label=f"random ranking ({prevalence:.3f})")
        if len(baseline):
            value = float(baseline[f"{metric}_mean"].iloc[0])
            ax.axhline(value, color=SECONDARY, lw=0.8, ls=(0, (1, 1.5)), label=f"degree only ({value:.3f})")
        ax.set_xticks(np.arange(len(names)), [MODEL_LABELS[n] for n in names])
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(0, 1.08)
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
    for ax in axes:
        ax.legend(loc="upper right", ncols=1)
    writer.save(fig, "fig02_test_metrics", title="Test performance by model and feature set",
                caption=("Illicit-class PR-AUC and F1 on the test steps, mean over seeds; error bars = one "
                         "standard deviation (none for the single-seed random forest). The F1 threshold is "
                         "chosen on validation and applied unchanged. Dashed: prevalence (PR-AUC of a random "
                         "ranking). Dotted: degree-only baseline (score = node degree in the visible graph)."),
                split="test", sources=[source])


def plot_per_step_f1(per_step: pd.DataFrame, label_counts: pd.DataFrame, val_end: int,
                     writer: FigureWriter, sources: list[Path]) -> None:
    """Illicit F1 per test step: one panel per model, one line per feature set (band = ± std over
    seeds), with the number of illicit test transactions per step on top for context."""
    names = _model_names(per_step)
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(SINGLE * 1.3, 1.1 + 1.45 * len(names)), sharex=True,
                             height_ratios=[0.7] + [1] * len(names))
    counts = label_counts[label_counts["time_step"] > val_end]
    axes[0].bar(counts["time_step"], counts["illicit"], width=0.75, color=LABEL_COLORS["illicit"],
                edgecolor=SURFACE, lw=0.6)
    axes[0].set_ylabel("illicit\n(test)")
    axes[0].set_title("Illicit transactions per test step")
    for ax, name in zip(axes[1:], names):
        rows = per_step[per_step["model"] == name]
        for fs in _feature_sets(rows):
            r = rows[rows["feature_set"] == fs].sort_values("time_step")
            ax.plot(r["time_step"], r["f1_mean"], color=FEATURE_COLORS[fs], marker=FEATURE_MARKERS[fs],
                    markersize=3, markeredgecolor=SURFACE, markeredgewidth=0.4, label=FEATURE_LABELS[fs])
            if (r["n_seeds"] > 1).any():
                ax.fill_between(r["time_step"], r["f1_mean"] - r["f1_std"], r["f1_mean"] + r["f1_std"],
                                color=FEATURE_COLORS[fs], alpha=0.15, lw=0)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("illicit F1")
        ax.set_title(MODEL_LABELS[name])
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", ncols=1)
    axes[-1].set_xlabel("time step")
    axes[-1].set_xticks(counts["time_step"])
    writer.save(fig, "fig03_per_step_f1", title="Illicit F1 per test time step",
                caption=("Illicit F1 on each test time step with the validation-chosen threshold, mean over "
                         "seeds (band = ± one standard deviation). Top: number of illicit transactions per "
                         "step; F1 is undefined (gap) where a step has no illicit transaction and no alarm."),
                split="test", sources=sources)


def _representative_seed(metrics_all: pd.DataFrame, model: str, feature_set: str) -> int:
    """Seed whose VALIDATION PR-AUC is the median of its group (chosen without looking at test)."""
    val = metrics_all[(metrics_all["model"] == model) & (metrics_all["feature_set"] == feature_set)
                      & (metrics_all["split"] == "val")]
    order = val.sort_values("pr_auc").reset_index(drop=True)
    return int(order.loc[(len(order) - 1) // 2, "seed"])


def plot_confusion_matrices(metrics_all: pd.DataFrame, writer: FigureWriter, source: Path) -> None:
    """2x2 test confusion matrix for each model and feature set (representative seed), with counts
    and share of the actual class annotated."""
    names = _model_names(metrics_all)
    feature_sets = _feature_sets(metrics_all[~metrics_all["model"].isin(BASELINES)])
    fig, axes = plt.subplots(len(feature_sets), len(names), squeeze=False,
                             figsize=(1.75 * len(names) + 0.4, 1.75 * len(feature_sets)))
    sources = [source]
    for i, fs in enumerate(feature_sets):
        for j, name in enumerate(names):
            ax = axes[i, j]
            seed = _representative_seed(metrics_all, name, fs)
            path = writer.run_dir / "models" / f"{name}__{fs}__seed{seed}" / "metrics.json"
            sources.append(path)
            m = json.loads(path.read_text(encoding="utf-8"))["test"]
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
            share = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
            for (r, c), count in np.ndenumerate(cm):
                ax.text(c, r, f"{int(count):,}\n{share[r, c]:.0%}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if share[r, c] > 0.55 else INK)
            ax.set_xticks([0, 1], ["licit", "illicit"])
            ax.set_yticks([0, 1], ["licit", "illicit"])
            ax.tick_params(length=0)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            seeds = metrics_all.loc[(metrics_all["model"] == name) & (metrics_all["feature_set"] == fs), "seed"]
            suffix = f" (seed {seed})" if seeds.nunique() > 1 else ""
            ax.set_title(f"{MODEL_LABELS[name]}, {fs}{suffix}", fontsize=7)
            if i == len(feature_sets) - 1:
                ax.set_xlabel("predicted")
            if j == 0:
                ax.set_ylabel("actual")
    writer.save(fig, "fig04_confusion_test", title="Test confusion matrices",
                caption=("Confusion matrices on the test steps at the validation-chosen threshold. Cells show "
                         "counts and the share of the actual class (row); shading = that share. For multi-seed "
                         "models the seed with the median validation PR-AUC is shown."),
                split="test", sources=sources)


def plot_learning_curves(metrics_all: pd.DataFrame, model: str, writer: FigureWriter) -> None:
    """Training loss (top) and validation PR-AUC (bottom) per epoch, all seeds, best epoch marked."""
    rows = metrics_all[metrics_all["model"] == model]
    feature_sets = _feature_sets(rows)
    fig, axes = plt.subplots(2, len(feature_sets), figsize=(DOUBLE, 3.4), sharex="col", sharey="row", squeeze=False)
    sources = []
    for j, fs in enumerate(feature_sets):
        seeds = sorted(rows.loc[rows["feature_set"] == fs, "seed"].unique())
        for seed in seeds:
            path = writer.run_dir / "models" / f"{model}__{fs}__seed{seed}" / "history.csv"
            if not path.is_file():
                continue
            sources.append(path)
            h = pd.read_csv(path)
            best = h.loc[h["val_pr_auc"].idxmax()]
            axes[0, j].plot(h["epoch"], h["train_loss"], color=FEATURE_COLORS[fs], lw=0.8, alpha=0.75)
            axes[1, j].plot(h["epoch"], h["val_pr_auc"], color=FEATURE_COLORS[fs], lw=0.8, alpha=0.75)
            axes[1, j].plot(best["epoch"], best["val_pr_auc"], marker=FEATURE_MARKERS[fs], markersize=4,
                            color=FEATURE_COLORS[fs], markeredgecolor=SURFACE, markeredgewidth=0.5, ls="none")
        axes[0, j].set_title(f"{MODEL_LABELS[model]}, {FEATURE_LABELS[fs]} ({len(seeds)} seed{'s' if len(seeds) != 1 else ''})")
        axes[1, j].set_xlabel("epoch")
    axes[0, 0].set_ylabel("training loss\n(weighted BCE)")
    axes[1, 0].set_ylabel("validation PR-AUC")
    if not sources:
        plt.close(fig)
        return
    writer.save(fig, f"fig05_learning_{model}", title=f"{MODEL_LABELS[model]} learning curves",
                caption=("Full-batch training: one optimiser step per epoch. Top: class-weighted BCE on labelled "
                         "training nodes. Bottom: validation PR-AUC used for early stopping; the marker is the "
                         "best epoch, whose weights are kept. One line per seed."),
                split="train (loss) / validation (PR-AUC)", sources=sources)


def plot_homophily_by_phase(homophily: dict[str, dict[str, float]], writer: FigureWriter, source: Path) -> None:
    """Per phase: P(neighbour illicit | illicit) vs P(neighbour illicit | licit), with the no-homophily
    level (illicit share of labelled endpoints) as a dashed reference and assortativity annotated."""
    phases = [p for p in ("train", "val", "test") if p in homophily]
    x, width = np.arange(len(phases)), 0.36
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    bases = [homophily[p]["illicit_endpoint_share"] for p in phases]
    for offset, (own, key) in ((-width / 2, ("illicit", "p_neighbour_illicit_given_illicit")),
                               (width / 2, ("licit", "p_neighbour_illicit_given_licit"))):
        values = [homophily[p][key] for p in phases]
        ax.bar(x + offset, values, width=width, color=LABEL_COLORS[own], edgecolor=SURFACE, lw=0.6,
               label=f"neighbour of {'an' if own == 'illicit' else 'a'} {own} tx")
        for xi, v, base in zip(x + offset, values, bases):   # value above the bar and the dashed reference
            ax.text(xi, max(v, base) + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=SECONDARY)
    for i, (p, base) in enumerate(zip(phases, bases)):
        ax.plot([i - width, i + width], [base, base], color=MUTED, lw=0.9, ls=(0, (3, 2)),
                label="no homophily (illicit endpoint share)" if i == 0 else None)
        ax.text(i, 1.0, f"r = {homophily[p]['label_assortativity']:.2f}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color=SECONDARY)
    ax.set_xticks(x, [f"{p}\n({homophily[p]['n_edges_both_labelled']:,} edges)" for p in phases])
    ax.set_ylabel("P(neighbour is illicit)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=2)
    writer.save(fig, "fig06_homophily_by_phase", title="Label homophily along edges, per phase",
                caption=("Share of illicit neighbours for illicit and for licit transactions, on edges whose two "
                         "endpoints are both labelled (read as undirected; unknown endpoints excluded). The dashed "
                         "segment is the share expected without homophily; r is Newman's label assortativity "
                         "(0 = random, 1 = only same-class edges). Only the training phase informs design "
                         "choices; validation and test are descriptive."),
                split="train / validation / test (descriptive)", sources=[source])


def plot_homophily_by_step(by_step: pd.DataFrame, train_end: int, val_end: int, writer: FigureWriter,
                           source: Path) -> None:
    """Label assortativity per time step (top) and labelled edges touching an illicit tx (bottom)."""
    by_step = by_step.sort_values("time_step")
    steps = by_step["time_step"].to_numpy()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(DOUBLE, 3.4), sharex=True, height_ratios=[1.6, 1])
    top.axhline(0, color=AXIS, lw=0.7, zorder=1)
    top.plot(steps, by_step["label_assortativity"], color=LABEL_COLORS["illicit"], marker="o", markersize=2.6,
             markeredgecolor=SURFACE, markeredgewidth=0.4)
    few = (by_step["n_edges_with_illicit"] < MIN_HOMOPHILY_EVIDENCE).to_numpy()
    top.plot(steps[few], by_step["label_assortativity"].to_numpy()[few], ls="none", marker="o", markersize=3.4,
             markerfacecolor=SURFACE, markeredgecolor=LABEL_COLORS["illicit"], markeredgewidth=0.8,
             label=f"< {MIN_HOMOPHILY_EVIDENCE} labelled edges with an illicit end")
    top.legend(loc="lower right")
    top.set_ylabel("label assortativity r")
    top.set_ylim(min(-0.2, float(np.nanmin(by_step["label_assortativity"])) - 0.05), 1.0)
    # labelled edges with >= 1 illicit endpoint = how much evidence each step's r rests on
    bottom.bar(steps, by_step["n_edges_with_illicit"], width=0.8, color=LABEL_COLORS["illicit"], edgecolor=SURFACE, lw=0.6)
    bottom.set_ylabel("labelled edges\nwith an illicit end")
    bottom.set_xlabel("time step")
    bottom.set_xlim(steps.min() - 0.6, steps.max() + 0.6)
    for ax in (top, bottom):
        _split_lines(ax, train_end, val_end, label=ax is top)
    writer.save(fig, "fig07_homophily_by_step", title="Label assortativity per time step",
                caption=("Top: Newman's label assortativity of the edges inside each time step, on edges with both "
                         "endpoints labelled (gap = step without illicit endpoints). Bottom: number of such edges "
                         "with at least one illicit endpoint, i.e. the evidence behind each value; steps with few "
                         "edges give noisy r (hollow markers: fewer than "
                         f"{MIN_HOMOPHILY_EVIDENCE}). Dashed lines mark the chronological split; values outside the "
                         "training steps are descriptive only."),
                split="all steps (descriptive)", sources=[source])


def make_figures(run_dir: Path, config: StudyCConfig) -> list[Path]:
    """Draw every figure from the saved files of ``run_dir``; returns the files in ``figures/``."""
    apply_style()
    writer = FigureWriter(run_dir, config)
    counts_path = run_dir / "data" / "label_counts_by_step.csv"
    summary_path = run_dir / "summary" / "metrics_summary.csv"
    all_path = run_dir / "summary" / "metrics_all.csv"
    per_step_path = run_dir / "summary" / "per_step_summary.csv"
    counts = pd.read_csv(counts_path)
    summary, metrics_all, per_step = pd.read_csv(summary_path), pd.read_csv(all_path), pd.read_csv(per_step_path)

    plot_labels_by_step(counts, config.split.train_end, config.split.val_end, writer, counts_path)
    plot_metric_comparison(summary, writer, summary_path)
    plot_per_step_f1(per_step, counts, config.split.val_end, writer, [per_step_path, counts_path])
    plot_confusion_matrices(metrics_all, writer, all_path)
    homophily_path = run_dir / "data" / "edge_homophily.json"
    by_step_path = run_dir / "data" / "edge_homophily_by_step.csv"
    plot_homophily_by_phase(json.loads(homophily_path.read_text(encoding="utf-8")), writer, homophily_path)
    if by_step_path.is_file():   # absent in runs saved before this table existed
        plot_homophily_by_step(pd.read_csv(by_step_path), config.split.train_end, config.split.val_end, writer,
                               by_step_path)
    else:
        logger.warning("Skipping fig07: %s not found", by_step_path)
    for model in ("mlp", "graphsage", "gat"):
        if model in set(metrics_all["model"]):
            plot_learning_curves(metrics_all, model, writer)
    writer.write_captions()
    return sorted(writer.dir.iterdir())
