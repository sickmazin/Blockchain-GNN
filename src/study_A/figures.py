"""Publication figures for Study A, built only from the saved tables of a run.

Style: vector PDF (embedded TrueType) plus 300-dpi PNG, sized for a two-column
paper (single column 3.5 in, double column 7.16 in), sans-serif 8 pt text,
hairline axes and grid, thin marks. Colour roles follow a validated categorical
palette: tokens take slots 1-3 (all-pairs colour-vision-safe) and always add a
distinct marker shape; other nominal categories use later slots with direct
labels. Quantities on very different scales are never put on twin axes.

Each figure gets a caption entry in ``figures/captions.csv`` (and ``captions.md``)
stating the title, graph definition, time window, weight definition and the
source artifacts it was drawn from.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .centrality import MEASURE_LABELS  # noqa: E402
from .config import StudyAConfig  # noqa: E402
from .distributions import ccdf, model_ccdf  # noqa: E402
from .reporting import selected_tokens  # noqa: E402
from .topology import BOWTIE_ORDER  # noqa: E402
from .windows import FULL_PERIOD  # noqa: E402

logger = logging.getLogger("study_A")

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TOKEN_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
BOWTIE_COLORS = dict(zip(BOWTIE_ORDER, ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
BOWTIE_LABELS = {"CORE": "Core (LSCC)", "IN": "IN", "OUT": "OUT", "TENDRILS_TUBES": "Tendrils/tubes",
                 "DISCONNECTED": "Disconnected"}
COMMUNITY_COLORS = ["#008300", "#6250d6", "#e34948"]
COMMUNITY_MARKERS = ["o", "s", "^"]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#5598e7", "#f0efec",
                                                                "#ec835a", "#b3261e"])
SINGLE, DOUBLE = 3.5, 7.16
GRAPH_BASE = ("directed ERC-20 transfer graph of deduplicated transfers; nodes are addresses; "
              "self-loops and the zero address removed")
PROJECTION = "undirected projection (reciprocal arc weights summed)"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titlelocation": "left",
        "axes.titlepad": 4, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": SECONDARY, "text.color": INK, "axes.titlecolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.grid.which": "major", "grid.color": GRID, "grid.linewidth": 0.45, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04, "mathtext.default": "regular",
    })


@dataclass
class RunData:
    run_dir: Path
    config: StudyAConfig
    tokens: list[str] = field(default_factory=list)
    _cache: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)

    def table(self, token: str, name: str) -> pd.DataFrame:
        key = (token, name)
        if key not in self._cache:
            path = self.run_dir / token / name
            path = path if path.suffix in {".csv", ".gz"} else path.with_suffix(".csv")
            try:
                frame = pd.read_csv(path)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                frame = pd.DataFrame()
            self._cache[key] = frame
        return self._cache[key]

    def color(self, token: str) -> str:
        return SLOTS[self.tokens.index(token) % len(SLOTS)]

    def marker(self, token: str) -> str:
        return TOKEN_MARKERS[self.tokens.index(token) % len(TOKEN_MARKERS)]

    def windows(self, token: str) -> pd.DataFrame:
        manifest = self.table(token, "window_manifest")
        manifest = manifest.loc[manifest["window"] != FULL_PERIOD].copy()
        manifest["start"] = pd.to_datetime(manifest["start_inclusive"])
        manifest["mid"] = manifest["start"] + (pd.to_datetime(manifest["end_exclusive"]) - manifest["start"]) / 2
        return manifest


class FigureWriter:
    def __init__(self, run_dir: Path, config: StudyAConfig):
        self.dir = run_dir / "figures"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.captions: list[dict[str, Any]] = []

    def save(self, fig: plt.Figure, figure_id: str, *, title: str, caption: str, graph: str,
             window: str, weight: str, sources: list[str]) -> None:
        if self.config.figures.embed_titles:
            # A suptitle shares the top band with "outside upper" figure legends and overlaps them.
            # Lay the figure out once, pin every figure legend where the layout put it (otherwise it
            # snaps back to the figure edge once the layout engine is off), freeze, then put the
            # title above everything drawn.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for legend in fig.legends:
                extent = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
                legend.set_loc("center")
                legend.set_bbox_to_anchor(((extent.x0 + extent.x1) / 2, (extent.y0 + extent.y1) / 2),
                                          transform=fig.transFigure)
            fig.set_layout_engine("none")
            box = fig.get_tightbbox(renderer)
            gap = 8 / 72  # inches
            fig.text(box.x0 / fig.get_figwidth(), (box.y1 + gap) / fig.get_figheight(), title,
                     ha="left", va="bottom", fontsize=9, fontweight="bold", color=INK)
        files = []
        for fmt in self.config.figures.formats:
            path = self.dir / f"{figure_id}.{fmt}"
            fig.savefig(path, dpi=self.config.figures.dpi, metadata={"Title": title} if fmt == "pdf" else None)
            files.append(path.name)
        plt.close(fig)
        self.captions.append({"figure": figure_id, "files": "; ".join(files), "title": title,
                              "caption": caption, "graph_definition": graph, "time_window": window,
                              "weight_definition": weight, "source_artifacts": "; ".join(sources)})
        logger.info("Figure %s saved", figure_id)

    def write_captions(self) -> None:
        """Update captions.csv/md: rows of figures redrawn now replace their old rows, others are kept."""
        table = pd.DataFrame(self.captions)
        path = self.dir / "captions.csv"
        if path.is_file() and not table.empty:
            previous = pd.read_csv(path)
            previous = previous.loc[~previous["figure"].isin(table["figure"])]
            table = pd.concat([previous, table], ignore_index=True)
        if not table.empty:
            table = table.sort_values("figure", kind="stable").reset_index(drop=True)
        table.to_csv(path, index=False)
        lines = ["# Study A figures", ""]
        for row in table.fillna("").to_dict("records"):
            lines += [f"## {row['figure']}: {row['title']}", "", row["caption"], "",
                      f"- Graph: {row['graph_definition']}", f"- Window: {row['time_window']}",
                      f"- Weight: {row['weight_definition']}", f"- Source: {row['source_artifacts']}",
                      f"- Files: {row['files']}", ""]
        (self.dir / "captions.md").write_text("\n".join(lines), encoding="utf-8")


# ----- shared helpers ----------------------------------------------------------------------

def _token_legend(fig: plt.Figure, data: RunData, *, extra: list[Any] | None = None,
                  loc: str = "outside upper center") -> None:
    handles = [Line2D([], [], color=data.color(t), marker=data.marker(t), markersize=4, lw=1.1,
                      markerfacecolor=data.color(t), markeredgecolor=SURFACE, markeredgewidth=0.4,
                      label=t) for t in data.tokens]
    handles += extra or []
    fig.legend(handles=handles, loc=loc, ncol=len(handles), frameon=False)


def _date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, show_offset=False))


def _windowed_line(ax: plt.Axes, data: RunData, token: str, x: pd.Series, y: pd.Series,
                   partial: pd.Series) -> None:
    color = data.color(token)
    ax.plot(x, y, color=color, lw=1.1, zorder=2)
    full = ~partial.astype(bool).to_numpy()
    ax.scatter(x[full], y[full], s=11, marker=data.marker(token), color=color,
               edgecolors=SURFACE, linewidths=0.4, zorder=3)
    if (~full).any():
        ax.scatter(x[~full], y[~full], s=11, marker=data.marker(token), facecolors=SURFACE,
                   edgecolors=color, linewidths=0.8, zorder=3)


def _partial_handle() -> Line2D:
    return Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, markeredgecolor=MUTED,
                  lw=0, markersize=4, label="partial month")


def _monthly(data: RunData, token: str, name: str, **filters: Any) -> pd.DataFrame:
    table = data.table(token, name)
    if table.empty:
        return table
    table = table.loc[table["window"] != FULL_PERIOD]
    for column, value in filters.items():
        table = table.loc[table[column] == value]
    windows = data.windows(token)[["window", "mid"]]
    return table.merge(windows, on="window", how="left").sort_values("mid")


def _ref_line(ax: plt.Axes, value: float, label: str | None = None) -> None:
    ax.axhline(value, color=MUTED, lw=0.8, zorder=1)
    if label:
        ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=SECONDARY)


def _compact(value: float) -> str:
    if value >= 1:
        for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
            if value >= limit:
                return f"{value / limit:g}{suffix}"
        return f"{value:g}"
    exponent = math.floor(math.log10(value))
    mantissa = value / 10 ** exponent
    if round(mantissa, 6) == 1:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}\\times10^{{{exponent}}}$"


def _log_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Log scale with 1-2-5 ticks when the range spans few decades, and compact labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    low, high = ax.get_ylim() if axis == "y" else ax.get_xlim()
    subs = (1.0, 2.0, 5.0) if high / max(low, 1e-300) < 1e3 else (1.0,)
    target.set_major_locator(LogLocator(base=10, subs=subs))
    target.set_major_formatter(FuncFormatter(lambda v, _: _compact(v) if v > 0 else ""))
    target.set_minor_formatter(NullFormatter())


def _short_address(address: str) -> str:
    return f"{address[:6]}…{address[-4:]}"


# ----- figures ---------------------------------------------------------------------------

def fig_activity(data: RunData, writer: FigureWriter) -> None:
    rows = len(data.tokens)
    fig, axes = plt.subplots(rows, 2, figsize=(DOUBLE, 1.35 * rows + 0.3), squeeze=False)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r, token in enumerate(data.tokens):
        daily = data.table(token, "daily_activity")
        daily["date"] = pd.to_datetime(daily["date"])
        for c, (column, label) in enumerate((("transfers", "Unique transfers per day"),
                                             ("active_addresses", "Active addresses per day"))):
            ax = axes[r, c]
            series = daily.set_index("date")[column].astype(float)
            rolling = series.rolling(7, center=True, min_periods=4).mean()
            ax.plot(series.index, series.where(series > 0), color=data.color(token), lw=0.5, alpha=0.35)
            ax.plot(rolling.index, rolling.where(rolling > 0), color=data.color(token), lw=1.2)
            _log_axis(ax)
            ax.set_title(f"({next(letters)}) {token}: {label.lower()}")
            ax.set_ylabel(label.split(" per")[0])
            _date_axis(ax)
    handles = [Line2D([], [], color=MUTED, lw=0.5, alpha=0.6, label="daily"),
               Line2D([], [], color=MUTED, lw=1.2, label="7-day centred mean")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    writer.save(fig, "fig01_daily_activity", title="Daily transfer activity",
                caption="Daily number of unique ERC-20 transfers and of distinct active addresses "
                        "(senders or receivers), after reconstructing unique transfers from the overlapping "
                        "TGS snapshots. Thin lines: daily values; thick lines: 7-day centred moving average. "
                        "Logarithmic y-axis; days without transfers are not drawn.",
                graph="event stream (no graph aggregation; filters not applied)", window="daily, full period",
                weight="none (counts)", sources=[f"{t}/daily_activity.csv" for t in data.tokens])


def fig_snapshot_replication(data: RunData, writer: FigureWriter) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    for token in data.tokens:
        by_date = data.table(token, "dedup_by_date")
        by_date["date"] = pd.to_datetime(by_date["date"])
        ratio = by_date["raw_rows"] / by_date["unique_events"].replace(0, np.nan)
        ax.plot(by_date["date"], ratio, color=data.color(token), lw=1.0)
        span = data.table(token, "snapshot_structure")["span_days"].mode().iloc[0]
    _ref_line(ax, float(span), f"snapshot span = {int(span)} days")
    ax.set_ylim(0, span + 1.2)
    ax.set_ylabel("Raw rows / unique transfers")
    _date_axis(ax)
    _token_legend(fig, data)
    writer.save(fig, "fig02_snapshot_replication", title="Replication of transfers across TGS snapshots",
                caption="Ratio between raw edge-list rows and reconstructed unique transfers per calendar day. "
                        "Each TGS snapshot is a 7-day sliding window with a 1-day stride, so interior days are "
                        "repeated in 7 snapshots and boundary days in fewer. Every transfer key was found with "
                        "identical multiplicity in all snapshots covering its date (see the deduplication audit).",
                graph="raw edge lists (before aggregation)", window="daily, full period", weight="none",
                sources=[f"{t}/dedup_by_date.csv" for t in data.tokens] + ["summary/dedup_audit.csv"])


def _tail_row(data: RunData, token: str, variant: str, quantity: str) -> pd.Series | None:
    fits = data.table(token, "tail_fits")
    if fits.empty:
        return None
    match = fits.loc[(fits["variant"] == variant) & (fits["quantity"] == quantity)]
    return match.iloc[0] if len(match) and pd.notna(match.iloc[0].get("xmin")) else None


def _node_values(data: RunData, token: str, quantity: str, variant: str = "frequency") -> np.ndarray:
    if quantity == "undirected_degree":
        nodes = data.table(token, f"nodes/{FULL_PERIOD}_node_topology.csv.gz")
        return nodes["undirected_degree"].to_numpy(dtype=float)
    table = data.table(token, f"centralities/{FULL_PERIOD}__{variant}.csv.gz")
    return table[quantity].to_numpy(dtype=float)


def fig_degree_ccdf(data: RunData, writer: FigureWriter) -> None:
    panels = [("in_degree", "In-degree $k_{in}$"), ("out_degree", "Out-degree $k_{out}$"),
              ("in_strength", "In-strength $s_{in}$ (transfers)"),
              ("out_strength", "Out-strength $s_{out}$ (transfers)")]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.4))
    for ax, letter, (quantity, label) in zip(axes.flat, "abcd", panels):
        floor = 1.0
        for token in data.tokens:
            x, y = ccdf(_node_values(data, token, quantity))
            ax.scatter(x, y, s=4, marker=data.marker(token), color=data.color(token), alpha=0.75,
                       edgecolors="none", rasterized=len(x) > 3000)
            floor = min(floor, y.min())
            fit = _tail_row(data, token, "frequency", quantity)
            if fit is not None:
                grid = np.unique(np.geomspace(fit["xmin"], x.max(), 80).round())
                model = fit["tail_share"] * model_ccdf("power_law", grid, fit)
                keep = model >= 0.5 * y.min()  # never extrapolate the fit below the data
                ax.plot(grid[keep], model[keep], color=data.color(token), lw=1.0, ls=(0, (4, 2)), zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.4 * floor)
        ax.set_xlabel(label)
        ax.set_ylabel("$P(X \\geq x)$")
        ax.set_title(f"({letter}) {label.split(' $')[0]}")
    _token_legend(fig, data, extra=[Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 2)),
                                           label="power-law MLE fit, $x \\geq x_{min}$")])
    writer.save(fig, "fig03_degree_strength_ccdf", title="Degree and strength distributions",
                caption="Empirical complementary cumulative distributions on log-log axes. Dashed lines show the "
                        "discrete power-law maximum-likelihood fit above the KS-optimal x_min (Clauset et al., 2009), "
                        "scaled by the tail share. A straight-looking tail is not evidence of a power law; the "
                        "formal comparison with alternative distributions is in tail_fits.csv and Figure 4.",
                graph=GRAPH_BASE, window="full period",
                weight="transfer count (strength = number of transfers)",
                sources=[f"{t}/centralities/{FULL_PERIOD}__frequency.csv.gz; {t}/tail_fits.csv" for t in data.tokens])


MODEL_STYLES = {"power_law": ("power law", INK, (0, (4, 2)), 1.0),
                "lognormal": ("log-normal", SECONDARY, (0, (6, 1.5, 1.5, 1.5)), 1.0),
                "cutoff": ("power law with cutoff", INK, (0, (1, 1.2)), 1.2),
                "exponential": ("exponential", MUTED, "-", 0.9)}


def fig_tail_models(data: RunData, writer: FigureWriter) -> None:
    quantities = [("in_degree", "In-degree"), ("out_degree", "Out-degree"), ("in_strength", "In-strength")]
    rows = len(data.tokens)
    fig, axes = plt.subplots(rows, len(quantities), figsize=(DOUBLE, 1.85 * rows + 0.35), squeeze=False)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r, token in enumerate(data.tokens):
        for c, (quantity, label) in enumerate(quantities):
            ax = axes[r, c]
            fit = _tail_row(data, token, "frequency", quantity)
            x, y = ccdf(_node_values(data, token, quantity))
            ax.set_title(f"({next(letters)}) {token}: {label.lower()}")
            if fit is None:
                ax.text(0.5, 0.5, "not assessed", transform=ax.transAxes, ha="center", color=SECONDARY)
                continue
            tail = x >= fit["xmin"]
            ax.scatter(x[tail], y[tail] / fit["tail_share"], s=5, marker=data.marker(token),
                       color=data.color(token), alpha=0.8, edgecolors="none")
            grid = np.unique(np.geomspace(fit["xmin"], x.max(), 120).round())
            for model, (name, color, style, width) in MODEL_STYLES.items():
                try:
                    values = model_ccdf(model, grid, fit)
                except (KeyError, ValueError, TypeError):
                    continue
                ax.plot(grid, values, color=color, ls=style, lw=width)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(max(0.5 / fit["n_tail"], 1e-7), 1.6)
            ax.set_xlabel(f"{label} $x$")
            if c == 0:
                ax.set_ylabel("$P(X \\geq x \\mid X \\geq x_{min})$")
            p = fit.get("gof_p")
            ax.text(0.97, 0.96, f"$\\hat{{\\alpha}}$ = {fit['alpha']:.2f}\n$x_{{min}}$ = {fit['xmin']:.0f}\n"
                                f"$p_{{GOF}}$ = {p:.2f}" if pd.notna(p) else f"$\\hat{{\\alpha}}$ = {fit['alpha']:.2f}",
                    transform=ax.transAxes, fontsize=6.5, color=SECONDARY, va="top", ha="right")
    handles = [Line2D([], [], color=color, ls=style, lw=width, label=name)
               for name, color, style, width in MODEL_STYLES.values()]
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    writer.save(fig, "fig04_tail_model_comparison", title="Tail model comparison",
                caption="Conditional CCDF of the tail (x >= x_min) with four maximum-likelihood models fitted on "
                        "the same tail. Bootstrap goodness-of-fit p-values (p_GOF >= 0.1: power law plausible) and "
                        "likelihood-ratio tests are reported in tail_fits.csv; the interpretation column there "
                        "uses heavy-tailed language only.",
                graph=GRAPH_BASE + "; undirected degree on the " + PROJECTION, window="full period",
                weight="transfer count", sources=[f"{t}/tail_fits.csv" for t in data.tokens])


def fig_volume_ccdf(data: RunData, writer: FigureWriter) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4))
    for ax, letter, (quantity, label) in zip(axes, "ab", (("in_strength", "In-strength"),
                                                          ("out_strength", "Out-strength"))):
        left = 1.0
        for token in data.tokens:
            values = _node_values(data, token, quantity, "raw_volume")
            total = np.nansum(values)
            # The axis starts at the 0.5th percentile; a few dust-sized amounts would stretch it.
            left = min(left, float(np.nanquantile(values[values > 0], 0.005) / total))
            x, y = ccdf(values / total)
            ax.scatter(x, y, s=3, marker=data.marker(token), color=data.color(token), alpha=0.6,
                       edgecolors="none", rasterized=True)
            fit = _tail_row(data, token, "raw_volume", quantity)
            if fit is not None:
                grid = np.geomspace(fit["xmin"], np.nanmax(values), 80)
                ax.plot(grid / total, fit["tail_share"] * model_ccdf("power_law", grid, fit),
                        color=data.color(token), lw=0.9, ls=(0, (4, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(left=left)
        ax.set_xlabel(f"{label} as share of total volume")
        ax.set_ylabel("$P(X \\geq x)$")
        ax.set_title(f"({letter}) {label} (raw token volume)")
    _token_legend(fig, data, extra=[Line2D([], [], color=MUTED, ls=(0, (4, 2)), label="power-law fit")])
    writer.save(fig, "fig05_volume_strength_ccdf", title="Volume strength distributions",
                caption="CCDF of address in- and out-strength in the raw-volume graph, expressed as a share of the "
                        "token's total transferred volume so that tokens with different units can share an axis "
                        "(x-axis from the 0.5th percentile of positive values). "
                        "Dashed: continuous power-law fit above x_min. Raw token amounts are on-chain integer "
                        "units and are not comparable in absolute terms across tokens.",
                graph=GRAPH_BASE + "; pairs with zero total volume removed", window="full period",
                weight="raw token volume", sources=[f"{t}/centralities/{FULL_PERIOD}__raw_volume.csv.gz"
                                                    for t in data.tokens])


def fig_concentration(data: RunData, writer: FigureWriter) -> None:
    panels = [("raw_volume", "out_strength", "Volume sent"), ("raw_volume", "in_strength", "Volume received"),
              ("frequency", "out_degree", "Out-degree"), ("frequency", "in_degree", "In-degree")]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.15), sharey=True)
    for ax, letter, (variant, quantity, label) in zip(axes, "abcd", panels):
        for token in data.tokens:
            curve = data.table(token, "concentration_curve")
            curve = curve.loc[(curve["variant"] == variant) & (curve["quantity"] == quantity)]
            stats_row = data.table(token, "concentration")
            stats_row = stats_row.loc[(stats_row["window"] == FULL_PERIOD) & (stats_row["variant"] == variant)
                                      & (stats_row["quantity"] == quantity)]
            gini = stats_row["gini"].iloc[0] if len(stats_row) else np.nan
            ax.plot(curve["top_fraction"], curve["share_held"], color=data.color(token), lw=1.2,
                    label=f"G = {gini:.2f}")
            ax.plot(curve["top_fraction"].iloc[::25], curve["share_held"].iloc[::25], lw=0,
                    marker=data.marker(token), color=data.color(token), markersize=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.4)
        equality = np.geomspace(1e-6, 1, 200)
        ax.plot(equality, equality, color=MUTED, lw=0.7)
        ax.set_xscale("log")
        ax.set_xlim(left=max(1e-5, ax.get_xlim()[0]), right=1.05)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Top fraction of addresses")
        ax.set_title(f"({letter}) {label}")
        ax.legend(loc="upper left", handlelength=1.0, fontsize=6.5)
    axes[0].set_ylabel("Share of total held")
    _token_legend(fig, data)
    writer.save(fig, "fig06_concentration_curves", title="Concentration of activity",
                caption="Share of total volume or degree held by the top fraction of addresses (log x-axis); the "
                        "grey curve marks perfect equality (share held = fraction of addresses). Legend values are Gini coefficients. Concentration "
                        "describes structural activity only and does not identify the role of any address.",
                graph=GRAPH_BASE, window="full period", weight="raw token volume (a, b); unweighted degree (c, d)",
                sources=[f"{t}/concentration_curve.csv; {t}/concentration.csv" for t in data.tokens])


def fig_temporal_structure(data: RunData, writer: FigureWriter) -> None:
    panels: list[tuple[str, str, Callable[[str], pd.DataFrame], str, bool]] = [
        ("nodes", "Active addresses", lambda t: _monthly(data, t, "topology"), "nodes", True),
        ("edges", "Directed ties", lambda t: _monthly(data, t, "topology"), "edges", True),
        ("density", "Density", lambda t: _monthly(data, t, "topology"), "density", True),
        ("reciprocity", "Reciprocity", lambda t: _monthly(data, t, "topology"), "reciprocity", False),
        ("lscc", "LSCC share", lambda t: _monthly(data, t, "topology"), "largest_scc_share", False),
        ("transitivity", "Global clustering", lambda t: _monthly(data, t, "topology"), "transitivity_global", False),
        ("modularity", "Leiden modularity Q", lambda t: _monthly(data, t, "communities", variant="frequency",
                                                                 algorithm="leiden"), "modularity", False),
        ("gini", "Gini, volume sent", lambda t: _monthly(data, t, "concentration", variant="raw_volume",
                                                         quantity="out_strength"), "gini", False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 3.7), sharex=True)
    for ax, letter, (_, label, loader, column, log) in zip(axes.flat, "abcdefgh", panels):
        for token in data.tokens:
            table = loader(token)
            if table.empty or column not in table:
                continue
            _windowed_line(ax, data, token, table["mid"], table[column], table["partial"])
        if log:
            _log_axis(ax)
        ax.set_title(f"({letter}) {label}")
        _date_axis(ax)
    _token_legend(fig, data, extra=[_partial_handle()])
    writer.save(fig, "fig07_temporal_structure", title="Monthly evolution of network structure",
                caption="Structural descriptors of disjoint calendar-month graphs. Hollow markers denote months "
                        "with less than 80% calendar coverage (first and last months), whose size is not "
                        "comparable. Clustering and modularity are computed on the undirected projection; "
                        "modularity uses Leiden at resolution 1 on transfer-count weights; Gini refers to "
                        "address out-strength in the raw-volume graph.",
                graph=GRAPH_BASE + "; clustering/modularity on the " + PROJECTION, window="calendar months",
                weight="unweighted topology (a-f), transfer count (g), raw volume (h)",
                sources=[f"{t}/topology.csv; {t}/communities.csv; {t}/concentration.csv" for t in data.tokens])


def fig_small_world(data: RunData, writer: FigureWriter) -> None:
    panels = [("transitivity_global_ratio", "Clustering ratio $C / C_{null}$"),
              ("average_path_length_ratio", "Path-length ratio $L / L_{null}$"),
              ("sigma_transitivity_global", "Small-world index $\\sigma$")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2), sharex=True)
    for ax, letter, (column, label) in zip(axes, "abc", panels):
        values = []
        for token in data.tokens:
            table = _monthly(data, token, "topology")
            if column not in table:
                continue
            _windowed_line(ax, data, token, table["mid"], table[column], table["partial"])
            values.extend(table[column].dropna().tolist())
        _ref_line(ax, 1.0, "null expectation")
        if values and max(values) / max(min(values), 1e-9) > 20:
            ax.set_yscale("log")
        ax.set_title(f"({letter}) {label}")
        _date_axis(ax)
    _token_legend(fig, data, extra=[_partial_handle()])
    writer.save(fig, "fig08_small_world", title="Comparison with degree-preserving null graphs",
                caption="Observed global clustering and average shortest-path length of each monthly undirected "
                        "projection divided by the mean over 20 degree-preserving rewired graphs (10 swaps per "
                        "edge); sigma = (C/C_null)/(L/L_null). Path lengths are measured on the largest connected "
                        "component, exactly up to 5,000 nodes and from 500 seeded sources above. A small-world "
                        "reading needs sigma clearly above 1 with L/L_null close to 1.",
                graph=PROJECTION + ", unweighted", window="calendar months", weight="none (topology)",
                sources=[f"{t}/topology.csv; {t}/null_samples.csv" for t in data.tokens])


def fig_rich_club(data: RunData, writer: FigureWriter) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    for token in data.tokens:
        table = data.table(token, "rich_club")
        if table.empty:
            continue
        table = table.loc[table["window"] == FULL_PERIOD]
        band = 2 * table["phi_null_sd"] / table["phi_null_mean"]
        ax.fill_between(table["k"], 1 - band, 1 + band, color=data.color(token), alpha=0.12, lw=0)
        ax.plot(table["k"], table["rho"], color=data.color(token), marker=data.marker(token), markersize=3,
                markeredgecolor=SURFACE, markeredgewidth=0.4)
    _ref_line(ax, 1.0)
    ax.set_xscale("log")
    low, high = ax.get_ylim()
    if high / max(low, 1e-9) > 20:
        _log_axis(ax)
    ax.set_xlabel("Degree threshold $k$")
    ax.set_ylabel("$\\rho(k) = \\phi(k) / \\phi_{null}(k)$")
    _token_legend(fig, data)
    writer.save(fig, "fig09_rich_club", title="Normalised rich-club coefficient",
                caption="Rich-club coefficient phi(k) of addresses with degree above k, divided by its mean over "
                        "the degree-preserving null graphs. Shaded bands: +-2 null standard deviations relative to "
                        "the null mean. rho > 1 indicates that high-degree addresses are more interconnected than "
                        "expected from the degree sequence alone.",
                graph=PROJECTION + ", unweighted", window="full period", weight="none (topology)",
                sources=[f"{t}/rich_club.csv" for t in data.tokens])


def fig_bowtie(data: RunData, writer: FigureWriter) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE, 0.55 + 0.42 * len(data.tokens)))
    for row, token in enumerate(data.tokens):
        topology = data.table(token, "topology")
        full = topology.loc[topology["window"] == FULL_PERIOD]
        if full.empty:
            continue
        left = 0.0
        for part in BOWTIE_ORDER:
            share = float(full[f"bowtie_{part.lower()}_share"].iloc[0])
            ax.barh(row, share, left=left, height=0.62, color=BOWTIE_COLORS[part], edgecolor=SURFACE,
                    linewidth=1.0)
            if share >= 0.07:
                ink = INK if part in {"CORE", "IN"} else SURFACE
                ax.text(left + share / 2, row, f"{100 * share:.0f}%", ha="center", va="center", fontsize=6.5,
                        color=ink)
            left += share
    ax.set_yticks(range(len(data.tokens)), data.tokens)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of addresses")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=BOWTIE_COLORS[p], label=BOWTIE_LABELS[p]) for p in BOWTIE_ORDER]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, handlelength=0.9, columnspacing=0.8,
               fontsize=6.5)
    writer.save(fig, "fig10_bowtie", title="Bow-tie structure",
                caption="Bow-tie decomposition of the directed graph around its largest strongly connected "
                        "component (core): IN reaches the core, OUT is reached from it, tendrils/tubes are the rest "
                        "of the largest weakly connected component, and disconnected nodes lie outside it.",
                graph=GRAPH_BASE, window="full period", weight="none (topology)",
                sources=[f"{t}/topology.csv" for t in data.tokens])


CORRELATION_MEASURES = ["in_degree_centrality", "out_degree_centrality", "in_strength", "out_strength",
                        "pagerank", "authority", "hub", "eigenvector_undirected_lcc", "betweenness",
                        "harmonic_in", "harmonic_out"]


def fig_centrality_correlation(data: RunData, writer: FigureWriter) -> None:
    n = len(data.tokens)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE, 2.75), squeeze=False)
    image = None
    for ax, letter, token in zip(axes[0], "abcdefgh", data.tokens):
        corr = data.table(token, "centrality_correlation")
        corr = corr.loc[corr["variant"] == "frequency"]
        matrix = (corr.pivot(index="measure_a", columns="measure_b", values="spearman")
                  .reindex(index=CORRELATION_MEASURES, columns=CORRELATION_MEASURES))
        values = matrix.to_numpy(dtype=float, copy=True)
        values[np.triu_indices_from(values, k=1)] = np.nan
        image = ax.imshow(values, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
        labels = [MEASURE_LABELS[m].replace(" closeness", "").replace(" (undirected)", " (undir.)")
                  .replace("Eigenvector", "Eigenvector") for m in CORRELATION_MEASURES]
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0][0] else [], fontsize=6)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"({letter}) {token}")
    if image is not None:
        bar = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.75, aspect=25, pad=0.02)
        bar.set_label("Spearman rank correlation")
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, color=AXIS)
    writer.save(fig, "fig11_centrality_correlation", title="Agreement between centrality measures",
                caption="Spearman rank correlations between node centralities, computed over the nodes where both "
                        "measures are defined (harmonic closeness is restricted to the candidate set when the graph "
                        "exceeds the exact-computation threshold; see centrality_status.csv). Betweenness and "
                        "harmonic closeness use distance = 1/log(1 + w).",
                graph=GRAPH_BASE, window="full period", weight="transfer count",
                sources=[f"{t}/centrality_correlation.csv" for t in data.tokens])


def fig_rank_stability(data: RunData, writer: FigureWriter) -> None:
    panels = [("frequency", "pagerank", "PageRank (transfer count)"),
              ("frequency", "betweenness", "Betweenness (transfer count)"),
              ("raw_volume", "out_strength", "Volume sent")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2), sharey=True, sharex=True)
    k = None
    for ax, letter, (variant, measure, label) in zip(axes, "abc", panels):
        for token in data.tokens:
            table = data.table(token, "rank_stability")
            if table.empty:
                continue
            table = table.loc[(table["variant"] == variant) & (table["measure"] == measure)]
            windows = data.windows(token).set_index("window")
            order = list(windows.index)
            current = [order[order.index(w) + 1] for w in table["previous_window"]]
            x = windows.loc[current, "mid"].to_numpy()
            partial = (windows.loc[current, "partial"].to_numpy() | table["previous_partial"].to_numpy())
            _windowed_line(ax, data, token, pd.Series(x), table["topk_jaccard"].reset_index(drop=True),
                           pd.Series(partial))
            k = int(table["k"].iloc[0]) if len(table) else k
        ax.set_ylim(0, 1)
        ax.set_title(f"({letter}) {label}")
        _date_axis(ax)
    axes[0].set_ylabel(f"Top-{k or 100} Jaccard with previous month")
    _token_legend(fig, data, extra=[_partial_handle()])
    writer.save(fig, "fig12_rank_stability", title="Persistence of the most central addresses",
                caption="Jaccard overlap between the top-k addresses of consecutive months for three centrality "
                        "measures (Kendall's tau on the union of both top-k sets is in rank_stability.csv). Hollow "
                        "markers involve a partial month.",
                graph=GRAPH_BASE, window="consecutive calendar months", weight="transfer count (a, b); raw volume (c)",
                sources=[f"{t}/rank_stability.csv" for t in data.tokens])


def fig_turnover(data: RunData, writer: FigureWriter) -> None:
    panels = [("retained_share_of_previous", "Addresses retained from previous month"),
              ("repeated_share_of_previous", "Ties repeated from previous month")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), sharex=True)
    for ax, letter, (column, label) in zip(axes, "ab", panels):
        for token in data.tokens:
            table = data.table(token, "turnover")
            if table.empty:
                continue
            windows = data.windows(token).set_index("window")
            order = list(windows.index)
            current = [order[order.index(w) + 1] for w in table["previous_window"]]
            partial = windows.loc[current, "partial"].to_numpy() | table["previous_partial"].to_numpy()
            _windowed_line(ax, data, token, pd.Series(windows.loc[current, "mid"].to_numpy()),
                           table[column].reset_index(drop=True), pd.Series(partial))
        ax.set_ylim(0, None)
        ax.set_ylabel("Share")
        ax.set_title(f"({letter}) {label}")
        _date_axis(ax)
    _token_legend(fig, data, extra=[_partial_handle()])
    writer.save(fig, "fig13_turnover", title="Address and tie turnover",
                caption="Share of the previous month's active addresses (a) and directed ties (b) that are observed "
                        "again in the following month.",
                graph=GRAPH_BASE, window="consecutive calendar months", weight="none",
                sources=[f"{t}/turnover.csv" for t in data.tokens])


def fig_communities(data: RunData, writer: FigureWriter) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.2))
    ax = axes[0, 0]
    for token in data.tokens:
        sizes = data.table(token, "community_sizes")
        sizes = sizes.loc[(sizes["window"] == FULL_PERIOD) & (sizes["variant"] == "frequency")
                          & (sizes["algorithm"] == "leiden")]
        x, y = ccdf(sizes["size"].to_numpy())
        ax.plot(x, y, color=data.color(token), marker=data.marker(token), markersize=3, lw=1.0,
                markeredgecolor=SURFACE, markeredgewidth=0.4, drawstyle="steps-post")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Community size (addresses)")
    ax.set_ylabel("$P(S \\geq s)$")
    ax.set_title("(a) Leiden community sizes")
    for ax, letter, column, label, log in ((axes[0, 1], "b", "modularity", "Modularity $Q_{\\gamma}$", False),
                                           (axes[1, 0], "c", "communities", "Number of communities", True)):
        for token in data.tokens:
            sweep = data.table(token, "resolution_sweep")
            sweep = sweep.loc[sweep["variant"] == "frequency"]
            ax.plot(sweep["resolution"], sweep[column], color=data.color(token), marker=data.marker(token),
                    markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sweep["resolution"], [f"{v:g}" for v in sweep["resolution"]])
        ax.minorticks_off()
        if log:
            _log_axis(ax)
        ax.set_xlabel("Resolution $\\gamma$")
        ax.set_title(f"({letter}) {label} vs resolution")
    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    for position, token in enumerate(data.tokens):
        stability = data.table(token, "seed_stability")
        stability = stability.loc[stability["variant"] == "frequency", "ari"].dropna()
        if stability.empty:
            continue
        ax.boxplot(stability, positions=[position], widths=0.45, showfliers=False,
                   medianprops={"color": INK, "lw": 1.0}, boxprops={"color": AXIS, "lw": 0.7},
                   whiskerprops={"color": AXIS, "lw": 0.7}, capprops={"color": AXIS, "lw": 0.7})
        ax.scatter(position + rng.uniform(-0.12, 0.12, len(stability)), stability, s=7,
                   marker=data.marker(token), color=data.color(token), edgecolors="none", alpha=0.8, zorder=3)
    ax.set_xticks(range(len(data.tokens)), data.tokens)
    ax.set_ylabel("Pairwise ARI across seeds")
    ax.set_title("(d) Leiden seed stability")
    _token_legend(fig, data)
    writer.save(fig, "fig14_communities", title="Community structure",
                caption="Leiden communities of the weighted undirected projection (transfer-count weights). "
                        "(a) CCDF of community sizes at resolution 1. (b, c) Resolution sweep with a fixed seed. "
                        "(d) Adjusted Rand index between Leiden partitions obtained with different seeds. "
                        "Communities are algorithmic groupings of addresses with dense mutual transfers, not "
                        "real-world groups.",
                graph=PROJECTION, window="full period", weight="transfer count",
                sources=[f"{t}/community_sizes.csv; {t}/resolution_sweep.csv; {t}/seed_stability.csv"
                         for t in data.tokens])


def fig_community_persistence(data: RunData, writer: FigureWriter) -> None:
    panels = [("jaccard", "All addresses"), ("jaccard_shared", "Addresses active in both months")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), sharey=True, sharex=True)
    for ax, letter, (column, label) in zip(axes, "ab", panels):
        for token in data.tokens:
            table = data.table(token, "community_persistence")
            if table.empty:
                continue
            table = table.loc[(table["variant"] == "frequency") & (table["algorithm"] == "leiden")]
            summary = table.groupby(["previous_window", "previous_partial"])[column].median().reset_index()
            windows = data.windows(token).set_index("window")
            order = list(windows.index)
            current = [order[order.index(w) + 1] for w in summary["previous_window"]]
            partial = windows.loc[current, "partial"].to_numpy() | summary["previous_partial"].to_numpy()
            _windowed_line(ax, data, token, pd.Series(windows.loc[current, "mid"].to_numpy()),
                           summary[column], pd.Series(partial))
        ax.set_ylim(0, 1)
        ax.set_title(f"({letter}) {label}")
        _date_axis(ax)
    axes[0].set_ylabel("Median best-match Jaccard")
    _token_legend(fig, data, extra=[_partial_handle()])
    writer.save(fig, "fig15_community_persistence", title="Persistence of communities",
                caption="For every Leiden community with at least 10 addresses in a month, the Jaccard similarity "
                        "with its best-matching community in the next month; the median over communities is shown. "
                        "(b) restricts both communities to addresses active in both months, separating community "
                        "change from address turnover.",
                graph=PROJECTION, window="consecutive calendar months", weight="transfer count",
                sources=[f"{t}/community_persistence.csv" for t in data.tokens])


def fig_quotient(data: RunData, writer: FigureWriter, token: str, index: int) -> None:
    sizes = data.table(token, "community_sizes")
    sizes = sizes.loc[(sizes["window"] == FULL_PERIOD) & (sizes["variant"] == "frequency")
                      & (sizes["algorithm"] == "leiden")]
    edges = data.table(token, "community_quotient")
    edges = edges.loc[(edges["variant"] == "frequency") & (edges["algorithm"] == "leiden")]
    top = data.config.figures.quotient_communities
    shown = set(sizes.nsmallest(top, "community")["community"])
    total = sizes["size"].sum()
    graph = nx.Graph()
    for row in sizes.itertuples():
        node = row.community if row.community in shown else "other"
        graph.add_node(node)
        graph.nodes[node]["size"] = graph.nodes[node].get("size", 0) + row.size
    for row in edges.itertuples():
        a = row.community_a if row.community_a in shown else "other"
        b = row.community_b if row.community_b in shown else "other"
        if a == b:
            continue
        weight = graph.edges[a, b]["weight"] + row.weight if graph.has_edge(a, b) else row.weight
        graph.add_edge(a, b, weight=weight)
    for u, v, attrs in graph.edges(data=True):
        attrs["layout_weight"] = math.log1p(attrs["weight"])
    position = nx.spring_layout(graph, weight="layout_weight", seed=data.config.seed, iterations=300,
                                k=1.6 / math.sqrt(max(len(graph), 1)))
    fig, ax = plt.subplots(figsize=(SINGLE, 3.1))
    weights = np.array([math.log1p(a["weight"]) for _, _, a in graph.edges(data=True)])
    if len(weights):
        scaled = 0.2 + 1.8 * (weights - weights.min()) / max(np.ptp(weights), 1e-9)
        nx.draw_networkx_edges(graph, position, ax=ax, width=scaled, edge_color=MUTED, alpha=0.35)
    color = data.color(token)
    radius = {n: math.sqrt(25 + 900 * math.sqrt(graph.nodes[n]["size"] / total)) for n in graph}
    nx.draw_networkx_nodes(graph, position, ax=ax, node_size=[radius[n] ** 2 for n in graph],
                           node_color=[AXIS if n == "other" else color for n in graph],
                           edgecolors=SURFACE, linewidths=0.8)
    for node, (x, y) in position.items():
        share = 100 * graph.nodes[node]["size"] / total
        name = "other" if node == "other" else f"C{node}"
        if share >= 4 and node != "other":
            ax.annotate(f"{name}\n{share:.1f}%", (x, y), ha="center", va="center", fontsize=5.5, color=SURFACE)
        else:  # small nodes: label below the marker, in ink, so it stays legible
            ax.annotate(f"{name} {share:.1f}%", (x, y), xytext=(0, -radius[node] / 2 - 2),
                        textcoords="offset points", ha="center", va="top", fontsize=5.5, color=INK)
    ax.set_axis_off()
    writer.save(fig, f"fig16_{index}_community_quotient_{token}",
                title=f"Community quotient graph ({token})",
                caption=f"{token}: the {top} largest Leiden communities of the weighted undirected projection as "
                        "nodes (area proportional to the share of addresses, label = share), all remaining "
                        "communities merged into 'other' for display only. Edge width is proportional to "
                        "log(1 + total transfers between communities). Layout: seeded spring layout; distances "
                        "between nodes carry no meaning.",
                graph=PROJECTION, window="full period", weight="transfer count",
                sources=[f"{token}/community_sizes.csv; {token}/community_quotient.csv"])


def _grouped_layout(graph: nx.Graph, group: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    """Place groups with a spring layout of the group graph, then each group's nodes in its own disc."""
    members: dict[Any, list[str]] = {}
    for node in sorted(graph):
        members.setdefault(group[node], []).append(node)
    between = nx.Graph()
    between.add_nodes_from(members)
    for u, v in graph.edges():
        a, b = group[u], group[v]
        if a != b:
            between.add_edge(a, b, weight=between.edges[a, b]["weight"] + 1 if between.has_edge(a, b) else 1)
    largest = max(len(m) for m in members.values())
    radius = {g: 0.06 + 0.32 * math.sqrt(len(m) / largest) for g, m in members.items()}
    centres = nx.spring_layout(between, seed=seed, iterations=300, weight=None,
                               k=2.2 / math.sqrt(max(len(members), 1)))
    position: dict[str, np.ndarray] = {}
    for index, (g, nodes) in enumerate(members.items()):
        local = (nx.spring_layout(graph.subgraph(nodes), seed=seed + index, iterations=120)
                 if len(nodes) > 1 else {nodes[0]: np.zeros(2)})
        coords = np.array([local[n] for n in nodes], dtype=float)
        coords -= coords.mean(axis=0)
        spread = np.abs(coords).max() or 1.0
        for node, xy in zip(nodes, coords / spread):
            position[node] = centres[g] + radius[g] * xy
    return position


def fig_network_sample(data: RunData, writer: FigureWriter, token: str, index: int) -> None:
    import matplotlib.patheffects as effects

    fc = data.config.figures
    pairs = data.table(token, f"graphs/{FULL_PERIOD}_pairs.csv.gz")
    members = data.table(token, f"communities/{FULL_PERIOD}__frequency__leiden.csv.gz")
    cent = data.table(token, f"centralities/{FULL_PERIOD}__frequency.csv.gz")
    graph = nx.Graph()
    for row in pairs.itertuples():
        if graph.has_edge(row.source, row.destination):
            graph[row.source][row.destination]["weight"] += row.event_count
        else:
            graph.add_edge(row.source, row.destination, weight=row.event_count)
    community = dict(zip(members["node"], members["community"]))
    strength = dict(graph.degree(weight="weight"))
    sizes = members["community"].value_counts()
    coloured = list(range(fc.drawing_communities))
    # Grey context: the next largest communities, so the drawing shows separate groups.
    context = list(range(fc.drawing_communities, fc.quotient_communities))
    shown = coloured + context
    shares = np.array([sizes.get(c, 0) for c in shown], dtype=float)
    quota = np.maximum(6, np.round(fc.drawing_max_nodes * shares / max(shares.sum(), 1))).astype(int)
    by_community: dict[int, list[str]] = {}
    for node in graph:
        by_community.setdefault(community.get(node, -1), []).append(node)
    selected: list[str] = []
    for c, q in zip(shown, quota):
        nodes = sorted(by_community.get(c, []), key=lambda n: (-strength[n], n))
        selected.extend(nodes[:q])
    sub = graph.subgraph(selected).copy()
    sub = sub.subgraph(max(nx.connected_components(sub), key=len)).copy()
    position = _grouped_layout(sub, {n: community[n] for n in sub}, data.config.seed)
    pagerank = dict(zip(cent["node"], cent["pagerank"]))

    fig, ax = plt.subplots(figsize=(DOUBLE * 0.66, 3.7))
    intra = [(u, v) for u, v in sub.edges() if community[u] == community[v]]
    inter = [(u, v) for u, v in sub.edges() if community[u] != community[v]]
    nx.draw_networkx_edges(sub, position, edgelist=inter, ax=ax, width=0.2, edge_color=AXIS, alpha=0.35)
    nx.draw_networkx_edges(sub, position, edgelist=intra, ax=ax, width=0.3, edge_color=MUTED, alpha=0.45)
    values = np.array([pagerank.get(n, 0.0) for n in sub])
    scale = np.sqrt(values / values.max()) if values.max() > 0 else np.zeros(len(values))
    size_of = dict(zip(sub.nodes, 4 + 70 * scale))
    grey = [n for n in sub if community[n] not in coloured]
    nx.draw_networkx_nodes(sub, position, nodelist=grey, ax=ax, node_size=[size_of[n] for n in grey],
                           node_color=AXIS, edgecolors=SURFACE, linewidths=0.2)
    handles = []
    for c, colour, marker in zip(coloured, COMMUNITY_COLORS, COMMUNITY_MARKERS):
        nodes = [n for n in sub if community[n] == c]
        if not nodes:
            continue
        nx.draw_networkx_nodes(sub, position, nodelist=nodes, ax=ax, node_size=[size_of[n] for n in nodes],
                               node_color=colour, node_shape=marker, edgecolors=SURFACE, linewidths=0.3)
        share = 100 * sizes.get(c, 0) / sizes.sum()
        handles.append(Line2D([], [], lw=0, marker=marker, color=colour, markersize=5,
                              label=f"C{c} ({share:.1f}% of addresses)"))
    handles.append(Line2D([], [], lw=0, marker="o", color=AXIS, markersize=5,
                          label=f"C{len(coloured)}-C{fc.quotient_communities - 1} (grey)"))
    leaders = sorted(sub, key=lambda n: (-pagerank.get(n, 0.0), n))[:5]
    nx.draw_networkx_nodes(sub, position, nodelist=leaders, ax=ax, node_size=[size_of[n] for n in leaders],
                           node_color="none", edgecolors=INK, linewidths=0.9)
    halo = [effects.withStroke(linewidth=2.0, foreground=SURFACE)]
    for rank, node in enumerate(leaders, start=1):
        ax.annotate(str(rank), position[node], xytext=(3, 3), textcoords="offset points", fontsize=6.5,
                    fontweight="bold", color=INK, path_effects=halo)
        handles.append(Line2D([], [], lw=0, marker=f"${rank}$", color=INK, markersize=5,
                              label=f"{_short_address(node)} (PageRank rank {rank})"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5,
              title="Leiden community / top PageRank", title_fontsize=6.5, alignment="left")
    ax.set_axis_off()
    writer.save(fig, f"fig17_{index}_network_sample_{token}", title=f"Sampled transfer network ({token})",
                caption=f"{token}: induced subgraph (largest connected part) of {sub.number_of_nodes():,} of "
                        f"{graph.number_of_nodes():,} addresses, sampled deterministically as the highest-strength "
                        f"addresses of the {fc.quotient_communities} largest Leiden communities in proportion to "
                        f"their size; the {fc.drawing_communities} largest are coloured. Communities are placed by "
                        "a seeded spring layout of the community graph and each community's addresses inside its "
                        "own disc, so positions between groups carry no metric meaning. Node area is proportional "
                        "to the square root of PageRank; the five highest-PageRank addresses are numbered "
                        "(truncated identifiers, not identities).",
                graph=PROJECTION, window="full period", weight="transfer count",
                sources=[f"{token}/graphs/{FULL_PERIOD}_pairs.csv.gz; {token}/communities/"
                         f"{FULL_PERIOD}__frequency__leiden.csv.gz; {token}/centralities/{FULL_PERIOD}__frequency.csv.gz"])


PAIR_COLORS = ["#008300", "#6250d6", "#e34948"]
PAIR_MARKERS = ["D", "v", "P"]


def _cross(data: RunData, name: str) -> pd.DataFrame:
    path = data.run_dir / "cross_token" / name
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def fig_cross_token_overlap(data: RunData, writer: FigureWriter) -> None:
    monthly = _cross(data, "overlap_monthly.csv")
    placement = _cross(data, "shared_address_placement.csv.gz")
    if monthly.empty:
        raise ValueError("no cross-token overlap tables (needs >= 2 tokens)")
    pairs = list(dict.fromkeys(monthly["pair"]))
    style = {p: (PAIR_COLORS[i % 3], PAIR_MARKERS[i % 3]) for i, p in enumerate(pairs)}
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    for pair in pairs:
        table = monthly.loc[monthly["pair"] == pair].copy()
        table["mid"] = pd.to_datetime(table["window"]) + pd.Timedelta(days=15)
        color, marker = style[pair]
        for ax, column in ((axes[0], "address_jaccard"), (axes[1], "ari")):
            if column not in table:
                continue
            valid = table.dropna(subset=[column])
            ax.plot(valid["mid"], valid[column], color=color, lw=1.0)
            full = ~valid["partial"].astype(bool)
            ax.scatter(valid.loc[full, "mid"], valid.loc[full, column], s=11, marker=marker, color=color,
                       edgecolors=SURFACE, linewidths=0.4, zorder=3)
            ax.scatter(valid.loc[~full, "mid"], valid.loc[~full, column], s=11, marker=marker,
                       facecolors=SURFACE, edgecolors=color, linewidths=0.8, zorder=3)
        if "ari_null_q95" in table:
            valid = table.dropna(subset=["ari_null_q95"])
            axes[1].plot(valid["mid"], valid["ari_null_q95"], color=color, lw=0.7, alpha=0.6, ls=(0, (1, 1.2)))
    _log_axis(axes[0])
    axes[0].set_title("(a) Address Jaccard, same month")
    axes[1].set_title("(b) Community agreement")
    axes[1].set_ylabel("ARI on shared addresses")
    _ref_line(axes[1], 0.0)
    for ax in axes[:2]:
        _date_axis(ax)
    ax = axes[2]
    positions, labels = [], []
    for i, pair in enumerate(pairs):
        for j, side in enumerate(("a", "b")):
            values = placement.loc[(placement["pair"] == pair) & (placement["side"] == side)
                                   & placement["shared"], "pagerank_percentile"]
            if values.empty:
                continue
            position = i * 2.6 + j
            color, marker = style[pair]
            ax.boxplot(values, positions=[position], widths=0.6, showfliers=False,
                       medianprops={"color": color, "lw": 1.4}, boxprops={"color": AXIS, "lw": 0.7},
                       whiskerprops={"color": AXIS, "lw": 0.7}, capprops={"color": AXIS, "lw": 0.7})
            token = placement.loc[(placement["pair"] == pair) & (placement["side"] == side), "token"].iloc[0]
            positions.append(position)
            labels.append(token)
    _ref_line(ax, 0.5)
    ax.set_xticks(positions, labels, rotation=90, fontsize=6.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("PageRank percentile")
    ax.set_title("(c) Shared addresses")
    handles = [Line2D([], [], color=style[p][0], marker=style[p][1], markersize=4, lw=1.0,
                      markeredgecolor=SURFACE, markeredgewidth=0.4, label=p) for p in pairs]
    handles += [Line2D([], [], color=MUTED, lw=0.7, ls=(0, (1, 1.2)), label="ARI null 95th percentile"),
                _partial_handle()]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    writer.save(fig, "fig18_cross_token_overlap", title="Addresses shared between token networks",
                caption="(a) Jaccard similarity of the address sets of two tokens' graphs in the same calendar month "
                        "(log scale). (b) Adjusted Rand index between the two tokens' Leiden partitions restricted to "
                        "shared addresses (months with at least 30 shared addresses); dotted lines give the 95th "
                        "percentile of 200 label permutations, so points above them indicate that shared addresses "
                        "co-locate in communities more than chance. (c) PageRank percentile, within each token's "
                        "common-period graph, of the addresses shared by the pair (boxes without outliers; the grey line is the median over all addresses, 0.5). Shared "
                        "addresses are identifiers active in both networks, not verified common actors.",
                graph=PROJECTION + " for communities; " + GRAPH_BASE + " for PageRank",
                window="common calendar months (a, b); common period of each pair (c)", weight="transfer count",
                sources=["cross_token/overlap_monthly.csv", "cross_token/shared_address_placement.csv.gz"])


def _diagonal_column_order(shares: pd.DataFrame) -> list[int]:
    """Order columns so that each row's best-matching column sits as close to the diagonal as possible."""
    named = [c for c in shares.columns if c >= 0]
    best_row = {c: int(np.nanargmax(shares[c].to_numpy())) for c in named}
    peak = {c: float(np.nanmax(shares[c].to_numpy())) for c in named}
    ordered = sorted(named, key=lambda c: (best_row[c], -peak[c], c))
    return ordered + [c for c in shares.columns if c < 0]


def fig_community_overlap(data: RunData, writer: FigureWriter) -> None:
    import string

    matrix = _cross(data, "community_overlap_matrix.csv")
    period = _cross(data, "overlap_common_period.csv")
    if matrix.empty:
        raise ValueError("no community overlap matrix")
    pairs = list(dict.fromkeys(matrix["pair"]))
    ncols = min(3, len(pairs))
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE if ncols > 1 else SINGLE, 2.45 * nrows + 0.35),
                             squeeze=False)
    blues = LinearSegmentedColormap.from_list("blue_ramp", ["#f4f8fd", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
    image = None
    for ax, letter, pair in zip(axes.flat, string.ascii_lowercase, pairs):
        table = matrix.loc[matrix["pair"] == pair]
        counts = table.pivot(index="community_a", columns="community_b", values="shared_addresses").fillna(0.0)
        rows = sorted(counts.index, key=lambda c: (c < 0, -counts.loc[c].sum(), c))
        counts = counts.loc[rows]
        shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        cols = _diagonal_column_order(shares)
        shares = shares[cols]
        image = ax.imshow(shares.to_numpy(dtype=float, copy=True), cmap=blues, vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
        token_a, token_b = table["token_a"].iloc[0], table["token_b"].iloc[0]
        ax.set_xticks(range(len(cols)), ["other" if c < 0 else f"C{c}" for c in cols], rotation=90, fontsize=5.5)
        ax.set_yticks(range(len(rows)), ["other" if r < 0 else f"C{r}" for r in rows], fontsize=5.5)
        ax.set_xlabel(f"{token_b} community", fontsize=7)
        ax.set_ylabel(f"{token_a} community", fontsize=7)
        info = period.loc[period["pair"] == pair]
        detail = f"n = {int(counts.to_numpy().sum()):,}"
        if len(info) and "ari" in info and pd.notna(info["ari"].iloc[0]):
            detail += f", ARI = {info['ari'].iloc[0]:.2f}"
        ax.set_title(f"({letter}) {pair}\n{detail}", fontsize=7.5, linespacing=1.3)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=1.5)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    if image is not None:
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, aspect=30, pad=0.015)
        bar.set_label("Share of the row community's shared addresses")
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, color=AXIS)
    writer.save(fig, "fig19_community_overlap", title="Community overlap between token networks",
                caption="For each pair of tokens, the addresses active in both networks over the pair's common "
                        "period are cross-tabulated by their Leiden community in each token's graph. Rows are the "
                        "communities of the first token holding the most shared addresses, columns those of the "
                        "second ('other' merges the rest); each cell is the share of the row community's shared "
                        "addresses that fall in the column community (rows sum to 1, common 0-1 scale). Columns are "
                        "ordered so that each row's best match lies near the diagonal: a dark diagonal means that "
                        "addresses grouped together in one token are also grouped together in the other. n is the "
                        "number of shared addresses; ARI is computed on all of them (overlap_common_period.csv). "
                        "Absolute counts are in community_overlap_matrix.csv.",
                graph=PROJECTION, window="common period of each pair", weight="transfer count",
                sources=["cross_token/community_overlap_matrix.csv", "cross_token/overlap_common_period.csv"])


FIGURES: list[Callable[[RunData, FigureWriter], None]] = [
    fig_activity, fig_snapshot_replication, fig_degree_ccdf, fig_tail_models, fig_volume_ccdf,
    fig_concentration, fig_temporal_structure, fig_small_world, fig_rich_club, fig_bowtie,
    fig_centrality_correlation, fig_rank_stability, fig_turnover, fig_communities, fig_community_persistence,
]


def make_all_figures(run_dir: Path, config: StudyAConfig, only: list[str] | None = None) -> pd.DataFrame:
    apply_style()
    data = RunData(run_dir, config, selected_tokens(run_dir))
    writer = FigureWriter(run_dir, config)
    jobs: list[tuple[str, Callable[[], None]]] = [(f.__name__, lambda f=f: f(data, writer)) for f in FIGURES]
    if len(data.tokens) > 1 and (run_dir / "cross_token").is_dir():
        jobs.append(("fig_cross_token_overlap", lambda: fig_cross_token_overlap(data, writer)))
        jobs.append(("fig_community_overlap", lambda: fig_community_overlap(data, writer)))
    for i, token in enumerate(data.tokens, start=1):
        jobs.append((f"fig_quotient_{token}", lambda t=token, i=i: fig_quotient(data, writer, t, i)))
        jobs.append((f"fig_network_sample_{token}", lambda t=token, i=i: fig_network_sample(data, writer, t, i)))
    status = []
    for name, job in jobs:
        if only and not any(name.startswith(prefix) for prefix in only):
            continue
        try:
            job()
            status.append({"figure_function": name, "status": "ok", "error": None})
        except Exception as exc:  # A broken figure is reported, the others are still produced.
            plt.close("all")
            logger.exception("Figure %s failed", name)
            status.append({"figure_function": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    writer.write_captions()
    table = pd.DataFrame(status)
    path = writer.dir / "figure_status.csv"
    if only and path.is_file() and not table.empty:  # partial rebuild: keep the other figures' status
        previous = pd.read_csv(path)
        previous = previous.loc[~previous["figure_function"].isin(table["figure_function"])]
        pd.concat([previous, table], ignore_index=True).to_csv(path, index=False)
    else:
        table.to_csv(path, index=False)
    return table
