"""Cross-token summary tables (CSV) and their LaTeX booktabs versions.

Every table here is derived only from the per-token CSV artifacts of a run, so
it can be regenerated with ``ACTION = "summarize"`` in ``main.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import pandas as pd

from .windows import FULL_PERIOD


def _read(run_dir: Path, token: str, name: str) -> pd.DataFrame:
    path = run_dir / token / f"{name}.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect(run_dir: Path, tokens: list[str], name: str) -> pd.DataFrame:
    frames = [_read(run_dir, t, name).assign(token=t) for t in tokens]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    return table[["token", *[c for c in table.columns if c != "token"]]]


def selected_tokens(run_dir: Path) -> list[str]:
    selection = pd.read_csv(run_dir / "selection" / "token_selection.csv")
    return selection.loc[selection["selected"], "token"].astype(str).tolist()


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) >= 1000):
        return f"{int(value):,}".replace(",", "{,}")
    if isinstance(value, float):
        if value != 0 and (abs(value) < 10 ** -digits or abs(value) >= 1e5):
            mantissa, exponent = f"{value:.2e}".split("e")
            return f"${mantissa}\\times10^{{{int(exponent)}}}$"
        return f"{value:.{digits}f}"
    return str(value).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def to_latex(table: pd.DataFrame, columns: dict[str, str], path: Path, caption: str, label: str,
             formatters: dict[str, Callable[[object], str]] | None = None) -> None:
    """Write a booktabs table with the given column -> header mapping (no jinja2 needed)."""
    formatters = formatters or {}
    available = [c for c in columns if c in table.columns]
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" + "r" * (len(available) - 1) + "}", "\\toprule",
             " & ".join(columns[c] for c in available) + " \\\\", "\\midrule"]
    for _, row in table.iterrows():
        cells = [formatters.get(c, _fmt)(row[c]) for c in available]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summaries(run_dir: Path, tokens: list[str] | None = None) -> None:
    tokens = tokens or selected_tokens(run_dir)
    summary, latex = run_dir / "summary", run_dir / "tables"
    summary.mkdir(parents=True, exist_ok=True)

    dedup = pd.DataFrame([json.loads((run_dir / t / "dedup_audit.json").read_text(encoding="utf-8"))
                          for t in tokens if (run_dir / t / "dedup_audit.json").is_file()])
    if not dedup.empty:
        dedup.to_csv(summary / "dedup_audit.csv", index=False)
        to_latex(dedup, {"token": "Token", "raw_rows": "Raw rows", "unique_events": "Unique transfers",
                         "raw_to_unique_ratio": "Ratio", "consistent_key_share": "Consistent keys",
                         "snapshot_span_days_mode": "Snapshot span (d)", "date_min": "First day",
                         "date_max": "Last day"},
                 latex / "dedup_audit.tex",
                 "Reconstruction of unique transfers from overlapping TGS snapshots. A key is consistent "
                 "when every snapshot covering its date contains it with the same multiplicity.",
                 "tab:dedup")

    topology = collect(run_dir, tokens, "topology")
    if not topology.empty:
        topology.to_csv(summary / "topology_all_windows.csv", index=False)
        full = topology.loc[topology["window"] == FULL_PERIOD]
        full.to_csv(summary / "topology_full_period.csv", index=False)
        to_latex(full, {"token": "Token", "nodes": "Addresses", "edges": "Directed ties",
                        "density": "Density", "reciprocity": "Reciprocity",
                        "largest_wcc_share": "LWCC share", "largest_scc_share": "LSCC share",
                        "assortativity_out_in": "$r_{\\mathrm{out,in}}$",
                        "assortativity_undirected": "$r$ (undir.)"},
                 latex / "network_overview.tex",
                 "Full-period directed transfer-count graphs after removing self-loops and the zero address.",
                 "tab:overview")
        to_latex(full, {"token": "Token", "transitivity_global": "$C_{\\mathrm{glob}}$",
                        "transitivity_global_null_mean": "$C_{\\mathrm{null}}$",
                        "transitivity_global_ratio": "$C/C_{\\mathrm{null}}$",
                        "average_path_length": "$L$", "average_path_length_null_mean": "$L_{\\mathrm{null}}$",
                        "average_path_length_ratio": "$L/L_{\\mathrm{null}}$",
                        "sigma_transitivity_global": "$\\sigma$", "diameter": "Diameter",
                        "max_core": "Max core"},
                 latex / "small_world.tex",
                 "Undirected projection (largest connected component for path lengths) versus "
                 "degree-preserving null graphs. Path lengths are sampled where stated in the run metadata.",
                 "tab:smallworld")

    tails = collect(run_dir, tokens, "tail_fits")
    if not tails.empty:
        tails.to_csv(summary / "tail_fits.csv", index=False)
        to_latex(tails, {"token": "Token", "variant": "Graph", "quantity": "Quantity", "xmin": "$x_{\\min}$",
                         "alpha": "$\\hat\\alpha$", "n_tail": "$n_{\\mathrm{tail}}$", "gof_p": "$p_{\\mathrm{GOF}}$",
                         "lr_vs_lognormal": "$\\mathcal{R}_{\\mathrm{LN}}$", "p_vs_lognormal": "$p$",
                         "lr_vs_exponential": "$\\mathcal{R}_{\\mathrm{Exp}}$", "p_vs_exponential": "$p$",
                         "lr_vs_cutoff": "$\\mathcal{R}_{\\mathrm{cut}}$", "p_vs_cutoff": "$p$"},
                 latex / "tail_fits.tex",
                 "Tail analysis (full period). $\\mathcal{R}<0$ favours the alternative; "
                 "power-law plausibility requires $p_{\\mathrm{GOF}}\\geq0.1$.", "tab:tails")

    communities = collect(run_dir, tokens, "communities")
    if not communities.empty:
        communities.to_csv(summary / "communities_all_windows.csv", index=False)
        full = communities.loc[communities["window"] == FULL_PERIOD]
        full.to_csv(summary / "communities_full_period.csv", index=False)
        to_latex(full, {"token": "Token", "variant": "Graph", "algorithm": "Algorithm",
                        "communities": "Communities", "modularity": "$Q$",
                        "largest_community_share": "Largest share", "top10_coverage": "Top-10 cover",
                        "ari_leiden_louvain": "ARI (L/L)"},
                 latex / "communities.tex",
                 "Community detection on the weighted undirected projection (full period, resolution 1).",
                 "tab:communities")

    concentration = collect(run_dir, tokens, "concentration")
    if not concentration.empty:
        concentration.to_csv(summary / "concentration_all_windows.csv", index=False)
        full = concentration.loc[concentration["window"] == FULL_PERIOD]
        full.to_csv(summary / "concentration_full_period.csv", index=False)
        to_latex(full.loc[full["variant"].isin(["frequency", "raw_volume"])],
                 {"token": "Token", "variant": "Graph", "quantity": "Quantity", "gini": "Gini",
                  "hhi": "HHI", "top1pct_share": "Top 1\\%", "top10pct_share": "Top 10\\%",
                  "addresses_for_half": "$N_{50\\%}$"},
                 latex / "concentration.tex",
                 "Concentration of degree and strength across addresses (full period).", "tab:concentration")

    overlap_path = run_dir / "cross_token" / "overlap_common_period.csv"
    if overlap_path.is_file() and overlap_path.stat().st_size > 1:
        overlap = pd.read_csv(overlap_path)
        overlap.to_csv(summary / "overlap_common_period.csv", index=False)
        to_latex(overlap, {"pair": "Pair", "days": "Days", "addresses_a": "$|A|$", "addresses_b": "$|B|$",
                           "shared_addresses": "Shared", "shared_share_of_a": "Share of $A$",
                           "shared_share_of_b": "Share of $B$", "ari": "ARI",
                           "ari_null_q95": "ARI null $q_{95}$", "nmi": "NMI"},
                 latex / "cross_token_overlap.tex",
                 "Addresses active in both token networks over their common period, and agreement of the "
                 "two Leiden partitions on those addresses (label-permutation null).", "tab:overlap")

    status = collect(run_dir, tokens, "analysis_status")
    if not status.empty:
        status.to_csv(summary / "analysis_status.csv", index=False)
        status.loc[status["status"] != "ok"].to_csv(summary / "analysis_failures.csv", index=False)
    centrality_status = collect(run_dir, tokens, "centrality_status")
    if not centrality_status.empty:
        centrality_status.to_csv(summary / "centrality_status.csv", index=False)
