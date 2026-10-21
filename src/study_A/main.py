"""Run the Study A ERC-20 graph and community analysis for selected tokens.

Edit ``TOKENS`` below or pass tokens on the command line. Temporal windows use
shared calendar-date boundaries across the selected files, making token-to-
token overlap comparisons time-aligned. Outputs are written under reports/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import logging
from pathlib import Path
import time
from typing import Any, Sequence

import networkx as nx
import pandas as pd
from math import comb
try:
    from .draw import draw_leiden, draw_louvain, draw_overlap
    from .graph import (
        WEIGHT_COLUMNS,
        analyze_windows,
        build_graphs_from_events,
        community_central_nodes,
        community_overlap,
        community_persistence,
        compute_communities,
        describe_graph,
        describe_projection,
        make_undirected_projection,
    )
    from .preprocessing import PREPROCESSING_VERSION, assign_windows, load_token_events
    from .utils import save_tables
except ImportError:
    from draw import draw_leiden, draw_louvain, draw_overlap
    from graph import (
        WEIGHT_COLUMNS,
        analyze_windows,
        build_graphs_from_events,
        community_central_nodes,
        community_overlap,
        community_persistence,
        compute_communities,
        describe_graph,
        describe_projection,
        make_undirected_projection,
    )
    from preprocessing import PREPROCESSING_VERSION, assign_windows, load_token_events
    from utils import save_tables



# Configure the token set here.
TOKENS = ["BITCOIN", "SWAP"]
# Choose only the edge-weight analyses needed for this run. The command-line
# option --edge-types can override this list, e.g. --edge-types frequency.
EDGE_TYPES_TO_ANALYZE = ["frequency", "raw_volume", "log1p_volume"]
NUM_DATE_WINDOWS = 10
SEED = 42
BETWEENNESS_K = 32
INCLUDE_EXPENSIVE_PATH_CENTRALITIES = False
TOP_CENTRAL_NODES_PER_COMMUNITY = 5
TOP_RANKED_NODES_TO_HIGHLIGHT = 5
MAX_DRAW_NODES = 400
OVERLAP_DRAW_TOP_K = 40

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "TGS_edgelists"


def _shared_date_windows(
    token_paths: dict[str, Path],
    requested_windows: int,
) -> tuple[dict[str, tuple[pd.Timestamp, pd.Timestamp]], pd.DataFrame, pd.DataFrame]:
    """Build one set of equal-duration calendar windows over the selected data."""
    if requested_windows < 1:
        raise ValueError("requested_windows must be at least 1")
    limits: list[dict[str, Any]] = []
    for token, path in token_paths.items():
        date_column = pd.read_csv(path, usecols=["date"])["date"]
        parsed_all = pd.to_datetime(date_column, errors="coerce")
        parsed = parsed_all.dropna()
        if parsed.empty:
            raise ValueError(f"{token} has no valid dates: {path}")
        limits.append({"token": token, "source_file": str(path.resolve()),
                       "date_min": parsed.min(), "date_max": parsed.max(),
                       "valid_date_rows": int(parsed.size),
                       "invalid_date_rows": int(parsed_all.isna().sum())})

    start = min(row["date_min"] for row in limits).normalize()
    end_exclusive = max(row["date_max"] for row in limits).normalize() + pd.Timedelta(days=1)
    total_days = max(1, int((end_exclusive - start) / pd.Timedelta(days=1)))
    number_of_windows = min(requested_windows, total_days)
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for index in range(number_of_windows):
        start_offset = (total_days * index) // number_of_windows
        end_offset = (total_days * (index + 1)) // number_of_windows
        window_start = start + pd.Timedelta(days=start_offset)
        window_end = start + pd.Timedelta(days=end_offset)
        last_included_day = window_end - pd.Timedelta(days=1)
        label = f"{window_start:%Y%m%d}_to_{last_included_day:%Y%m%d}"
        windows[label] = (window_start, window_end)

    manifest = pd.DataFrame([
        {"window": label, "start_inclusive": bounds[0],
         "end_exclusive": bounds[1], "end_inclusive_date": bounds[1] - pd.Timedelta(days=1),
         "window_index": index}
        for index, (label, bounds) in enumerate(windows.items())
    ])
    return windows, pd.DataFrame(limits), manifest


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _snapshot_windows(events: pd.DataFrame, requested: int) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    """Split each token's observed snapshots into near-equal chronological groups."""
    snapshots = sorted(pd.to_numeric(events["snapshot"], errors="coerce").dropna().unique())
    if not snapshots:
        raise ValueError("token has no valid snapshots")
    chunks = [chunk for chunk in __import__("numpy").array_split(snapshots, min(requested, len(snapshots))) if len(chunk)]
    windows: dict[str, tuple[float, float]] = {}
    rows = []
    dates = pd.to_datetime(events["date"], errors="coerce")
    for index, chunk in enumerate(chunks, 1):
        first, last = float(chunk[0]), float(chunk[-1])
        label = f"window_{index:02d}_snap_{int(first)}_to_{int(last)}"
        end = float(snapshots[snapshots.index(chunk[-1]) + 1]) if chunk[-1] != snapshots[-1] else last + 1
        windows[label] = (first, end)
        selected_dates = dates.loc[pd.to_numeric(events["snapshot"], errors="coerce").isin(chunk)].dropna()
        rows.append({"window": label, "window_index": index, "snapshot_start": first,
                     "snapshot_end": last, "date_start": selected_dates.min() if len(selected_dates) else None,
                     "date_end": selected_dates.max() if len(selected_dates) else None,
                     "snapshot_count": len(chunk)})
    return windows, pd.DataFrame(rows)


def _common_calendar_windows(events_a: pd.DataFrame, events_b: pd.DataFrame,
                            requested: int) -> tuple[list[tuple[str, pd.Timestamp, pd.Timestamp]], pd.Timestamp | None, pd.Timestamp | None]:
    """Make equal-duration calendar windows within two tokens' observed overlap."""
    dates_a = pd.to_datetime(events_a["date"], errors="coerce").dropna()
    dates_b = pd.to_datetime(events_b["date"], errors="coerce").dropna()
    if dates_a.empty or dates_b.empty:
        return [], None, None
    start = max(dates_a.min().normalize(), dates_b.min().normalize())
    end = min(dates_a.max().normalize() + pd.Timedelta(days=1),
              dates_b.max().normalize() + pd.Timedelta(days=1))
    days = int((end - start) / pd.Timedelta(days=1))
    if days <= 0:
        return [], start, end
    count = min(max(1, requested), days)
    windows = []
    for i in range(count):
        offset_start = (days * i) // count
        offset_end = (days * (i + 1)) // count
        left, right = start + pd.Timedelta(days=offset_start), start + pd.Timedelta(days=offset_end)
        windows.append((f"calendar_{i + 1:02d}_{left:%Y%m%d}_to_{(right - pd.Timedelta(days=1)):%Y%m%d}", left, right))
    return windows, start, end


def _adjusted_rand(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Adjusted Rand index for two partitions on their shared node set."""
    joined = left[["node", "community"]].merge(right[["node", "community"]], on="node", suffixes=("_a", "_b"))
    n = len(joined)
    if n < 2:
        return 1.0
    cells = joined.groupby(["community_a", "community_b"], dropna=False).size().tolist()
    rows = joined.groupby("community_a", dropna=False).size().tolist()
    cols = joined.groupby("community_b", dropna=False).size().tolist()
    index = sum(comb(int(v), 2) for v in cells)
    expected = sum(comb(int(v), 2) for v in rows) * sum(comb(int(v), 2) for v in cols) / comb(n, 2)
    maximum = 0.5 * (sum(comb(int(v), 2) for v in rows) + sum(comb(int(v), 2) for v in cols))
    return float((index - expected) / (maximum - expected)) if maximum != expected else 1.0


def _save_quotient(projection: nx.Graph, memberships: pd.DataFrame, output: Path,
                   title: str, *, algorithm: str = "leiden", top_k: int = 40, seed: int = SEED) -> None:
    """Save complete quotient CSVs and a top-community plus Other figure."""
    member = memberships.set_index("node")["community"].to_dict()
    sizes = memberships.groupby("community").size().sort_values(ascending=False)
    quotient = nx.Graph()
    quotient.add_nodes_from((int(c), {"address_count": int(n)}) for c, n in sizes.items())
    for u, v, data in projection.edges(data=True):
        a, b = member.get(u), member.get(v)
        if a is None or b is None or a == b:
            continue
        if quotient.has_edge(a, b): quotient[a][b]["weight"] += float(data.get("weight", 1.0))
        else: quotient.add_edge(a, b, weight=float(data.get("weight", 1.0)))
    pd.DataFrame([{"community": c, "address_count": d["address_count"]} for c, d in quotient.nodes(data=True)]).to_csv(output / f"{algorithm}_quotient_nodes.csv", index=False)
    pd.DataFrame([{"community_a": a, "community_b": b, "weight": d["weight"]} for a,b,d in quotient.edges(data=True)]).to_csv(output / f"{algorithm}_quotient_edges.csv", index=False)
    shown = set(sizes.head(top_k).index)
    view = nx.Graph(); view.add_nodes_from((c, quotient.nodes[c]) for c in shown)
    other_count = int(sum(n for c,n in sizes.items() if c not in shown))
    if other_count: view.add_node("Other", address_count=other_count)
    for a,b,d in quotient.edges(data=True):
        x = a if a in shown else ("Other" if other_count else None)
        y = b if b in shown else ("Other" if other_count else None)
        if x is None or y is None or x == y: continue
        if view.has_edge(x,y): view[x][y]["weight"] += d["weight"]
        else: view.add_edge(x,y,weight=d["weight"])
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(view, seed=seed, weight="weight")
    sizes_draw = [60 + 16 * __import__("math").sqrt(view.nodes[n]["address_count"]) for n in view]
    widths = [0.4 + __import__("math").log1p(view[u][v]["weight"]) for u,v in view.edges]
    fig, ax = plt.subplots(figsize=(14, 11)); nx.draw_networkx_edges(view,pos,ax=ax,width=widths,alpha=.35)
    nx.draw_networkx_nodes(view,pos,ax=ax,node_size=sizes_draw,node_color="#4c78a8",alpha=.85)
    labels = {n: (str(n) if n == "Other" else f"C{n}\n{view.nodes[n]['address_count']:,}") for n in view}
    nx.draw_networkx_labels(view,pos,labels=labels,font_size=7,ax=ax)
    ax.set_title(title + f" — top {len(shown)} communities + Other; seed={seed}")
    ax.text(.01,.01,"Undirected projection: reciprocal weights summed; self-loops and zero address excluded. Edge width ∝ log1p(aggregate weight). Other is display-only.",transform=ax.transAxes,fontsize=8)
    ax.axis("off"); fig.tight_layout(); fig.savefig(output / f"{algorithm}_quotient.png", dpi=220); plt.close(fig)


def _save_address_ego(projection: nx.Graph, directed: nx.DiGraph, leiden: pd.DataFrame,
                     pagerank: pd.DataFrame, output: Path, title: str, seed: int = SEED) -> None:
    """Draw five directed-PageRank leaders and up to five strongest projected neighbors each."""
    ranked = pagerank.sort_values(["pagerank", "node"], ascending=[False, True], kind="stable").head(5)
    leaders = ranked.node.tolist(); chosen = set(leaders)
    for node in leaders:
        neighbors = sorted(projection[node].items(), key=lambda item: (-float(item[1].get("weight", 0)), str(item[0])))[:5]
        chosen.update(neighbor for neighbor, _ in neighbors)
    subgraph = projection.subgraph(chosen).copy()
    membership = leiden.set_index("node").community.to_dict()
    rank = pagerank.set_index("node").pagerank.to_dict()
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(subgraph, seed=seed, weight="weight")
    colors = [membership.get(n, -1) for n in subgraph]
    sizes = [80 + 1800 * float(rank.get(n, 0)) for n in subgraph]
    widths = [0.4 + __import__("math").log1p(float(subgraph[u][v].get("weight", 0))) for u,v in subgraph.edges]
    fig, ax = plt.subplots(figsize=(12,9)); nx.draw_networkx_edges(subgraph,pos,ax=ax,width=widths,alpha=.4)
    nx.draw_networkx_nodes(subgraph,pos,ax=ax,node_size=sizes,node_color=colors,cmap="tab20",alpha=.9)
    labels = {n: str(n)[:12] for n in subgraph}; nx.draw_networkx_labels(subgraph,pos,labels=labels,font_size=7,ax=ax)
    ax.set_title(title + " — address level, deterministic induced sample")
    ax.text(.01,.01,f"Selected {len(leaders)} highest PageRank addresses + up to 5 strongest undirected-projection neighbors each; {len(subgraph):,} addresses/{subgraph.number_of_edges():,} edges; node size ∝ directed PageRank; edge width ∝ log1p(weight); seed={seed}.",transform=ax.transAxes,fontsize=8)
    ax.axis("off"); fig.tight_layout(); fig.savefig(output / "address_ego_top5.png",dpi=220); plt.close(fig)


def _configure_logging(run_dir: Path) -> logging.Logger:
    """Write timestamped progress messages to both console and the run log."""
    logger = logging.getLogger("study_A")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "analysis.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def run_study(
    tokens: Sequence[str] = TOKENS,
    *,
    edge_types: Sequence[str] = EDGE_TYPES_TO_ANALYZE,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = None,
    number_of_windows: int = NUM_DATE_WINDOWS,
    include_expensive_path_centralities: bool = INCLUDE_EXPENSIVE_PATH_CENTRALITIES,
) -> dict[str, pd.DataFrame]:
    """Run graph, centrality, projection, community, temporal and overlap analyses.

    Betweenness and both closeness calculations are disabled by default because
    they can be prohibitive on large ERC-20 graphs. Their omission is written to
    the centrality-status table. Pass ``include_expensive_path_centralities=True``
    to enable the existing implementation (betweenness uses ``BETWEENNESS_K``).
    """
    selected_tokens = list(dict.fromkeys(str(token).strip() for token in tokens if str(token).strip()))
    if not selected_tokens:
        raise ValueError("provide at least one token in TOKENS or --tokens")
    selected_edge_types = list(dict.fromkeys(str(edge_type).strip() for edge_type in edge_types))
    invalid_edge_types = [edge_type for edge_type in selected_edge_types if edge_type not in WEIGHT_COLUMNS]
    if invalid_edge_types:
        raise ValueError(f"unknown edge types {invalid_edge_types}; choose from {list(WEIGHT_COLUMNS)}")
    if not selected_edge_types:
        raise ValueError(f"select at least one edge type from {list(WEIGHT_COLUMNS)}")
    data_path = Path(data_dir)
    token_paths = {token: data_path / f"{token}_edgelist.txt" for token in selected_tokens}
    missing = [str(path) for path in token_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"TGS files not found: {missing}")

    token_date_profile = pd.DataFrame()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "reports" / "study_A" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(run_dir)
    run_started = time.perf_counter()
    logger.info("Run %s avviato | token=%s | tipi arco=%s | finestre richieste=%d | output=%s",
                run_id, ", ".join(selected_tokens), ", ".join(selected_edge_types),
                number_of_windows, run_dir)
    logger.info("Le finestre saranno create separatamente per token raggruppando gli snapshot cronologici.")
    save_tables(run_dir, tables={"token_date_profile": token_date_profile}, metadata={
        "run_id": run_id,
        "tokens": selected_tokens,
        "edge_types": selected_edge_types,
        "source_directory": str(data_path.resolve()),
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": SEED,
        "number_of_date_windows": number_of_windows,
        "window_policy": "10 chronological groups of observed snapshots per token; plus full-period aggregate",
        "cross_token_community_comparisons": "recomputed per token pair and shared calendar window",
        "include_expensive_path_centralities": include_expensive_path_centralities,
    })

    graph_summaries: list[pd.DataFrame] = []
    projection_summaries: list[pd.DataFrame] = []
    centrality_tops: list[pd.DataFrame] = []
    centrality_statuses: list[pd.DataFrame] = []
    persistence_tables: list[pd.DataFrame] = []
    overlap_summaries: list[pd.DataFrame] = []
    membership_store: dict[tuple[str, str, str], dict[str, pd.DataFrame]] = {}
    event_store: dict[str, pd.DataFrame] = {}

    for token, source_path in token_paths.items():
        token_started = time.perf_counter()
        logger.info("Token %s: caricamento eventi da %s", token, source_path.name)
        events, event_audit = load_token_events(source_path)
        event_store[token] = events
        windows, window_manifest = _snapshot_windows(events, number_of_windows)
        token_date_profile = pd.DataFrame([{
            "token": token, "source_file": str(source_path.resolve()),
            "snapshot_count": events["snapshot"].nunique(),
            "date_min": events["date"].min(), "date_max": events["date"].max(),
        }])
        save_tables(run_dir / _safe_name(token), tables={"window_manifest": window_manifest,
                     "token_date_profile": token_date_profile}, metadata={
            "token": token, "window_policy": "equal-count chronological observed snapshots",
            "token_label_note": "BITCOIN denotes the local TGS ERC-20 filename only; no contract or Bitcoin-network identity is inferred." if token == "BITCOIN" else None,
            "seed": SEED, "source_file": str(source_path.resolve()),
        })
        # Snapshot IDs are not shared across tokens and some are repeated on dates;
        # assign the temporal groups by snapshot values for this token.
        windowed_events, window_audit = assign_windows(events, windows, column="snapshot")
        windowed_events["window"] = "FULL_PERIOD"
        for label, (start_snapshot, end_snapshot) in windows.items():
            mask = events["snapshot"].ge(start_snapshot) & events["snapshot"].lt(end_snapshot)
            windowed_events.loc[mask, "window"] = label
        full_events = events.copy()
        full_events["window"] = "FULL_PERIOD"
        windowed_events = pd.concat([windowed_events, full_events], ignore_index=True)
        token_dir = run_dir / _safe_name(token)
        token_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Token %s: caricati %s eventi; assegnati %s eventi a finestre (audit: %s)",
                    token, f"{len(events):,}",
                    f"{int(window_audit.loc[window_audit['window'].ne('UNASSIGNED'), 'rows'].sum()):,}",
                    token_dir)
        save_tables(token_dir, tables={
            "window_audit": window_audit,
            "event_audit": pd.DataFrame([event_audit]),
        }, metadata={
            "token": token,
            "source_file": str(source_path.resolve()),
            "preprocessing_version": event_audit.get("preprocessing_version", PREPROCESSING_VERSION),
            "window_policy": "token-local snapshot groups; intervals [start, end)",
            "seed": SEED,
        })

        # Run all directed graph summaries and centralities once per window/variant.
        centrality_started = time.perf_counter()
        logger.info("Token %s: avvio riepiloghi diretti e centralità; betweenness/closeness costose=%s",
                    token, "abilitate" if include_expensive_path_centralities else "disabilitate")
        core = analyze_windows(
            windowed_events,
            variants=selected_edge_types,
            graph_info={"token": token, "source_file": str(source_path.resolve())},
            centrality_seed=SEED,
            betweenness_k=BETWEENNESS_K,
            calculate_path_centralities=include_expensive_path_centralities,
        )
        logger.info("Token %s: riepiloghi diretti e centralità completati in %.1f s",
                    token, time.perf_counter() - centrality_started)
        if not core["graph_summary"].empty:
            graph_summaries.append(core["graph_summary"])
        if not core["centrality_top_k"].empty:
            centrality_tops.append(core["centrality_top_k"].assign(token=token))
        if not core["centrality_status"].empty:
            centrality_statuses.append(core["centrality_status"].assign(token=token))

        token_memberships: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
        token_persistence_tables: list[pd.DataFrame] = []
        analysis_windows = {**windows, "FULL_PERIOD": (min(events["snapshot"]), max(events["snapshot"]) + 1)}
        for window_index, (window_label, (window_start, window_end)) in enumerate(analysis_windows.items(), start=1):
            events_in_window = windowed_events.loc[windowed_events["window"].eq(window_label)].copy()
            if events_in_window.empty:
                logger.warning("Token %s: finestra %d/%d %s senza eventi assegnati; saltata",
                               token, window_index, len(windows), window_label)
                continue
            logger.info("Token %s: finestra %d/%d %s | %s eventi",
                        token, window_index, len(windows), window_label, f"{len(events_in_window):,}")
            actual_dates = events_in_window["date"].dropna()
            graph_info = {
                "token": token,
                "source_file": str(source_path.resolve()),
                "preprocessing_version": event_audit.get("preprocessing_version", PREPROCESSING_VERSION),
                "window": window_label,
                "window_start_inclusive": str(window_start),
                "window_end_exclusive": str(window_end),
                "observed_date_min": str(actual_dates.min()) if not actual_dates.empty else None,
                "observed_date_max": str(actual_dates.max()) if not actual_dates.empty else None,
                "window_policy": "full observed period" if window_label == "FULL_PERIOD" else "equal-count chronological snapshots within token",
                "seed": SEED,
            }
            views, aggregates = build_graphs_from_events(
                events_in_window, variants=selected_edge_types, graph_info=graph_info,
            )
            for variant_index, (variant, view) in enumerate(views.items(), start=1):
                variant_started = time.perf_counter()
                # Hierarchy: run/token/edge_type/window/analysis_files.
                variant_dir = token_dir / _safe_name(variant) / _safe_name(window_label)
                logger.info("Token %s | finestra %s | arco %s (%d/%d): avvio | nodi=%s archi=%s",
                            token, window_label, variant, variant_index, len(views),
                            f"{view.graph.number_of_nodes():,}", f"{view.graph.number_of_edges():,}")
                graph_summary_rows = core["graph_summary"]
                node_descriptives = core["node_descriptives"]
                if not graph_summary_rows.empty and "window" in graph_summary_rows.columns:
                    graph_summary_rows = graph_summary_rows.loc[
                        graph_summary_rows["window"].astype(str).eq(window_label)
                        & graph_summary_rows["graph"].eq(variant)
                    ]
                if not node_descriptives.empty and "window" in node_descriptives.columns:
                    node_descriptives = node_descriptives.loc[
                        node_descriptives["window"].astype(str).eq(window_label)
                        & node_descriptives["graph"].eq(variant)
                    ].copy()
                graph_summary = (graph_summary_rows.iloc[0].to_dict()
                                 if not graph_summary_rows.empty else describe_graph(view)[0])
                # analyze_windows has already produced these measures; select its
                # rows to avoid running a second, expensive centrality pass.
                centralities = core["centralities"]
                if not centralities.empty and "window" in centralities.columns:
                    centralities = centralities.loc[
                        centralities["window"].astype(str).eq(window_label)
                        & centralities["graph"].eq(variant)
                    ].copy()
                centrality_status = core["centrality_status"]
                if not centrality_status.empty and "window" in centrality_status.columns:
                    centrality_status = centrality_status.loc[
                        centrality_status["window"].astype(str).eq(window_label)
                        & centrality_status["variant"].eq(variant)
                    ].copy()
                centrality_top_k = core["centrality_top_k"]
                if not centrality_top_k.empty:
                    centrality_top_k = centrality_top_k.loc[
                        centrality_top_k["window"].astype(str).eq(window_label)
                        & centrality_top_k["graph"].eq(variant)
                    ].copy()

                projection_started = time.perf_counter()
                logger.info("Proiezione non diretta | %s | %s | %s: avvio",
                            token, window_label, variant)
                projection = make_undirected_projection(
                    view, weight="weight", name=f"undirected_sum_{variant}", keep_self_loops=False,
                )
                projection_summary = describe_projection(projection)
                logger.info("Proiezione non diretta | %s | %s | %s: completata | nodi=%s archi=%s | %.1f s",
                            token, window_label, variant,
                            f"{projection.number_of_nodes():,}",
                            f"{projection.number_of_edges():,}",
                            time.perf_counter() - projection_started)
                projection_clustering = projection_summary.pop("clustering_binary_by_node", {})
                projection_summaries.append(pd.DataFrame([{
                    **graph_info, "variant": variant, **projection_summary,
                }]))
                projection_clustering_table = pd.DataFrame([
                    {**graph_info, "variant": variant, "node": node,
                     "clustering_binary": coefficient}
                    for node, coefficient in projection_clustering.items()
                ], columns=[*graph_info.keys(), "variant", "node", "clustering_binary"])
                communities_started = time.perf_counter()
                logger.info("Community detection | %s | %s | %s: avvio Louvain e Leiden",
                            token, window_label, variant)
                louvain_memberships, leiden_memberships, community_metadata = compute_communities(
                    projection, seed=SEED, resolution=1.0, weight="weight",
                )
                logger.info("Community detection | %s | %s | %s: completata | Louvain=%d Leiden=%d | %.1f s",
                            token, window_label, variant,
                            louvain_memberships["community"].nunique() if not louvain_memberships.empty else 0,
                            leiden_memberships["community"].nunique() if not leiden_memberships.empty else 0,
                            time.perf_counter() - communities_started)

                page_rank_ok = (
                    not centrality_status.empty
                    and centrality_status.loc[
                        centrality_status["measure"].eq("pagerank"), "status"
                    ].eq("ok").any()
                    and "pagerank" in centralities.columns
                )
                rank_measure = "pagerank" if page_rank_ok else "out_strength"
                rank_table = centralities[["node", rank_measure]].copy() if rank_measure in centralities else node_descriptives[["node", "out_strength"]].copy()
                top_ranked_nodes = (
                    rank_table.loc[rank_table["node"].isin(
                        max(nx.connected_components(projection), key=len, default=set())
                    )].dropna(subset=[rank_measure])
                    .sort_values([rank_measure, "node"], ascending=[False, True], kind="stable")
                    .head(TOP_RANKED_NODES_TO_HIGHLIGHT)["node"].tolist()
                )
                louvain_top = community_central_nodes(
                    louvain_memberships, rank_table, measure=rank_measure,
                    top_k=TOP_CENTRAL_NODES_PER_COMMUNITY,
                )
                leiden_top = community_central_nodes(
                    leiden_memberships, rank_table, measure=rank_measure,
                    top_k=TOP_CENTRAL_NODES_PER_COMMUNITY,
                )
                ari = _adjusted_rand(leiden_memberships, louvain_memberships)
                community_sizes = pd.concat([
                    leiden_memberships.groupby("community").size().rename("address_count").reset_index().assign(algorithm="leiden"),
                    louvain_memberships.groupby("community").size().rename("address_count").reset_index().assign(algorithm="louvain"),
                ], ignore_index=True)
                top5 = centralities.sort_values(["pagerank", "node"], ascending=[False, True], kind="stable").head(5).copy() if "pagerank" in centralities else pd.DataFrame()
                if not top5.empty:
                    top5["in_strength"] = top5["node"].map(dict(view.graph.in_degree(weight="weight")))
                    top5["out_strength"] = top5["node"].map(dict(view.graph.out_degree(weight="weight")))
                    top5 = top5.merge(leiden_memberships[["node", "community"]].rename(columns={"community":"leiden_community"}),on="node",how="left")
                    top5 = top5.merge(louvain_memberships[["node", "community"]].rename(columns={"community":"louvain_community"}),on="node",how="left")
                community_metadata.update({**graph_info, "variant": variant,
                                           "centrality_measure_for_highlights": rank_measure})
                save_tables(variant_dir, tables={
                    "aggregated_edges_audit": aggregates,
                    "graph_summary": pd.DataFrame([graph_summary]),
                    "node_descriptives": node_descriptives,
                    "centralities": centralities,
                    "centrality_top_k": centrality_top_k,
                    "centrality_status": centrality_status,
                    "projection_summary": pd.DataFrame([projection_summary]),
                    "projection_clustering_by_node": projection_clustering_table,
                    "louvain_communities": louvain_memberships,
                    "leiden_communities": leiden_memberships,
                    "louvain_top_central_nodes": louvain_top,
                    "leiden_top_central_nodes": leiden_top,
                    "community_size_distribution": community_sizes,
                    "top5_pagerank_both_algorithms": top5,
                    "partition_comparison": pd.DataFrame([{"token": token, "window": window_label,
                        "variant": variant, "shared_node_count": len(leiden_memberships), "adjusted_rand_index": ari,
                        "leiden_modularity": community_metadata.get("leiden_modularity"),
                        "louvain_modularity": community_metadata.get("louvain_modularity")}]),
                }, metadata=community_metadata)
                _save_quotient(projection, leiden_memberships, variant_dir,
                               f"{token} — Leiden — {variant} — {window_label}")
                _save_quotient(projection, louvain_memberships, variant_dir,
                               f"{token} — Louvain — {variant} — {window_label}", algorithm="louvain")
                if "pagerank" in centralities:
                    _save_address_ego(projection, view.graph, leiden_memberships,
                                      centralities[["node", "pagerank"]], variant_dir,
                                      f"{token} — {variant} — {window_label}")

                for algorithm, memberships, top_nodes, draw_function in (
                    ("louvain", louvain_memberships, louvain_top, draw_louvain),
                    ("leiden", leiden_memberships, leiden_top, draw_leiden),
                ):
                    membership_store.setdefault((token, variant, algorithm), {})[window_label] = memberships
                    stored_memberships = memberships.assign(
                        token=token, window=window_label, variant=variant, algorithm=algorithm,
                    )
                    token_memberships.setdefault((variant, algorithm), {})[window_label] = stored_memberships
                    # Save both a plain community map and one highlighting only
                    # the five highest-ranked nodes across the whole graph.
                    plot_variants = (
                        ("senza evidenziazione", None, f"{algorithm}_communities.png",
                         f"{token} — {algorithm.title()} — {variant} — {window_label}"),
                        ("top 5", top_ranked_nodes, f"{algorithm}_communities_top5.png",
                         f"{token} — {algorithm.title()} — {variant} — {window_label} — Top 5 {rank_measure}"),
                    )
                    for plot_label, highlights, filename, plot_title in plot_variants:
                        logger.info("Disegno community | %s | %s | %s | %s (%s): avvio",
                                    token, window_label, variant, algorithm, plot_label)
                        draw_function(
                            projection,
                            memberships,
                            title=plot_title,
                            save_path=variant_dir / filename,
                            show=False,
                            max_nodes=MAX_DRAW_NODES,
                            seed=SEED,
                            highlight_nodes=highlights,
                            highlight_measure=rank_measure,
                        )
                        logger.info("Disegno community | %s | %s | %s | %s (%s): salvato",
                                    token, window_label, variant, algorithm, plot_label)
                logger.info("Token %s | finestra %s | arco %s: completato; comunità Louvain=%d, Leiden=%d; durata=%.1f s",
                            token, window_label, variant,
                            louvain_memberships["community"].nunique() if not louvain_memberships.empty else 0,
                            leiden_memberships["community"].nunique() if not leiden_memberships.empty else 0,
                            time.perf_counter() - variant_started)

        # Temporal persistence compares consecutive shared-date windows within
        # each token, graph variant, and community algorithm.
        for (variant, algorithm), by_window in token_memberships.items():
            ordered_windows = list(windows)
            for previous_label, current_label in zip(ordered_windows, ordered_windows[1:]):
                if previous_label not in by_window or current_label not in by_window:
                    continue
                persistence = community_persistence(
                    by_window[previous_label], by_window[current_label],
                )
                if not persistence.empty:
                    tagged_persistence = persistence.assign(
                        token=token, variant=variant, algorithm=algorithm,
                        previous_window=previous_label, current_window=current_label,
                    )
                    persistence_tables.append(tagged_persistence)
                    token_persistence_tables.append(tagged_persistence)
        # Save temporal persistence within the same edge-type directory.
        for variant in selected_edge_types:
            variant_persistence = [table for table in token_persistence_tables
                                   if not table.empty and table["variant"].eq(variant).all()]
            if variant_persistence:
                save_tables(token_dir / _safe_name(variant), tables={
                    "community_persistence": pd.concat(variant_persistence, ignore_index=True),
                }, metadata={"token": token, "variant": variant, "seed": SEED})
        logger.info("Token %s: analisi completate in %.1f s", token, time.perf_counter() - token_started)

    # Rebuild each pair's graphs on common calendar windows. Snapshot IDs are
    # token-local, so the partitions from the token-specific evolution analysis
    # are deliberately not reused here.
    overlap_dir = run_dir / "cross_token_overlap"
    token_pairs = list(combinations(selected_tokens, 2))
    total_overlap_tasks = len(token_pairs) * number_of_windows * len(selected_edge_types) * 4
    overlap_task = 0
    for token_a, token_b in token_pairs:
        common_windows, common_start, common_end = _common_calendar_windows(
            event_store[token_a], event_store[token_b], number_of_windows,
        )
        pair_root = overlap_dir / f"{_safe_name(token_a)}__{_safe_name(token_b)}"
        if not common_windows:
            no_overlap = pd.DataFrame([{"token_a": token_a, "token_b": token_b,
                "common_start": common_start, "common_end_exclusive": common_end,
                "status": "no_shared_calendar_dates"}])
            save_tables(pair_root, tables={"comparison_status": no_overlap}, metadata={
                "token_a": token_a, "token_b": token_b,
                "comparison_policy": "equal-duration calendar windows over shared observed date range",
                "seed": SEED,
            })
            continue
        manifest = pd.DataFrame([{"window": label, "start_inclusive": left,
            "end_exclusive": right, "end_inclusive_date": right - pd.Timedelta(days=1),
            "window_index": index, "token_a": token_a, "token_b": token_b,
            "shared_range_start": common_start, "shared_range_end_exclusive": common_end}
            for index, (label, left, right) in enumerate(common_windows, start=1)])
        save_tables(pair_root, tables={"common_calendar_window_manifest": manifest}, metadata={
            "token_a": token_a, "token_b": token_b,
            "comparison_policy": "equal-duration calendar windows inside pairwise shared observed date range",
            "window_count": len(common_windows), "seed": SEED,
        })
        for window_label, window_start, window_end in common_windows:
            token_window_events = {}
            for token in (token_a, token_b):
                frame = event_store[token]
                parsed = pd.to_datetime(frame["date"], errors="coerce")
                token_window_events[token] = frame.loc[parsed.ge(window_start) & parsed.lt(window_end)].copy()
            for variant in selected_edge_types:
                memberships_by_algorithm: dict[str, dict[str, pd.DataFrame]] = {token_a: {}, token_b: {}}
                for token in (token_a, token_b):
                    overlap_task += 1
                    current_events = token_window_events[token]
                    info = {"token": token, "source_file": str(token_paths[token].resolve()),
                        "window": window_label, "window_start_inclusive": str(window_start),
                        "window_end_exclusive": str(window_end),
                        "window_policy": "pairwise shared calendar interval",
                        "comparison_pair": f"{token_a}__{token_b}", "seed": SEED}
                    views, _ = build_graphs_from_events(current_events, variants=[variant], graph_info=info)
                    projection = make_undirected_projection(views[variant], weight="weight",
                        name=f"undirected_sum_{variant}", keep_self_loops=False)
                    if projection.number_of_nodes():
                        louvain_part, leiden_part, _ = compute_communities(
                            projection, seed=SEED, resolution=1.0, weight="weight")
                    else:
                        louvain_part = pd.DataFrame(columns=["node", "community", "community_size"])
                        leiden_part = pd.DataFrame(columns=["node", "community", "community_size"])
                    memberships_by_algorithm[token]["louvain"] = louvain_part
                    memberships_by_algorithm[token]["leiden"] = leiden_part
                for algorithm in ("louvain", "leiden"):
                    overlap_task += 1
                    summary, overlaps, shared_addresses = community_overlap(
                        memberships_by_algorithm[token_a][algorithm],
                        memberships_by_algorithm[token_b][algorithm],
                        token_a=token_a, token_b=token_b,
                        window_a=window_label, window_b=window_label,
                        algorithm_a=algorithm, algorithm_b=algorithm,
                        variant_a=variant, variant_b=variant,
                    )
                    summary["window_start_inclusive"] = window_start
                    summary["window_end_exclusive"] = window_end
                    summary["shared_calendar_days"] = int((window_end-window_start)/pd.Timedelta(days=1))
                    overlap_summaries.append(summary)
                    pair_dir = pair_root / _safe_name(variant) / _safe_name(window_label) / algorithm
                    save_tables(pair_dir, tables={
                        "overlap_summary": summary, "community_overlaps": overlaps,
                        "shared_addresses": shared_addresses,
                        "memberships_token_a": memberships_by_algorithm[token_a][algorithm],
                        "memberships_token_b": memberships_by_algorithm[token_b][algorithm],
                    }, metadata={"token_a": token_a, "token_b": token_b,
                        "window": window_label, "window_start_inclusive": str(window_start),
                        "window_end_exclusive": str(window_end), "variant": variant,
                        "algorithm": algorithm, "projection": "undirected; reciprocal weights summed",
                        "filters": ["self-loops excluded", "zero address excluded"],
                        "seed": SEED, "resolution": 1.0})
                    draw_overlap(summary, overlaps,
                        metric="overlap_coefficient_on_shared_graph_nodes",
                        top_k=OVERLAP_DRAW_TOP_K,
                        save_path=pair_dir / "community_overlap.png", show=False)
                    logger.info("Confronto calendario %d/%d | %s vs %s | %s | %s | %s | nodi condivisi=%s",
                        overlap_task, total_overlap_tasks, token_a, token_b, window_label,
                        variant, algorithm, f"{int(summary.iloc[0]['shared_node_count']):,}")

    results = {
        "graph_summaries": pd.concat(graph_summaries, ignore_index=True) if graph_summaries else pd.DataFrame(),
        "projection_summaries": pd.concat(projection_summaries, ignore_index=True) if projection_summaries else pd.DataFrame(),
        "centrality_top_k": pd.concat(centrality_tops, ignore_index=True) if centrality_tops else pd.DataFrame(),
        "centrality_status": pd.concat(centrality_statuses, ignore_index=True) if centrality_statuses else pd.DataFrame(),
        "community_persistence": pd.concat(persistence_tables, ignore_index=True) if persistence_tables else pd.DataFrame(),
        "community_overlap_summary": pd.concat(overlap_summaries, ignore_index=True) if overlap_summaries else pd.DataFrame(),
    }
    # Store aggregate summaries under token/edge-type folders as well, rather
    # than writing cross-token CSVs that mix the three weight definitions.
    summary_tables = {
        "graph_summary": results["graph_summaries"],
        "projection_summary": results["projection_summaries"],
        "centrality_top_k": results["centrality_top_k"],
        "centrality_status": results["centrality_status"],
        "community_persistence": results["community_persistence"],
    }
    for token in selected_tokens:
        for variant in selected_edge_types:
            selected_tables: dict[str, pd.DataFrame] = {}
            for name, table in summary_tables.items():
                if table.empty:
                    continue
                token_column = "token" if "token" in table.columns else None
                variant_column = "variant" if "variant" in table.columns else "graph"
                if token_column is None or variant_column not in table.columns:
                    continue
                selected = table.loc[
                    table[token_column].eq(token) & table[variant_column].eq(variant)
                ].copy()
                if not selected.empty:
                    selected_tables[name] = selected
            if selected_tables:
                summary_path = run_dir / _safe_name(token) / _safe_name(variant)
                save_tables(summary_path, tables=selected_tables, metadata={
                    "run_id": run_id, "token": token, "variant": variant, "seed": SEED,
                    "edge_types": selected_edge_types,
                    "window_policy": "token-local equal-count snapshot windows plus full period",
                    "include_expensive_path_centralities": include_expensive_path_centralities,
                })

    # Keep overlap summaries split by token pair, weight definition and method.
    overlap_summary_table = results["community_overlap_summary"]
    if not overlap_summary_table.empty:
        for (token_a, token_b, variant, algorithm), overlap_rows in overlap_summary_table.groupby(
            ["token_a", "token_b", "variant_a", "algorithm_a"], sort=True,
        ):
            summary_path = (overlap_dir / f"{_safe_name(token_a)}__{_safe_name(token_b)}"
                            / _safe_name(variant) / algorithm)
            save_tables(summary_path, tables={"overlap_summary_all_windows": overlap_rows}, metadata={
                "run_id": run_id, "token_a": token_a, "token_b": token_b,
                "variant": variant, "algorithm": algorithm, "seed": SEED,
            })
    logger.info("Run %s completato in %.1f s | risultati=%s",
                run_id, time.perf_counter() - run_started, run_dir)
    return results

if __name__ == "__main__":

    # VALUE OF STUDY
    tokens = TOKENS
    data_dir = DEFAULT_DATA_DIR
    output_dir = None
    windows =  NUM_DATE_WINDOWS
    include_expensive_path_centralities = INCLUDE_EXPENSIVE_PATH_CENTRALITIES
    edge_types = EDGE_TYPES_TO_ANALYZE

    try:
        run_study(
            tokens,
            edge_types=edge_types,
            data_dir=data_dir,
            output_dir=output_dir,
            number_of_windows=windows,
            include_expensive_path_centralities=include_expensive_path_centralities,
        )
    except Exception:
        logging.getLogger("study_A").exception("Run interrotto da un errore non gestito")
        raise
