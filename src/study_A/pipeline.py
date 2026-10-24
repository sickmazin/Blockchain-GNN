"""End-to-end Study A run: selection, per-token analysis, cross-token summaries, figures.

Output layout (``reports/study_A/runs/<run_id>/``)::

    config.toml, resolved_config.json, run_metadata.json, run.log
    selection/        token_profile.csv, token_selection.csv
    <TOKEN>/          tables (CSV), token.log
        graphs/       analytical edge lists per window (filtered pairs, gzip CSV)
        centralities/ node-level centralities per window and weight variant
        communities/  Leiden and Louvain memberships per window and variant
        nodes/        node-level topology (bow-tie class, coreness) per window
    summary/          cross-token CSV tables
    tables/           LaTeX (booktabs) versions of the summary tables
    figures/          PDF + PNG figures and captions.csv

Every analysis step writes a row to ``analysis_status.csv``; a failing step is
recorded with its error and the run continues.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from . import PREPROCESSING_VERSION
from .centrality import centrality_correlations, compute_centralities, top_k
from .communities import (adjusted_rand, leiden, louvain, normalized_mutual_info, partition_summary,
                          persistence, quotient, resolution_sweep, seed_stability)
from .concentration import QUANTITIES, concentration_curve, concentration_table
from .config import PROJECT_ROOT, StudyAConfig
from .data import (_file_fingerprint, address_format, daily_activity, load_token, profile_all, select_tokens,
                   token_path)
from .distributions import analyse_tail
from .graphs import DISTANCE_RULE, VARIANT_LABELS, aggregate_pairs, build_bundle, filter_pairs, variant_edges
from .nullmodels import null_samples, small_world_summary
from .temporal import address_turnover, rank_stability, tie_persistence
from .topology import (bowtie, directed_assortativity, directed_descriptors, rich_club_curve,
                       rich_club_ks, undirected_descriptors)
from .windows import FULL_PERIOD, assign_windows, calendar_windows

logger = logging.getLogger("study_A")

STABILITY_MEASURES = ["pagerank", "in_strength", "out_strength", "betweenness", "hub", "authority"]
PACKAGES = ["numpy", "pandas", "scipy", "networkx", "igraph", "leidenalg", "matplotlib"]


def configure_logging(log_file: Path | None = None) -> None:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def _write(table: pd.DataFrame | list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(table) if isinstance(table, list) else table
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


class StepLog:
    """Collects one status row per analysis step (status, duration, error)."""

    def __init__(self, token: str):
        self.token = token
        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def step(self, name: str, **context: Any) -> Iterator[None]:
        started = time.perf_counter()
        row = {"token": self.token, "step": name, **context}
        try:
            yield
            row.update(status="ok", error=None)
        except Exception as exc:  # Recorded, logged and skipped; never silently zero-filled.
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            logger.exception("%s | %s %s failed", self.token, name, context)
        finally:
            row["seconds"] = round(time.perf_counter() - started, 2)
            self.rows.append(row)


def _window_rows(events: pd.DataFrame, config: StudyAConfig) -> pd.DataFrame:
    manifest = calendar_windows(events["date"], config.windows.freq, config.windows.min_coverage)
    if config.windows.include_full_period:
        first, last = events["date"].min(), events["date"].max()
        days = (last - first).days + 1
        manifest = pd.concat([manifest, pd.DataFrame([{
            "window": FULL_PERIOD, "window_index": 0, "start_inclusive": first,
            "end_exclusive": last + pd.Timedelta(days=1), "calendar_days": days,
            "covered_days": days, "coverage": 1.0, "partial": False}])], ignore_index=True)
    return manifest


def analyse_token(token: str, config: StudyAConfig, run_dir: Path) -> dict[str, Any]:
    """Run every Study A analysis for one token and write its tables."""
    out = run_dir / _safe(token)
    out.mkdir(parents=True, exist_ok=True)
    configure_logging(out / "token.log")
    steps = StepLog(token)
    started = time.perf_counter()
    loaded = load_token(token, config.paths.data_dir, config.paths.cache_dir)
    events = loaded.events
    identifiers = address_format(pd.concat([events["source"], events["destination"]], ignore_index=True))
    audit = {**loaded.audit, **identifiers,
             "zero_address_identifiable": identifiers["address_format"] == "hex"}
    if not audit["zero_address_identifiable"]:
        logger.warning("%s: node IDs are %s; the zero-address filter cannot apply", token,
                       identifiers["address_format"])
    (out / "dedup_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    _write(loaded.by_date, out / "dedup_by_date.csv")
    _write(loaded.by_snapshot, out / "snapshot_structure.csv")
    _write(daily_activity(events), out / "daily_activity.csv")

    manifest = _window_rows(events, config)
    _write(manifest, out / "window_manifest.csv")
    calendar = manifest.loc[manifest["window"] != FULL_PERIOD]
    events = events.assign(window=assign_windows(events, calendar))

    collect: dict[str, list] = {name: [] for name in (
        "filter_audit", "variant_audit", "topology", "null_samples", "rich_club", "centrality_status",
        "centrality_topk", "centrality_correlation", "concentration", "concentration_curve", "communities",
        "community_sizes", "community_top_nodes", "community_quotient", "seed_stability",
        "resolution_sweep", "tail_fits", "turnover", "rank_stability", "community_persistence")}
    previous: dict[str, Any] = {}
    cc, tc = config.centrality, config.topology
    windows = list(calendar["window"]) + ([FULL_PERIOD] if config.windows.include_full_period else [])

    for window in windows:
        info = manifest.set_index("window").loc[window]
        context = {"window": window, "partial": bool(info["partial"])}
        subset = events if window == FULL_PERIOD else events.loc[events["window"] == window]
        logger.info("%s | window %s | %s unique events", token, window, f"{int(subset['multiplicity'].sum()):,}")
        pairs, audit = filter_pairs(aggregate_pairs(subset),
                                    exclude_self_loops=config.filters.exclude_self_loops,
                                    exclude_zero_address=config.filters.exclude_zero_address)
        collect["filter_audit"].append({**context, **audit})
        _write(pairs, out / "graphs" / f"{_safe(window)}_pairs.csv.gz")
        if pairs.empty:
            steps.rows.append({"token": token, "step": "window", **context, "status": "skipped",
                               "error": "no edges after filters", "seconds": 0})
            continue

        base_edges, _ = variant_edges(pairs, "frequency")
        base = build_bundle(base_edges, "frequency")
        topo: dict[str, Any] = dict(context)
        with steps.step("topology", **context):
            topo.update(directed_descriptors(base.directed))
            shares, labels = bowtie(base.directed)
            topo.update(shares)
            topo.update(directed_assortativity(base.directed))
            undirected, coreness = undirected_descriptors(
                base.undirected, exact_max_nodes=tc.exact_path_max_nodes,
                n_sources=tc.path_sources, seed=config.seed)
            topo.update(undirected)
            _write(pd.DataFrame({"node": base.names, "bowtie": labels, "coreness": coreness,
                                 "undirected_degree": base.undirected.degree()}),
                   out / "nodes" / f"{_safe(window)}_node_topology.csv.gz")
        run_nulls = window != FULL_PERIOD or config.null_model.run_on_full_period
        if run_nulls and "average_path_length" in topo:
            with steps.step("null_models", **context):
                degrees = np.asarray(base.undirected.degree())
                ks = rich_club_ks(degrees, tc.rich_club_points)
                samples, rich_null = null_samples(
                    base.undirected, n_null=config.null_model.n_null,
                    swaps_per_edge=config.null_model.swaps_per_edge, seed=config.seed,
                    exact_max_nodes=tc.exact_path_max_nodes, n_sources=tc.path_sources, rich_ks=ks)
                topo.update(small_world_summary(topo, samples))
                topo["null_model"] = (f"degree-preserving double-edge swaps, "
                                      f"{config.null_model.swaps_per_edge} x edges, simple graph")
                collect["null_samples"].extend(samples.assign(**context).to_dict("records"))
                observed = rich_club_curve(base.undirected, ks)
                for i, k in enumerate(ks):
                    null_values = rich_null[:, i] if rich_null.size else np.array([np.nan])
                    mean = float(np.nanmean(null_values))
                    collect["rich_club"].append({
                        **context, "k": int(k), "nodes_above_k": int((degrees > k).sum()),
                        "phi_observed": float(observed[i]), "phi_null_mean": mean,
                        "phi_null_sd": float(np.nanstd(null_values, ddof=1)),
                        "rho": float(observed[i] / mean) if mean > 0 else np.nan})
        collect["topology"].append(topo)

        current: dict[str, Any] = {"nodes": set(base.names), "pairs": pairs}
        for variant in cc.variants:
            vcontext = {**context, "variant": variant}
            edges, vaudit = variant_edges(pairs, variant,
                                          drop_zero_volume=config.filters.drop_zero_volume_pairs_in_volume_graphs)
            collect["variant_audit"].append({**context, **vaudit})
            if edges.empty:
                continue
            bundle = build_bundle(edges, variant)
            table = None
            with steps.step("centralities", **vcontext):
                table, status = compute_centralities(
                    bundle, seed=config.seed, exact_max_nodes=cc.exact_max_nodes,
                    betweenness_pivots=cc.betweenness_pivots,
                    harmonic_top_candidates=cc.harmonic_top_candidates,
                    harmonic_random_candidates=cc.harmonic_random_candidates,
                    damping=cc.pagerank_damping)
                _write(table, out / "centralities" / f"{_safe(window)}__{variant}.csv.gz")
                collect["centrality_status"].extend(status.assign(**vcontext).to_dict("records"))
                collect["centrality_topk"].extend(top_k(table, cc.top_k).assign(**vcontext).to_dict("records"))
                for row in concentration_table(table).to_dict("records"):
                    collect["concentration"].append({**vcontext, **row})
                if window == FULL_PERIOD:
                    corr = centrality_correlations(table)
                    collect["centrality_correlation"].extend(
                        corr.rename_axis("measure_a").reset_index()
                        .melt(id_vars="measure_a", var_name="measure_b", value_name="spearman")
                        .assign(variant=variant).to_dict("records"))
                    for quantity in QUANTITIES:
                        collect["concentration_curve"].extend(concentration_curve(table[quantity].to_numpy())
                                                 .assign(variant=variant, quantity=quantity).to_dict("records"))
            current[f"centralities_{variant}"] = table

            memberships: dict[str, np.ndarray] = {}
            with steps.step("communities", **vcontext):
                u = bundle.undirected
                gamma = config.communities.resolution
                memberships = {"leiden": leiden(u, resolution=gamma, seed=config.seed),
                               "louvain": louvain(u, resolution=gamma, seed=config.seed)}
                agreement = {"ari_leiden_louvain": adjusted_rand(memberships["leiden"], memberships["louvain"]),
                             "nmi_leiden_louvain": normalized_mutual_info(memberships["leiden"], memberships["louvain"])}
                for algorithm, membership in memberships.items():
                    collect["communities"].append({**vcontext, "algorithm": algorithm, "resolution": gamma,
                                                   "seed": config.seed, **partition_summary(u, membership, gamma),
                                                   **agreement})
                    sizes = np.bincount(membership)
                    collect["community_sizes"].extend({**vcontext, "algorithm": algorithm, "community": c,
                                                       "size": int(s)} for c, s in enumerate(sizes))
                    frame = pd.DataFrame({"node": u.vs["name"], "community": membership})
                    _write(frame, out / "communities" / f"{_safe(window)}__{variant}__{algorithm}.csv.gz")
                    current[f"communities_{variant}_{algorithm}"] = frame
                    if table is not None:
                        joined = frame.merge(table[["node", "pagerank", "in_strength", "out_strength"]], on="node")
                        top = (joined.loc[joined["community"] < 20]
                               .sort_values(["community", "pagerank", "node"], ascending=[True, False, True])
                               .groupby("community").head(config.communities.top_central_per_community))
                        collect["community_top_nodes"].extend(top.assign(**vcontext, algorithm=algorithm)
                                                              .to_dict("records"))
                if window == FULL_PERIOD:
                    q = quotient(u, memberships["leiden"])
                    collect["community_quotient"].extend(q.assign(variant=variant, algorithm="leiden").to_dict("records"))
            if window == FULL_PERIOD and memberships:
                with steps.step("community_stability", **vcontext):
                    seeds = [config.seed + i for i in range(config.communities.stability_seeds)]
                    collect["seed_stability"].extend(seed_stability(u, resolution=gamma, seeds=seeds)
                                                     .assign(variant=variant).to_dict("records"))
                    collect["resolution_sweep"].extend(resolution_sweep(
                        u, resolutions=config.communities.resolution_sweep, seed=config.seed)
                        .assign(variant=variant).to_dict("records"))
            if window == FULL_PERIOD and table is not None:
                _tail_fits(token, table, base.undirected.degree(), variant, config, collect, steps)

        if previous and window != FULL_PERIOD:
            _compare_windows(previous, current, config, collect, steps, token)
        if window != FULL_PERIOD:
            previous = {**current, "window": window, "partial": bool(info["partial"])}

    for name, rows in collect.items():
        _write(rows, out / f"{name}.csv")
    _write(steps.rows, out / "analysis_status.csv")
    elapsed = time.perf_counter() - started
    logger.info("%s | finished in %.1f s", token, elapsed)
    failed = sum(1 for r in steps.rows if r["status"] == "failed")
    result = {"token": token, "seconds": round(elapsed, 1), "failed_steps": failed,
              "steps": len(steps.rows), "unique_events": loaded.audit["unique_events"]}
    # Written last: a token folder without this marker is an interrupted analysis.
    (out / COMPLETE_MARKER).write_text(json.dumps({
        "analysis_fingerprint": analysis_fingerprint(config),
        "finished_utc": datetime.now(timezone.utc).isoformat(), **result}, indent=2), encoding="utf-8")
    return result


def _tail_fits(token: str, table: pd.DataFrame, undirected_degree: list[int], variant: str,
               config: StudyAConfig, collect: dict[str, list], steps: StepLog) -> None:
    dc = config.distributions
    series: list[tuple[str, np.ndarray, bool]] = []
    if variant == "frequency":
        series += [("in_degree", table["in_degree"].to_numpy(), True),
                   ("out_degree", table["out_degree"].to_numpy(), True),
                   ("undirected_degree", np.asarray(undirected_degree), True),
                   ("in_strength", table["in_strength"].to_numpy(), True),
                   ("out_strength", table["out_strength"].to_numpy(), True)]
    elif variant == "raw_volume":
        series += [("in_strength", table["in_strength"].to_numpy(), False),
                   ("out_strength", table["out_strength"].to_numpy(), False)]
    for quantity, values, discrete in series:
        with steps.step("tail_fit", window=FULL_PERIOD, variant=variant, quantity=quantity):
            result = analyse_tail(values, discrete=discrete, min_tail=dc.min_tail,
                                  max_candidates=dc.max_xmin_candidates, n_boot=dc.gof_bootstrap,
                                  seed=config.seed)
            collect["tail_fits"].append({"window": FULL_PERIOD, "variant": variant,
                                         "quantity": quantity, **result})
            logger.info("%s | tail %s/%s: %s", token, variant, quantity, result.get("interpretation"))


def _compare_windows(previous: dict[str, Any], current: dict[str, Any], config: StudyAConfig,
                     collect: dict[str, list], steps: StepLog, token: str) -> None:
    context = {"previous_window": previous["window"], "previous_partial": previous["partial"]}
    with steps.step("temporal_comparison", **context):
        collect["turnover"].append({**context, **address_turnover(previous["nodes"], current["nodes"]),
                                    **tie_persistence(previous["pairs"], current["pairs"])})
        for variant in config.centrality.variants:
            prev_table, curr_table = previous.get(f"centralities_{variant}"), current.get(f"centralities_{variant}")
            if prev_table is not None and curr_table is not None:
                collect["rank_stability"].extend(rank_stability(
                    prev_table, curr_table, STABILITY_MEASURES, config.centrality.rank_stability_k)
                    .assign(**context, variant=variant).to_dict("records"))
            for algorithm in ("leiden", "louvain"):
                prev_c = previous.get(f"communities_{variant}_{algorithm}")
                curr_c = current.get(f"communities_{variant}_{algorithm}")
                if prev_c is not None and curr_c is not None:
                    detail = persistence(prev_c, curr_c, min_size=config.communities.persistence_min_size)
                    collect["community_persistence"].extend(
                        detail.assign(**context, variant=variant, algorithm=algorithm).to_dict("records"))


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def _versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def load_or_build_profile(config: StudyAConfig, workers: int) -> pd.DataFrame:
    path = config.paths.output_root / "profile" / "token_profile.csv"
    if path.is_file():
        profile = pd.read_csv(path)
        if ("preprocessing_version" in profile and "address_format" in profile and
                set(profile["preprocessing_version"].dropna().astype(str)) == {PREPROCESSING_VERSION}):
            logger.info("Using token profile %s", path)
            return profile
    profile = profile_all(config.paths.data_dir, config.paths.cache_dir, workers=workers)
    _write(profile, path)
    return profile


COMPLETE_MARKER = "token_complete.json"
# Sections that change per-token results; [run], [selection], [overlap] and [figures] do not.
ANALYSIS_SECTIONS = ["seed", "windows", "filters", "centrality", "topology", "null_model",
                     "communities", "distributions"]


def analysis_fingerprint(config: StudyAConfig | dict[str, Any],
                         preprocessing_version: str = PREPROCESSING_VERSION) -> str:
    """Hash of every setting that affects a token's analysis (plus the preprocessing version)."""
    values = config.to_dict() if isinstance(config, StudyAConfig) else config
    payload = {key: values[key] for key in ANALYSIS_SECTIONS}
    payload["preprocessing_version"] = preprocessing_version
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def completed_tokens(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Tokens of a run whose analysis finished.

    A token is complete when its folder holds the completion marker. Runs made before
    the marker existed are recognised by ``analysis_status.csv`` (always the last file
    written by the analysis); their fingerprint is derived from the run's saved
    ``resolved_config.json`` and preprocessing version, and the marker is written.
    """
    done: dict[str, dict[str, Any]] = {}
    if not run_dir.is_dir():
        return done
    for marker in run_dir.glob(f"*/{COMPLETE_MARKER}"):
        done[marker.parent.name] = json.loads(marker.read_text(encoding="utf-8"))
    resolved = run_dir / "resolved_config.json"
    metadata = run_dir / "run_metadata.json"
    if not (resolved.is_file() and metadata.is_file()):
        return done
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    for status in run_dir.glob("*/analysis_status.csv"):
        token = status.parent.name
        if token in done or token not in saved.get("selected_tokens", []):
            continue
        info = {"analysis_fingerprint": analysis_fingerprint(
                    json.loads(resolved.read_text(encoding="utf-8")),
                    str(saved.get("preprocessing_version", PREPROCESSING_VERSION))),
                "finished_utc": saved.get("finished_utc"), "recognised_from": "analysis_status.csv",
                **next((r for r in saved.get("token_results", []) if r.get("token") == token), {"token": token})}
        (status.parent / COMPLETE_MARKER).write_text(json.dumps(info, indent=2), encoding="utf-8")
        done[token] = info
    return done


def run(config: StudyAConfig, *, config_path: Path | None, workers: int = 1, make_figures: bool = True,
        run_id: str | None = None) -> Path:
    """Analyse the selected tokens into ``runs/<run_id>``.

    If the run directory already exists, its completed tokens are kept and reused
    (not recomputed) provided they were analysed with the same analytical settings;
    selected tokens not yet completed are analysed, and the cross-token overlap,
    summaries and figures are rebuilt over all tokens of the run.
    """
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir / "run.log")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc)
    fingerprint = analysis_fingerprint(config)
    existing = completed_tokens(run_dir)
    mismatched = sorted(t for t, info in existing.items() if info.get("analysis_fingerprint") != fingerprint)
    if mismatched:
        raise ValueError(
            f"run {run_id!r} already contains {mismatched} analysed with different analytical settings "
            f"({', '.join(ANALYSIS_SECTIONS)}). Restore those settings or use a new run_id.")
    previous_path = run_dir / "run_metadata.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    invocation_dir = run_dir / "invocations"
    invocation_dir.mkdir(exist_ok=True)
    stamp = started_utc.strftime("%Y%m%dT%H%M%SZ")
    if config_path is not None:
        shutil.copyfile(config_path, run_dir / "config.toml")
        shutil.copyfile(config_path, invocation_dir / f"{stamp}_config.toml")
    (run_dir / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    metadata: dict[str, Any] = {"run_id": run_id, "started_utc": previous.get("started_utc", started_utc.isoformat()),
                                "preprocessing_version": PREPROCESSING_VERSION, "seed": config.seed,
                                "analysis_fingerprint": fingerprint,
                                "git": _git_state(), "environment": _versions(),
                                "distance_rule": DISTANCE_RULE, "variants": VARIANT_LABELS}

    profile = load_or_build_profile(config, workers=max(workers, config.selection.profile_workers))
    sc = config.selection
    selection = select_tokens(profile, mode=sc.mode, tokens=sc.tokens, n_tokens=sc.n_tokens,
                              min_coverage_days=sc.min_coverage_days, max_nodes=sc.max_nodes,
                              rank_by=sc.rank_by, require_hex_addresses=sc.require_hex_addresses)
    requested = selection.loc[selection["selected"], "token"].tolist()
    earlier = [t for t in previous.get("selected_tokens", sorted(existing)) if t in existing]
    earlier += sorted(t for t in existing if t not in earlier)
    to_analyse = [t for t in requested if t not in existing]
    tokens = earlier + to_analyse
    for token in to_analyse:  # an interrupted earlier attempt leaves an incomplete folder
        if (run_dir / _safe(token)).is_dir() and not (run_dir / _safe(token) / COMPLETE_MARKER).is_file():
            logger.info("%s: removing incomplete folder from an interrupted attempt", token)
            shutil.rmtree(run_dir / _safe(token))
    kept = selection["token"].isin(earlier)
    selection.loc[kept, "selected"] = True
    selection.loc[kept, "reason"] = "completed in an earlier invocation of this run (reused)"
    _write(profile, run_dir / "selection" / "token_profile.csv")
    _write(selection, run_dir / "selection" / "token_selection.csv")
    metadata["selected_tokens"] = tokens
    metadata["selection_criterion"] = (
        f"mode={sc.mode}; hex addresses required={sc.require_hex_addresses}; "
        f"coverage >= {sc.min_coverage_days} days; addresses <= {sc.max_nodes:,}; "
        f"top {sc.n_tokens} by {sc.rank_by}" if sc.mode == "ranked" else f"explicit: {tokens}")
    sources = dict(previous.get("source_files", {}))
    sources.update({t: _file_fingerprint(token_path(config.paths.data_dir, t), with_hash=True)
                    for t in to_analyse})
    metadata["source_files"] = {t: sources[t] for t in tokens if t in sources}
    logger.info("Run %s | reused: %s | to analyse: %s", run_id, ", ".join(earlier) or "-",
                ", ".join(to_analyse) or "-")

    if workers > 1 and len(to_analyse) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(to_analyse))) as pool:
            list(pool.map(analyse_token, to_analyse, [config] * len(to_analyse), [run_dir] * len(to_analyse)))
    else:
        for token in to_analyse:
            analyse_token(token, config, run_dir)
    configure_logging(run_dir / "run.log")
    metadata["token_results"] = [{k: v for k, v in info.items() if k != "analysis_fingerprint"}
                                 for t, info in completed_tokens(run_dir).items() if t in tokens]
    metadata["invocations"] = previous.get("invocations", []) + [{
        "started_utc": started_utc.isoformat(), "config": f"invocations/{stamp}_config.toml",
        "requested": requested, "analysed": to_analyse, "reused": earlier, "git": metadata["git"]}]

    if config.overlap.enabled and len(tokens) > 1:
        from .overlap import cross_token_overlap
        try:
            cross_token_overlap(run_dir, tokens, config)
            metadata["cross_token_overlap"] = "ok"
        except Exception as exc:  # Recorded; per-token results remain valid.
            logger.exception("Cross-token overlap failed")
            metadata["cross_token_overlap"] = f"failed: {type(exc).__name__}: {exc}"

    from .reporting import build_summaries
    build_summaries(run_dir, tokens)
    if make_figures:
        from .figures import make_all_figures
        make_all_figures(run_dir, config)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["seconds"] = round(time.perf_counter() - started, 1)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Run %s finished in %.1f s -> %s", run_id, metadata["seconds"], run_dir)
    return run_dir


def rebuild_figures(config: StudyAConfig, run_id: str) -> Path:
    """Regenerate summary tables and figures of an existing run from its saved tables only.

    Nothing is recomputed: figures read the CSV artifacts, so the current [figures]
    settings (formats, dpi, titles, drawing sizes) can be changed and re-applied.
    """
    if not run_id:
        raise ValueError("set [run] run_id to the run whose figures should be rebuilt")
    run_dir = config.paths.output_root / "runs" / run_id
    if not (run_dir / "selection" / "token_selection.csv").is_file():
        raise FileNotFoundError(f"no completed run at {run_dir}")
    configure_logging(run_dir / "figures.log")
    from .figures import make_all_figures
    from .reporting import build_summaries, selected_tokens

    tokens = selected_tokens(run_dir)
    missing = [t for t in tokens if t not in completed_tokens(run_dir)]
    if missing:
        raise ValueError(f"tokens {missing} of run {run_id!r} are not complete; run the analysis first")
    build_summaries(run_dir, tokens)
    status = make_all_figures(run_dir, config)
    failed = status.loc[status["status"] != "ok", "figure_function"].tolist()
    logger.info("Figures of %s rebuilt (%d ok, failed: %s)", run_id, int((status["status"] == "ok").sum()),
                failed or "none")
    return run_dir
