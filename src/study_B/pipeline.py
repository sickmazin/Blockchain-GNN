"""End-to-end Study B run. Saves ONLY the config and CSV tables; figures are drawn from the CSVs.

reports/study_B/runs/<run_id>/
    config.toml                      copy of the TOML used
    data/filter_counts.csv           mint / burn / self-loop removal (events and volume)
    splits/windows.csv               every window: dates, coverage, phase, n_events
    splits/targets.csv               every target window: history size, recent pool, positives, cold-start
    samples/pairs_<neg>_<phase>.csv  fixed evaluation pairs per negative set (same for every model)
    predictions/<model>[_seed<k>]_<neg>_<phase>.csv   pairs + score
    metrics/metrics.csv              one row per model, seed, negative set, phase
    metrics/per_window.csv           test metrics per window
    metrics/by_source_degree.csv     test metrics per source out-degree bin
    metrics/training_curves.csv      GNN epoch, train_loss, val_ap
    metrics/by_dst_recent.csv        test metrics split by true destination in / not in the recent pool
    metrics/node2vec_grid.csv        Node2Vec (p, q) grid: uniform-val AP / MRR, selected pair
    figures/

<neg> is a negative set of ``sampling.negative_sets`` ("uniform", "recent"; see samples.py).

Partial rerun: with the ``run_id`` of an existing run, only the models in ``models.enabled`` are
recomputed. The evaluation pairs are rebuilt and must equal the saved ones (else the run stops), the
other models' rows and prediction files are kept, ``config.toml`` stays the original and the TOML
used is saved as ``config_<model>.toml`` for every rerun model; all figures are redrawn.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import baselines, data, evaluate, features, figures, gnn, samples, splits
from .config import GNNS, HEURISTICS, StudyBConfig

logger = logging.getLogger("study_B")
DEGREE_BINS = [0, 1, 5, 20, 100, np.inf]   # source out-degree bins for the error analysis
EVAL_PHASES = ("val", "test")
# GNN name suffix -> (training negative pools, validation negative sets for early stopping)
GNN_TRAINING = {"": (("history_nodes",), ("uniform",)),
                "_recent": (("recent_receivers",), ("recent",)),
                "_mixed": (("history_nodes", "recent_receivers"), ("uniform", "recent"))}


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _score_heuristic(name: str, target: samples.TargetWindow, pairs: pd.DataFrame, n_nodes: int,
                     x: np.ndarray) -> np.ndarray:
    src, dst = pairs["src"].to_numpy(), pairs["dst"].to_numpy()
    if name == "dst_popularity":
        return baselines.dst_popularity(x, dst)
    fn = baselines.preferential_attachment if name == "preferential_attachment" else baselines.common_neighbours
    return fn(target, src, dst, n_nodes)


def _node2vec_embeddings(eval_targets: list[samples.TargetWindow], n_nodes: int, config: StudyBConfig,
                         device: torch.device, p: float, q: float) -> dict[int, np.ndarray]:
    """window_index -> embeddings. Refit every ``refit_every`` targets on the history before the
    FIRST target of the block, so every window of the block sees only its past."""
    cfg = config.node2vec
    out = {}
    for start in range(0, len(eval_targets), cfg.refit_every):
        block = eval_targets[start:start + cfg.refit_every]
        emb = baselines.fit_node2vec(block[0], n_nodes, cfg, device, config.seed, p, q)
        out |= {t.window_index: emb for t in block}
    return out


def _window_tensors(target: samples.TargetWindow, x: np.ndarray, pairs: pd.DataFrame | None,
                    device: torch.device) -> dict:
    """Tensors of one target window for gnn.py (pairs=None for train windows)."""
    out = {"x": torch.as_tensor(x, device=device),
           "edge_index": torch.as_tensor(target.history_edges, device=device),
           "history_nodes": torch.as_tensor(target.history_nodes, device=device),
           # empty only in the first windows: fall back to the uniform pool there
           "recent_receivers": torch.as_tensor(target.recent_receivers if len(target.recent_receivers)
                                               else target.history_nodes, device=device),
           "positives": torch.as_tensor(target.positives, device=device)}
    if pairs is not None:
        for col in ("src", "dst", "label"):
            out[col] = torch.as_tensor(pairs[col].to_numpy().copy(), device=device)
    return out


PAIR_COLUMNS = ["window_index", "src", "dst", "label", "group", "src_out_degree"]


_PRED_FILE = re.compile(r"^(?P<model>.+?)(?:_seed(?P<seed>\d+))?_(?P<negatives>uniform|recent)_test$")


def _by_dst_recent(pred_dir: Path, recent_pool: dict[int, np.ndarray], hits_k: list[int]) -> pd.DataFrame:
    """Test metrics split by whether the TRUE destination is itself in the recent-receiver pool.

    All recent negatives are in that pool, so a positive outside it is separable by activity alone;
    the "dst_recent" subset is the like-for-like comparison. Built from every saved test prediction.
    """
    rows = []
    for path in sorted(pred_dir.glob("*_test.csv")):
        keys = _PRED_FILE.match(path.stem).groupdict()
        pred = pd.read_csv(path)
        pos = pred[pred["label"] == 1]
        in_pool = {g for w, g, d in zip(pos["window_index"], pos["group"], pos["dst"])
                   if np.isin(d, recent_pool[w])}
        flag = pred["group"].isin(in_pool)
        for subset, part in (("dst_recent", pred[flag]), ("dst_not_recent", pred[~flag])):
            rows.append(keys | {"subset": subset} | evaluate.metrics(part, hits_k))
    return pd.DataFrame(rows)


def _check_same_pairs(table: pd.DataFrame, path: Path) -> None:
    """Partial rerun: the rebuilt evaluation pairs must equal the saved ones, or models are not comparable."""
    if not path.is_file():
        raise ValueError(f"partial rerun: {path} is missing; use a new run_id for a full run")
    saved = pd.read_csv(path)
    if not saved[PAIR_COLUMNS].equals(table[PAIR_COLUMNS].reset_index(drop=True).astype(saved[PAIR_COLUMNS].dtypes)):
        raise ValueError(f"partial rerun: evaluation pairs differ from {path}. Data, split or sampling "
                         "settings changed since the original run; use a new run_id for a full run")


def _replace_models(new: pd.DataFrame, path: Path, models: list[str], resume: bool) -> pd.DataFrame:
    """Partial rerun: saved rows of the other models + the new rows of the rerun models."""
    if not resume or not path.is_file():
        return new
    old = pd.read_csv(path)
    return pd.concat([old[~old["model"].isin(models)], new], ignore_index=True)


def _by_window(table: pd.DataFrame, targets: list[samples.TargetWindow]) -> list[pd.DataFrame]:
    """Rows of ``table`` split per target window, in the order of ``targets``."""
    return [table[table["window_index"] == t.window_index] for t in targets]


def run(config: StudyBConfig, config_path: Path) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    run_id = config.run.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.output_root / "runs" / run_id
    resume = (run_dir / "config.toml").is_file()
    enabled = list(config.models.enabled)
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        logger.info("partial rerun of %s: recomputing only %s", run_dir.name, enabled)
    else:
        shutil.copyfile(config_path, run_dir / "config.toml")
    device = _device(config.run.device)
    rng = np.random.default_rng(config.seed)

    # 1. Data and filters
    events, filter_counts, addresses = data.load_events(config.data.token, config.paths.data_dir,
                                                        config.paths.cache_dir)
    n_nodes = len(addresses)
    if not resume:
        _save(filter_counts, run_dir / "data" / "filter_counts.csv")

    # 2. Windows and phases
    manifest, events["window_index"] = splits.make_windows(events, config.data.freq)
    manifest = splits.assign_phases(manifest, config.split)
    manifest["n_events"] = manifest["window_index"].map(events.groupby("window_index")["multiplicity"].sum()).fillna(0).astype(int)
    if not resume:
        _save(manifest, run_dir / "splits" / "windows.csv")

    # 3. Target windows, features and fixed evaluation pairs
    pairs, node_first = samples.first_seen(events)
    targets = [samples.build_target(events, pairs, node_first, int(row.window_index), row.phase,
                                    recent_windows=features.RECENT_WINDOWS)
               for row in manifest.itertuples() if row.phase in splits.PHASES]
    if not resume:
        _save(samples.target_summary(targets), run_dir / "splits" / "targets.csv")
    logger.info("%d target windows, %d nodes", len(targets), n_nodes)

    x_raw = {t.window_index: features.node_features(events, t, n_nodes) for t in targets}
    scaler = features.fit_scaler(x_raw, targets)

    by_phase: dict[str, list[samples.TargetWindow]] = {p: [t for t in targets if t.phase == p] for p in splits.PHASES}
    # Negative sets in the outer loop: the "uniform" pairs do not depend on which other sets are enabled.
    eval_pairs: dict[tuple[str, str], pd.DataFrame] = {}
    for neg_set in config.sampling.negative_sets:
        for phase in EVAL_PHASES:
            tables, offset = [], 0
            for t in by_phase[phase]:
                negatives = samples.sample_negatives(t, config.sampling.n_neg_eval, rng, pool=neg_set)
                table = samples.eval_pairs(t, negatives, offset)
                # raw out-degree (the feature column is log1p) for the DEGREE_BINS error analysis
                table["src_out_degree"] = np.bincount(t.history_edges[0], minlength=n_nodes)[table["src"].to_numpy()]
                # windows since dst was last active (historical feature): why uniform negatives are easy
                table["dst_recency"] = x_raw[t.window_index][table["dst"].to_numpy(), features.FEATURES.index("recency")]
                offset += len(t.positives)
                tables.append(table)
            eval_pairs[(neg_set, phase)] = pd.concat(tables, ignore_index=True)
            path = run_dir / "samples" / f"pairs_{neg_set}_{phase}.csv"
            if resume:
                _check_same_pairs(eval_pairs[(neg_set, phase)], path)
            else:
                _save(eval_pairs[(neg_set, phase)], path)

    if resume:   # only now: a rerun stopped by the pair check must not overwrite a model's config
        for name in enabled:
            shutil.copyfile(config_path, run_dir / f"config_{name}.toml")

    # 4. Models: every model scores exactly the same saved pairs
    predictions: dict[tuple[str, int | None, str, str], pd.DataFrame] = {}
    curves = []
    for name in enabled:
        logger.info("model %s", name)
        if name in HEURISTICS:
            for (neg_set, phase), table in eval_pairs.items():
                scores = [_score_heuristic(name, t, part, n_nodes, x_raw[t.window_index])
                          for t, part in zip(by_phase[phase], _by_window(table, by_phase[phase]))]
                predictions[(name, None, neg_set, phase)] = table.assign(score=np.concatenate(scores))
        elif name == "node2vec":
            def n2v_predictions(phase: str, p: float, q: float) -> dict[tuple, pd.DataFrame]:
                emb = _node2vec_embeddings(by_phase[phase], n_nodes, config, device, p, q)
                out = {}
                for (neg_set, ph), table in eval_pairs.items():
                    if ph != phase:
                        continue
                    scores = [baselines.node2vec_score(emb[t.window_index], part["src"].to_numpy(),
                                                       part["dst"].to_numpy())
                              for t, part in zip(by_phase[phase], _by_window(table, by_phase[phase]))]
                    out[(name, None, neg_set, phase)] = table.assign(score=np.concatenate(scores))
                return out

            # (p, q) chosen on the uniform validation pairs only; test sees the chosen pair alone
            grid, best = [], None
            for p, q in config.node2vec.pq_pairs:
                val = n2v_predictions("val", p, q)
                m = evaluate.metrics(val[(name, None, "uniform", "val")], [])
                grid.append({"p": p, "q": q, "val_uniform_ap": m["ap"], "val_uniform_mrr": m["mrr"]})
                logger.info("node2vec p=%s q=%s val uniform AP=%.4f", p, q, m["ap"])
                if best is None or m["ap"] > best[0]:
                    best = (m["ap"], p, q, val)
            _, p, q, val = best
            grid_table = pd.DataFrame(grid).assign(selected=lambda d: (d.p == p) & (d.q == q))
            _save(grid_table, run_dir / "metrics" / "node2vec_grid.csv")
            predictions |= val | n2v_predictions("test", p, q)
        elif name in GNNS:
            scaled = {w: scaler.transform(x).astype(np.float32) for w, x in x_raw.items()}
            train_w = [_window_tensors(t, scaled[t.window_index], None, device) for t in by_phase["train"]]
            kind, _, variant = name.partition("_")
            pools, val_sets = GNN_TRAINING["_" + variant if variant else ""]
            if "recent_receivers" in pools:
                n_empty = sum(len(t.recent_receivers) == 0 for t in by_phase["train"])
                if n_empty:
                    logger.info("%s: %d train windows without recent receivers use uniform negatives", name, n_empty)
            eval_w = {key: [_window_tensors(t, scaled[t.window_index], part, device)
                            for t, part in zip(by_phase[key[1]], _by_window(table, by_phase[key[1]]))]
                      for key, table in eval_pairs.items()}
            for seed in config.neural.seeds:
                torch.manual_seed(seed)   # weight initialisation depends on the seed too
                model = gnn.LinkPredictor(len(features.FEATURES), config.neural, kind, config.gat).to(device)
                # early stopping on the validation pairs with the training negative distribution; for
                # *_mixed the mean AP over uniform + recent val windows (same windows -> mean of the two sets)
                val_w = [w for s in val_sets for w in eval_w[(s, "val")]]
                model, curve = gnn.train_model(model, train_w, val_w, config.neural,
                                               config.sampling.n_neg_train, seed, pools)
                curves.append(curve.assign(model=name, seed=seed))
                for key, windows in eval_w.items():
                    scores = np.concatenate([gnn.predict(model, w) for w in windows])
                    predictions[(name, seed, *key)] = eval_pairs[key].assign(score=scores)

    # 5. Predictions and metrics
    rows, per_window, by_degree = [], [], []
    hits_k = config.evaluation.hits_k
    for name in enabled:   # stale files of a rerun model (e.g. a seed no longer in the config)
        for old in (run_dir / "predictions").glob(f"{name}_*.csv"):
            old.unlink()
    for (name, seed, neg_set, phase), pred in predictions.items():
        suffix = f"_seed{seed}" if seed is not None else ""
        _save(pred, run_dir / "predictions" / f"{name}{suffix}_{neg_set}_{phase}.csv")
        keys = {"model": name, "seed": seed, "negatives": neg_set, "phase": phase}
        rows.append(keys | evaluate.metrics(pred, hits_k))
        if phase == "test":
            per_window.append(evaluate.per_window(pred, hits_k).assign(**keys))
            by_degree.append(evaluate.by_source_degree(pred, DEGREE_BINS, hits_k).assign(**keys))
    tables = {"metrics.csv": pd.DataFrame(rows),
              "per_window.csv": pd.concat(per_window, ignore_index=True),
              "by_source_degree.csv": pd.concat(by_degree, ignore_index=True),
              "training_curves.csv": pd.concat(curves, ignore_index=True) if curves else None}
    for file, new in tables.items():
        path = run_dir / "metrics" / file
        if new is None:   # no GNN in this run: keep the saved curves of a previous run, if any
            continue
        _save(_replace_models(new, path, enabled, resume), path)

    recent_pool = {t.window_index: t.recent_receivers for t in targets}
    _save(_by_dst_recent(run_dir / "predictions", recent_pool, hits_k), run_dir / "metrics" / "by_dst_recent.csv")

    # 6. Figures (only from the CSV files above)
    figures.make_all(run_dir, config.figures, n_neg=config.sampling.n_neg_eval)
    logger.info("run saved in %s", run_dir)
    return run_dir
