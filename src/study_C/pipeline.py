"""
End-to-end Study C run: validate data, split, train every model, evaluate, report.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import data as data_module,models,train,evaluate
from . import features, figures, graph, reporting, splits
from .config import StudyCConfig
from .data import ILLICIT, LICIT

logger = logging.getLogger("study_C")
TORCH_MODELS = ("mlp", "graphsage", "gat")
BASELINES = ("degree_baseline",)


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


def _evaluate_and_save(run_dir: Path, model_name: str, feature_set: str, seed: int, pred_val: pd.DataFrame,
                       pred_test: pd.DataFrame, nodes: pd.DataFrame,
                       threshold_metric: str) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Same evaluation for every model: threshold on validation, applied unchanged to test."""
    threshold = evaluate.select_threshold(pred_val["y_true"].to_numpy(), pred_val["y_score"].to_numpy(),
                                          threshold_metric)
    metrics = {split: evaluate.classification_metrics(pred["y_true"].to_numpy(), pred["y_score"].to_numpy(),
                                                      threshold)
               for split, pred in (("val", pred_val), ("test", pred_test))}
    per_step = evaluate.per_step_metrics(pred_test, nodes, threshold)
    keys = {"model": model_name, "feature_set": feature_set, "seed": seed}

    out = reporting.model_directory(run_dir, model_name, feature_set, seed)
    reporting.write_table(pred_val, out / "predictions_val.csv")
    reporting.write_table(pred_test, out / "predictions_test.csv")
    reporting.write_table(per_step, out / "per_step_test.csv")
    reporting.write_json(keys | {"threshold_metric": threshold_metric, "threshold": threshold} | metrics,
                         out / "metrics.json")
    logger.info("%s / %s / seed %d: threshold %.4f, val PR-AUC %.4f F1 %.4f, test PR-AUC %.4f F1 %.4f",
                model_name, feature_set, seed, threshold, metrics["val"]["pr_auc"], metrics["val"]["f1"],
                metrics["test"]["pr_auc"], metrics["test"]["f1"])
    rows = [keys | {"split": split} | values for split, values in metrics.items()]
    return rows, per_step.assign(**keys)


def _train_and_save_torch(run_dir: Path, config: StudyCConfig, model_name: str, feature_set: str, seed: int,
                          graphs: graph.PhaseGraphs, pos_weight: float, device: torch.device,
                          nodes: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Train one torch model with one seed, evaluate it like the RF, save history and weights."""
    train.set_seed(seed)   # before creating the model: the seed also fixes the weight initialisation
    model = models.make_torch_model(model_name, graphs.train.num_features, config.neural, config.gat)
    start = time.perf_counter()
    result = train.train_torch(model, graphs.train, graphs.val, config.neural, pos_weight, device)
    seconds = time.perf_counter() - start
    pred_val = evaluate.predict_graph(result.model, graphs.val)
    pred_test = evaluate.predict_graph(result.model, graphs.test.to(device))
    model_rows, per_step = _evaluate_and_save(run_dir, model_name, feature_set, seed, pred_val, pred_test, nodes,
                                              config.evaluation.threshold_metric)
    out = reporting.model_directory(run_dir, model_name, feature_set, seed)
    reporting.write_table(result.history, out / "history.csv")
    torch.save(result.model.state_dict(), out / "model.pt")
    info = {"best_epoch": result.best_epoch, "n_epochs": len(result.history), "train_seconds": round(seconds, 1)}
    logger.info("%s / %s / seed %d: best epoch %d of %d, %.1f s", model_name, feature_set, seed,
                info["best_epoch"], info["n_epochs"], seconds)
    return model_rows, per_step, info


def _per_step_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean ± std over seeds of the per-step test metrics, per model, feature set and step."""
    grouped = pd.concat(frames, ignore_index=True).groupby(["model", "feature_set", "time_step"], sort=True)
    summary = grouped[["precision", "recall", "f1"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    # support and n depend only on the step, not on the seed: report them once.
    summary.insert(0, "support_illicit", grouped["support_illicit"].first())
    summary.insert(0, "n", grouped["n"].first())
    summary.insert(0, "n_seeds", grouped["seed"].nunique())
    return summary.reset_index()


def _select_model(summary: pd.DataFrame) -> dict[str, Any]:
    """Best model/feature set by mean VALIDATION PR-AUC; the test set plays no part in the choice."""
    # non-learning references (degree_baseline) are not candidates
    val = summary[(summary["split"] == "val") & ~summary["model"].isin(BASELINES)]
    val = val.sort_values("pr_auc_mean", ascending=False)
    best = val.iloc[0]
    test = summary[(summary["split"] == "test") & (summary["model"] == best["model"])
                   & (summary["feature_set"] == best["feature_set"])].iloc[0]
    return {
        "criterion": "highest mean validation PR-AUC over seeds (test metrics not used for selection)",
        "model": best["model"],
        "feature_set": best["feature_set"],
        "n_seeds": int(best["n_seeds"]),
        "val_pr_auc_mean": float(best["pr_auc_mean"]),
        "test_pr_auc_mean": float(test["pr_auc_mean"]),
        "test_f1_mean": float(test["f1_mean"]),
        "ranking_by_val_pr_auc": [
            {"model": r.model, "feature_set": r.feature_set, "val_pr_auc_mean": float(r.pr_auc_mean)}
            for r in val.itertuples()],
    }


def run(config: StudyCConfig, config_path: Path) -> Path:
    """Execute the whole protocol and return the run directory."""
    # 0. Setup
    run_dir = reporting.run_directory(config)
    configure_logging(run_dir / "run.log")
    shutil.copy(config_path, run_dir / "config.toml")
    reporting.write_json(config.to_dict(), run_dir / "resolved_config.json")
    reporting.write_json(reporting.run_metadata(config, config.paths.data_dir), run_dir / "run_metadata.json")
    logger.info("Study C run in %s", run_dir)

    # 1. Data -> data/
    data = data_module.load_elliptic(config.paths.data_dir, config.features.n_local)
    profile = data_module.validate_elliptic(data)
    reporting.write_json(profile, run_dir / "data" / "data_profile.json")
    reporting.write_table(data_module.label_counts_by_step(data.nodes).reset_index(),
                          run_dir / "data" / "label_counts_by_step.csv")
    logger.info("Data: %d nodes, %d edges, labels %s", profile["n_nodes"], profile["n_edges"],
                profile["label_counts"])
    if profile["n_nodes_without_class"]:
        logger.warning("%d nodes have no row in %s", profile["n_nodes_without_class"], data_module.CLASSES_FILE)

    # 2. Split -> splits/
    data.nodes["phase"] = splits.assign_phases(data.nodes, config.split)
    audit = splits.check_split(data.nodes, data.edges)
    reporting.write_table(splits.split_manifest(data.nodes), run_dir / "splits" / "split_manifest.csv")
    reporting.write_table(splits.split_summary(data.nodes), run_dir / "splits" / "split_summary.csv")
    reporting.write_json(audit, run_dir / "splits" / "split_audit.json")
    logger.info("Split steps %s, cross-phase edges %d", audit["phase_steps"], audit["n_cross_phase_edges"])
    # Label homophily along edges, overall and per phase: design choices rest on train.
    label_pairs, homophily = data_module.edge_label_homophily(data.nodes, data.edges)
    reporting.write_table(label_pairs, run_dir / "data" / "edge_label_pairs.csv")
    reporting.write_json(homophily, run_dir / "data" / "edge_homophily.json")
    reporting.write_table(data_module.edge_homophily_by_step(data.nodes, data.edges),
                          run_dir / "data" / "edge_homophily_by_step.csv")
    logger.info("Edge label assortativity (labelled-labelled edges): %s",
                {scope: round(s["label_assortativity"], 3) for scope, s in homophily.items()})

    # 3. Experiment loop
    phase = data.nodes["phase"].astype(str).to_numpy()
    train_mask, val_mask, test_mask = (phase == "train"), (phase == "val"), (phase == "test")
    # isin (not ``!= -1``): a node without a row in the classes file has label NaN and is not labelled.
    labelled = data.nodes["label"].isin([ILLICIT, LICIT]).to_numpy()
    y = data.nodes["label"].fillna(-1).astype(np.int64).to_numpy()
    rows: list[dict[str, Any]] = []            # one row per model/feature set/seed/split
    per_step_frames: list[pd.DataFrame] = []   # per-step test metrics of every model
    status: list[dict[str, Any]] = []          # outcome of every training -> summary/model_status.csv
    device = train.resolve_device(config.run.device)
    logger.info("Torch device: %s%s", device,
                f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")

    # 3.0 Structure-only reference: score = degree in the graph visible to the phase (no training).
    if "degree_baseline" in config.models.enabled:
        tx_ids = data.nodes["txId"].to_numpy()
        last_step = int(data.nodes["time_step"].max())
        preds = {}
        for split_name, mask, max_step in (("val", val_mask, config.split.val_end), ("test", test_mask, last_step)):
            rows_mask = mask & labelled
            degree = graph.visible_degree(data.nodes, data.edges, max_step)
            preds[split_name] = pd.DataFrame({"txId": tx_ids[rows_mask], "y_true": y[rows_mask],
                                              "y_score": degree.reindex(tx_ids[rows_mask]).to_numpy(np.float64)})
        model_rows, model_per_step = _evaluate_and_save(run_dir, "degree_baseline", "structure", config.seed,
                                                        preds["val"], preds["test"], data.nodes,
                                                        config.evaluation.threshold_metric)
        rows += model_rows
        per_step_frames.append(model_per_step)
        status.append({"model": "degree_baseline", "feature_set": "structure", "seed": config.seed, "status": "ok"})

    for feature_set in config.features.feature_sets:
        columns = features.select_feature_columns(data, feature_set)
        logger.info("Feature set %r: %d columns", feature_set, len(columns))

        # 3a. Random Forest (if enabled) — a single seed (config.seed)
        #     X without scaler; fit on train & labelled; predict val and test.
        if "random_forest" in config.models.enabled:
            X_rf = features.build_matrix(data.nodes, columns, None)
            fit_rows = train_mask & labelled
            rf = models.make_random_forest(config.random_forest, config.seed)
            result = train.train_random_forest(rf, X_rf[fit_rows], y[fit_rows])
            val_pred= evaluate.predict_sklearn(result.model, X_rf[val_mask & labelled], data.nodes["txId"].to_numpy()[val_mask & labelled], y[val_mask & labelled])
            test_pred= evaluate.predict_sklearn(result.model, X_rf[test_mask & labelled], data.nodes["txId"].to_numpy()[test_mask & labelled], y[test_mask & labelled])
            model_rows, model_per_step = _evaluate_and_save(run_dir, "random_forest", feature_set, config.seed,
                                                            val_pred, test_pred, data.nodes,
                                                            config.evaluation.threshold_metric)
            rows += model_rows
            per_step_frames.append(model_per_step)
            status.append({"model": "random_forest", "feature_set": feature_set, "seed": config.seed, "status": "ok"})

        # 3b. Torch models (if enabled)
        # The scaler is fit on ALL train nodes (UNKNOWN too: features known at training time).
        scaler = (features.fit_scaler(data.nodes, columns, train_mask)
                  if config.features.standardize else None)
        X = features.build_matrix(data.nodes, columns, scaler)
        w = features.class_weights(y[train_mask & labelled])
        pos_weight = w[ILLICIT] / w[LICIT]
        logger.info("Feature set %r: X %s %s, class weights %s, pos_weight %.3f",
                    feature_set, X.shape, X.dtype, w, pos_weight)
        torch_models = [name for name in TORCH_MODELS if name in config.models.enabled]
        if not torch_models:
            continue
        graphs = graph.build_phase_graphs(data.nodes, data.edges, X, config.split.train_end, config.split.val_end)
        logger.info("Feature set %r graphs (nodes/edges): train %d/%d, val %d/%d, test %d/%d", feature_set,
                    *(n for g in (graphs.train, graphs.val, graphs.test) for n in (g.num_nodes, g.num_edges)))
        for model_name in torch_models:
            for seed in config.neural.seeds:
                keys = {"model": model_name, "feature_set": feature_set, "seed": seed}
                try:
                    model_rows, model_per_step, info = _train_and_save_torch(
                        run_dir, config, model_name, feature_set, seed, graphs, pos_weight, device, data.nodes)
                except Exception as error:   # a failing model (e.g. GAT OOM) does not stop the run
                    logger.exception("%s / %s / seed %d failed", model_name, feature_set, seed)
                    status.append(keys | {"status": "failed", "error": f"{type(error).__name__}: {error}"})
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                rows += model_rows
                per_step_frames.append(model_per_step)
                status.append(keys | {"status": "ok"} | info)

    # 4. Riepilogo -> summary/
    reporting.write_table(pd.DataFrame(status), run_dir / "summary" / "model_status.csv")
    if not rows:
        logger.warning("No model was trained: nothing to summarise")
        return run_dir
    all_rows, summary = reporting.summarise_metrics(rows)
    reporting.write_table(all_rows, run_dir / "summary" / "metrics_all.csv")
    reporting.write_table(summary, run_dir / "summary" / "metrics_summary.csv")
    reporting.write_table(_per_step_summary(per_step_frames), run_dir / "summary" / "per_step_summary.csv")
    selection = _select_model(summary)
    reporting.write_json(selection, run_dir / "summary" / "model_selection.json")
    logger.info("Selected %s / %s (val PR-AUC %.4f)", selection["model"], selection["feature_set"],
                selection["val_pr_auc_mean"])

    # 5. Figures -> figures/ (read from the files just saved: they can be redrawn without retraining)
    try:
        files = figures.make_figures(run_dir, config)
        logger.info("Figures (%d files) in %s", len(files), run_dir / "figures")
    except Exception:   # results are already saved: a failing figure does not invalidate the run
        logger.exception("Figures failed")
    logger.info("Run complete: %s", run_dir)
    return run_dir
