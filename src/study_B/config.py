"""Typed, file-driven configuration for Study B (same pattern as Study C)."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "study_B.toml"

HEURISTICS = ("dst_popularity", "preferential_attachment", "common_neighbours")
# *_recent: same GNN, trained and early-stopped on recent-receiver negatives; *_mixed: each training
# negative drawn from the uniform or the recent pool with probability 1/2 (assistant.md §6.3)
GNNS = ("graphsage", "gat", "graphsage_recent", "gat_recent", "graphsage_mixed", "gat_mixed")
MODELS = HEURISTICS + ("node2vec",) + GNNS


@dataclass
class RunConfig:
    run_id: str = ""
    device: str = "auto"


@dataclass
class PathsConfig:
    data_dir: Path = PROJECT_ROOT / "datasets" / "TGS_edgelists"
    cache_dir: Path = PROJECT_ROOT / "reports" / "study_A" / "cache"
    output_root: Path = PROJECT_ROOT / "reports" / "study_B"


@dataclass
class DataConfig:
    token: str = "SWAP"
    freq: str = "W-MON"


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    min_history_windows: int = 4
    drop_partial_targets: bool = True


@dataclass
class SamplingConfig:
    n_neg_eval: int = 50
    n_neg_train: int = 5
    negative_sets: list[str] = field(default_factory=lambda: ["uniform", "recent"])


@dataclass
class ModelsConfig:
    enabled: list[str] = field(default_factory=lambda: list(HEURISTICS) + ["node2vec", "graphsage"])


@dataclass
class Node2VecConfig:
    dim: int = 64
    walk_length: int = 20
    walks_per_node: int = 10
    context_size: int = 5
    epochs: int = 1
    lr: float = 0.01
    refit_every: int = 4
    batch_size: int = 128
    # (p, q) pairs; the pair with the best uniform-val AP is the only one scored on test
    pq_pairs: list[list[float]] = field(default_factory=lambda: [[0.25, 1.0], [1.0, 1.0], [0.25, 4.0],
                                                                 [1.0, 4.0], [4.0, 4.0]])

@dataclass
class NeuralConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 20
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class GATConfig:
    heads: int = 2


@dataclass
class EvaluationConfig:
    hits_k: list[int] = field(default_factory=lambda: [1, 10])


@dataclass
class FigureConfig:
    formats: list[str] = field(default_factory=lambda: ["png"])
    dpi: int = 300


@dataclass
class StudyBConfig:
    seed: int = 42
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    node2vec: Node2VecConfig = field(default_factory=Node2VecConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value
        return convert(asdict(self))


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown keys for [{cls.__name__}]: {unknown}")
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        current = getattr(defaults, name)
        if is_dataclass(current):
            kwargs[name] = _build(type(current), value)
        elif isinstance(current, Path):
            path = Path(value)
            kwargs[name] = path if path.is_absolute() else PROJECT_ROOT / path
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> StudyBConfig:
    """Read the TOML file into a validated ``StudyBConfig`` (missing keys keep defaults)."""
    with open(Path(path) if path is not None else DEFAULT_CONFIG, mode="rb") as f:
        config = _build(StudyBConfig, tomllib.load(f))
    validate(config)
    return config


def validate(config: StudyBConfig) -> None:
    """Fail fast on inconsistent settings, before any expensive step runs."""
    unknown = [m for m in config.models.enabled if m not in MODELS]
    if unknown or not config.models.enabled:
        raise ValueError(f"models.enabled must be a non-empty subset of {list(MODELS)}")
    s = config.split
    if not (0 < s.train_frac < 1 and 0 < s.val_frac < 1 and s.train_frac + s.val_frac < 1):
        raise ValueError("split: need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    if config.sampling.n_neg_eval < 1 or config.sampling.n_neg_train < 1:
        raise ValueError("sampling.n_neg_eval and n_neg_train must be >= 1")
    sets = config.sampling.negative_sets
    if "uniform" not in sets or not set(sets) <= {"uniform", "recent"} or len(set(sets)) != len(sets):
        raise ValueError("sampling.negative_sets: distinct values from {uniform, recent}, must include "
                         "'uniform' (GNN early stopping uses the uniform validation pairs)")
    if any(m.endswith(("_recent", "_mixed")) for m in config.models.enabled) and "recent" not in sets:
        raise ValueError("*_recent / *_mixed GNNs need 'recent' in sampling.negative_sets "
                         "(their early stopping uses the recent validation pairs)")
    # torch-cluster 1.6.3 chains its rejection tests (if / else if), so the walk is node2vec-correct
    # only for q >= 1 and p <= q (assistant.md §6.3)
    pairs = config.node2vec.pq_pairs
    if not pairs or not all(len(pq) == 2 and 0 < pq[0] <= pq[1] and pq[1] >= 1 for pq in pairs):
        raise ValueError("node2vec.pq_pairs: non-empty list of [p, q] with q >= 1 and 0 < p <= q")
    if config.run.device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"run.device must be auto/cuda/cpu, got {config.run.device!r}")
    if not config.neural.seeds or not 0 <= config.neural.dropout < 1:
        raise ValueError("neural.seeds must be non-empty and neural.dropout in [0, 1)")
