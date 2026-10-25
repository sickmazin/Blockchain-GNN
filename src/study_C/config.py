"""Typed, file-driven configuration for Study C.

Every tunable choice lives in ``study_C.toml``. The resolved
configuration is saved with each run so that metrics can be traced back to the
exact parameters.

"""

from __future__ import annotations

import tomllib
from dataclasses import asdict,dataclass,field,fields,is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "study_C.toml"

MODELS = ("degree_baseline", "random_forest", "mlp", "graphsage", "gat")
FEATURE_SETS = ("local", "all")


@dataclass
class RunConfig:
    run_id: str = ""
    device: str = "auto"


@dataclass
class PathsConfig:
    data_dir: Path = PROJECT_ROOT / "datasets" / "elliptic_bitcoin_dataset"
    output_root: Path = PROJECT_ROOT / "reports" / "study_C"


@dataclass
class SplitConfig:
    train_end: int = 34
    val_end: int = 39


@dataclass
class FeatureConfig:
    feature_sets: list[str] = field(default_factory=lambda: ["local", "all"])
    n_local: int = 93
    standardize: bool = True


@dataclass
class ModelsConfig:
    enabled: list[str] = field(default_factory=lambda: list(MODELS))


@dataclass
class RandomForestConfig:
    n_estimators: int = 500
    max_features: int | str = 50
    class_weight: str = "balanced"
    n_jobs: int = -1


@dataclass
class NeuralConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 5e-4
    max_epochs: int = 1000
    patience: int = 50
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclass
class GATConfig:
    heads: int = 4
    attention_dropout : bool = False
    ELU_activation: bool = False

@dataclass
class EvaluationConfig:
    threshold_metric: str = "f1"


@dataclass
class FigureConfig:
    formats: list[str] = field(default_factory=lambda: ["png"])
    dpi: int = 300


@dataclass
class StudyCConfig:
    seed: int = 42
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (Path -> str), saved as resolved_config.json."""
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
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"unknown keys for [{cls.__name__}]: {unknown}")
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in values:
            continue
        current = getattr(defaults, name)
        value = values[name]
        if is_dataclass(current):
            kwargs[name] = _build(type(current), value)
        elif isinstance(current, Path):
            path = Path(value)
            kwargs[name] = path if path.is_absolute() else PROJECT_ROOT / path
        else:
            kwargs[name] = value
    return cls(**kwargs)



def load_config(path: str | Path | None = None) -> StudyCConfig:
    """Read the TOML file into a validated ``StudyCConfig`` (missing keys keep defaults)."""
    with open(Path(path) if path is not None else DEFAULT_CONFIG, mode="rb") as config:
        values = tomllib.load(config)
    config = _build(StudyCConfig, values)
    validate(config)
    return config



def validate(config: StudyCConfig) -> None:
    """Fail fast on inconsistent settings, before any expensive step runs."""
    unknown = [v for v in config.models.enabled if v not in MODELS]
    if unknown or not config.models.enabled:
        raise ValueError(f"config.models.enabled must be a non-empty subset of {list(MODELS)}")
    unknown = [v for v in config.features.feature_sets if v not in FEATURE_SETS]
    if unknown or not config.features.feature_sets:
        raise ValueError(f"config.features.feature_sets must be a non-empty subset of {list(FEATURE_SETS)}")
    if not 1 <= config.split.train_end < config.split.val_end:
        raise ValueError(f"split must satisfy 1 <= train_end < val_end, got "
                         f"train_end={config.split.train_end}, val_end={config.split.val_end}")
    if config.run.device not in ['auto', 'cuda', 'cpu']:
        raise ValueError(f"Run device not correct: {config.run.device} not valide")
    if not config.neural.seeds or not 0 <= config.neural.dropout < 1:
        raise ValueError("neural.seeds must be non-empty and neural.dropout in [0, 1)")
