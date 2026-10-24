"""Typed, file-driven configuration for Study A.

Every tunable choice of the analysis lives here and is read from a TOML file
(default: ``configs/study_A.toml``). The resolved configuration is saved with
each run so that results can be traced back to the exact parameters.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "study_A.toml"


@dataclass
class RunConfig:
    # "all": analyse (reusing completed tokens of an existing run_id), then tables and figures.
    # "figures": only rebuild summary tables and figures of run_id from its saved tables.
    task: str = "all"
    workers: int = 2
    make_figures: bool = True
    # Empty string = UTC timestamp.
    run_id: str = ""


@dataclass
class PathsConfig:
    data_dir: Path = PROJECT_ROOT / "datasets" / "TGS_edgelists"
    output_root: Path = PROJECT_ROOT / "reports" / "study_A"
    cache_dir: Path = PROJECT_ROOT / "reports" / "study_A" / "cache"


@dataclass
class SelectionConfig:
    # "ranked" applies the documented criterion below; "explicit" uses `tokens` verbatim.
    mode: str = "ranked"
    tokens: list[str] = field(default_factory=list)
    n_tokens: int = 3
    min_coverage_days: int = 365
    max_nodes: int = 250_000
    rank_by: str = "unique_addresses"
    # Integer-ID files cannot use the zero-address filter or cross-token matching.
    require_hex_addresses: bool = True
    profile_workers: int = 4


@dataclass
class WindowConfig:
    # pandas offset alias for disjoint calendar windows ("MS" = calendar months).
    freq: str = "MS"
    # Windows covering less than this share of their calendar span are flagged partial.
    min_coverage: float = 0.8
    include_full_period: bool = True


@dataclass
class FilterConfig:
    exclude_self_loops: bool = True
    exclude_zero_address: bool = True
    # A pair whose transfers all carry zero tokens has no volume tie strength.
    drop_zero_volume_pairs_in_volume_graphs: bool = True


@dataclass
class CentralityConfig:
    variants: list[str] = field(default_factory=lambda: ["frequency", "raw_volume", "log1p_volume"])
    exact_max_nodes: int = 5_000
    betweenness_pivots: int = 1_000
    harmonic_top_candidates: int = 1_000
    harmonic_random_candidates: int = 1_000
    pagerank_damping: float = 0.85
    top_k: int = 20
    rank_stability_k: int = 100


@dataclass
class TopologyConfig:
    exact_path_max_nodes: int = 5_000
    path_sources: int = 500
    rich_club_points: int = 40


@dataclass
class NullModelConfig:
    n_null: int = 20
    swaps_per_edge: int = 10
    # Null models on the full-period graph are the most expensive step.
    run_on_full_period: bool = True


@dataclass
class CommunityConfig:
    resolution: float = 1.0
    stability_seeds: int = 10
    resolution_sweep: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0])
    persistence_min_size: int = 10
    top_central_per_community: int = 5


@dataclass
class DistributionConfig:
    min_tail: int = 50
    max_xmin_candidates: int = 200
    gof_bootstrap: int = 100


@dataclass
class OverlapConfig:
    """Time-aligned cross-token comparison of addresses and communities (selected tokens only)."""

    enabled: bool = True
    min_shared: int = 30
    permutations: int = 200
    matrix_communities: int = 10


@dataclass
class FigureConfig:
    formats: list[str] = field(default_factory=lambda: ["pdf", "png"])
    dpi: int = 300
    # Paper figures carry their title in the caption file; set true to also draw it.
    embed_titles: bool = False
    # Display-only node budget for sampled network drawings.
    drawing_max_nodes: int = 600
    # Colour-coded communities in node-level drawings (3 = all-pairs colour-safe cap).
    drawing_communities: int = 3
    quotient_communities: int = 15


@dataclass
class StudyAConfig:
    seed: int = 42
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    centrality: CentralityConfig = field(default_factory=CentralityConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    null_model: NullModelConfig = field(default_factory=NullModelConfig)
    communities: CommunityConfig = field(default_factory=CommunityConfig)
    distributions: DistributionConfig = field(default_factory=DistributionConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)
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


def load_config(path: str | Path | None = None) -> StudyAConfig:
    """Read a TOML file into a validated configuration (missing keys keep defaults)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(config_path, "rb") as handle:
        values = tomllib.load(handle)
    config = _build(StudyAConfig, values)
    validate(config)
    return config


def validate(config: StudyAConfig) -> None:
    from .graphs import VARIANTS

    unknown = [v for v in config.centrality.variants if v not in VARIANTS]
    if unknown or not config.centrality.variants:
        raise ValueError(f"centrality.variants must be a non-empty subset of {list(VARIANTS)}")
    if config.run.task not in {"all", "figures"}:
        raise ValueError("run.task must be 'all' or 'figures'")
    if config.run.task == "figures" and not config.run.run_id:
        raise ValueError("run.task = 'figures' needs run.run_id (the run to redraw)")
    if config.selection.mode not in {"ranked", "explicit"}:
        raise ValueError("selection.mode must be 'ranked' or 'explicit'")
    if config.selection.mode == "explicit" and not config.selection.tokens:
        raise ValueError("selection.tokens is required when selection.mode = 'explicit'")
    if not 0 < config.windows.min_coverage <= 1:
        raise ValueError("windows.min_coverage must be in (0, 1]")
    if config.null_model.n_null < 1 or config.null_model.swaps_per_edge < 1:
        raise ValueError("null_model.n_null and swaps_per_edge must be positive")
