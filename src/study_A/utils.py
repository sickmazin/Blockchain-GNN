"""Shared output utilities for Study A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def save_tables(
    output_dir: str | Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist CSV tables without source paths and JSON run metadata."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if Path(name).name != name:
            raise ValueError("table names must be plain filenames without directories")
        filename = name if name.endswith(".csv") else f"{name}.csv"
        # Source paths duplicate information from run metadata and clutter
        # row-oriented result tables, so omit this column from every CSV.
        table.drop(columns=["source_file"], errors="ignore").to_csv(
            directory / filename, index=False,
        )
    if metadata is not None:
        with (directory / "run_metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(dict(metadata), stream, indent=2, ensure_ascii=False, default=str)
