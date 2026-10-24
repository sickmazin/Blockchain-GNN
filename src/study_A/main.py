"""Run the whole of Study A with the settings in ``configs/study_A.toml``.

Execute this file (from the IDE or ``venv/Scripts/python.exe src/study_A/main.py``).
With ``[run] task = "all"`` it profiles the TGS files if needed, selects the tokens,
analyses each of them (reusing tokens already completed in the same run_id),
compares them (cross-token overlap) and writes the summary tables and figures to
``reports/study_A/runs/<run_id>/``. With ``task = "figures"`` it only redraws the
tables and figures of an existing run. Every parameter is read from the TOML file.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script: make ``src.study_A`` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.study_A.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.study_A.pipeline import rebuild_figures, run  # noqa: E402


def main(config_path: Path = DEFAULT_CONFIG) -> Path:
    config = load_config(config_path)
    if config.run.task == "figures":
        return rebuild_figures(config, config.run.run_id)
    return run(config, config_path=config_path, workers=config.run.workers,
               make_figures=config.run.make_figures, run_id=config.run.run_id or None)


if __name__ == "__main__":
    main()
