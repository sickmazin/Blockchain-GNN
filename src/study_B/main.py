"""
Run the whole of Study B with the settings in ``study_B.toml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script: make ``src.study_B`` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.study_B.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.study_B.pipeline import run  # noqa: E402


def main(config_path: Path = DEFAULT_CONFIG) -> Path:
    config = load_config(config_path)
    return run(config, config_path=config_path)


if __name__ == "__main__":
    main()
