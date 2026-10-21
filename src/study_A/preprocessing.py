"""Reproducible preprocessing for Study A ERC-20 transfer networks.

This module covers data profiling, explicit token selection, event loading,
window assignment, event aggregation, and graph-table preparation (phases 1–4).
Addresses and transfers retain their on-chain meanings; no identity inference is
performed. Raw files are read-only and derived outputs belong outside datasets/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TGS_DATA = PROJECT_ROOT / "datasets" / "TGS_edgelists"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
EXPECTED_COLUMNS = ("source", "destination", "weight", "date", "snapshot")
PREPROCESSING_VERSION = "1.0"


def list_tgs_files(data_dir: str | Path = TGS_DATA) -> list[Path]:
    """List regular TGS files in stable filename order."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"TGS directory not found: {directory}")
    return sorted((p for p in directory.iterdir() if p.is_file()), key=lambda p: p.name.casefold())


def _token_from_path(path: Path) -> str:
    suffix = "_edgelist.txt"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def profile_file(path: str | Path) -> pd.DataFrame:
    """Return schema, scale, missingness, time, weight, and graph audit metrics."""
    path = Path(path)
    df = pd.read_csv(path)
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"{path.name}: missing columns {missing_columns}; found {df.columns.tolist()}")

    source = df["source"].astype("string").str.strip()
    destination = df["destination"].astype("string").str.strip()
    amount = pd.to_numeric(df["weight"], errors="coerce")
    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    addresses = pd.concat([source, destination], ignore_index=True).dropna()
    duplicates = df.duplicated(subset=list(EXPECTED_COLUMNS), keep=False)
    profile: dict[str, object] = {
        "token": _token_from_path(path), "file": path.name, "path": str(path.resolve()),
        "rows": len(df), "unique_sources": source.nunique(),
        "unique_destinations": destination.nunique(), "unique_nodes": addresses.nunique(),
        "unique_ordered_pairs": pd.DataFrame({"source": source, "destination": destination}).drop_duplicates().shape[0],
        "duplicate_event_rows": int(duplicates.sum()),
        "self_loops": int(source.eq(destination).fillna(False).sum()),
        "zero_address_rows": int((source.eq(ZERO_ADDRESS) | destination.eq(ZERO_ADDRESS)).fillna(False).sum()),
        "invalid_weight": int((amount.isna() & df["weight"].notna()).sum()),
        "zero_weight": int(amount.eq(0).sum()), "negative_weight": int(amount.lt(0).sum()),
        "weight_min": amount.min(), "weight_max": amount.max(),
        "unique_dates": df["date"].nunique(dropna=True),
        "invalid_dates": int((parsed_date.isna() & df["date"].notna()).sum()),
        "date_min": parsed_date.min(), "date_max": parsed_date.max(),
        "unique_snapshots": df["snapshot"].nunique(dropna=True),
        "snapshot_min": df["snapshot"].min(), "snapshot_max": df["snapshot"].max(),
        "missing_values_total": int(df[list(EXPECTED_COLUMNS)].isna().sum().sum()),
    }
    for column, count in df[list(EXPECTED_COLUMNS)].isna().sum().items():
        profile[f"missing_{column}"] = int(count)
    return pd.DataFrame([profile])


def profile_all_files(data_dir: str | Path = TGS_DATA, output_path: str | Path | None = None) -> pd.DataFrame:
    """Profile every file; retain unreadable-file errors in the result."""
    rows = []
    for path in list_tgs_files(data_dir):
        try:
            rows.append(profile_file(path))
        except Exception as exc:
            rows.append(pd.DataFrame([{"token": _token_from_path(path), "file": path.name,
                                       "path": str(path.resolve()), "error": str(exc)}]))
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if output_path is not None:
        output = Path(output_path)
        if output.is_dir() or output.suffix == "":
            output /= "profile.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    return result


def select_tokens(profile: pd.DataFrame, selected_tokens: Sequence[str]) -> pd.DataFrame:
    """Return requested token profile rows in the caller's order.

    Token identifiers are exact, case-sensitive names without
    ``_edgelist.txt`` (for example ``["0x0", "aDAI"]``). This function does
    not rank or silently substitute candidates.
    """
    if profile is None or not isinstance(profile, pd.DataFrame):
        raise TypeError("profile must be a pandas DataFrame")

    if not selected_tokens:
        raise ValueError("selected_tokens cannot be empty")

    available = set(profile["token"].dropna().astype(str))
    missing = [token for token in selected_tokens if token not in available]
    if missing:
        raise ValueError(f"Tokens not present in profile: {missing}")
    indexed = profile.drop_duplicates("token", keep="first").set_index("token", drop=False)
    return indexed.loc[list(selected_tokens)].reset_index(drop=True)


def load_token_events(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load source events and return them with parsing/quality audit counts."""
    path = Path(path)
    events = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {events.columns.tolist()}")
    events = events.copy()
    events["source"] = events["source"].astype("string").str.strip()
    events["destination"] = events["destination"].astype("string").str.strip()
    original_weight = events["weight"]
    events["weight"] = pd.to_numeric(original_weight, errors="coerce")
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events["snapshot"] = pd.to_numeric(events["snapshot"], errors="coerce")
    audit = {
        "source_file": str(path.resolve()), "preprocessing_version": PREPROCESSING_VERSION,
        "rows": len(events), "missing_required_values": int(events[list(EXPECTED_COLUMNS)].isna().sum().sum()),
        "invalid_weight": int((events["weight"].isna() & original_weight.notna()).sum()),
        "invalid_date": int(events["date"].isna().sum()),
        "invalid_snapshot": int(events["snapshot"].isna().sum()),
        "duplicate_event_rows": int(events.duplicated(list(EXPECTED_COLUMNS), keep=False).sum()),
        "self_loops": int(events["source"].eq(events["destination"]).fillna(False).sum()),
        "zero_address_rows": int((events["source"].eq(ZERO_ADDRESS) | events["destination"].eq(ZERO_ADDRESS)).fillna(False).sum()),
    }
    return events, audit


def assign_windows(
    events: pd.DataFrame,
    windows: Mapping[str, tuple[object, object]],
    column: str = "snapshot",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign events to disjoint half-open [start, end) windows and return audit.

    Windows must be chronologically ordered and non-overlapping. Rows with an
    invalid/missing time value or outside all intervals remain unassigned.
    """
    if column not in events:
        raise ValueError(f"events has no time column {column!r}")
    intervals = [(name, start, end) for name, (start, end) in windows.items()]
    for i, (name, start, end) in enumerate(intervals):
        if start >= end:
            raise ValueError(f"window {name!r} must satisfy start < end")
        if i and start < intervals[i - 1][2]:
            raise ValueError("windows must be ordered and non-overlapping")
    assigned = events.copy()
    assigned["window"] = pd.NA
    values = assigned[column]
    for name, start, end in intervals:
        mask = values.ge(start) & values.lt(end) & assigned["window"].isna()
        assigned.loc[mask, "window"] = name
    audit_rows = []
    for name, _, _ in intervals:
        subset = assigned.loc[assigned["window"].eq(name)]
        audit_rows.append({"window": name, "rows": len(subset),
                           "unique_nodes": pd.concat([subset["source"], subset["destination"]]).nunique(),
                           "total_weight": subset["weight"].sum(min_count=1), "unassigned_rows": 0})
    audit_rows.append({"window": "UNASSIGNED", "rows": int(assigned["window"].isna().sum()),
                       "unique_nodes": pd.concat([assigned.loc[assigned["window"].isna(), "source"],
                                                   assigned.loc[assigned["window"].isna(), "destination"]]).nunique(),
                       "total_weight": assigned.loc[assigned["window"].isna(), "weight"].sum(min_count=1),
                       "unassigned_rows": int(assigned["window"].isna().sum())})
    return assigned, pd.DataFrame(audit_rows)


def aggregate_window(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate events by ordered pair, preserving event frequency and volume."""
    required = {"source", "destination", "weight"}
    #if not required.issubset(events.columns):
    #    raise ValueError(f"events must contain {sorted(required)}")
    valid = events.dropna(subset=["source", "destination"])
    group_columns = ["source", "destination"]
    if "window" in valid:
        group_columns.insert(0, "window")
    result = (valid.groupby(group_columns, dropna=False, sort=True)
              .agg(event_count=("weight", "size"), volume=("weight", lambda x: x.sum(min_count=1)))
              .reset_index())
    return result


def _windows_from_snapshots(snapshots: Sequence[object], snapshots_per_window: int) -> dict[str, tuple[object, object]]:
    """Group sorted observed snapshot IDs into disjoint half-open intervals."""
    windows: dict[str, tuple[object, object]] = {}
    snapshots = sorted(set(snapshots))
    for offset in range(0, len(snapshots), snapshots_per_window):
        chunk = snapshots[offset:offset + snapshots_per_window]
        start = chunk[0]
        end = snapshots[offset + snapshots_per_window] if offset + snapshots_per_window < len(snapshots) else chunk[-1] + 1
        windows[f"snapshot_{start}_to_{chunk[-1]}"] = (start, end)
    return windows

#================================================================================
def main() -> None:
    """Print preprocessing previews for BITCOIN, SWAP, and SUPER."""
    # Esplora i dati preprocessati dello Studio A.
    data_dir=TGS_DATA
    snapshots_per_window=10     #Snapshot consecutivi per finestra (default: 10)
    top=5   #Righe da mostrare per anteprima (default: 5)
    
    tokens= ["BITCOIN", "SWAP", "SUPER"]
    
    for token in tokens:
        path = data_dir / f"{token}_edgelist.txt"
        if not path.is_file():
            print(f"\n=== {token}: file non trovato ({path}) ===")
            continue

        print(f"\n{'=' * 18} {token} {'=' * 18}")
        events, event_audit = load_token_events(path)
        print("Audit eventi:")
        print(pd.Series(event_audit).to_string())
        print(f"\nPrime {top} righe preprocessate:")
        print(events.head(top).to_string(index=False))

        snapshots = events["snapshot"].dropna().unique()
        if len(snapshots):
            windows = _windows_from_snapshots(snapshots, snapshots_per_window)
            windowed, window_audit = assign_windows(events, windows)
            print("\nRiepilogo finestre (intervalli [inizio, fine)):")
            print(window_audit.to_string(index=False))
            usable_events = windowed.loc[windowed["window"].notna()].copy()
        else:
            print("\nNessuno snapshot interpretabile: aggregazione senza finestre.")
            usable_events = events

        aggregates = aggregate_window(usable_events)
        print(f"\nCoppie dirette aggregate: {len(aggregates):,}")
        print(f"\nTop {top} coppie per frequenza:")
        print(aggregates.sort_values("event_count", ascending=False, kind="stable").head(top).to_string(index=False))
        print(f"\nTop {top} coppie per volume aggregato:")
        print(aggregates.sort_values("volume", ascending=False, kind="stable").head(top).to_string(index=False))


if __name__ == "__main__":
    main()

