"""Loading, validation, snapshot deduplication, profiling and token selection.

The TGS/MiNT edge lists (Zenodo record 15364297) repeat every transfer in each
``snapshot`` whose span contains its date: snapshots are overlapping sliding
windows (7 days, 1-day stride in the files inspected). Counting raw rows would
therefore multiply event frequencies and volumes up to sevenfold, unevenly at
the period boundaries. :func:`deduplicate_snapshots` reconstructs one unique
event stream and verifies, per date, that every snapshot covering that date
holds the same multiset of transfers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import PREPROCESSING_VERSION

EXPECTED_COLUMNS = ("source", "destination", "weight", "date", "snapshot")
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
FILE_SUFFIX = "_edgelist.txt"
EVENT_KEY = ["date", "source", "destination", "weight"]

logger = logging.getLogger("study_A")


def token_path(data_dir: Path, token: str) -> Path:
    return Path(data_dir) / f"{token}{FILE_SUFFIX}"


def list_tokens(data_dir: Path) -> list[str]:
    directory = Path(data_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"TGS directory not found: {directory}")
    names = [p.name[: -len(FILE_SUFFIX)] for p in directory.iterdir()
             if p.is_file() and p.name.endswith(FILE_SUFFIX)]
    return sorted(names, key=str.casefold)


def read_raw(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one raw file, validate its schema and drop only unparseable rows (audited)."""
    path = Path(path)
    raw = pd.read_csv(path, dtype={"source": "string", "destination": "string", "date": "string"})
    missing = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {raw.columns.tolist()}")
    # Hex addresses are case-insensitive (EIP-55 checksums only change case).
    source = raw["source"].str.strip()
    destination = raw["destination"].str.strip()
    case_changed = int((source.ne(source.str.lower()) | destination.ne(destination.str.lower())).sum())
    frame = pd.DataFrame({
        "source": source.str.lower(),
        "destination": destination.str.lower(),
        "weight": pd.to_numeric(raw["weight"], errors="coerce"),
        "date": pd.to_datetime(raw["date"], errors="coerce", format="%Y-%m-%d"),
        "snapshot": pd.to_numeric(raw["snapshot"], errors="coerce"),
    })
    invalid = {
        "missing_source": int(frame["source"].isna().sum()),
        "missing_destination": int(frame["destination"].isna().sum()),
        "invalid_weight": int(frame["weight"].isna().sum()),
        "invalid_date": int(frame["date"].isna().sum()),
        "invalid_snapshot": int(frame["snapshot"].isna().sum()),
        "negative_weight": int((frame["weight"] < 0).sum()),
    }
    bad = frame.isna().any(axis=1) | (frame["weight"] < 0)
    audit = {
        "source_file": str(path.resolve()),
        "raw_rows": int(len(frame)),
        "rows_with_case_normalised_address": case_changed,
        **invalid,
        "dropped_invalid_rows": int(bad.sum()),
    }
    frame = frame.loc[~bad].reset_index(drop=True)
    frame["snapshot"] = frame["snapshot"].astype(np.int64)
    return frame, audit


def deduplicate_snapshots(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Collapse overlapping snapshots into unique transfer events.

    A transfer is identified by ``(date, source, destination, weight)``; truly
    repeated identical transfers on one day are kept through ``multiplicity``.
    For a key, the multiplicity is its count within one snapshot. The key is
    *consistent* when it appears in every snapshot that covers its date, with
    the same count in each. Inconsistent keys keep their maximum per-snapshot
    count and are reported, never silently dropped.

    Snapshot coverage is taken from the nominal sliding-window structure (modal
    span and stride, anchored at the first snapshot) when every snapshot's
    observed dates fall inside its nominal interval; this also detects a
    snapshot that lost all rows of a date. Otherwise coverage falls back to the
    snapshots observed on each date and the structure is flagged irregular.

    Returns ``(events, by_date, by_snapshot, audit)``.
    """
    by_snapshot = raw.groupby("snapshot").agg(
        date_min=("date", "min"), date_max=("date", "max"),
        distinct_dates=("date", "nunique"), rows=("date", "size")).reset_index()
    by_snapshot["span_days"] = (by_snapshot["date_max"] - by_snapshot["date_min"]).dt.days + 1
    span = int(by_snapshot["span_days"].mode().iloc[0])
    stride_days = by_snapshot["date_min"].diff().dt.days.dropna()
    stride = int(stride_days.mode().iloc[0]) if len(stride_days) else 1
    offset = pd.to_timedelta((by_snapshot["snapshot"] - by_snapshot["snapshot"].iloc[0]) * stride, unit="D")
    # Earliest implied anchor, so snapshots whose first days had no transfers still fit.
    anchor = (by_snapshot["date_min"] - offset).min()
    by_snapshot["nominal_start"] = anchor + offset
    by_snapshot["nominal_end"] = by_snapshot["nominal_start"] + pd.Timedelta(days=span - 1)
    by_snapshot["within_nominal"] = ((by_snapshot["date_min"] >= by_snapshot["nominal_start"])
                                     & (by_snapshot["date_max"] <= by_snapshot["nominal_end"]))
    regular = bool(by_snapshot["within_nominal"].all())

    dates = np.sort(raw["date"].unique())
    if regular:
        starts = by_snapshot["nominal_start"].to_numpy()
        ends = by_snapshot["nominal_end"].to_numpy()
        covering = ((starts[None, :] <= dates[:, None]) & (ends[None, :] >= dates[:, None])).sum(axis=1)
        snapshots_per_date = pd.Series(covering, index=pd.DatetimeIndex(dates, name="date"))
    else:
        snapshots_per_date = raw.groupby("date")["snapshot"].nunique()

    per_snapshot = (raw.groupby(EVENT_KEY + ["snapshot"], sort=False, observed=True)
                    .size().rename("count").reset_index())
    stats = (per_snapshot.groupby(EVENT_KEY, sort=False, observed=True)["count"]
             .agg(n_snapshots="size", count_min="min", count_max="max").reset_index())
    stats["expected_snapshots"] = stats["date"].map(snapshots_per_date).astype(np.int64)
    stats["consistent"] = ((stats["n_snapshots"] == stats["expected_snapshots"])
                           & (stats["count_min"] == stats["count_max"]))
    events = (stats[EVENT_KEY].assign(multiplicity=stats["count_max"].astype(np.int64))
              .sort_values(["date", "source", "destination", "weight"], kind="stable")
              .reset_index(drop=True))

    by_date = pd.DataFrame({
        "snapshots_covering_date": snapshots_per_date,
        "raw_rows": raw.groupby("date").size(),
        "unique_keys": stats.groupby("date").size(),
        "inconsistent_keys": stats.loc[~stats["consistent"]].groupby("date").size(),
        "unique_events": events.groupby("date")["multiplicity"].sum(),
    }).fillna(0).astype(np.int64).rename_axis("date").reset_index()

    unique_events = int(events["multiplicity"].sum())
    audit = {
        "deduplication_rule": "one count per (date, source, destination, weight) within a snapshot; "
                              "snapshots covering the same date must agree",
        "rows_after_validation": int(len(raw)),
        "unique_event_keys": int(len(stats)),
        "unique_events": unique_events,
        "raw_to_unique_ratio": float(len(raw) / unique_events) if unique_events else float("nan"),
        "consistent_keys": int(stats["consistent"].sum()),
        "inconsistent_keys": int((~stats["consistent"]).sum()),
        "consistent_key_share": float(stats["consistent"].mean()) if len(stats) else float("nan"),
        "snapshots": int(by_snapshot["snapshot"].nunique()),
        "snapshot_span_days_mode": span,
        "snapshot_span_days_max": int(by_snapshot["span_days"].max()),
        "snapshot_stride_days_mode": stride,
        "snapshot_structure_regular": regular,
        "snapshot_coverage_rule": "nominal sliding windows" if regular else "observed snapshots per date",
        "date_min": str(events["date"].min().date()),
        "date_max": str(events["date"].max().date()),
    }
    return events, by_date, by_snapshot, audit


def _file_fingerprint(path: Path, with_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    fingerprint: dict[str, Any] = {"path": str(path.resolve()), "size_bytes": stat.st_size,
                                   "mtime_ns": stat.st_mtime_ns}
    if with_hash:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        fingerprint["sha256"] = digest.hexdigest()
    return fingerprint


@dataclass
class TokenEvents:
    token: str
    events: pd.DataFrame
    by_date: pd.DataFrame
    by_snapshot: pd.DataFrame
    audit: dict[str, Any]


def load_token(token: str, data_dir: Path, cache_dir: Path | None = None) -> TokenEvents:
    """Load the deduplicated events of one token, reusing a fingerprinted cache when valid."""
    path = token_path(data_dir, token)
    if not path.is_file():
        raise FileNotFoundError(path)
    fingerprint = _file_fingerprint(path)
    if cache_dir is not None:
        stem = Path(cache_dir) / token
        meta_path = stem.with_suffix(".json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta.get("fingerprint", {}).get("size_bytes") == fingerprint["size_bytes"]
                    and meta.get("fingerprint", {}).get("mtime_ns") == fingerprint["mtime_ns"]
                    and meta.get("preprocessing_version") == PREPROCESSING_VERSION):
                logger.info("%s: using cached deduplicated events (%s)", token, stem)
                events = pd.read_csv(f"{stem}_events.csv.gz", parse_dates=["date"],
                                     dtype={"source": "string", "destination": "string"})
                by_date = pd.read_csv(f"{stem}_by_date.csv", parse_dates=["date"])
                by_snapshot = pd.read_csv(f"{stem}_by_snapshot.csv", parse_dates=["date_min", "date_max"])
                return TokenEvents(token, events, by_date, by_snapshot, meta["audit"])
    logger.info("%s: reading %s", token, path.name)
    raw, read_audit = read_raw(path)
    events, by_date, by_snapshot, dedup_audit = deduplicate_snapshots(raw)
    audit = {"token": token, "preprocessing_version": PREPROCESSING_VERSION,
             **read_audit, **dedup_audit}
    logger.info("%s: %s raw rows -> %s unique events (ratio %.2f, consistent keys %.4f%%)",
                token, f"{audit['raw_rows']:,}", f"{audit['unique_events']:,}",
                audit["raw_to_unique_ratio"], 100 * audit["consistent_key_share"])
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(cache_dir) / token
        events.to_csv(f"{stem}_events.csv.gz", index=False, date_format="%Y-%m-%d")
        by_date.to_csv(f"{stem}_by_date.csv", index=False, date_format="%Y-%m-%d")
        by_snapshot.to_csv(f"{stem}_by_snapshot.csv", index=False, date_format="%Y-%m-%d")
        stem.with_suffix(".json").write_text(json.dumps({
            "fingerprint": fingerprint, "preprocessing_version": PREPROCESSING_VERSION,
            "audit": audit}, indent=2), encoding="utf-8")
    return TokenEvents(token, events, by_date, by_snapshot, audit)


def address_format(addresses: pd.Series) -> dict[str, Any]:
    """Classify node identifiers: 0x-prefixed 20-byte hex addresses or anonymised integer IDs.

    A few TGS files (e.g. ALBT, POLS) use integer node IDs instead of addresses. There the
    zero address cannot be identified and addresses cannot be matched across tokens.
    """
    unique = pd.Series(pd.unique(addresses.astype("string")))
    hex_share = float(unique.str.fullmatch(r"0x[0-9a-f]{40}").mean()) if len(unique) else float("nan")
    integer_share = float(unique.str.fullmatch(r"\d+").mean()) if len(unique) else float("nan")
    label = "hex" if hex_share == 1.0 else "integer_ids" if integer_share == 1.0 else "mixed"
    return {"address_format": label, "hex_address_share": hex_share}


def profile_token(token: str, data_dir: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    """Size, coverage and audit counts of one token after snapshot deduplication."""
    try:
        loaded = load_token(token, data_dir, cache_dir)
    except Exception as exc:  # A broken file is reported in the profile, not hidden.
        return {"token": token, "error": f"{type(exc).__name__}: {exc}"}
    ev = loaded.events
    mult = ev["multiplicity"]
    is_loop = ev["source"].eq(ev["destination"])
    is_zero = ev["source"].eq(ZERO_ADDRESS) | ev["destination"].eq(ZERO_ADDRESS)
    addresses = pd.unique(pd.concat([ev["source"], ev["destination"]], ignore_index=True))
    span = (ev["date"].max() - ev["date"].min()).days + 1
    return {
        **loaded.audit,
        "unique_addresses": int(len(addresses) - (ZERO_ADDRESS in set(addresses))),
        **address_format(pd.Series(addresses)),
        "unique_ordered_pairs": int(ev[["source", "destination"]].drop_duplicates().shape[0]),
        "coverage_days": int(span),
        "active_days": int(ev["date"].nunique()),
        "self_loop_events": int(mult[is_loop].sum()),
        "zero_address_events": int(mult[is_zero].sum()),
        "zero_weight_events": int(mult[ev["weight"].eq(0)].sum()),
        "error": None,
    }


def profile_all(data_dir: Path, cache_dir: Path | None, workers: int = 4) -> pd.DataFrame:
    tokens = list_tokens(data_dir)
    logger.info("Profiling %d token files with %d workers", len(tokens), workers)
    if workers <= 1:
        rows = [profile_token(t, data_dir, cache_dir) for t in tokens]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(profile_token, tokens, [data_dir] * len(tokens),
                                 [cache_dir] * len(tokens)))
    return pd.DataFrame(rows)


def select_tokens(profile: pd.DataFrame, *, mode: str, tokens: list[str], n_tokens: int,
                  min_coverage_days: int, max_nodes: int, rank_by: str,
                  require_hex_addresses: bool = True) -> pd.DataFrame:
    """Apply the documented selection criterion and return every token with its decision.

    ranked: keep files that loaded without error, whose node identifiers are hex
    addresses (if required: the zero-address filter and cross-token comparison need
    them), cover at least
    ``min_coverage_days`` calendar days and have at most ``max_nodes`` addresses
    (computational feasibility); rank the survivors by ``rank_by`` (descending,
    ties by token name) and select the first ``n_tokens``.
    """
    table = profile.copy()
    table["selected"] = False
    table["reason"] = ""
    ok = table["error"].isna()
    table.loc[~ok, "reason"] = "load error"
    if mode == "explicit":
        missing = [t for t in tokens if t not in set(table["token"])]
        if missing:
            raise ValueError(f"tokens not present in the profile: {missing}")
        table.loc[table["token"].isin(tokens) & ok, ["selected", "reason"]] = [True, "explicit"]
        table["rank"] = table["token"].map({t: i + 1 for i, t in enumerate(tokens)})
        return table.sort_values(["selected", "rank"], ascending=[False, True]).reset_index(drop=True)
    if rank_by not in table.columns:
        raise ValueError(f"rank_by column {rank_by!r} not in profile")
    if require_hex_addresses:
        not_hex = ok & table["address_format"].ne("hex")
        table.loc[not_hex, "reason"] = "node IDs are not hex addresses (" + table.loc[not_hex, "address_format"] + ")"
        ok &= ~not_hex
    short = ok & (table["coverage_days"] < min_coverage_days)
    large = ok & ~short & (table["unique_addresses"] > max_nodes)
    table.loc[short, "reason"] = f"coverage < {min_coverage_days} days"
    table.loc[large, "reason"] = f"addresses > {max_nodes:,} (feasibility)"
    eligible = table.loc[ok & ~short & ~large].sort_values([rank_by, "token"], ascending=[False, True])
    table["rank"] = table["token"].map({t: i + 1 for i, t in enumerate(eligible["token"])})
    chosen = eligible["token"].head(n_tokens)
    table.loc[table["token"].isin(chosen), ["selected", "reason"]] = [True, f"top {n_tokens} by {rank_by}"]
    table.loc[ok & ~short & ~large & ~table["selected"], "reason"] = f"eligible, rank > {n_tokens}"
    return table.sort_values(["selected", "rank"], ascending=[False, True], na_position="last").reset_index(drop=True)


def daily_activity(events: pd.DataFrame) -> pd.DataFrame:
    """Unique transfers, active addresses and senders/receivers per calendar day."""
    grouped = events.groupby("date")
    active = (pd.concat([events[["date", "source"]].rename(columns={"source": "address"}),
                         events[["date", "destination"]].rename(columns={"destination": "address"})])
              .drop_duplicates().groupby("date").size())
    table = pd.DataFrame({
        "transfers": grouped["multiplicity"].sum(),
        "active_addresses": active,
        "senders": grouped["source"].nunique(),
        "receivers": grouped["destination"].nunique(),
    }).rename_axis("date").reset_index()
    full = pd.date_range(table["date"].min(), table["date"].max(), freq="D")
    return (table.set_index("date").reindex(full, fill_value=0).rename_axis("date").reset_index())
