"""
Load the deduplicated transfer events of the Study B token and apply the edge filters.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.study_A.data import ZERO_ADDRESS, load_token

# Columns of the table returned by load_events (every other module relies on these names).
EVENT_COLUMNS = ["src", "dst", "weight", "multiplicity", "date"]


def load_events(token: str, data_dir: Path, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Return (events, filter_counts, addresses).

    events:        one row per unique (date, src, dst, weight) key, sorted by date; columns src, dst
                   (int node ids), weight (raw amount of ONE transfer), multiplicity (identical transfers
                   that day: events = sum of multiplicity, volume = sum of weight * multiplicity), date
    filter_counts: one row with events/volume before, removed as mint, burn, self-loop, and after
    addresses:     ``addresses[i]`` is the on-chain address of node id ``i``
    """
    raw = load_token(token, data_dir, cache_dir).events   # date, source, destination, weight, multiplicity
    n_events = raw["multiplicity"]
    volume = raw["weight"] * raw["multiplicity"]

    # Disjoint masks, so the removed counts add up: 0x0 -> 0x0 is counted once, as mint.
    mint = raw["source"] == ZERO_ADDRESS
    burn = (raw["destination"] == ZERO_ADDRESS) & ~mint
    self_loop = (raw["source"] == raw["destination"]) & ~mint
    keep = ~(mint | burn | self_loop)

    counts = {"token": token, "events_before": int(n_events.sum()), "volume_before": volume.sum()}
    for name, mask in (("mint", mint), ("burn", burn), ("self_loop", self_loop)):
        counts[f"{name}_events"] = int(n_events[mask].sum())
        counts[f"{name}_volume"] = volume[mask].sum()
    counts["events_after"] = int(n_events[keep].sum())
    counts["volume_after"] = volume[keep].sum()
    filter_counts = pd.DataFrame([counts])

    events = raw.loc[keep].reset_index(drop=True)
    # One factorize over both columns: the same address gets the same id as source and as destination.
    codes, addresses = pd.factorize(pd.concat([events["source"], events["destination"]], ignore_index=True))
    n = len(events)
    events = pd.DataFrame({"src": codes[:n], "dst": codes[n:],
                           "weight": events["weight"].to_numpy(), "multiplicity": events["multiplicity"].to_numpy(),
                           "date": events["date"].to_numpy()})
    events = events.sort_values("date", kind="stable", ignore_index=True)
    return events[EVENT_COLUMNS], filter_counts, pd.Index(addresses)