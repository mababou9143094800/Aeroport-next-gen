from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime

import pandas as pd

from .models import MovementEvent


def pseudonymize(raw_id: str, salt: str = "hia-flow360") -> str:
    payload = f"{salt}:{raw_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def events_to_dataframe(events: list[MovementEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "minute",
                "source",
                "entity_type",
                "entity_hash",
                "zone",
                "confidence",
                "flight_id",
                "direction",
            ]
        )

    rows = [
        {
            "timestamp": e.timestamp,
            "minute": e.timestamp.replace(second=0, microsecond=0),
            "source": e.source.value,
            "entity_type": e.entity_type.value,
            "entity_hash": pseudonymize(e.entity_hash),
            "zone": e.zone.value,
            "confidence": e.confidence,
            "flight_id": e.flight_id,
            "direction": e.direction,
        }
        for e in events
    ]
    return pd.DataFrame(rows)


def dedupe_events(df: pd.DataFrame, confidence_threshold: float = 0.75) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df[df["confidence"] >= confidence_threshold].copy()
    filtered = filtered.sort_values("confidence", ascending=False)
    deduped = filtered.drop_duplicates(subset=["minute", "entity_hash", "zone", "direction"], keep="first")
    return deduped.sort_values("timestamp")


def estimate_dwell_minutes(df: pd.DataFrame) -> dict[tuple[datetime, str], float]:
    if df.empty:
        return {}

    zone_time = defaultdict(list)
    for _, row in df.sort_values("timestamp").iterrows():
        zone_time[(row["entity_hash"], row["zone"])].append(row["timestamp"])

    dwell_samples = defaultdict(list)
    for (_, zone), times in zone_time.items():
        if len(times) < 2:
            continue
        deltas = [
            (times[i + 1] - times[i]).total_seconds() / 60.0
            for i in range(len(times) - 1)
        ]
        for t in times[1:]:
            dwell_samples[(t.replace(second=0, microsecond=0), zone)].append(max(0.1, min(30.0, sum(deltas) / len(deltas))))

    output = {}
    for key, vals in dwell_samples.items():
        output[key] = float(sum(vals) / len(vals))
    return output

