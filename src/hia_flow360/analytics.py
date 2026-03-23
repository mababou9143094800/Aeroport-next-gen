from __future__ import annotations

from datetime import datetime

import pandas as pd

from .fusion import dedupe_events, estimate_dwell_minutes
from .models import FlowSnapshot, ZoneType

ZONE_CAPACITY = {
    ZoneType.CHECKIN.value: 900,
    ZoneType.SECURITY.value: 700,
    ZoneType.IMMIGRATION.value: 520,
    ZoneType.RETAIL.value: 450,
    ZoneType.GATE.value: 1200,
    ZoneType.TRANSFER.value: 600,
    ZoneType.BAGGAGE.value: 500,
}


def compute_snapshots(df: pd.DataFrame) -> list[FlowSnapshot]:
    if df.empty:
        return []

    clean = dedupe_events(df)
    dwell_lookup = estimate_dwell_minutes(clean)
    clean["signed_flow"] = clean["direction"].map({"in": 1, "out": -1, "cross": 0}).fillna(0)

    grouped = (
        clean.groupby(["minute", "zone"])
        .agg(
            inflow=("direction", lambda x: float((x == "in").sum())),
            outflow=("direction", lambda x: float((x == "out").sum())),
            net=("signed_flow", "sum"),
        )
        .reset_index()
        .sort_values(["zone", "minute"])
    )

    snapshots: list[FlowSnapshot] = []
    occupancy_per_zone: dict[str, int] = {}

    for _, row in grouped.iterrows():
        zone = row["zone"]
        minute = row["minute"]
        prev = occupancy_per_zone.get(zone, 0)
        current = max(0, prev + int(row["net"]))
        occupancy_per_zone[zone] = current

        capacity = ZONE_CAPACITY.get(zone, 500)
        congestion = min(1.0, current / capacity)
        dwell = dwell_lookup.get((minute, zone), max(1.0, min(18.0, current / 35.0)))

        snapshots.append(
            FlowSnapshot(
                timestamp=minute.to_pydatetime() if isinstance(minute, pd.Timestamp) else minute,
                zone=ZoneType(zone),
                occupancy=current,
                inflow_per_min=round(float(row["inflow"]), 2),
                outflow_per_min=round(float(row["outflow"]), 2),
                avg_dwell_minutes=round(float(dwell), 2),
                congestion_score=round(congestion, 3),
            )
        )
    return snapshots


def latest_snapshot_map(snapshots: list[FlowSnapshot]) -> dict[str, FlowSnapshot]:
    latest: dict[str, FlowSnapshot] = {}
    for item in snapshots:
        latest[item.zone.value] = item
    return latest


def airport_kpis(snapshots: list[FlowSnapshot]) -> dict[str, float]:
    if not snapshots:
        return {
            "total_occupancy": 0.0,
            "avg_congestion": 0.0,
            "avg_dwell_minutes": 0.0,
            "max_zone_congestion": 0.0,
        }

    total_occupancy = sum(s.occupancy for s in snapshots)
    avg_congestion = sum(s.congestion_score for s in snapshots) / len(snapshots)
    avg_dwell = sum(s.avg_dwell_minutes for s in snapshots) / len(snapshots)
    max_zone = max(s.congestion_score for s in snapshots)
    return {
        "total_occupancy": float(total_occupancy),
        "avg_congestion": round(float(avg_congestion), 3),
        "avg_dwell_minutes": round(float(avg_dwell), 2),
        "max_zone_congestion": round(float(max_zone), 3),
    }

