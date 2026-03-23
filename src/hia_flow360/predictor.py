from __future__ import annotations

from collections import defaultdict

from .models import FlowSnapshot, PredictionPoint, ZoneType


def _risk(wait_minutes: float, congestion: float) -> str:
    # Wider risk bands to make category separation clearer in the dashboard.
    if wait_minutes >= 30 or congestion >= 0.9:
        return "high"
    if wait_minutes >= 15 or congestion >= 0.65:
        return "medium"
    return "low"


def forecast(snapshots: list[FlowSnapshot], horizon_min: int = 30) -> list[PredictionPoint]:
    if not snapshots:
        return []

    by_zone = defaultdict(list)
    for s in snapshots:
        by_zone[s.zone.value].append(s)

    preds: list[PredictionPoint] = []
    for zone, series in by_zone.items():
        series = sorted(series, key=lambda x: x.timestamp)[-20:]
        if len(series) < 3:
            continue

        avg_delta = sum(series[i].occupancy - series[i - 1].occupancy for i in range(1, len(series))) / (len(series) - 1)
        current = series[-1]
        projected = max(0, int(current.occupancy + avg_delta * horizon_min))
        projected_wait = max(1.0, (projected / max(1.0, current.outflow_per_min + 1)) * 0.9)
        projected_congestion = min(1.0, current.congestion_score + (avg_delta / 200.0))

        preds.append(
            PredictionPoint(
                zone=ZoneType(zone),
                horizon_min=horizon_min,
                predicted_occupancy=projected,
                predicted_wait_minutes=round(projected_wait, 2),
                risk_level=_risk(projected_wait, projected_congestion),
            )
        )

    return sorted(preds, key=lambda x: (x.risk_level, x.predicted_wait_minutes), reverse=True)
