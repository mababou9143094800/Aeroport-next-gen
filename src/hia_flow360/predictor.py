from __future__ import annotations

from collections import defaultdict

from .models import FlowSnapshot, PredictionPoint, ZoneType


def _risk(wait_minutes: float, congestion: float) -> str:
    if wait_minutes >= 30 or congestion >= 0.9:
        return "high"
    if wait_minutes >= 15 or congestion >= 0.65:
        return "medium"
    return "low"


def _median(values: list[float]) -> float:
    """Médiane robuste pour lisser les données caméra bruitées."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def forecast(snapshots: list[FlowSnapshot], horizon_min: int = 30) -> list[PredictionPoint]:
    """
    Prédit l'occupancy, le temps d'attente et le niveau de risque par zone.

    Fonctionne avec deux sources de données :
    - Snapshots simulés  : série temporelle régulière (1 entrée / minute)
    - Snapshots caméras  : série irrégulière, potentiellement bruitée

    Robustesse caméra :
    - Utilise la médiane des deltas d'occupancy (résistant aux pics)
    - Élimine les deltas aberrants (> 3× l'écart-type)
    - Requiert au moins 3 snapshots par zone pour établir une tendance
    """
    if not snapshots:
        return []

    by_zone: dict[str, list[FlowSnapshot]] = defaultdict(list)
    for s in snapshots:
        by_zone[s.zone.value].append(s)

    preds: list[PredictionPoint] = []
    for zone, series in by_zone.items():
        series = sorted(series, key=lambda x: x.timestamp)[-20:]
        if len(series) < 3:
            continue

        # Calcul des deltas d'occupancy entre snapshots consécutifs
        raw_deltas = [
            float(series[i].occupancy - series[i - 1].occupancy)
            for i in range(1, len(series))
        ]

        # Filtrage des deltas aberrants (données caméra bruitées)
        # On conserve les valeurs dans ±2.5 fois l'écart absolu médian
        if len(raw_deltas) >= 3:
            med = _median(raw_deltas)
            mad = _median([abs(d - med) for d in raw_deltas]) or 1.0
            filtered = [d for d in raw_deltas if abs(d - med) <= 2.5 * mad]
            deltas = filtered if filtered else raw_deltas
        else:
            deltas = raw_deltas

        # Tendance : médiane des deltas (robuste aux pics caméra)
        avg_delta = _median(deltas)

        current = series[-1]
        projected = max(0, int(current.occupancy + avg_delta * horizon_min))

        # Temps d'attente estimé : occupancy projetée / débit de sortie
        projected_wait = max(
            1.0,
            (projected / max(1.0, current.outflow_per_min + 1)) * 0.9,
        )

        # Score de congestion projeté (ajustement conservateur)
        projected_congestion = min(
            1.0,
            current.congestion_score + (avg_delta / 200.0),
        )

        preds.append(
            PredictionPoint(
                zone=ZoneType(zone),
                horizon_min=horizon_min,
                predicted_occupancy=projected,
                predicted_wait_minutes=round(projected_wait, 2),
                risk_level=_risk(projected_wait, projected_congestion),
            )
        )

    return sorted(
        preds,
        key=lambda x: (x.risk_level, x.predicted_wait_minutes),
        reverse=True,
    )
