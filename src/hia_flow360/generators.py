from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random
import uuid

from .models import EntityType, MovementEvent, SourceType, ZoneType

ZONES = list(ZoneType)
FLIGHTS = ["QR120", "QR451", "BA124", "LH621", "EY401", "AF103"]


def _weighted_zone() -> ZoneType:
    return random.choices(
        population=ZONES,
        weights=[0.2, 0.22, 0.12, 0.08, 0.2, 0.1, 0.08],
        k=1,
    )[0]


def _weighted_source() -> SourceType:
    return random.choices(
        population=[SourceType.WIFI, SourceType.XOVIS, SourceType.AODB],
        weights=[0.45, 0.4, 0.15],
        k=1,
    )[0]


def generate_events(minutes: int = 120, avg_events_per_minute: int = 40) -> list[MovementEvent]:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=minutes)
    events: list[MovementEvent] = []

    entity_pool = [f"u_{i:05d}" for i in range(4000)]
    staff_pool = [f"s_{i:04d}" for i in range(500)]

    for minute_index in range(minutes):
        current_ts = start + timedelta(minutes=minute_index)
        count = max(5, int(random.gauss(avg_events_per_minute, 6)))
        for _ in range(count):
            entity_type = EntityType.PASSENGER if random.random() < 0.86 else EntityType.STAFF
            entity_hash = random.choice(entity_pool if entity_type == EntityType.PASSENGER else staff_pool)
            zone = _weighted_zone()
            source = _weighted_source()

            events.append(
                MovementEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=current_ts + timedelta(seconds=random.randint(0, 59)),
                    source=source,
                    entity_type=entity_type,
                    entity_hash=entity_hash,
                    zone=zone,
                    confidence=round(random.uniform(0.72, 0.99), 2),
                    flight_id=random.choice(FLIGHTS) if entity_type == EntityType.PASSENGER and random.random() < 0.6 else None,
                    direction=random.choices(["in", "out", "cross"], [0.52, 0.32, 0.16], k=1)[0],
                )
            )

    _enforce_realistic_outflow(events)
    return sorted(events, key=lambda x: x.timestamp)


def _enforce_realistic_outflow(events: list[MovementEvent]) -> None:
    """Ensure each active zone-minute has at least one in and one out movement."""
    bucket: dict[tuple[datetime, ZoneType], list[int]] = {}
    for idx, event in enumerate(events):
        minute = event.timestamp.replace(second=0, microsecond=0)
        key = (minute, event.zone)
        bucket.setdefault(key, []).append(idx)

    for _, indices in bucket.items():
        # Keep sparse buckets untouched; forcing outflow on single-point data is too artificial.
        if len(indices) < 2:
            continue

        has_out = any(events[i].direction == "out" for i in indices)
        has_in = any(events[i].direction == "in" for i in indices)

        if not has_out:
            out_candidates = [i for i in indices if events[i].direction != "out"]
            if out_candidates:
                pick = random.choice(out_candidates)
                events[pick] = events[pick].model_copy(update={"direction": "out"})

        if not has_in:
            in_candidates = [i for i in indices if events[i].direction != "in"]
            if in_candidates:
                pick = random.choice(in_candidates)
                events[pick] = events[pick].model_copy(update={"direction": "in"})
