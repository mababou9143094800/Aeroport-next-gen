from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    PASSENGER = "passenger"
    STAFF = "staff"


class SourceType(str, Enum):
    WIFI = "wifi"
    XOVIS = "xovis"
    AODB = "aodb"


class ZoneType(str, Enum):
    CHECKIN = "checkin"
    SECURITY = "security"
    IMMIGRATION = "immigration"
    RETAIL = "retail"
    GATE = "gate"
    TRANSFER = "transfer"
    BAGGAGE = "baggage"


class MovementEvent(BaseModel):
    event_id: str
    timestamp: datetime
    source: SourceType
    entity_type: EntityType
    entity_hash: str = Field(description="Pseudonymized stable identifier")
    zone: ZoneType
    confidence: float = Field(ge=0.0, le=1.0)
    flight_id: str | None = None
    direction: Literal["in", "out", "cross"] = "in"


class FlowSnapshot(BaseModel):
    timestamp: datetime
    zone: ZoneType
    occupancy: int
    inflow_per_min: float
    outflow_per_min: float
    avg_dwell_minutes: float
    congestion_score: float = Field(ge=0.0, le=1.0)


class PredictionPoint(BaseModel):
    zone: ZoneType
    horizon_min: int
    predicted_occupancy: int
    predicted_wait_minutes: float
    risk_level: Literal["low", "medium", "high"]


class CameraCreate(BaseModel):
    name: str
    x: float
    y: float
    angle: float = 0.0
    zone: str


class CameraUpdate(BaseModel):
    name: str | None = None
    x: float | None = None
    y: float | None = None
    angle: float | None = None
    zone: str | None = None

