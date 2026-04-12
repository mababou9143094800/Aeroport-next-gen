from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

_DATA_DIR = Path(__file__).parent.parent.parent  # Aeroport-next-gen/
_ZONES_FILE = _DATA_DIR / "zones.json"
_lock = Lock()

_DEFAULT_ZONES: list[dict] = [
    {"id": "checkin",     "label": "Enregistrement",  "type": "checkin",     "x": 10,  "y": 10,  "w": 220, "h": 390, "color": "#3B82F6"},
    {"id": "baggage",     "label": "Bagages",          "type": "baggage",     "x": 10,  "y": 410, "w": 220, "h": 220, "color": "#F97316"},
    {"id": "security",    "label": "Sécurité",         "type": "security",    "x": 240, "y": 10,  "w": 170, "h": 200, "color": "#EF4444"},
    {"id": "immigration", "label": "Douane",           "type": "immigration", "x": 240, "y": 220, "w": 170, "h": 180, "color": "#8B5CF6"},
    {"id": "transfer",    "label": "Transit",          "type": "transfer",    "x": 420, "y": 10,  "w": 160, "h": 200, "color": "#06B6D4"},
    {"id": "retail",      "label": "Boutiques",        "type": "retail",      "x": 420, "y": 220, "w": 160, "h": 180, "color": "#10B981"},
    {"id": "gate",        "label": "Embarquement",     "type": "gate",        "x": 590, "y": 10,  "w": 580, "h": 620, "color": "#F59E0B"},
]


def load_zones() -> list[dict]:
    with _lock:
        if _ZONES_FILE.exists():
            return json.loads(_ZONES_FILE.read_text(encoding="utf-8"))
        return [z.copy() for z in _DEFAULT_ZONES]


def save_zones(zones: list[dict]) -> None:
    with _lock:
        _ZONES_FILE.write_text(json.dumps(zones, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_to_default() -> list[dict]:
    zones = [z.copy() for z in _DEFAULT_ZONES]
    save_zones(zones)
    return zones
