from __future__ import annotations

import json
import uuid
from pathlib import Path
from threading import Lock

_DATA_DIR = Path(__file__).parent.parent.parent  # Aeroport-next-gen/
_CAMERAS_FILE = _DATA_DIR / "cameras.json"
_VIDEOS_BASE = _DATA_DIR / "videos"

_lock = Lock()


def _load() -> list[dict]:
    if _CAMERAS_FILE.exists():
        return json.loads(_CAMERAS_FILE.read_text(encoding="utf-8"))
    return []


def _save(cameras: list[dict]) -> None:
    _CAMERAS_FILE.write_text(json.dumps(cameras, indent=2, ensure_ascii=False), encoding="utf-8")


def list_cameras() -> list[dict]:
    with _lock:
        return _load()


def get_camera(camera_id: str) -> dict | None:
    with _lock:
        return next((c for c in _load() if c["id"] == camera_id), None)


def create_camera(
    name: str, x: float, y: float, angle: float, zone: str,
    source_type: str = "file", usb_index: int | None = None,
) -> dict:
    with _lock:
        cameras = _load()
        camera_id = uuid.uuid4().hex[:8]
        folder = str(_VIDEOS_BASE / camera_id)
        Path(folder).mkdir(parents=True, exist_ok=True)
        camera = {
            "id": camera_id,
            "name": name,
            "x": x,
            "y": y,
            "angle": angle,
            "zone": zone,
            "folder": folder,
            "active": True,
            "person_count": 0,
            "source_type": source_type,
            "usb_index": usb_index,
        }
        cameras.append(camera)
        _save(cameras)
        return camera


def update_camera(camera_id: str, **kwargs) -> dict | None:
    with _lock:
        cameras = _load()
        for i, c in enumerate(cameras):
            if c["id"] == camera_id:
                for k, v in kwargs.items():
                    # usb_index peut légitimement être None (source_type=file)
                    if v is not None or k == "usb_index":
                        cameras[i][k] = v
                _save(cameras)
                return cameras[i]
        return None


def set_person_count(camera_id: str, count: int) -> None:
    with _lock:
        cameras = _load()
        for c in cameras:
            if c["id"] == camera_id:
                c["person_count"] = count
                break
        _save(cameras)


def delete_camera(camera_id: str) -> bool:
    with _lock:
        cameras = _load()
        new_cameras = [c for c in cameras if c["id"] != camera_id]
        if len(new_cameras) == len(cameras):
            return False
        _save(new_cameras)
        return True
