from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Generator

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    _yolo_model: object | None = YOLO("yolov8n.pt")
    _YOLO_AVAILABLE = True
except Exception:
    _yolo_model = None
    _YOLO_AVAILABLE = False

_processors: dict[str, "CameraProcessor"] = {}
_plock = threading.Lock()

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Minimum proportion of pixels in a bounding box that must be "in motion"
# for the person to count as moving.
_MOTION_THRESHOLD = 0.12
# Number of frames needed to warm up the background model before filtering.
_WARMUP_FRAMES = 50


def _blank_jpeg(label: str, sub: str = "") -> bytes:
    if not _CV2_AVAILABLE:
        return b""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (28, 28, 28)
    cv2.putText(frame, label, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    if sub:
        cv2.putText(frame, sub, (30, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 110), 1)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


class CameraProcessor:
    def __init__(self, camera_id: str, folder: str) -> None:
        self.camera_id = camera_id
        self.folder = Path(folder)
        self.person_count: int = 0          # moving people only
        self.total_detected: int = 0        # all detected (moving + static)
        self._latest_frame: bytes = (
            _blank_jpeg("En attente de vidéo…", self.folder.name)
            if _CV2_AVAILABLE else b""
        )
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        # Background subtractor – learns background over first frames
        self._bg_sub = (
            cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=20, detectShadows=False)
            if _CV2_AVAILABLE else None
        )
        self._frame_count = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"cam-{camera_id}"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_frame(self) -> bytes:
        with self._frame_lock:
            return self._latest_frame

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if not _CV2_AVAILABLE:
            return
        while not self._stop.is_set():
            videos = (
                sorted(p for p in self.folder.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
                if self.folder.exists() else []
            )
            if not videos:
                frame_bytes = _blank_jpeg("Aucune vidéo dans le dossier", str(self.folder.name))
                with self._frame_lock:
                    self._latest_frame = frame_bytes
                self._stop.wait(2.0)
                continue
            for path in videos:
                if self._stop.is_set():
                    return
                self._process_video(path)

    def _process_video(self, path: Path) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        delay = 1.0 / fps
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, moving, total = self._detect(frame)
            self.person_count = moving
            self.total_detected = total
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with self._frame_lock:
                self._latest_frame = buf.tobytes()
            time.sleep(delay)
        cap.release()

    # ------------------------------------------------------------------
    # Detection with motion filtering
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Return (annotated frame, moving_count, total_detected)."""
        self._frame_count += 1
        moving_count = 0
        total_count = 0
        warmed_up = self._frame_count > _WARMUP_FRAMES

        # --- Background subtraction → motion mask ---
        motion_mask: np.ndarray | None = None
        if self._bg_sub is not None:
            fgmask = self._bg_sub.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.dilate(fgmask, kernel, iterations=2)
            _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            motion_mask = fgmask

        # --- YOLO person detection ---
        if _YOLO_AVAILABLE and _yolo_model is not None:
            results = _yolo_model(frame, classes=[0], conf=0.45, verbose=False)  # type: ignore[call-arg]
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    total_count += 1

                    # Determine if person is moving
                    is_moving = True  # default during warmup
                    if motion_mask is not None and warmed_up:
                        cx1 = max(x1, 0)
                        cy1 = max(y1, 0)
                        cx2 = min(x2, frame.shape[1])
                        cy2 = min(y2, frame.shape[0])
                        roi = motion_mask[cy1:cy2, cx1:cx2]
                        motion_ratio = float(np.sum(roi > 0)) / max(roi.size, 1)
                        is_moving = motion_ratio >= _MOTION_THRESHOLD

                    if is_moving:
                        moving_count += 1
                        color = (0, 210, 90)     # green = moving
                        label = f"mvt {conf:.0%}"
                    else:
                        color = (110, 110, 110)  # grey = stationary
                        label = f"fixe"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_moving else 1)
                    cv2.putText(
                        frame, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    )
        else:
            cv2.putText(
                frame, "YOLO non disponible", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2,
            )

        # --- Status overlay ---
        if not warmed_up:
            status = f"Calibration ({self._frame_count}/{_WARMUP_FRAMES})…"
            cv2.putText(frame, status, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

        label = f"En mvt: {moving_count}  /  Détectés: {total_count}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.rectangle(frame, (6, 6), (tw + 14, th + 18), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        return frame, moving_count, total_count


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def start_processor(camera_id: str, folder: str) -> None:
    with _plock:
        if camera_id not in _processors:
            _processors[camera_id] = CameraProcessor(camera_id, folder)


def stop_processor(camera_id: str) -> None:
    with _plock:
        proc = _processors.pop(camera_id, None)
    if proc:
        proc.stop()


def get_count(camera_id: str) -> int:
    """Returns count of moving people only."""
    with _plock:
        proc = _processors.get(camera_id)
    return proc.person_count if proc else 0


def get_total_detected(camera_id: str) -> int:
    """Returns total detected (moving + stationary)."""
    with _plock:
        proc = _processors.get(camera_id)
    return proc.total_detected if proc else 0


def frame_generator(camera_id: str) -> Generator[bytes, None, None]:
    while True:
        with _plock:
            proc = _processors.get(camera_id)
        frame = proc.get_frame() if proc else _blank_jpeg("Caméra introuvable")
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.04)  # ~25 fps


def start_all(cameras: list[dict]) -> None:
    for cam in cameras:
        start_processor(cam["id"], cam["folder"])
