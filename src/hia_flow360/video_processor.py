from __future__ import annotations

import math
import threading
import time
from collections import deque
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

# ── Filtrage mouvement ────────────────────────────────────────────────────────
_MOTION_THRESHOLD    = 0.20   # proportion de pixels en mouvement dans la bbox
_MIN_DISPLACEMENT_PX = 15     # déplacement centroïde min pour valider le mvt

# ── Tracking ─────────────────────────────────────────────────────────────────
_TRACK_HISTORY  = 12
_TRACK_MAX_DIST = 90
_TRACK_MAX_GONE = 15
_WARMUP_FRAMES  = 50

# ── Stream ────────────────────────────────────────────────────────────────────
_STREAM_MAX_W  = 854   # largeur max pour le navigateur
_STREAM_JPEG_Q = 70    # qualité JPEG stream

# ── YOLO ─────────────────────────────────────────────────────────────────────
# On passe imgsz directement à YOLO sans pre-resize manuel.
# YOLO gère lui-même le letterboxing (indispensable pour les vidéos portrait).
# 640 au lieu de 320 : les personnes éloignées font ~16 px au lieu de ~8 px,
# ce qui passe le seuil de détection fiable de YOLOv8.
_YOLO_IMGSZ = 640

# Délai minimal entre deux soumissions YOLO pour la même caméra (ms).
# Évite de spammer le dispatcher quand la vidéo tourne vite.
_SUBMIT_MIN_INTERVAL = 0.12   # ~8 soumissions/s max par caméra (640 est plus lent)


# ─────────────────────────────────────────────────────────────────────────────
# Tracker centroïde
# ─────────────────────────────────────────────────────────────────────────────

class _PersonTracker:
    """
    Tracker nearest-neighbour avec historique de déplacement.
    Retourne [(track_id, dx, dy, history_len)] par bbox.
    Exécuté exclusivement dans le thread YOLO.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, deque] = {}
        self._gone:   dict[int, int]   = {}
        self._next_id = 0

    def update(
        self, bboxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, float, float, int]]:
        centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in bboxes]

        if not bboxes:
            for tid in list(self._gone):
                self._gone[tid] += 1
                if self._gone[tid] > _TRACK_MAX_GONE:
                    self._tracks.pop(tid, None)
                    self._gone.pop(tid, None)
            return []

        track_ids   = list(self._tracks.keys())
        matched_det: dict[int, int] = {}
        matched_trk: set[int]       = set()

        for det_i, (cx, cy) in enumerate(centroids):
            best_tid, best_dist = None, float("inf")
            for tid in track_ids:
                if tid in matched_trk:
                    continue
                tx, ty = self._tracks[tid][-1]
                d = math.hypot(cx - tx, cy - ty)
                if d < best_dist and d <= _TRACK_MAX_DIST:
                    best_dist, best_tid = d, tid
            if best_tid is not None:
                matched_det[det_i] = best_tid
                matched_trk.add(best_tid)

        for tid in track_ids:
            if tid not in matched_trk:
                self._gone[tid] = self._gone.get(tid, 0) + 1
                if self._gone[tid] > _TRACK_MAX_GONE:
                    self._tracks.pop(tid, None)
                    self._gone.pop(tid, None)
            else:
                self._gone[tid] = 0

        for det_i, tid in matched_det.items():
            self._tracks[tid].append(centroids[det_i])

        for det_i in range(len(bboxes)):
            if det_i not in matched_det:
                nid = self._next_id
                self._tracks[nid] = deque([centroids[det_i]], maxlen=_TRACK_HISTORY + 2)
                self._gone[nid]   = 0
                matched_det[det_i] = nid
                self._next_id += 1

        result: list[tuple[int, float, float, int]] = []
        for det_i in range(len(bboxes)):
            tid  = matched_det[det_i]
            hist = self._tracks[tid]
            hlen = len(hist)
            dx   = float(hist[-1][0] - hist[0][0]) if hlen >= 2 else 0.0
            dy   = float(hist[-1][1] - hist[0][1]) if hlen >= 2 else 0.0
            result.append((tid, dx, dy, hlen))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher YOLO avec priorité aux caméras visionnées
# ─────────────────────────────────────────────────────────────────────────────

class _YoloDispatcher:
    """
    Thread YOLO unique partagé entre toutes les caméras.

    Priorité dynamique :
      1. Les caméras avec un viewer actif (stream ouvert dans le navigateur)
         sont traitées EN PREMIER et de façon continue jusqu'à épuisement
         de leurs slots → refresh YOLO à ~500 ms pour la caméra regardée.
      2. Une fois les prioritaires traités, on traite UN slot de caméra
         "background" (comptage, pas de viewer actif).

    Chaque caméra dispose d'un slot unique : si une nouvelle frame arrive
    avant que la précédente soit traitée, elle l'écrase (latest-wins).
    """

    def __init__(self) -> None:
        self._slots: dict[str, tuple | None] = {}
        self._lock  = threading.Lock()
        self._event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="yolo-worker"
        )
        self._thread.start()

    def register(self, camera_id: str) -> None:
        with self._lock:
            self._slots.setdefault(camera_id, None)

    def unregister(self, camera_id: str) -> None:
        with self._lock:
            self._slots.pop(camera_id, None)

    def submit(
        self,
        camera_id: str,
        frame: np.ndarray,
        motion_mask: np.ndarray | None,
        warmed_up: bool,
    ) -> None:
        """Non bloquant. Écrase l'ancien slot si le worker est encore occupé."""
        with self._lock:
            self._slots[camera_id] = (frame, motion_mask, warmed_up)
        self._event.set()

    def _pick(self, priority_only: bool) -> tuple[str, tuple] | None:
        """Extrait le prochain slot à traiter (prioritaire si demandé)."""
        with self._lock:
            for cam_id, slot in self._slots.items():
                if slot is None:
                    continue
                if priority_only:
                    with _plock:
                        proc = _processors.get(cam_id)
                    if not (proc and proc._viewer_count > 0):
                        continue
                self._slots[cam_id] = None
                return cam_id, slot
        return None

    def _process(self, cam_id: str, frame: np.ndarray,
                 motion_mask: np.ndarray | None, warmed_up: bool) -> None:
        with _plock:
            proc = _processors.get(cam_id)
        if proc is None or not _YOLO_AVAILABLE or _yolo_model is None:
            return
        detections = proc._run_yolo(frame, motion_mask, warmed_up)
        with proc._det_lock:
            proc._detections    = detections
            proc.person_count   = sum(1 for d in detections if d[5])
            proc.inflow_count   = sum(1 for d in detections if d[6])
            proc.outflow_count  = sum(1 for d in detections if d[7])
            proc.total_detected = len(detections)

    def _run(self) -> None:
        while True:
            self._event.wait(timeout=0.05)
            self._event.clear()

            # 1. Traiter TOUTES les caméras avec viewer actif d'abord
            while True:
                job = self._pick(priority_only=True)
                if job is None:
                    break
                self._process(job[0], *job[1])

            # 2. Traiter UNE caméra background (comptage en arrière-plan)
            job = self._pick(priority_only=False)
            if job:
                self._process(job[0], *job[1])


_dispatcher: _YoloDispatcher | None = None
_disp_lock  = threading.Lock()


def _get_dispatcher() -> _YoloDispatcher:
    global _dispatcher
    if _dispatcher is None:
        with _disp_lock:
            if _dispatcher is None:
                _dispatcher = _YoloDispatcher()
    return _dispatcher


# ─────────────────────────────────────────────────────────────────────────────
# Frame vide
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Processeur caméra
# ─────────────────────────────────────────────────────────────────────────────

class CameraProcessor:
    """
    Architecture à deux niveaux découplés :

    Thread vidéo (_process_video)
        Lit les frames à la cadence native de la vidéo.
        Exécute MOG2 (~2 ms), throttle les soumissions YOLO,
        annote avec les DERNIÈRES détections disponibles,
        encode et stocke le JPEG pour le stream.

    Thread YOLO (_YoloDispatcher, unique et partagé)
        Reçoit les frames via les slots.
        Traite en priorité les caméras avec un viewer actif.
        Met à jour person_count / inflow_count / outflow_count.

    Résultat : le stream vidéo tourne à plein FPS sans jamais attendre YOLO.
    La détection se rafraîchit à ~500 ms pour la caméra regardée.
    """

    def __init__(self, camera_id: str, folder: str, angle: float = 0.0) -> None:
        self.camera_id = camera_id
        self.folder    = Path(folder)
        self.angle     = angle

        # Compteurs publics (mis à jour par le dispatcher)
        self.person_count:   int = 0
        self.total_detected: int = 0
        self.inflow_count:   int = 0
        self.outflow_count:  int = 0

        # Sens d'entrée selon l'angle de la caméra
        # cos(angle) >= 0 → caméra orientée vers la droite → droite = entrée
        self._inflow_is_right: bool = math.cos(math.radians(angle)) >= 0

        # Résultats YOLO (dispatcher → video loop)
        self._det_lock   = threading.Lock()
        self._detections: list[tuple] = []

        # Frame JPEG pour le stream (video loop → frame_generator)
        self._frame_lock   = threading.Lock()
        self._latest_frame = (
            _blank_jpeg("En attente de vidéo…", Path(folder).name)
            if _CV2_AVAILABLE else b""
        )

        # Nombre de viewers actifs (frame_generator ouvert dans le navigateur)
        # Utilisé par le dispatcher pour la priorité YOLO.
        self._viewer_count: int = 0
        self._viewer_lock       = threading.Lock()

        self._stop        = threading.Event()
        self._frame_count = 0
        self._tracker     = _PersonTracker()
        self._last_submit: float = 0.0   # throttle des soumissions YOLO

        self._bg_sub = (
            cv2.createBackgroundSubtractorMOG2(
                history=400, varThreshold=20, detectShadows=False
            )
            if _CV2_AVAILABLE else None
        )

        _get_dispatcher().register(camera_id)

        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"cam-{camera_id}"
        )
        self._thread.start()

    # ── API publique ──────────────────────────────────────────────────────────

    def get_frame(self) -> bytes:
        with self._frame_lock:
            return self._latest_frame

    def add_viewer(self) -> None:
        with self._viewer_lock:
            self._viewer_count += 1

    def remove_viewer(self) -> None:
        with self._viewer_lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    def stop(self) -> None:
        self._stop.set()
        _get_dispatcher().unregister(self.camera_id)

    # ── Thread vidéo ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        if not _CV2_AVAILABLE:
            return
        while not self._stop.is_set():
            videos = (
                sorted(p for p in self.folder.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
                if self.folder.exists() else []
            )
            if not videos:
                with self._frame_lock:
                    self._latest_frame = _blank_jpeg(
                        "Aucune vidéo dans le dossier", self.folder.name
                    )
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

        # Rotation automatique selon les métadonnées (vidéos portrait téléphone)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        delay = 1.0 / fps

        while not self._stop.is_set():
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            self._frame_count += 1
            warmed_up = self._frame_count > _WARMUP_FRAMES

            # ── MOG2 (rapide, chaque frame) ───────────────────────────────
            motion_mask: np.ndarray | None = None
            if self._bg_sub is not None:
                fgmask = self._bg_sub.apply(frame)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fgmask = cv2.dilate(fgmask, kernel, iterations=2)
                _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
                motion_mask = fgmask

            # ── Soumission au dispatcher YOLO (throttlée, non bloquante) ──
            now = time.perf_counter()
            if now - self._last_submit >= _SUBMIT_MIN_INTERVAL:
                _get_dispatcher().submit(
                    self.camera_id, frame.copy(), motion_mask, warmed_up
                )
                self._last_submit = now

            # ── Annotation avec les dernières détections disponibles ───────
            with self._det_lock:
                detections = list(self._detections)

            annotated = self._annotate(frame.copy(), detections, warmed_up)

            # ── Encodage JPEG réduit pour le stream ───────────────────────
            sh, sw = annotated.shape[:2]
            if sw > _STREAM_MAX_W:
                sf       = _STREAM_MAX_W / sw
                stream_f = cv2.resize(
                    annotated, (_STREAM_MAX_W, int(sh * sf)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                stream_f = annotated

            _, buf = cv2.imencode(
                ".jpg", stream_f, [cv2.IMWRITE_JPEG_QUALITY, _STREAM_JPEG_Q]
            )
            with self._frame_lock:
                self._latest_frame = buf.tobytes()

            # Respect de la cadence vidéo (soustrait le temps de traitement)
            elapsed = time.perf_counter() - t0
            wait    = delay - elapsed
            if wait > 0:
                time.sleep(wait)

        cap.release()

    # ── Inférence YOLO (appelée par le dispatcher dans son thread) ────────────

    def _run_yolo(
        self,
        frame: np.ndarray,
        motion_mask: np.ndarray | None,
        warmed_up: bool,
    ) -> list[tuple]:
        """
        Détection YOLO + tracking + classification.

        On passe la frame ORIGINALE à YOLO avec imgsz=_YOLO_IMGSZ.
        YOLO gère lui-même le letterboxing (resize proportionnel + padding)
        et retourne les coordonnées dans l'espace de la frame originale.
        Ceci est INDISPENSABLE pour les vidéos portrait : un pre-resize manuel
        faussait les coordonnées après le letterboxing interne de YOLO.

        Retourne list[(x1, y1, x2, y2, conf, is_moving, flag_inflow, flag_outflow)].
        """
        if not (_YOLO_AVAILABLE and _yolo_model is not None):
            return []

        h, w = frame.shape[:2]

        results = _yolo_model(  # type: ignore[call-arg]
            frame, classes=[0], conf=0.30, verbose=False, imgsz=_YOLO_IMGSZ
        )

        all_bboxes: list[tuple[int, int, int, int]] = []
        all_confs:  list[float] = []

        for r in results:
            for box in r.boxes:
                # Coordonnées dans l'espace de la frame originale (YOLO gère l'inverse transform)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_bboxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
                all_confs.append(float(box.conf[0]))

        track_info = self._tracker.update(all_bboxes)
        detections: list[tuple] = []

        for (x1, y1, x2, y2), conf, (tid, dx, dy, hlen) in zip(
            all_bboxes, all_confs, track_info
        ):
            # Vérification mouvement : masque pixel + déplacement centroïde
            is_moving = True
            if motion_mask is not None and warmed_up:
                roi         = motion_mask[y1:y2, x1:x2]
                pixel_ok    = float(np.sum(roi > 0)) / max(roi.size, 1) >= _MOTION_THRESHOLD
                if hlen >= 3:
                    is_moving = pixel_ok and math.hypot(dx, dy) >= _MIN_DISPLACEMENT_PX
                else:
                    is_moving = pixel_ok

            # Classification inflow / outflow
            flag_in = flag_out = False
            if is_moving and abs(dx) >= 8:
                if (dx > 0) == self._inflow_is_right:
                    flag_in = True
                else:
                    flag_out = True

            detections.append((x1, y1, x2, y2, conf, is_moving, flag_in, flag_out))

        return detections

    # ── Annotation (thread vidéo, non bloquant) ───────────────────────────────

    def _annotate(
        self, frame: np.ndarray, detections: list[tuple], warmed_up: bool
    ) -> np.ndarray:
        moving = inflow = outflow = total = 0

        for (x1, y1, x2, y2, conf, is_moving, flag_in, flag_out) in detections:
            total += 1
            if is_moving:
                moving += 1
                if flag_in:
                    inflow += 1
                    color, lbl = (0, 210, 90),  f"IN {conf:.0%}"
                elif flag_out:
                    outflow += 1
                    color, lbl = (30, 140, 255), f"OUT {conf:.0%}"
                else:
                    color, lbl = (0, 210, 90),  f"mvt {conf:.0%}"
            else:
                color, lbl = (110, 110, 110), "fixe"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_moving else 1)
            cv2.putText(frame, lbl, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        if not warmed_up:
            cv2.putText(
                frame, f"Calibration ({self._frame_count}/{_WARMUP_FRAMES})…",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1,
            )

        label = f"Mvt:{moving}  IN:{inflow}  OUT:{outflow}  |  Tot:{total}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (6, 6), (tw + 14, th + 18), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# API publique
# ─────────────────────────────────────────────────────────────────────────────

def start_processor(camera_id: str, folder: str, angle: float = 0.0) -> None:
    with _plock:
        if camera_id not in _processors:
            _processors[camera_id] = CameraProcessor(camera_id, folder, angle)


def stop_processor(camera_id: str) -> None:
    with _plock:
        proc = _processors.pop(camera_id, None)
    if proc:
        proc.stop()


def get_count(camera_id: str) -> int:
    with _plock:
        proc = _processors.get(camera_id)
    return proc.person_count if proc else 0


def get_total_detected(camera_id: str) -> int:
    with _plock:
        proc = _processors.get(camera_id)
    return proc.total_detected if proc else 0


def get_flow_counts(camera_id: str) -> dict[str, int]:
    with _plock:
        proc = _processors.get(camera_id)
    if proc:
        return {
            "moving":  proc.person_count,
            "inflow":  proc.inflow_count,
            "outflow": proc.outflow_count,
        }
    return {"moving": 0, "inflow": 0, "outflow": 0}


def frame_generator(camera_id: str) -> Generator[bytes, None, None]:
    """
    Génère les frames MJPEG à 10 FPS max.
    Enregistre un viewer actif pour que le dispatcher YOLO priorise cette caméra.
    """
    proc_ref: CameraProcessor | None = None
    with _plock:
        proc_ref = _processors.get(camera_id)
    if proc_ref:
        proc_ref.add_viewer()
    try:
        while True:
            with _plock:
                proc = _processors.get(camera_id)
            frame = proc.get_frame() if proc else _blank_jpeg("Caméra introuvable")
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.10)   # 10 fps
    finally:
        # Décrémenter le compteur de viewers quand le navigateur ferme la connexion
        if proc_ref:
            proc_ref.remove_viewer()


def start_all(cameras: list[dict]) -> None:
    for cam in cameras:
        start_processor(cam["id"], cam["folder"], cam.get("angle", 0.0))
