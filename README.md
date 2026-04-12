# HIA Flow360

Plateforme Python/FastAPI de supervision des flux passagers aéroportuaires.

Le projet combine:
- un mode simule (generation d'evenements Wi-Fi/XOVIS/AODB),
- un mode reel via cameras video avec detection YOLOv8,
- un dashboard web integre (overview, zones, predictions, carte interactive).

## Fonctionnalites

### 1) Simulation operationnelle
- Generation d'evenements synthetiques multi-sources
- Fusion et calcul de snapshots par zone
- KPIs: occupancy, inflow/min, outflow/min, dwell moyen, congestion
- Predictions court terme (occupancy + risque d'attente)

### 2) Video analytics (mode reel)
- Gestion CRUD des cameras
- Lecture de videos par camera (dossiers dans `videos/<camera_id>/`)
- Detection de personnes avec `ultralytics` (YOLOv8n)
- Filtrage mouvement (background subtraction) pour compter les personnes en mouvement
- Streaming MJPEG par camera

### 3) Dashboard web
- `/` : vue overview KPI
- `/dashboard/zones` : analytics par zone
- `/dashboard/predictions` : previsions
- `/dashboard/map` : carte interactive (zones + cameras)

Un switch global dans l'UI permet de passer de la source `simulated` a `real`.

## Structure du projet

```text
Aeroport-next-gen/
  cameras.json
  zones.json
  yolov8n.pt
  requirements.txt
  src/
    hia_flow360/
      app.py             # API FastAPI + pages dashboard
      main.py            # point d'entree local (uvicorn)
      analytics.py       # KPIs et snapshots
      fusion.py          # fusion/pseudonymisation
      generators.py      # generation d'evenements simules
      predictor.py       # previsions
      camera_manager.py  # persistance des cameras (cameras.json)
      video_processor.py # pipeline video + YOLO + streaming
      zones_config.py    # gestion des zones (zones.json)
      models.py          # schemas Pydantic
  videos/
    <camera_id>/         # videos associees a chaque camera
```

## Prerequis
- Python 3.10+
- Environnement virtuel recommande
- Le fichier `yolov8n.pt` a la racine du projet

## Installation et lancement (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m hia_flow360.main
```

L'application demarre sur `http://127.0.0.1:8000`.

## Documentation API
- Swagger UI: `http://127.0.0.1:8000/docs`

## Endpoints principaux

### Sante et simulation
- `GET /health`
- `POST /refresh?minutes=120&avg_events_per_minute=40`
- `GET /kpis`
- `GET /zones/latest`
- `GET /predictions?horizon_min=30`

### Dashboard HTML
- `GET /`
- `GET /dashboard/zones`
- `GET /dashboard/predictions`
- `GET /dashboard/map`

### Cameras
- `GET /cameras`
- `POST /cameras`
- `PUT /cameras/{camera_id}`
- `DELETE /cameras/{camera_id}`
- `GET /cameras/{camera_id}/stream`
- `GET /cameras/{camera_id}/count`

### Zones et agregration live
- `GET /zones-config`
- `POST /zones-config`
- `POST /zones-config/reset`
- `GET /zones/camera-live`

## Notes d'utilisation
- Au demarrage, la simulation est initialisee automatiquement.
- Les processors video se lancent automatiquement pour les cameras deja presentes dans `cameras.json`.
- Pour tester le mode reel, ajoutez une camera puis deposez une ou plusieurs videos dans son dossier `videos/<camera_id>/`.
- En mode `real`, la page predictions reste basee sur le moteur de prediction de la simulation (indique dans l'UI).

## Perspectives d'evolution
- Connecteurs reels HIA (Wi-Fi, XOVIS, AODB) en remplacement des generateurs
- Stockage persistant des evenements/snapshots (stream + base analytique)
- Modeles ML calibres sur donnees historiques aeroport
- Authentification, RBAC et gouvernance privacy

