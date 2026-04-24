# HIA Flow360

Plateforme Python/FastAPI de supervision des flux passagers aéroportuaires.

Le projet combine :
- un mode simulé (génération d'événements Wi-Fi/XOVIS/AODB),
- un mode réel via caméras vidéo avec détection YOLOv8,
- un dashboard web intégré (overview, zones, prédictions, carte interactive).

## Fonctionnalités

### 1) Simulation opérationnelle
- Génération d'événements synthétiques multi-sources
- Fusion et calcul de snapshots par zone
- KPIs : occupancy, inflow/min, outflow/min, dwell moyen, congestion
- Prédictions court terme (occupancy + risque d'attente)

### 2) Video analytics (mode réel)
- Gestion CRUD des caméras
- Lecture de vidéos par caméra (dossiers dans `videos/<camera_id>/`)
- Détection de personnes avec YOLOv8n à `imgsz=640` (personnes éloignées incluses)
- Support des vidéos portrait (smartphones) — rotation automatique via métadonnées
- **Filtrage mouvement** : double condition — soustraction de fond MOG2 (≥ 20 % pixels actifs dans la bbox) **et** déplacement centroïde ≥ 15 px sur l'historique (évite les faux positifs dus au bruit vidéo)
- **Direction inflow/outflow** : basée sur le mouvement horizontal (dx) corrigé par l'angle de la caméra (`cos(angle) ≥ 0` → droite = entrée)
- **Dispatcher YOLO prioritaire** : thread YOLO unique partagé ; les caméras avec un stream ouvert sont traitées en continu (~500 ms refresh), les autres en arrière-plan
- Streaming MJPEG par caméra (10 fps, qualité 70 %, largeur max 854 px)
- Snapshots par zone toutes les 60 s pour alimenter le modèle prédictif en mode réel

### 3) Modèle prédictif
- En mode **simulé** : basé sur les snapshots de simulation
- En mode **réel** : basé sur les snapshots caméras dès que ≥ 3 points sont disponibles (indiqué `source: "cameras"` dans l'API)
- Robustesse aux données caméras bruitées : médiane des deltas + filtrage MAD (écart absolu médian)

### 4) Dashboard web
- `/` : vue overview KPIs
- `/dashboard/zones` : analytics par zone (inflow/outflow/dwell/congestion en mode réel)
- `/dashboard/predictions` : prévisions avec niveau de risque
- `/dashboard/map` : carte interactive (zones + caméras repositionnables)

Un switch global dans l'UI permet de passer de la source `simulated` à `real`.

## Structure du projet

```text
Aeroport-next-gen/
  cameras.json           # configuration et état des caméras
  zones.json             # configuration des zones
  yolov8n.pt             # poids YOLOv8n (téléchargé automatiquement si absent)
  requirements.txt
  src/
    hia_flow360/
      app.py             # API FastAPI + pages dashboard
      main.py            # point d'entrée local (uvicorn)
      analytics.py       # KPIs et snapshots
      fusion.py          # fusion/pseudonymisation
      generators.py      # génération d'événements simulés
      predictor.py       # prévisions (simulation + caméras réelles)
      camera_manager.py  # persistance des caméras (cameras.json)
      video_processor.py # pipeline vidéo + YOLO + streaming MJPEG
      zones_config.py    # gestion des zones (zones.json)
      models.py          # schémas Pydantic
  videos/
    <camera_id>/         # vidéos associées à chaque caméra
```

## Prérequis

- **Python 3.13** (obligatoire — `pydantic-core` et `numpy` n'ont pas de wheels pour Python 3.14)
- Environnement virtuel recommandé
- Le fichier `yolov8n.pt` est téléchargé automatiquement au premier lancement

## Installation et lancement (Windows)

```powershell
# Créer le venv avec Python 3.13 (py launcher recommandé)
py -3.13 -m venv .venv

# Activer le venv
.venv\Scripts\Activate.ps1

# Installer les dépendances (wheels binaires pré-compilés)
pip install --prefer-binary -r requirements.txt

# Lancer le serveur
$env:PYTHONPATH = "src"
python -m hia_flow360.main
```

L'application démarre sur `http://127.0.0.1:8000`.

> **Note Windows** : si le chemin du projet contient des caractères accentués (é, à…), utiliser `py -3.13` plutôt que `python` pour créer le venv afin d'éviter les erreurs de compilation.

## Documentation API

- Swagger UI : `http://127.0.0.1:8000/docs`

## Endpoints principaux

### Santé et simulation
| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | État du service |
| POST | `/refresh?minutes=120&avg_events_per_minute=40` | Regénère les événements simulés |
| GET | `/kpis` | KPIs globaux |
| GET | `/zones/latest` | Dernier snapshot par zone (simulé) |
| GET | `/predictions?horizon_min=30` | Prévisions (simulé ou caméras selon disponibilité) |

### Dashboard HTML
| Méthode | Endpoint |
|---|---|
| GET | `/` |
| GET | `/dashboard/zones` |
| GET | `/dashboard/predictions` |
| GET | `/dashboard/map` |

### Caméras
| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/cameras` | Liste des caméras |
| POST | `/cameras` | Ajouter une caméra |
| PUT | `/cameras/{camera_id}` | Modifier une caméra |
| DELETE | `/cameras/{camera_id}` | Supprimer une caméra |
| GET | `/cameras/{camera_id}/stream` | Stream MJPEG |
| GET | `/cameras/{camera_id}/count` | Comptage en temps réel |

### Zones et agrégation live
| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/zones-config` | Configuration des zones |
| POST | `/zones-config` | Modifier une zone |
| POST | `/zones-config/reset` | Réinitialiser les zones |
| GET | `/zones/camera-live` | Données en direct (occupancy, inflow, outflow, dwell, congestion) |

## Notes d'utilisation

- Au démarrage, la simulation est initialisée automatiquement.
- Les processors vidéo se lancent automatiquement pour les caméras présentes dans `cameras.json`.
- Pour tester le mode réel, ajoutez une caméra via la carte interactive puis déposez une ou plusieurs vidéos dans son dossier `videos/<camera_id>/`.
- En mode `real`, les prédictions utilisent les données caméras dès que 3 snapshots sont disponibles (environ 3 minutes après le démarrage). Avant ce délai, la simulation est utilisée en fallback.
- Les vidéos portrait (filmées en vertical) sont supportées nativement.

## Perspectives d'évolution

- Connecteurs réels HIA (Wi-Fi, XOVIS, AODB) en remplacement des générateurs
- Stockage persistant des événements/snapshots (stream + base analytique)
- Modèles ML calibrés sur données historiques aéroport
- Authentification, RBAC et gouvernance privacy
