# HIA Flow360 (Python MVP)

Python prototype for a unified airport movement intelligence platform inspired by the HIA challenge.

## What this MVP includes
- Multi-source synthetic ingestion (`Wi-Fi`, `XOVIS`, `AODB`)
- Data fusion + confidence filtering + event deduplication
- Real-time operational KPIs by zone:
  - occupancy
  - inflow/outflow per minute
  - estimated dwell time
  - congestion score
- Short-horizon prediction (occupancy + wait risk) per zone
- FastAPI endpoints suitable for dashboard integration

## Project structure
```text
src/hia_flow360/
  app.py          # FastAPI service
  analytics.py    # KPI and snapshot calculations
  fusion.py       # pseudonymization and deduplication
  generators.py   # synthetic event generation
  models.py       # Pydantic data contracts
  predictor.py    # short-term forecasting
  main.py         # local runner
requirements.txt
```

## Quick start
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m hia_flow360.main
```

API docs:
- Swagger UI: `http://127.0.0.1:8000/docs`

## Endpoints
- `GET /health`
- `POST /refresh?minutes=120&avg_events_per_minute=40`
- `GET /kpis`
- `GET /zones/latest`
- `GET /predictions?horizon_min=30`

## Notes for real deployment
- Replace synthetic generator with real connectors for HIA data sources.
- Move state to a streaming backbone (Kafka/Pulsar) + time-series/OLAP storage.
- Train ML models on historical airport data for zone-specific forecasts.
- Add role-based access controls and full privacy governance controls.

