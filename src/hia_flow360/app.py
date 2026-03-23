from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .analytics import airport_kpis, compute_snapshots, latest_snapshot_map
from .fusion import events_to_dataframe
from .generators import generate_events
from .predictor import forecast

app = FastAPI(
    title="HIA Flow360 API",
    description="Airport movement intelligence MVP in Python",
    version="0.1.0",
)

_lock = Lock()
_state: dict[str, object] = {"events": [], "snapshots": [], "updated_at": None}

_DASHBOARD_STYLE = """
<style>
  :root {
    --bg: #f4f6f8;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --text: #0f172a;
    --muted: #475569;
    --line: #dbe2ea;
    --nav-bg: linear-gradient(180deg, #0f172a 0%, #13243f 100%);
    --nav-text: #d6deea;
    --nav-active: #22c55e;
    --accent: #0ea5e9;
    --warning: #f59e0b;
    --danger: #ef4444;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    color: var(--text);
    background: radial-gradient(circle at top right, #dbeafe, var(--bg) 45%);
  }

  .layout {
    display: grid;
    grid-template-columns: 250px 1fr;
    min-height: 100vh;
  }

  .sidebar {
    background: var(--nav-bg);
    color: var(--nav-text);
    padding: 1.25rem 1rem;
    border-right: 1px solid #1f2d48;
    position: sticky;
    top: 0;
    height: 100vh;
  }

  .brand {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-bottom: 1.2rem;
  }

  .tag {
    display: inline-block;
    font-size: 0.78rem;
    padding: 0.2rem 0.45rem;
    border-radius: 999px;
    margin-top: 0.4rem;
    color: #d1fae5;
    background: rgba(34, 197, 94, 0.16);
    border: 1px solid rgba(34, 197, 94, 0.35);
  }

  .nav {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    margin-top: 1.3rem;
  }

  .nav a {
    color: var(--nav-text);
    text-decoration: none;
    padding: 0.58rem 0.7rem;
    border-radius: 0.55rem;
    border: 1px solid transparent;
    transition: 120ms ease-in;
  }

  .nav a:hover {
    background: rgba(148, 163, 184, 0.15);
    border-color: rgba(148, 163, 184, 0.18);
  }

  .nav a.active {
    background: rgba(34, 197, 94, 0.15);
    border-color: rgba(34, 197, 94, 0.35);
    color: #ecfdf5;
  }

  .main {
    padding: 1.35rem 1.55rem;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  h1 {
    margin: 0;
    font-size: 1.35rem;
  }

  .sub {
    margin: 0.35rem 0 0;
    color: var(--muted);
    font-size: 0.92rem;
  }

  .btn {
    background: var(--accent);
    color: white;
    border: 0;
    border-radius: 0.55rem;
    padding: 0.48rem 0.8rem;
    cursor: pointer;
    font-weight: 600;
  }

  .btn:hover { filter: brightness(0.95); }

  .grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(12, 1fr);
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0.85rem;
    padding: 0.85rem;
    box-shadow: 0 6px 24px rgba(15, 23, 42, 0.05);
  }

  .kpi {
    grid-column: span 3;
    min-height: 88px;
  }

  .kpi .label {
    color: var(--muted);
    font-size: 0.84rem;
    margin-bottom: 0.22rem;
  }

  .kpi .value {
    font-size: 1.42rem;
    font-weight: 700;
  }

  .span-8 { grid-column: span 8; }
  .span-4 { grid-column: span 4; }
  .span-12 { grid-column: span 12; }

  canvas {
    width: 100% !important;
    height: 330px !important;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }

  th, td {
    border-bottom: 1px solid var(--line);
    text-align: left;
    padding: 0.5rem 0.35rem;
  }

  th { color: var(--muted); font-weight: 600; background: var(--surface-2); }

  .badge {
    padding: 0.18rem 0.42rem;
    border-radius: 999px;
    color: white;
    font-size: 0.75rem;
  }
  .low { background: #10b981; }
  .medium { background: #f59e0b; }
  .high { background: #ef4444; }

  @media (max-width: 980px) {
    .layout { grid-template-columns: 1fr; }
    .sidebar { position: static; height: auto; }
    .kpi { grid-column: span 6; }
    .span-8, .span-4 { grid-column: span 12; }
  }
</style>
"""


def _shell(active: str, title: str, subtitle: str, body: str, page_script: str = "") -> str:
    nav = f"""
      <nav class="nav">
        <a href="/" {"class='active'" if active == "overview" else ""}>Overview</a>
        <a href="/dashboard/zones" {"class='active'" if active == "zones" else ""}>Zone Analytics</a>
        <a href="/dashboard/predictions" {"class='active'" if active == "predictions" else ""}>Predictions</a>
        <a href="/docs">API Docs</a>
      </nav>
    """
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>HIA Flow360 - {title}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
        {_DASHBOARD_STYLE}
      </head>
      <body>
        <div class="layout">
          <aside class="sidebar">
            <div class="brand">HIA Flow360<div class="tag">movement intelligence</div></div>
            {nav}
          </aside>
          <main class="main">
            <div class="header">
              <div>
                <h1>{title}</h1>
                <p class="sub">{subtitle}</p>
              </div>
              <button class="btn" onclick="refreshData()">Refresh Simulation</button>
            </div>
            {body}
          </main>
        </div>
        <script>
          async function refreshData() {{
            await fetch('/refresh', {{ method: 'POST' }});
            location.reload();
          }}
        </script>
        {page_script}
      </body>
    </html>
    """


def _rebuild(minutes: int = 120, avg_events_per_minute: int = 40) -> None:
    events = generate_events(minutes=minutes, avg_events_per_minute=avg_events_per_minute)
    df = events_to_dataframe(events)
    snapshots = compute_snapshots(df)
    _state["events"] = events
    _state["snapshots"] = snapshots
    _state["updated_at"] = datetime.now(timezone.utc)


@app.on_event("startup")
def _startup() -> None:
    with _lock:
        _rebuild()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    body = """
    <section class="grid">
      <article class="card kpi"><div class="label">Total Occupancy</div><div id="kpi-total" class="value">-</div></article>
      <article class="card kpi"><div class="label">Avg Congestion</div><div id="kpi-congestion" class="value">-</div></article>
      <article class="card kpi"><div class="label">Avg Dwell (min)</div><div id="kpi-dwell" class="value">-</div></article>
      <article class="card kpi"><div class="label">Peak Zone Congestion</div><div id="kpi-peak" class="value">-</div></article>
      <article class="card span-8">
        <h3>Zone Occupancy Snapshot</h3>
        <canvas id="occupancyChart"></canvas>
      </article>
      <article class="card span-4">
        <h3>Congestion Distribution</h3>
        <canvas id="congestionChart"></canvas>
      </article>
    </section>
    """
    script = """
    <script>
      let occupancyChart, congestionChart;
      async function loadOverview() {
        const [kpisRes, zonesRes] = await Promise.all([fetch('/kpis'), fetch('/zones/latest')]);
        const kpisData = await kpisRes.json();
        const zonesData = await zonesRes.json();
        const k = kpisData.kpis;
        document.getElementById('kpi-total').textContent = Math.round(k.total_occupancy);
        document.getElementById('kpi-congestion').textContent = (k.avg_congestion * 100).toFixed(1) + '%';
        document.getElementById('kpi-dwell').textContent = k.avg_dwell_minutes.toFixed(2);
        document.getElementById('kpi-peak').textContent = (k.max_zone_congestion * 100).toFixed(1) + '%';

        const zones = zonesData.zones || [];
        const labels = zones.map(z => z.zone);
        const occupancy = zones.map(z => z.occupancy);
        const congestion = zones.map(z => +(z.congestion_score * 100).toFixed(2));

        if (occupancyChart) occupancyChart.destroy();
        occupancyChart = new Chart(document.getElementById('occupancyChart'), {
          type: 'bar',
          data: {
            labels,
            datasets: [{ label: 'Occupancy', data: occupancy, backgroundColor: '#0ea5e9', borderRadius: 6 }]
          },
          options: { responsive: true, plugins: { legend: { display: false } } }
        });

        if (congestionChart) congestionChart.destroy();
        congestionChart = new Chart(document.getElementById('congestionChart'), {
          type: 'doughnut',
          data: {
            labels,
            datasets: [{ data: congestion, backgroundColor: ['#0369a1','#0ea5e9','#38bdf8','#7dd3fc','#0284c7','#0891b2','#22d3ee'] }]
          },
          options: { responsive: true }
        });
      }
      loadOverview();
    </script>
    """
    return _shell("overview", "Operations Overview", "Live KPIs and airport-wide flow health", body, script)


@app.get("/dashboard/zones", response_class=HTMLResponse)
def dashboard_zones() -> str:
    body = """
    <section class="grid">
      <article class="card span-8">
        <h3>Inflow vs Outflow by Zone</h3>
        <canvas id="flowChart"></canvas>
      </article>
      <article class="card span-4">
        <h3>Average Dwell Time by Zone</h3>
        <canvas id="dwellChart"></canvas>
      </article>
      <article class="card span-12">
        <h3>Latest Zone Table</h3>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr><th>Zone</th><th>Occupancy</th><th>Inflow/min</th><th>Outflow/min</th><th>Dwell(min)</th><th>Congestion</th></tr>
            </thead>
            <tbody id="zones-table"></tbody>
          </table>
        </div>
      </article>
    </section>
    """
    script = """
    <script>
      let flowChart, dwellChart;
      async function loadZones() {
        const zonesRes = await fetch('/zones/latest');
        const zonesData = await zonesRes.json();
        const zones = zonesData.zones || [];
        const labels = zones.map(z => z.zone);
        const inflow = zones.map(z => z.inflow_per_min);
        const outflow = zones.map(z => z.outflow_per_min);
        const dwell = zones.map(z => z.avg_dwell_minutes);

        if (flowChart) flowChart.destroy();
        flowChart = new Chart(document.getElementById('flowChart'), {
          type: 'bar',
          data: {
            labels,
            datasets: [
              { label: 'Inflow/min', data: inflow, backgroundColor: '#10b981', borderRadius: 5 },
              { label: 'Outflow/min', data: outflow, backgroundColor: '#f97316', borderRadius: 5 }
            ]
          },
          options: { responsive: true }
        });

        if (dwellChart) dwellChart.destroy();
        dwellChart = new Chart(document.getElementById('dwellChart'), {
          type: 'doughnut',
          data: {
            labels,
            datasets: [{
              label: 'Dwell min',
              data: dwell,
              backgroundColor: ['#106b9a','#239ccf','#42acdb','#6fb8dd','#1584bf','#1b94b2','#33b9cf'],
              borderColor: '#e2e8f0',
              borderWidth: 2
            }]
          },
          options: {
            responsive: true,
            cutout: '48%',
            plugins: {
              legend: {
                display: true,
                position: 'top',
                labels: { boxWidth: 18, padding: 14 }
              }
            }
          }
        });

        const tbody = document.getElementById('zones-table');
        tbody.innerHTML = zones.map(z => `
          <tr>
            <td>${z.zone}</td>
            <td>${z.occupancy}</td>
            <td>${z.inflow_per_min.toFixed(2)}</td>
            <td>${z.outflow_per_min.toFixed(2)}</td>
            <td>${z.avg_dwell_minutes.toFixed(2)}</td>
            <td>${(z.congestion_score * 100).toFixed(1)}%</td>
          </tr>
        `).join('');
      }
      loadZones();
    </script>
    """
    return _shell("zones", "Zone Analytics", "Flow, dwell and congestion patterns by operational area", body, script)


@app.get("/dashboard/predictions", response_class=HTMLResponse)
def dashboard_predictions() -> str:
    body = """
    <section class="grid">
      <article class="card span-8">
        <h3>Predicted Occupancy (30 min horizon)</h3>
        <canvas id="predictionChart"></canvas>
      </article>
      <article class="card span-4">
        <h3>Predicted Wait Time</h3>
        <canvas id="waitChart"></canvas>
      </article>
      <article class="card span-12">
        <h3>Risk Table</h3>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr><th>Zone</th><th>Horizon (min)</th><th>Predicted Occupancy</th><th>Predicted Wait</th><th>Risk</th></tr>
            </thead>
            <tbody id="pred-table"></tbody>
          </table>
        </div>
      </article>
    </section>
    """
    script = """
    <script>
      let predictionChart, waitChart;
      async function loadPredictions() {
        const predsRes = await fetch('/predictions?horizon_min=30');
        const predsData = await predsRes.json();
        const preds = predsData.predictions || [];
        const labels = preds.map(p => p.zone);
        const occupancy = preds.map(p => p.predicted_occupancy);
        const wait = preds.map(p => p.predicted_wait_minutes);

        if (predictionChart) predictionChart.destroy();
        predictionChart = new Chart(document.getElementById('predictionChart'), {
          type: 'bar',
          data: {
            labels,
            datasets: [{ label: 'Predicted occupancy', data: occupancy, backgroundColor: '#0284c7', borderRadius: 5 }]
          },
          options: { responsive: true, plugins: { legend: { display: false } } }
        });

        if (waitChart) waitChart.destroy();
        waitChart = new Chart(document.getElementById('waitChart'), {
          type: 'radar',
          data: {
            labels,
            datasets: [{ label: 'Wait (min)', data: wait, borderColor: '#dc2626', backgroundColor: 'rgba(248,113,113,0.30)' }]
          },
          options: { responsive: true }
        });

        const tbody = document.getElementById('pred-table');
        tbody.innerHTML = preds.map(p => `
          <tr>
            <td>${p.zone}</td>
            <td>${p.horizon_min}</td>
            <td>${p.predicted_occupancy}</td>
            <td>${p.predicted_wait_minutes.toFixed(2)}</td>
            <td><span class="badge ${p.risk_level}">${p.risk_level}</span></td>
          </tr>
        `).join('');
      }
      loadPredictions();
    </script>
    """
    return _shell("predictions", "Predictive Intelligence", "Near-term congestion and wait risk forecast", body, script)


@app.post("/refresh")
def refresh(minutes: int = 120, avg_events_per_minute: int = 40) -> dict[str, object]:
    with _lock:
        _rebuild(minutes=minutes, avg_events_per_minute=avg_events_per_minute)
        return {
            "message": "data refreshed",
            "minutes": minutes,
            "avg_events_per_minute": avg_events_per_minute,
            "updated_at": _state["updated_at"],
        }


@app.get("/kpis")
def get_kpis() -> dict[str, object]:
    with _lock:
        snapshots = _state["snapshots"]
        latest = latest_snapshot_map(snapshots)
        latest_list = list(latest.values())
        return {
            "updated_at": _state["updated_at"],
            "kpis": airport_kpis(latest_list),
            "zones_included": len(latest_list),
        }


@app.get("/zones/latest")
def get_latest_zone_states() -> dict[str, object]:
    with _lock:
        latest = latest_snapshot_map(_state["snapshots"])
        return {
            "updated_at": _state["updated_at"],
            "zones": [v.model_dump() for v in latest.values()],
        }


@app.get("/predictions")
def get_predictions(horizon_min: int = 30) -> dict[str, object]:
    with _lock:
        preds = forecast(_state["snapshots"], horizon_min=horizon_min)
        return {
            "updated_at": _state["updated_at"],
            "horizon_min": horizon_min,
            "predictions": [p.model_dump() for p in preds],
        }
