from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from .analytics import ZONE_CAPACITY, airport_kpis, compute_snapshots, latest_snapshot_map
from .fusion import events_to_dataframe
from .generators import generate_events
from .models import CameraCreate, CameraUpdate, FlowSnapshot, ZoneType
from .predictor import forecast
from . import camera_manager, video_processor, zones_config

app = FastAPI(
    title="HIA Flow360 API",
    description="Airport movement intelligence MVP in Python",
    version="0.1.0",
)

_lock = Lock()
_state: dict[str, object] = {
    "events": [],
    "snapshots": [],
    "updated_at": None,
    # Snapshots générés à partir des données caméras réelles
    "camera_snapshots": [],
    # Occupancy courante par zone (calculée à partir des flux caméra)
    "camera_zone_occ": {},
}

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

  /* ── Global mode toggle (sidebar) ─────────────── */
  .sidebar-footer {
    margin-top: auto;
    padding-top: 1.1rem;
    border-top: 1px solid #1f2d48;
    margin-top: 1.5rem;
  }
  .mode-section-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.45rem;
  }
  .mode-btns-sidebar {
    display: flex;
    gap: 0.3rem;
  }
  .sbtn {
    flex: 1;
    background: transparent;
    color: #94a3b8;
    border: 1px solid #2d3f5c;
    border-radius: 0.4rem;
    padding: 0.32rem 0.2rem;
    cursor: pointer;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .sbtn.active {
    background: rgba(14,165,233,0.18);
    color: #38bdf8;
    border-color: #0ea5e9;
  }

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
        <a href="/dashboard/predictions" {"class='active'" if active == "predictions" else ""}>Prédictions</a>
        <a href="/dashboard/map" {"class='active'" if active == "map" else ""}>Carte Interactive</a>
        <a href="/docs">API Docs</a>
      </nav>
      <div class="sidebar-footer">
        <div class="mode-section-label">Source de données</div>
        <div class="mode-btns-sidebar">
          <button class="sbtn" data-mode="simulated" onclick="setGlobalMode('simulated')">Simulé</button>
          <button class="sbtn" data-mode="real" onclick="setGlobalMode('real')">Réel</button>
        </div>
      </div>
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
              <button class="btn" id="btnRefresh" onclick="refreshData()">Actualiser</button>
            </div>
            {body}
          </main>
        </div>
        <script>
          /* ── Global data mode (localStorage) ── */
          function getGlobalMode() {{
            return localStorage.getItem('hia_data_mode') || 'simulated';
          }}
          function setGlobalMode(m) {{
            localStorage.setItem('hia_data_mode', m);
            _syncModeBtns(m);
            window.dispatchEvent(new CustomEvent('dataModeChange', {{ detail: {{ mode: m }} }}));
          }}
          function _syncModeBtns(m) {{
            document.querySelectorAll('.sbtn').forEach(b =>
              b.classList.toggle('active', b.dataset.mode === m));
          }}
          _syncModeBtns(getGlobalMode());

          async function refreshData() {{
            if (getGlobalMode() === 'simulated') {{
              await fetch('/refresh', {{ method: 'POST' }});
            }}
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


def _capture_camera_snapshot() -> None:
    """
    Échantillonne les compteurs inflow/outflow de chaque caméra active et
    construit un FlowSnapshot par zone pour alimenter le modèle prédictif.

    Logique d'occupancy :
      - On part de l'occupancy de la période précédente.
      - On l'ajuste avec (inflow - outflow) observés sur la caméra.
      - On ancre la valeur au nombre de personnes en mouvement visible
        (never less than what the camera actually sees).
    """
    cameras = camera_manager.list_cameras()
    zone_flow: dict[str, dict[str, int]] = {}

    for cam in cameras:
        zid = cam.get("zone", "gate")
        counts = video_processor.consume_flow_counts(cam["id"])
        if zid not in zone_flow:
            zone_flow[zid] = {"inflow": 0, "outflow": 0, "moving": 0}
        zone_flow[zid]["inflow"] += counts["inflow"]
        zone_flow[zid]["outflow"] += counts["outflow"]
        zone_flow[zid]["moving"] += counts["moving"]

    # N'enregistre un snapshot que si au moins une caméra voit des gens
    has_data = any(v["moving"] > 0 for v in zone_flow.values())
    if not has_data:
        return

    now = datetime.now(timezone.utc)
    new_snaps: list[FlowSnapshot] = []

    with _lock:
        cam_occ: dict[str, int] = dict(_state.get("camera_zone_occ", {}))  # type: ignore[arg-type]

        for zid, flow in zone_flow.items():
            try:
                zone_type = ZoneType(zid)
            except ValueError:
                continue

            inflow = flow["inflow"]
            outflow = flow["outflow"]
            moving = flow["moving"]

            # Mise à jour de l'occupancy courante
            prev_occ = cam_occ.get(zid, moving)
            net = inflow - outflow
            new_occ = max(moving, int(prev_occ + net))
            cam_occ[zid] = new_occ

            capacity = ZONE_CAPACITY.get(zid, 500)
            congestion = min(1.0, new_occ / max(1, capacity))
            dwell = round(
                min(new_occ / max(1.0, float(outflow)), 60.0), 2
            ) if outflow > 0 else max(1.0, round(new_occ / 35.0, 2))

            new_snaps.append(
                FlowSnapshot(
                    timestamp=now,
                    zone=zone_type,
                    occupancy=new_occ,
                    inflow_per_min=float(inflow),
                    outflow_per_min=float(outflow),
                    avg_dwell_minutes=dwell,
                    congestion_score=round(congestion, 3),
                )
            )

        _state["camera_zone_occ"] = cam_occ
        existing: list[FlowSnapshot] = _state.get("camera_snapshots", [])  # type: ignore[assignment]
        # Fenêtre glissante ~30 min (7 zones × 30 échantillons)
        _state["camera_snapshots"] = (existing + new_snaps)[-300:]


def _camera_snapshot_worker() -> None:
    """Thread de fond : capture un snapshot caméra toutes les 60 secondes."""
    time.sleep(30)  # délai initial pour laisser les caméras démarrer
    while True:
        try:
            _capture_camera_snapshot()
        except Exception:
            pass
        time.sleep(60)


@app.on_event("startup")
def _startup() -> None:
    with _lock:
        _rebuild()
    video_processor.start_all(camera_manager.list_cameras())
    t = threading.Thread(target=_camera_snapshot_worker, daemon=True, name="cam-snapshot")
    t.start()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    body = """
    <section class="grid">
      <article class="card kpi"><div class="label" id="kpi1lbl">Occupancy totale</div><div id="kpi1val" class="value">-</div></article>
      <article class="card kpi"><div class="label" id="kpi2lbl">Congestion moy.</div><div id="kpi2val" class="value">-</div></article>
      <article class="card kpi"><div class="label" id="kpi3lbl">Dwell moy. (min)</div><div id="kpi3val" class="value">-</div></article>
      <article class="card kpi"><div class="label" id="kpi4lbl">Congestion pic</div><div id="kpi4val" class="value">-</div></article>
      <article class="card span-8">
        <h3 id="chart1title">Occupancy par zone</h3>
        <canvas id="occupancyChart"></canvas>
      </article>
      <article class="card span-4">
        <h3 id="chart2title">Distribution</h3>
        <canvas id="congestionChart"></canvas>
      </article>
    </section>
    """
    script = """
    <script>
      let occupancyChart, congestionChart;
      const CHART_COLORS = ['#0369a1','#0ea5e9','#38bdf8','#7dd3fc','#0284c7','#0891b2','#22d3ee'];
      let _overviewBusy = false;

      async function loadOverview() {
        /* Guard : on ne lance pas un nouvel appel si le précédent est en cours */
        if (_overviewBusy) return;
        _overviewBusy = true;
        try {
          getGlobalMode() === 'simulated' ? await loadSimulated() : await loadReal();
        } catch(e) { /* réseau temporairement indisponible, on réessaie au prochain tour */ }
        finally { _overviewBusy = false; }
      }

      async function loadSimulated() {
        const [kr, zr] = await Promise.all([fetch('/kpis'), fetch('/zones/latest')]);
        const kd = await kr.json(); const zd = await zr.json();
        const k = kd.kpis;
        setKpis(
          'Occupancy totale', Math.round(k.total_occupancy),
          'Congestion moy.', (k.avg_congestion*100).toFixed(1)+'%',
          'Dwell moy. (min)', k.avg_dwell_minutes.toFixed(2),
          'Congestion pic', (k.max_zone_congestion*100).toFixed(1)+'%'
        );
        const zones = zd.zones || [];
        renderCharts(
          zones.map(z => z.zone), zones.map(z => z.occupancy),
          zones.map(z => +(z.congestion_score*100).toFixed(2)),
          'Occupancy par zone', 'Répartition congestion'
        );
      }

      async function loadReal() {
        const r = await fetch('/zones/camera-live');
        const d = await r.json();
        const zones = d.zones || [];
        const peak = zones.reduce((a,b) => a.occupancy>b.occupancy?a:b, {zone:'-',occupancy:0});
        setKpis(
          'Personnes en mvt.', d.total_persons || 0,
          'Caméras actives', (d.active_cameras||0)+' / '+(d.total_cameras||0),
          'Zone la + chargée', peak.zone || '-',
          'Caméras configurées', d.total_cameras || 0
        );
        renderCharts(
          zones.map(z => z.zone), zones.map(z => z.occupancy), zones.map(z => z.occupancy),
          'Personnes en mvt. par zone', 'Répartition par zone'
        );
      }

      function setKpis(l1,v1,l2,v2,l3,v3,l4,v4) {
        ['kpi1lbl','kpi2lbl','kpi3lbl','kpi4lbl'].forEach((id,i)=>
          document.getElementById(id).textContent=[l1,l2,l3,l4][i]);
        ['kpi1val','kpi2val','kpi3val','kpi4val'].forEach((id,i)=>
          document.getElementById(id).textContent=[v1,v2,v3,v4][i]);
      }

      function renderCharts(labels, occupancy, dist, title1, title2) {
        document.getElementById('chart1title').textContent = title1;
        document.getElementById('chart2title').textContent = title2;
        /* Mise à jour des données sans destroy/recreate → pas de fuite mémoire */
        if (occupancyChart) {
          occupancyChart.data.labels = labels;
          occupancyChart.data.datasets[0].data = occupancy;
          occupancyChart.update('none');
        } else {
          occupancyChart = new Chart(document.getElementById('occupancyChart'), {
            type: 'bar',
            data: { labels, datasets: [{ label: 'Personnes', data: occupancy, backgroundColor: '#0ea5e9', borderRadius: 6 }] },
            options: { responsive: true, animation: false, plugins: { legend: { display: false } } }
          });
        }
        if (congestionChart) {
          congestionChart.data.labels = labels;
          congestionChart.data.datasets[0].data = dist;
          congestionChart.update('none');
        } else {
          congestionChart = new Chart(document.getElementById('congestionChart'), {
            type: 'doughnut',
            data: { labels, datasets: [{ data: dist, backgroundColor: CHART_COLORS }] },
            options: { responsive: true, animation: false }
          });
        }
      }

      window.addEventListener('dataModeChange', () => {
        /* Réinitialise les charts lors d'un changement de mode */
        if (occupancyChart) { occupancyChart.destroy(); occupancyChart = null; }
        if (congestionChart) { congestionChart.destroy(); congestionChart = null; }
        loadOverview();
      });
      loadOverview();
      /* setTimeout récursif : l'intervalle démarre APRÈS la fin du fetch précédent */
      function scheduleOverview() { setTimeout(async () => { await loadOverview(); scheduleOverview(); }, 10000); }
      scheduleOverview();
    </script>
    """
    return _shell("overview", "Operations Overview", "Live KPIs — source de données dans la barre latérale", body, script)


@app.get("/dashboard/zones", response_class=HTMLResponse)
def dashboard_zones() -> str:
    body = """
    <section class="grid">
      <article class="card span-8">
        <h3 id="flowTitle">Inflow vs Outflow par zone</h3>
        <canvas id="flowChart"></canvas>
      </article>
      <article class="card span-4">
        <h3 id="dwellTitle">Temps de séjour moyen par zone</h3>
        <canvas id="dwellChart"></canvas>
      </article>
      <article class="card span-12">
        <h3>Tableau des zones</h3>
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
      let _lastFlowMode = null;
      const DWELL_COLORS = ['#106b9a','#239ccf','#42acdb','#6fb8dd','#1584bf','#1b94b2','#33b9cf'];

      function _updateOrCreate(chartRef, canvasId, type, data, options) {
        if (chartRef && chartRef.config.type === type) {
          chartRef.data.labels = data.labels;
          data.datasets.forEach((ds, i) => {
            if (chartRef.data.datasets[i]) {
              chartRef.data.datasets[i].data = ds.data;
            }
          });
          chartRef.update('none');
          return chartRef;
        }
        if (chartRef) chartRef.destroy();
        return new Chart(document.getElementById(canvasId), { type, data, options });
      }

      async function loadZones() {
        getGlobalMode() === 'simulated' ? await loadZonesSim() : await loadZonesReal();
      }

      async function loadZonesSim() {
        document.getElementById('flowTitle').textContent = 'Inflow vs Outflow par zone';
        document.getElementById('dwellTitle').textContent = 'Temps de séjour moyen par zone';
        const r = await fetch('/zones/latest');
        const d = await r.json();
        const zones = d.zones || [];
        const labels = zones.map(z => z.zone);
        const flowOpts = { responsive: true, animation: false };
        const dwellOpts = { responsive: true, animation: false, cutout: '48%',
          plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 18, padding: 14 } } } };
        if (_lastFlowMode !== 'sim') { if (flowChart) { flowChart.destroy(); flowChart = null; }
                                       if (dwellChart) { dwellChart.destroy(); dwellChart = null; } }
        _lastFlowMode = 'sim';
        flowChart = _updateOrCreate(flowChart, 'flowChart', 'bar', { labels, datasets: [
          { label: 'Inflow/min', data: zones.map(z => z.inflow_per_min), backgroundColor: '#10b981', borderRadius: 5 },
          { label: 'Outflow/min', data: zones.map(z => z.outflow_per_min), backgroundColor: '#f97316', borderRadius: 5 }
        ]}, flowOpts);
        dwellChart = _updateOrCreate(dwellChart, 'dwellChart', 'doughnut', { labels, datasets: [{
          label: 'Dwell min', data: zones.map(z => z.avg_dwell_minutes),
          backgroundColor: DWELL_COLORS, borderColor: '#e2e8f0', borderWidth: 2
        }]}, dwellOpts);
        document.getElementById('zones-table').innerHTML = zones.map(z => `
          <tr><td>${z.zone}</td><td>${z.occupancy}</td>
              <td>${z.inflow_per_min.toFixed(2)}</td><td>${z.outflow_per_min.toFixed(2)}</td>
              <td>${z.avg_dwell_minutes.toFixed(2)}</td><td>${(z.congestion_score*100).toFixed(1)}%</td></tr>
        `).join('');
      }

      async function loadZonesReal() {
        document.getElementById('flowTitle').textContent = 'Inflow vs Outflow par zone (caméras)';
        document.getElementById('dwellTitle').textContent = 'Dwell estimé par zone (caméras)';
        const r = await fetch('/zones/camera-live');
        const d = await r.json();
        const zones = d.zones || [];
        const labels = zones.map(z => z.zone);
        const barOpts = { responsive: true, animation: false, plugins: { legend: { display: true } },
                          scales: { x: { stacked: false }, y: { beginAtZero: true } } };
        const dwellOpts = { responsive: true, animation: false, plugins: { legend: { display: false } },
                            scales: { y: { beginAtZero: true, title: { display: true, text: 'min' } } } };
        if (_lastFlowMode !== 'real') { if (flowChart) { flowChart.destroy(); flowChart = null; }
                                        if (dwellChart) { dwellChart.destroy(); dwellChart = null; } }
        _lastFlowMode = 'real';
        flowChart = _updateOrCreate(flowChart, 'flowChart', 'bar', { labels, datasets: [
          { label: 'Inflow', data: zones.map(z => z.inflow_per_min || 0), backgroundColor: '#10b981', borderRadius: 5 },
          { label: 'Outflow', data: zones.map(z => z.outflow_per_min || 0), backgroundColor: '#f97316', borderRadius: 5 }
        ]}, barOpts);
        dwellChart = _updateOrCreate(dwellChart, 'dwellChart', 'bar', { labels, datasets: [{
          label: 'Dwell (min)', data: zones.map(z => z.avg_dwell_minutes || 0),
          backgroundColor: DWELL_COLORS.slice(0, labels.length), borderRadius: 5
        }]}, dwellOpts);
        document.getElementById('zones-table').innerHTML = zones.map(z => `
          <tr>
            <td>${z.zone}</td><td>${z.occupancy}</td>
            <td>${(z.inflow_per_min || 0).toFixed(1)}</td>
            <td>${(z.outflow_per_min || 0).toFixed(1)}</td>
            <td>${(z.avg_dwell_minutes || 0).toFixed(2)}</td>
            <td>${((z.congestion_score || 0) * 100).toFixed(1)}%</td>
          </tr>
        `).join('');
      }

      window.addEventListener('dataModeChange', () => loadZones());
      loadZones();
    </script>
    """
    return _shell("zones", "Zone Analytics", "Flow et congestion — source de données dans la barre latérale", body, script)


@app.get("/dashboard/predictions", response_class=HTMLResponse)
def dashboard_predictions() -> str:
    body = """
    <div id="realBanner" style="display:none;background:#0f172a;color:#94a3b8;border-radius:0.6rem;
         padding:0.55rem 0.9rem;font-size:0.84rem;margin-bottom:0.85rem;"></div>
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
      async function loadRealBanner() {
        const [cr, pr] = await Promise.all([
          fetch('/zones/camera-live'),
          fetch('/predictions?horizon_min=30'),
        ]);
        const d = await cr.json();
        const pd = await pr.json();
        const b = document.getElementById('realBanner');
        const src = pd.source || d.prediction_source || 'simulation';
        const srcLabel = src === 'cameras'
          ? '<span style="color:#4ade80">&#9679; Prédictions basées sur les caméras réelles</span>'
          : '<span style="color:#f59e0b">&#9679; Prédictions basées sur la simulation (caméras en cours de chauffe)</span>';
        b.innerHTML =
          '<strong>Caméras actives&nbsp;: ' + (d.active_cameras||0) + ' / ' + (d.total_cameras||0) +
          '</strong> &nbsp;&bull;&nbsp; Personnes en mvt.&nbsp;: <strong>' + (d.total_persons||0) +
          '</strong> &nbsp;&bull;&nbsp; ' + srcLabel;
        b.style.display = '';
      }

      async function init() {
        const banner = document.getElementById('realBanner');
        if (getGlobalMode() === 'real') { await loadRealBanner(); } else { banner.style.display = 'none'; }
        loadPredictions();
      }

      window.addEventListener('dataModeChange', e => {
        const banner = document.getElementById('realBanner');
        if (e.detail.mode === 'real') loadRealBanner(); else banner.style.display = 'none';
      });
      init();
    </script>
    """
    return _shell("predictions", "Predictive Intelligence", "Prévisions de congestion — source de données dans la barre latérale", body, script)


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
        cam_snaps: list[FlowSnapshot] = _state.get("camera_snapshots", [])  # type: ignore[assignment]
        # Utilise les données caméra réelles si on a suffisamment d'historique
        # (>= 3 snapshots = au moins 3 zones ou 3 périodes de capture)
        if len(cam_snaps) >= 3:
            preds = forecast(cam_snaps, horizon_min=horizon_min)
            source = "cameras"
        else:
            preds = forecast(_state["snapshots"], horizon_min=horizon_min)
            source = "simulation"
        return {
            "updated_at": _state["updated_at"],
            "horizon_min": horizon_min,
            "source": source,
            "predictions": [p.model_dump() for p in preds],
        }


# ---------------------------------------------------------------------------
# Camera CRUD endpoints
# ---------------------------------------------------------------------------

@app.get("/cameras")
def list_cameras_api() -> dict[str, object]:
    cams = camera_manager.list_cameras()
    for c in cams:
        c["person_count"] = video_processor.get_count(c["id"])
    return {"cameras": cams}


@app.get("/cameras/usb-devices")
def list_usb_devices_api() -> dict[str, object]:
    return {"devices": video_processor.list_usb_cameras()}


@app.post("/cameras", status_code=201)
def create_camera_api(body: CameraCreate) -> dict[str, object]:
    cam = camera_manager.create_camera(
        name=body.name,
        x=body.x,
        y=body.y,
        angle=body.angle,
        zone=body.zone,
        source_type=body.source_type,
        usb_index=body.usb_index,
    )
    video_processor.start_processor(
        cam["id"], cam["folder"], cam.get("angle", 0.0),
        cam.get("source_type", "file"), cam.get("usb_index"),
    )
    return cam


@app.put("/cameras/{camera_id}")
def update_camera_api(camera_id: str, body: CameraUpdate) -> dict[str, object]:
    cam = camera_manager.update_camera(
        camera_id,
        name=body.name,
        x=body.x,
        y=body.y,
        angle=body.angle,
        zone=body.zone,
        source_type=body.source_type,
        usb_index=body.usb_index,
    )
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    video_processor.update_processor(
        camera_id,
        angle=body.angle,
        source_type=body.source_type,
        usb_index=body.usb_index,
        folder=cam.get("folder"),
    )
    return cam


@app.delete("/cameras/{camera_id}")
def delete_camera_api(camera_id: str) -> dict[str, object]:
    video_processor.stop_processor(camera_id)
    ok = camera_manager.delete_camera(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"deleted": camera_id}


@app.get("/cameras/{camera_id}/stream")
def stream_camera(camera_id: str):
    cam = camera_manager.get_camera(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        video_processor.frame_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            # Empêche tout cache ou buffering côté navigateur/proxy
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/cameras/{camera_id}/count")
def camera_count(camera_id: str) -> dict[str, object]:
    return {"camera_id": camera_id, "person_count": video_processor.get_count(camera_id)}


# ---------------------------------------------------------------------------
# Interactive map dashboard
# ---------------------------------------------------------------------------

_MAP_STYLE = """
<style>
  /* ── Toolbar ─────────────────────────────────────────────── */
  .map-toolbar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.85rem;
  }
  .btn-add  { background: #10b981; }
  .btn-edit { background: #8b5cf6; }
  .btn-add.active, .btn-edit.active { background: #f59e0b; }
  .btn-reset { background: #64748b; font-size: 0.82rem; padding: 0.38rem 0.65rem; }
  .toolbar-sep { width: 1px; height: 24px; background: #dbe2ea; margin: 0 0.2rem; }
  .hint-text { font-size: 0.82rem; color: #f59e0b; font-weight: 600; display: none; }

  .legend {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-left: auto;
  }
  .legend-item { display: flex; align-items: center; gap: 0.3rem; font-size: 0.76rem; color: #475569; }
  .legend-dot  { width: 11px; height: 11px; border-radius: 3px; flex-shrink: 0; }

  /* ── Map layout ──────────────────────────────────────────── */
  .map-area { display: flex; gap: 1rem; align-items: flex-start; }
  .canvas-wrap {
    flex: 1; border: 1px solid #dbe2ea; border-radius: 0.85rem;
    overflow: hidden; background: #e0e5ea; cursor: default;
  }
  #airportMap { display: block; width: 100%; height: auto; }

  /* ── Side panels ─────────────────────────────────────────── */
  .side-panel {
    width: 230px; flex-shrink: 0;
    background: #fff; border: 1px solid #dbe2ea;
    border-radius: 0.85rem; padding: 1rem; display: none;
  }
  .side-panel.visible { display: block; }
  .side-panel h3 { margin: 0 0 0.6rem; font-size: 0.97rem; }
  .info-row { font-size: 0.83rem; color: #475569; margin-bottom: 0.35rem; }
  .info-row strong { color: #0f172a; }
  .panel-btns { display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.75rem; }
  .panel-btns button { width: 100%; }
  .btn-danger    { background: #ef4444; }
  .btn-secondary { background: #475569; }
  .count-big { font-size: 1.5rem; font-weight: 700; color: #0ea5e9; }
  .form-group { margin-bottom: 0.7rem; }
  .form-group label { display: block; font-size: 0.82rem; color: #475569; margin-bottom: 0.25rem; }
  .form-group input, .form-group select {
    width: 100%; padding: 0.4rem 0.55rem; border: 1px solid #cbd5e1;
    border-radius: 0.45rem; font-size: 0.9rem;
  }
  .form-group input[type=range]  { padding: 0; }
  .form-group input[type=color]  { padding: 2px; height: 34px; cursor: pointer; }

  /* ── Modals ──────────────────────────────────────────────── */
  .modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.45); z-index: 1000;
    align-items: center; justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal-box {
    background: #fff; border-radius: 1rem; padding: 1.5rem;
    width: 360px; max-width: 95vw;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
  }
  .modal-box.large { width: 740px; }
  .modal-box h3 { margin: 0 0 1rem; }
  .modal-footer { display: flex; gap: 0.5rem; justify-content: flex-end; margin-top: 1rem; }
  .video-wrap {
    background: #000; border-radius: 0.6rem; overflow: hidden;
    text-align: center; min-height: 300px;
  }
  .video-wrap img { max-width: 100%; height: auto; }
  .count-badge {
    display: inline-block; background: #0ea5e9; color: #fff;
    border-radius: 999px; padding: 0.2rem 0.75rem;
    font-weight: 700; font-size: 0.9rem; margin-top: 0.5rem;
  }

  /* ── Croix directionnelle (D-pad) ────────────────────────── */
  .dpad {
    display: grid;
    grid-template-columns: repeat(3, 38px);
    grid-template-rows: repeat(3, 38px);
    gap: 3px;
    margin: 0.35rem auto;
    width: fit-content;
  }
  .dpad-btn {
    width: 38px; height: 38px;
    border: 1.5px solid #cbd5e1; border-radius: 6px;
    background: #f1f5f9; cursor: pointer; font-size: 1.1rem; padding: 0;
    transition: background 0.12s, border-color 0.12s;
  }
  .dpad-btn:hover { background: #dbeafe; border-color: #93c5fd; }
  .dpad-btn.selected { background: #0ea5e9; color: #fff; border-color: #0369a1; }
  .dpad-center { width: 38px; height: 38px; }
  .dpad-info { font-size: 0.78rem; color: #64748b; text-align: center; margin-top: 0.2rem; }

  /* ── Source vidéo ────────────────────────────────────────── */
  .source-toggle { display: flex; gap: 1rem; }
  .source-toggle label {
    display: flex; align-items: center; gap: 0.3rem;
    font-size: 0.88rem; cursor: pointer;
  }
  .btn-refresh {
    font-size: 0.74rem; padding: 1px 6px;
    background: #64748b; color: #fff;
    border: none; border-radius: 4px; cursor: pointer; margin-left: 4px;
  }
  .btn-refresh:hover { background: #475569; }
</style>
"""

_MAP_BODY = """
<div class="map-toolbar">
  <button class="btn btn-add" id="btnAddCam" onclick="toggleMode('addCam')">+ Caméra</button>
  <button class="btn btn-edit" id="btnEditZones" onclick="toggleMode('editZones')">✏ Zones</button>
  <div class="toolbar-sep"></div>
  <button class="btn btn-reset" onclick="confirmResetZones()">Réinitialiser carte</button>
  <span id="hintText" class="hint-text"></span>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#3B82F6"></div>Enregistrement</div>
    <div class="legend-item"><div class="legend-dot" style="background:#EF4444"></div>Sécurité</div>
    <div class="legend-item"><div class="legend-dot" style="background:#8B5CF6"></div>Douane</div>
    <div class="legend-item"><div class="legend-dot" style="background:#06B6D4"></div>Transit</div>
    <div class="legend-item"><div class="legend-dot" style="background:#10B981"></div>Boutiques</div>
    <div class="legend-item"><div class="legend-dot" style="background:#F59E0B"></div>Embarquement</div>
    <div class="legend-item"><div class="legend-dot" style="background:#F97316"></div>Bagages</div>
  </div>
</div>
<div class="map-area">
  <div class="canvas-wrap"><canvas id="airportMap" width="1180" height="640"></canvas></div>

  <!-- Camera info panel -->
  <div id="camPanel" class="side-panel">
    <h3 id="cpName">Caméra</h3>
    <div class="info-row">Zone&nbsp;: <strong id="cpZone"></strong></div>
    <div class="info-row">Inflo&nbsp;: <strong id="cpInflow"></strong></div>
    <div class="info-row">Source&nbsp;: <strong id="cpSource"></strong></div>
    <div class="info-row">En mouvement&nbsp;: <span class="count-big" id="cpCount">0</span></div>
    <div class="panel-btns">
      <button class="btn" onclick="openVideoModal()">Voir le flux</button>
      <button class="btn btn-secondary" onclick="openCamEditModal()">Modifier</button>
      <button class="btn btn-danger" onclick="confirmDeleteCam()">Supprimer</button>
    </div>
  </div>

  <!-- Zone edit panel -->
  <div id="zonePanel" class="side-panel">
    <h3>Zone sélectionnée</h3>
    <div class="form-group">
      <label>Nom</label>
      <input type="text" id="zpLabel" oninput="onZoneLabelChange()" />
    </div>
    <div class="form-group">
      <label>Type</label>
      <select id="zpType" onchange="onZoneTypeChange()">
        <option value="checkin">Enregistrement</option>
        <option value="security">Sécurité</option>
        <option value="immigration">Douane / Immigration</option>
        <option value="transfer">Transit</option>
        <option value="retail">Boutiques</option>
        <option value="gate">Embarquement</option>
        <option value="baggage">Bagages</option>
        <option value="custom">Personnalisé</option>
      </select>
    </div>
    <div class="form-group">
      <label>Couleur</label>
      <input type="color" id="zpColor" oninput="onZoneColorChange()" />
    </div>
    <div class="panel-btns">
      <button class="btn btn-danger" onclick="confirmDeleteZone()">Supprimer ce bloc</button>
    </div>
  </div>
</div>

<!-- Camera add / edit modal -->
<div class="modal-overlay" id="camModal">
  <div class="modal-box" style="width:410px">
    <h3 id="camModalTitle">Ajouter une caméra</h3>
    <div class="form-group">
      <label>Nom</label>
      <input type="text" id="cmName" placeholder="ex: Cam Sécurité Nord" />
    </div>
    <div class="form-group">
      <label>Zone</label>
      <select id="cmZone">
        <option value="checkin">Enregistrement</option>
        <option value="security">Sécurité</option>
        <option value="immigration">Douane / Immigration</option>
        <option value="transfer">Transit</option>
        <option value="retail">Boutiques</option>
        <option value="gate">Embarquement</option>
        <option value="baggage">Bagages</option>
      </select>
    </div>
    <div class="form-group">
      <label>Sens de l'inflo</label>
      <div class="dpad" id="cmDpad">
        <button type="button" class="dpad-btn" onclick="setDpadDir('nw')" id="dpad-nw" title="Nord-Ouest">↖</button>
        <button type="button" class="dpad-btn" onclick="setDpadDir('n')"  id="dpad-n"  title="Nord">↑</button>
        <button type="button" class="dpad-btn" onclick="setDpadDir('ne')" id="dpad-ne" title="Nord-Est">↗</button>
        <button type="button" class="dpad-btn" onclick="setDpadDir('w')"  id="dpad-w"  title="Ouest">←</button>
        <div class="dpad-center"></div>
        <button type="button" class="dpad-btn" onclick="setDpadDir('e')"  id="dpad-e"  title="Est">→</button>
        <button type="button" class="dpad-btn" onclick="setDpadDir('sw')" id="dpad-sw" title="Sud-Ouest">↙</button>
        <button type="button" class="dpad-btn" onclick="setDpadDir('s')"  id="dpad-s"  title="Sud">↓</button>
        <button type="button" class="dpad-btn" onclick="setDpadDir('se')" id="dpad-se" title="Sud-Est">↘</button>
      </div>
      <div class="dpad-info">
        Inflo&nbsp;: <strong id="cmInflowLbl">→ Est</strong>
        &nbsp;|&nbsp;
        Outflow&nbsp;: <strong id="cmOutflowLbl">← Ouest</strong>
      </div>
    </div>
    <div class="form-group">
      <label>Source vidéo</label>
      <div class="source-toggle">
        <label><input type="radio" name="cmSource" value="file" checked onchange="onSourceChange()"> Fichiers</label>
        <label><input type="radio" name="cmSource" value="usb" onchange="onSourceChange()"> Caméra USB</label>
      </div>
    </div>
    <div class="form-group" id="cmUsbGroup" style="display:none">
      <label>Périphérique <button type="button" class="btn-refresh" onclick="refreshUsbDevices()">↺ Actualiser</button></label>
      <select id="cmUsbIndex"><option value="">Recherche…</option></select>
    </div>
    <div class="modal-footer">
      <button class="btn btn-secondary" onclick="closeCamModal()">Annuler</button>
      <button class="btn" onclick="saveCam()">Enregistrer</button>
    </div>
  </div>
</div>

<!-- New zone config modal -->
<div class="modal-overlay" id="newZoneModal">
  <div class="modal-box">
    <h3>Nouveau bloc de zone</h3>
    <div class="form-group"><label>Nom</label>
      <input type="text" id="nzLabel" placeholder="ex: Salle VIP" /></div>
    <div class="form-group"><label>Type</label>
      <select id="nzType" onchange="onNewZoneTypeChange()">
        <option value="gate">Embarquement</option>
        <option value="checkin">Enregistrement</option>
        <option value="security">Sécurité</option>
        <option value="immigration">Douane</option>
        <option value="transfer">Transit</option>
        <option value="retail">Boutiques</option>
        <option value="baggage">Bagages</option>
        <option value="custom">Personnalisé</option>
      </select></div>
    <div class="form-group"><label>Couleur</label>
      <input type="color" id="nzColor" value="#F59E0B" /></div>
    <div class="modal-footer">
      <button class="btn btn-secondary" onclick="closeNewZoneModal()">Annuler</button>
      <button class="btn" onclick="confirmNewZone()">Créer</button>
    </div>
  </div>
</div>

<!-- Video feed modal -->
<div class="modal-overlay" id="videoModal">
  <div class="modal-box large">
    <h3>Flux caméra&nbsp;: <span id="videoTitle"></span></h3>
    <div class="video-wrap"><img id="videoStream" src="" alt="Chargement…" /></div>
    <div style="text-align:center;margin-top:0.5rem">
      <span class="count-badge">En mouvement&nbsp;: <span id="vCount">0</span></span>
    </div>
    <div class="modal-footer">
      <button class="btn btn-secondary" onclick="closeVideoModal()">Fermer</button>
    </div>
  </div>
</div>
"""

_MAP_SCRIPT = """
<script>
const MAP_W = 1180, MAP_H = 640, HS = 9;

const ZONE_TYPES = {
  checkin:     {label:'Enregistrement', color:'#3B82F6'},
  security:    {label:'Sécurité',       color:'#EF4444'},
  immigration: {label:'Douane',         color:'#8B5CF6'},
  transfer:    {label:'Transit',        color:'#06B6D4'},
  retail:      {label:'Boutiques',      color:'#10B981'},
  gate:        {label:'Embarquement',   color:'#F59E0B'},
  baggage:     {label:'Bagages',        color:'#F97316'},
  custom:      {label:'Personnalisé',   color:'#94A3B8'},
};

// ── State ──────────────────────────────────────────────────────────────────────
let zones = [], cameras = [];
let currentMode = 'normal';
let selectedCam = null, selectedZone = null;
let dragState = null;
let pendingCamPos = null, pendingNewZone = null;
let videoCountInterval = null;

const canvas = document.getElementById('airportMap');
const ctx = canvas.getContext('2d');

// ── Drawing ────────────────────────────────────────────────────────────────────
function drawMap() {
  ctx.clearRect(0, 0, MAP_W, MAP_H);
  ctx.fillStyle = '#d1d9e0'; ctx.fillRect(0, 0, MAP_W, MAP_H);

  zones.forEach(z => {
    const sel = currentMode === 'editZones' && selectedZone && selectedZone.id === z.id;
    ctx.fillStyle = z.color + 'bb';
    ctx.beginPath(); ctx.roundRect(z.x, z.y, z.w, z.h, 7); ctx.fill();
    ctx.strokeStyle = sel ? '#0ea5e9' : z.color;
    ctx.lineWidth = sel ? 3 : 2;
    ctx.setLineDash(sel ? [6,3] : []);
    ctx.beginPath(); ctx.roundRect(z.x, z.y, z.w, z.h, 7); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#fff'; ctx.font = 'bold 12px Segoe UI'; ctx.textAlign = 'center';
    wrapText(z.label, z.x + z.w/2, z.y + z.h/2, z.w - 14, 17);
  });

  if (currentMode === 'editZones' && selectedZone) {
    getHandles(selectedZone).forEach(h => {
      ctx.fillStyle = '#fff'; ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 1.5;
      ctx.fillRect(h.x-HS/2, h.y-HS/2, HS, HS);
      ctx.strokeRect(h.x-HS/2, h.y-HS/2, HS, HS);
    });
  }

  if (dragState && dragState.type === 'draw') {
    const {startX:sx, startY:sy, currX:cx, currY:cy} = dragState;
    const [rx,ry,rw,rh] = [Math.min(sx,cx),Math.min(sy,cy),Math.abs(cx-sx),Math.abs(cy-sy)];
    ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 2; ctx.setLineDash([6,3]);
    ctx.strokeRect(rx,ry,rw,rh); ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(14,165,233,0.08)'; ctx.fillRect(rx,ry,rw,rh);
  }

  if (currentMode !== 'editZones') cameras.forEach(drawCamera);
}

function wrapText(text, cx, cy, maxW, lh) {
  const words = text.split(' '), lines = [];
  let line = '';
  words.forEach(w => {
    const t = line ? line+' '+w : w;
    if (ctx.measureText(t).width > maxW && line) { lines.push(line); line = w; }
    else line = t;
  });
  if (line) lines.push(line);
  const sy = cy - (lines.length-1)*lh/2;
  lines.forEach((l,i) => ctx.fillText(l, cx, sy+i*lh));
}

function drawCamera(cam) {
  const sel = selectedCam && selectedCam.id === cam.id;
  ctx.save(); ctx.translate(cam.x, cam.y); ctx.rotate(cam.angle * Math.PI/180);
  ctx.beginPath(); ctx.moveTo(10,0); ctx.lineTo(56,-23); ctx.lineTo(56,23); ctx.closePath();
  ctx.fillStyle = sel ? 'rgba(251,191,36,0.35)' : 'rgba(14,165,233,0.28)'; ctx.fill();
  ctx.fillStyle = sel ? '#fbbf24' : '#1e293b';
  ctx.beginPath(); ctx.roundRect(-13,-8,23,16,4); ctx.fill();
  ctx.fillStyle = sel ? '#f59e0b' : '#38bdf8';
  ctx.beginPath(); ctx.arc(12,0,6,0,Math.PI*2); ctx.fill();
  ctx.restore();
  ctx.font = 'bold 11px Segoe UI'; ctx.textAlign = 'center';
  ctx.fillStyle = '#0f172a'; ctx.fillText(cam.name, cam.x, cam.y+24);
  ctx.fillStyle = '#0ea5e9'; ctx.fillText('👥 '+(cam.person_count||0), cam.x, cam.y+37);
}

// ── Hit testing ────────────────────────────────────────────────────────────────
function zoneAt(x, y) {
  for (let i=zones.length-1; i>=0; i--) {
    const z=zones[i];
    if (x>=z.x && x<=z.x+z.w && y>=z.y && y<=z.y+z.h) return z;
  }
  return null;
}
function camAt(x, y) { return cameras.find(c => Math.hypot(x-c.x, y-c.y) < 18) || null; }
function getHandles(z) {
  return [
    {id:'nw',x:z.x,       y:z.y      }, {id:'n', x:z.x+z.w/2, y:z.y      },
    {id:'ne',x:z.x+z.w,   y:z.y      }, {id:'e', x:z.x+z.w,   y:z.y+z.h/2},
    {id:'se',x:z.x+z.w,   y:z.y+z.h  }, {id:'s', x:z.x+z.w/2, y:z.y+z.h  },
    {id:'sw',x:z.x,        y:z.y+z.h  }, {id:'w', x:z.x,        y:z.y+z.h/2},
  ];
}
function handleAt(x, y, z) {
  if (!z) return null;
  return getHandles(z).find(h => Math.abs(x-h.x)<=HS/2+2 && Math.abs(y-h.y)<=HS/2+2) || null;
}
function applyResize(z, id, dx, dy) {
  const n={...z};
  if(id==='nw'){n.x+=dx;n.y+=dy;n.w-=dx;n.h-=dy;}
  else if(id==='n'){n.y+=dy;n.h-=dy;}
  else if(id==='ne'){n.w+=dx;n.y+=dy;n.h-=dy;}
  else if(id==='e'){n.w+=dx;}
  else if(id==='se'){n.w+=dx;n.h+=dy;}
  else if(id==='s'){n.h+=dy;}
  else if(id==='sw'){n.x+=dx;n.w-=dx;n.h+=dy;}
  else if(id==='w'){n.x+=dx;n.w-=dx;}
  n.w=Math.max(n.w,50); n.h=Math.max(n.h,35);
  return n;
}

// ── Coords ─────────────────────────────────────────────────────────────────────
function coords(e) {
  const r = canvas.getBoundingClientRect();
  return {x:(e.clientX-r.left)*(MAP_W/r.width), y:(e.clientY-r.top)*(MAP_H/r.height)};
}

// ── Mouse events ───────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  const {x,y} = coords(e);
  if (currentMode === 'addCam') {
    pendingCamPos = {x,y};
    const z = zoneAt(x,y);
    if (z) document.getElementById('cmZone').value = z.type || z.id;
    document.getElementById('cmName').value = '';
    setDpadDir('e');
    document.querySelectorAll('input[name="cmSource"]').forEach(r => { r.checked = r.value === 'file'; });
    document.getElementById('cmUsbGroup').style.display = 'none';
    document.getElementById('camModalTitle').textContent = 'Ajouter une caméra';
    openModal('camModal'); setMode('normal'); return;
  }
  if (currentMode === 'editZones') {
    const h = handleAt(x,y,selectedZone);
    if (h && selectedZone) { dragState={type:'resize',handleId:h.id,startX:x,startY:y,origZone:{...selectedZone}}; return; }
    const z = zoneAt(x,y);
    if (z) { selectedZone=z; showZonePanel(z); dragState={type:'move',startX:x,startY:y,origX:z.x,origY:z.y}; drawMap(); return; }
    selectedZone=null; hideZonePanel();
    dragState={type:'draw',startX:x,startY:y,currX:x,currY:y}; drawMap(); return;
  }
  // Normal mode: select camera + start drag
  const c = camAt(x,y); selectedCam=c;
  if (c) {
    showCamPanel(c);
    dragState = { type:'moveCam', cam:c, startX:x, startY:y, origX:c.x, origY:c.y, moved:false };
  } else hideCamPanel();
  drawMap();
});

canvas.addEventListener('mousemove', e => {
  const {x,y} = coords(e);
  if (dragState) {
    const dx=x-dragState.startX, dy=y-dragState.startY;
    if (dragState.type==='move' && selectedZone) {
      const i=zones.findIndex(z=>z.id===selectedZone.id);
      if(i>=0){zones[i].x=dragState.origX+dx; zones[i].y=dragState.origY+dy; selectedZone=zones[i];}
    } else if (dragState.type==='resize' && selectedZone) {
      const i=zones.findIndex(z=>z.id===selectedZone.id);
      if(i>=0){zones[i]=applyResize(dragState.origZone,dragState.handleId,dx,dy); selectedZone=zones[i];}
    } else if (dragState.type==='draw') {
      dragState.currX=x; dragState.currY=y;
    } else if (dragState.type==='moveCam') {
      const i=cameras.findIndex(c=>c.id===dragState.cam.id);
      if(i>=0){ cameras[i].x=dragState.origX+dx; cameras[i].y=dragState.origY+dy; }
      dragState.moved = Math.hypot(dx,dy) > 4;
    }
    drawMap();
  }
  updateCursor(x,y);
});

canvas.addEventListener('mouseup', e => {
  if (!dragState) return;
  const {x,y} = coords(e);
  if (dragState.type==='move'||dragState.type==='resize') {
    autoSaveZones();
  } else if (dragState.type==='draw') {
    const rx=Math.min(dragState.startX,x), ry=Math.min(dragState.startY,y);
    const rw=Math.abs(x-dragState.startX), rh=Math.abs(y-dragState.startY);
    if(rw>40&&rh>30){pendingNewZone={x:rx,y:ry,w:rw,h:rh}; onNewZoneTypeChange(); openModal('newZoneModal');}
  } else if (dragState.type==='moveCam' && dragState.moved) {
    const cam = cameras.find(c=>c.id===dragState.cam.id);
    if (cam) {
      fetch('/cameras/'+cam.id, {
        method:'PUT', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ x: Math.round(cam.x), y: Math.round(cam.y) })
      }).then(()=>loadCameras());
    }
  }
  dragState=null; drawMap();
});

function updateCursor(x, y) {
  if (currentMode==='addCam'){canvas.style.cursor='crosshair';return;}
  if (currentMode==='editZones'){
    const h=handleAt(x,y,selectedZone);
    if(h){const m={nw:'nw-resize',n:'n-resize',ne:'ne-resize',e:'e-resize',se:'se-resize',s:'s-resize',sw:'sw-resize',w:'w-resize'};canvas.style.cursor=m[h.id]||'pointer';return;}
    canvas.style.cursor=zoneAt(x,y)?'move':'crosshair';return;
  }
  // Normal mode: grab cursor when over a camera
  canvas.style.cursor = dragState && dragState.type==='moveCam' ? 'grabbing'
                       : camAt(x,y) ? 'grab' : 'default';
}

// ── Mode ───────────────────────────────────────────────────────────────────────
function setMode(m) {
  currentMode=m;
  const hints={normal:'',addCam:'Cliquez sur la carte pour placer la caméra',
    editZones:'Sélectionner • Glisser pour déplacer • Poignées pour redimensionner • Tracer pour créer'};
  const ht=document.getElementById('hintText');
  ht.textContent=hints[m]; ht.style.display=m!=='normal'?'':'none';
  document.getElementById('btnAddCam').classList.toggle('active', m==='addCam');
  document.getElementById('btnEditZones').classList.toggle('active', m==='editZones');
  if(m==='editZones'){hideCamPanel();selectedCam=null;}
  if(m!=='editZones'){hideZonePanel();selectedZone=null;}
  drawMap();
}
function toggleMode(m){setMode(currentMode===m?'normal':m);}

// ── Zone panel ─────────────────────────────────────────────────────────────────
function showZonePanel(z) {
  document.getElementById('zpLabel').value=z.label;
  document.getElementById('zpType').value=z.type||'custom';
  document.getElementById('zpColor').value=z.color;
  document.getElementById('zonePanel').classList.add('visible');
}
function hideZonePanel(){document.getElementById('zonePanel').classList.remove('visible');}
function onZoneLabelChange(){
  if(!selectedZone)return;
  const i=zones.findIndex(z=>z.id===selectedZone.id);
  if(i>=0){zones[i].label=document.getElementById('zpLabel').value;selectedZone=zones[i];}
  drawMap(); autoSaveZones();
}
function onZoneTypeChange(){
  if(!selectedZone)return;
  const type=document.getElementById('zpType').value, i=zones.findIndex(z=>z.id===selectedZone.id);
  if(i>=0){zones[i].type=type; if(ZONE_TYPES[type]){zones[i].color=ZONE_TYPES[type].color;document.getElementById('zpColor').value=ZONE_TYPES[type].color;} selectedZone=zones[i];}
  drawMap(); autoSaveZones();
}
function onZoneColorChange(){
  if(!selectedZone)return;
  const i=zones.findIndex(z=>z.id===selectedZone.id);
  if(i>=0){zones[i].color=document.getElementById('zpColor').value;selectedZone=zones[i];}
  drawMap(); autoSaveZones();
}
function confirmDeleteZone(){
  if(!selectedZone||!confirm('Supprimer le bloc "'+selectedZone.label+'" ?'))return;
  zones=zones.filter(z=>z.id!==selectedZone.id); selectedZone=null; hideZonePanel(); drawMap(); autoSaveZones();
}

// ── New zone modal ─────────────────────────────────────────────────────────────
function onNewZoneTypeChange(){
  const t=document.getElementById('nzType').value;
  if(ZONE_TYPES[t])document.getElementById('nzColor').value=ZONE_TYPES[t].color;
}
function closeNewZoneModal(){pendingNewZone=null;closeModal('newZoneModal');drawMap();}
function confirmNewZone(){
  if(!pendingNewZone)return;
  const label=document.getElementById('nzLabel').value.trim()||'Nouvelle zone';
  const type=document.getElementById('nzType').value;
  const color=document.getElementById('nzColor').value;
  const z={id:'zone_'+Date.now(),label,type,color,x:Math.round(pendingNewZone.x),y:Math.round(pendingNewZone.y),w:Math.round(pendingNewZone.w),h:Math.round(pendingNewZone.h)};
  zones.push(z); selectedZone=z; showZonePanel(z); pendingNewZone=null; closeModal('newZoneModal'); drawMap(); autoSaveZones();
}
async function confirmResetZones(){
  if(!confirm('Réinitialiser la carte aux zones par défaut ?'))return;
  const r=await fetch('/zones-config/reset',{method:'POST'}); const d=await r.json(); zones=d.zones; selectedZone=null; hideZonePanel(); drawMap();
}
async function autoSaveZones(){
  await fetch('/zones-config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({zones})});
}

// ── D-pad directionnel ─────────────────────────────────────────────────────────
const DIR_ANGLES   = {nw:225, n:270, ne:315, w:180, e:0, sw:135, s:90, se:45};
const DIR_LABELS   = {nw:'↖ NW', n:'↑ N', ne:'↗ NE', w:'← O', e:'→ E', sw:'↙ SO', s:'↓ S', se:'↘ SE'};
const DIR_OPPOSITES= {n:'s', s:'n', e:'w', w:'e', ne:'sw', sw:'ne', nw:'se', se:'nw'};
let currentDpadDir = 'e';

function setDpadDir(dir) {
  currentDpadDir = dir;
  document.querySelectorAll('.dpad-btn').forEach(b => b.classList.remove('selected'));
  const btn = document.getElementById('dpad-' + dir);
  if (btn) btn.classList.add('selected');
  document.getElementById('cmInflowLbl').textContent  = DIR_LABELS[dir]            || dir;
  document.getElementById('cmOutflowLbl').textContent = DIR_LABELS[DIR_OPPOSITES[dir]] || '—';
}

function angleToDir(angle) {
  const a = ((angle % 360) + 360) % 360;
  let best = 'e', bestDiff = 360;
  for (const [d, da] of Object.entries(DIR_ANGLES)) {
    const diff = Math.abs(((a - da + 180 + 360) % 360) - 180);
    if (diff < bestDiff) { bestDiff = diff; best = d; }
  }
  return best;
}

// ── Source USB ─────────────────────────────────────────────────────────────────
function onSourceChange() {
  const isUsb = document.querySelector('input[name="cmSource"]:checked').value === 'usb';
  document.getElementById('cmUsbGroup').style.display = isUsb ? '' : 'none';
  if (isUsb) refreshUsbDevices();
}

async function refreshUsbDevices() {
  const sel = document.getElementById('cmUsbIndex');
  sel.innerHTML = '<option value="">Recherche…</option>';
  try {
    const r = await fetch('/cameras/usb-devices');
    const d = await r.json();
    sel.innerHTML = '';
    if (!d.devices.length) {
      sel.innerHTML = '<option value="">Aucune caméra détectée</option>';
      return;
    }
    d.devices.forEach(dev => {
      const opt = document.createElement('option');
      opt.value = dev.index;
      opt.textContent = dev.label;
      sel.appendChild(opt);
    });
  } catch(e) {
    sel.innerHTML = '<option value="">Erreur de détection</option>';
  }
}

// ── Camera panel ───────────────────────────────────────────────────────────────
function showCamPanel(cam) {
  document.getElementById('cpName').textContent   = cam.name;
  document.getElementById('cpZone').textContent   = zoneLabel(cam.zone);
  document.getElementById('cpInflow').textContent = DIR_LABELS[angleToDir(cam.angle || 0)] || (cam.angle + '°');
  document.getElementById('cpSource').textContent = (cam.source_type === 'usb')
    ? 'USB ' + cam.usb_index : 'Fichiers';
  document.getElementById('cpCount').textContent  = cam.person_count || 0;
  document.getElementById('camPanel').classList.add('visible');
}
function hideCamPanel(){document.getElementById('camPanel').classList.remove('visible');}
function zoneLabel(id){const z=zones.find(z=>(z.type||z.id)===id||z.id===id);return z?z.label:id;}

// ── Camera modal ───────────────────────────────────────────────────────────────
function openCamEditModal() {
  if (!selectedCam) return;
  document.getElementById('camModalTitle').textContent = 'Modifier la caméra';
  document.getElementById('cmName').value  = selectedCam.name;
  document.getElementById('cmZone').value  = selectedCam.zone;
  setDpadDir(angleToDir(selectedCam.angle || 0));
  const src = selectedCam.source_type || 'file';
  document.querySelectorAll('input[name="cmSource"]').forEach(r => { r.checked = r.value === src; });
  if (src === 'usb') {
    document.getElementById('cmUsbGroup').style.display = '';
    refreshUsbDevices().then(() => {
      if (selectedCam.usb_index != null)
        document.getElementById('cmUsbIndex').value = selectedCam.usb_index;
    });
  } else {
    document.getElementById('cmUsbGroup').style.display = 'none';
  }
  pendingCamPos = null;
  openModal('camModal');
}
function closeCamModal(){closeModal('camModal');}

async function saveCam() {
  const name = document.getElementById('cmName').value.trim();
  if (!name) { alert('Saisissez un nom.'); return; }
  const zone        = document.getElementById('cmZone').value;
  const angle       = DIR_ANGLES[currentDpadDir] ?? 0;
  const source_type = document.querySelector('input[name="cmSource"]:checked').value;
  const usb_index   = source_type === 'usb'
    ? (parseInt(document.getElementById('cmUsbIndex').value) || 0)
    : null;
  if (selectedCam && !pendingCamPos) {
    await fetch('/cameras/'+selectedCam.id, {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, zone, angle, source_type, usb_index})
    });
  } else if (pendingCamPos) {
    await fetch('/cameras', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, x:pendingCamPos.x, y:pendingCamPos.y, angle, zone, source_type, usb_index})
    });
    pendingCamPos = null;
  }
  closeCamModal(); await loadCameras();
}
async function confirmDeleteCam(){
  if(!selectedCam||!confirm('Supprimer "'+selectedCam.name+'" ?'))return;
  await fetch('/cameras/'+selectedCam.id,{method:'DELETE'}); selectedCam=null; hideCamPanel(); await loadCameras();
}

// ── Video modal ────────────────────────────────────────────────────────────────
function openVideoModal(){
  if(!selectedCam)return;
  document.getElementById('videoTitle').textContent=selectedCam.name;
  document.getElementById('videoStream').src='/cameras/'+selectedCam.id+'/stream';
  document.getElementById('vCount').textContent=selectedCam.person_count||0;
  openModal('videoModal');
  videoCountInterval=setInterval(async()=>{
    if(!selectedCam)return;
    const r=await fetch('/cameras/'+selectedCam.id+'/count');
    const d=await r.json(); document.getElementById('vCount').textContent=d.person_count;
  },2000);
}
function closeVideoModal(){clearInterval(videoCountInterval);document.getElementById('videoStream').src='';closeModal('videoModal');}

// ── Modal helpers ──────────────────────────────────────────────────────────────
function openModal(id){document.getElementById(id).classList.add('open');}
function closeModal(id){document.getElementById(id).classList.remove('open');}
document.querySelectorAll('.modal-overlay').forEach(el=>el.addEventListener('click',e=>{
  if(e.target===el){closeVideoModal();closeModal('camModal');closeNewZoneModal();}
}));

// ── Loaders ────────────────────────────────────────────────────────────────────
async function loadZones(){const r=await fetch('/zones-config');const d=await r.json();zones=d.zones;}
async function loadCameras(){
  const r=await fetch('/cameras');const d=await r.json();cameras=d.cameras;
  if(selectedCam){selectedCam=cameras.find(c=>c.id===selectedCam.id)||null;if(selectedCam)showCamPanel(selectedCam);else hideCamPanel();}
  drawMap();
}

(async()=>{
  await loadZones(); await loadCameras();
  let _camBusy=false;
  async function pollCameras(){
    if(!_camBusy){_camBusy=true;try{await loadCameras();}catch(e){}finally{_camBusy=false;}}
    setTimeout(pollCameras,10000);
  }
  pollCameras();
})();
</script>
"""


@app.get("/zones-config")
def get_zones_config() -> dict[str, object]:
    return {"zones": zones_config.load_zones()}


@app.post("/zones-config")
def save_zones_config(body: dict) -> dict[str, object]:
    zones_config.save_zones(body.get("zones", []))
    return {"ok": True}


@app.post("/zones-config/reset")
def reset_zones_config() -> dict[str, object]:
    return {"zones": zones_config.reset_to_default()}


@app.get("/zones/camera-live")
def camera_live_data() -> dict[str, object]:
    cameras = camera_manager.list_cameras()
    zone_map: dict[str, dict] = {}
    for cam in cameras:
        zid = cam.get("zone", "gate")
        counts = video_processor.get_flow_counts(cam["id"])
        if zid not in zone_map:
            zone_map[zid] = {
                "zone": zid,
                "occupancy": 0,
                "inflow": 0,
                "outflow": 0,
                "camera_count": 0,
            }
        zone_map[zid]["occupancy"] += counts["moving"]
        zone_map[zid]["inflow"] += counts["inflow"]
        zone_map[zid]["outflow"] += counts["outflow"]
        zone_map[zid]["camera_count"] += 1

    # Enrichissement : dwell estimé + congestion_score + noms normalisés pour le frontend
    for zid, z in zone_map.items():
        occ = z["occupancy"]
        out = z["outflow"]
        capacity = ZONE_CAPACITY.get(zid, 500)
        z["congestion_score"] = round(min(1.0, occ / max(1, capacity)), 3)
        # Dwell = occupancy / débit de sortie (borné entre 1 et 60 min)
        z["avg_dwell_minutes"] = round(
            min(60.0, occ / max(1.0, float(out))), 2
        ) if out > 0 else round(min(60.0, max(1.0, occ / 35.0)), 2)
        # Alias inflow_per_min / outflow_per_min pour cohérence avec les snapshots simulés
        z["inflow_per_min"] = float(z["inflow"])
        z["outflow_per_min"] = float(z["outflow"])

    total_persons = sum(z["occupancy"] for z in zone_map.values())
    active_cameras = sum(
        1 for cam in cameras if video_processor.get_count(cam["id"]) > 0
    )
    with _lock:
        cam_snaps_count = len(_state.get("camera_snapshots", []))
    return {
        "zones": list(zone_map.values()),
        "total_cameras": len(cameras),
        "active_cameras": active_cameras,
        "total_persons": total_persons,
        "prediction_source": "cameras" if cam_snaps_count >= 3 else "simulation",
    }


@app.get("/dashboard/map", response_class=HTMLResponse)
def dashboard_map() -> str:
    return _shell(
        "map",
        "Carte Interactive",
        "Plan de l'aéroport — ajoutez/modifiez zones et caméras en temps réel",
        _MAP_STYLE + _MAP_BODY,
        _MAP_SCRIPT,
    )
