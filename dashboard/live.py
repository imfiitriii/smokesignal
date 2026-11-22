"""Live data for the SmokeSignal dashboard + the forecast itself.

Uses the SAME feature code as training (experiments/features.py).
"""
import datetime as dt, io, os, sys
import numpy as np, pandas as pd, requests

try:                                   # use Windows' certificate store (fixes SSL errors on some PCs)
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, EXP)
from features import (POINTS, SELECTED, FIRE_BBOX, cat, daily_air, daily_weather, prep_fires,
                      locate_fires, mark_upwind, fire_features, build_X, wind_from_strait)

TZ = "Asia/Kuala_Lumpur"
STATIONS = {"CA12A": "Pegoh Ipoh (22 km)", "CA11A": "Tasek Ipoh (31 km)", "CA13A": "Seri Manjung (40 km)"}


def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_air():
    """Open-Meteo Air Quality (CAMS): last 7 days + 4 forecast days, hourly."""
    j = get_json("https://air-quality-api.open-meteo.com/v1/air-quality", dict(
        latitude=POINTS["local"][0], longitude=POINTS["local"][1],
        hourly="pm2_5,aerosol_optical_depth", past_days=7, forecast_days=4, timezone=TZ))
    return pd.DataFrame(j["hourly"])


def fetch_weather():
    """Open-Meteo Forecast at the 3 points: last 60 days (for dry-day streaks) + 4 forecast days."""
    j = get_json("https://api.open-meteo.com/v1/forecast", dict(
        latitude=",".join(str(p[0]) for p in POINTS.values()),
        longitude=",".join(str(p[1]) for p in POINTS.values()),
        hourly="precipitation,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height",
        past_days=60, forecast_days=4, timezone=TZ))
    return {name: pd.DataFrame(loc["hourly"]) for name, loc in zip(POINTS, j)}


def fetch_fires(key, days=5):
    """NASA FIRMS VIIRS NOAA-20 near-real-time detections for the last `days` days (max 5)."""
    start = (dt.datetime.utcnow().date() - dt.timedelta(days=days - 1)).isoformat()
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_NOAA20_NRT/{FIRE_BBOX}/{days}/{start}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    if not r.text.startswith("latitude"):
        raise RuntimeError("FIRMS did not return data (check the key)")
    return pd.read_csv(io.StringIO(r.text))


def fetch_doe():
    """Current official DOE Air Pollutant Index for the 3 nearest stations (display only)."""
    j = get_json("https://eqms.doe.gov.my/api3/publicportalapims/mapcaqm")
    rows = [d for d in j.get("data", []) if d.get("STATION_ID") in STATIONS]
    return pd.DataFrame([{"Station": STATIONS[d["STATION_ID"]], "API": d.get("API"),
                          "Category": d.get("CLASS"), "Updated": d.get("RECORD_DT", "").replace("T", " ")[:16],
                          "lat": d.get("LATITUDE"), "lon": d.get("LONGITUDE")} for d in rows])


def forecast(bundle, static_cells, air_hourly, wx_frames, fire_raw, base_date):
    """Build today's features (base_date = day D) and predict PM2.5 for D+1..D+3."""
    daily = daily_air(air_hourly)
    wd = daily_weather(wx_frames)

    fire, typed_static, raw_days = prep_fires(fire_raw)
    fire = locate_fires(fire)
    fire = fire[~fire.cell.isin(static_cells | typed_static)]          # drop known gas flares / factories
    fire = mark_upwind(fire, wd)
    idx = pd.date_range(base_date - pd.Timedelta(days=6), base_date + pd.Timedelta(days=3), freq="D")
    ff = fire_features(fire, idx, raw_days)

    out = {}
    for h, model in bundle["models"].items():
        X, _ = build_X(daily, wd, ff, h)
        row = X.loc[[base_date], SELECTED]
        missing = [c for c in SELECTED if pd.isna(row[c].iloc[0])]
        if missing:
            out[h] = {"error": "missing inputs: " + ", ".join(missing)}
            continue
        pm_today = float(row["pm"].iloc[0])
        pm = max(0.0, pm_today + float(model.predict(row)[0]))
        out[h] = {"date": base_date + pd.Timedelta(days=h), "pm": pm, "cat": int(cat(pm))}

    X1, _ = build_X(daily, wd, ff, 1)
    feats = X1.loc[base_date]
    return out, feats, daily, wd, fire
