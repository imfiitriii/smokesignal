"""SmokeSignal feature code, shared by training (model_fires.py) and the dashboard (dashboard/app.py).

Keeping ONE copy of this code guarantees the dashboard builds exactly the same features the model learned from.
"""
import numpy as np, pandas as pd

UTP = (4.385, 100.979)                       # Seri Iskandar / UTP
POINTS = {"local": (4.385, 100.979),         # weather points along the smoke path
          "strait": (3.0, 100.5),
          "riau": (0.5, 101.5)}
PTS = list(POINTS)
FIRE_BBOX = "95,-6,120,8"                    # west,south,east,north

# The 28 features kept after feature selection (see feature_check.py)
SELECTED = (["pm", "pm_max", "aod", "pm_lag1", "pm_d1", "pm_3d", "sin", "cos",
             "rain_local_tgt", "rain_strait_tgt", "rain_riau_tgt", "rh_local_tgt", "blh_local_tgt",
             "u_strait_tgt", "v_strait_tgt", "u_riau_tgt", "v_riau_tgt", "drydays_riau", "drydays_local",
             "n_upwind", "frp_upwind", "n_upwind_3d", "frp_upwind_3d", "n_upwind800",
             "n_sumatra", "frp_sumatra_3d", "n_peninsula", "frp_r0_300"])

CATEGORIES = ["Good", "Moderate", "Unhealthy"]
def cat(x):
    """PM2.5 daily mean (ug/m3) -> 0 Good (<=15, WHO), 1 Moderate (15-35), 2 Unhealthy (>35, Malaysia 24-h standard)."""
    return np.where(np.asarray(x) <= 15, 0, np.where(np.asarray(x) <= 35, 1, 2))


# ---------------- air quality ----------------
def daily_air(hourly):
    """hourly: DataFrame with time, pm2_5, aerosol_optical_depth (local time)."""
    aq = hourly.copy()
    aq["date"] = pd.to_datetime(aq["time"]).dt.normalize()
    g = aq.groupby("date")
    daily = pd.DataFrame({"pm": g["pm2_5"].mean(), "pm_max": g["pm2_5"].max(),
                          "aod": g["aerosol_optical_depth"].mean(), "n": g["pm2_5"].count()})
    return daily[daily.n >= 18].drop(columns="n")


# ---------------- weather ----------------
def daily_weather(frames):
    """frames: dict point-name -> hourly DataFrame (time, precipitation, relative_humidity_2m,
    wind_speed_10m, wind_direction_10m, boundary_layer_height), local time."""
    parts = []
    for name, h in frames.items():
        h = h.copy(); h["pt"] = name; parts.append(h)
    wx = pd.concat(parts)
    wx["date"] = pd.to_datetime(wx["time"]).dt.normalize()
    rad = np.deg2rad(wx["wind_direction_10m"])
    wx["u"] = -wx["wind_speed_10m"] * np.sin(rad)   # wind blowing TOWARDS east
    wx["v"] = -wx["wind_speed_10m"] * np.cos(rad)   # wind blowing TOWARDS north
    wd = wx.groupby(["pt", "date"]).agg(rain=("precipitation", "sum"), rh=("relative_humidity_2m", "mean"),
                                        u=("u", "mean"), v=("v", "mean"),
                                        blh=("boundary_layer_height", "mean")).unstack(0)
    wd.columns = [f"{a}_{b}" for a, b in wd.columns]
    for p in PTS:
        dry = (wd[f"rain_{p}"] < 1).astype(int)
        wd[f"drydays_{p}"] = dry.groupby((dry == 0).cumsum()).cumsum()
    return wd

def wind_from_strait(wd):
    """Daily direction the wind over the Strait comes FROM (degrees, 0 = north)."""
    return (np.rad2deg(np.arctan2(-wd["u_strait"], -wd["v_strait"])) + 360) % 360


# ---------------- fires ----------------
def prep_fires(raw):
    """raw FIRMS rows -> confident detections with local date + 0.02-degree cell id.
    Returns (fire, typed_static_cells, raw_days)."""
    fire = raw[raw["confidence"].astype(str) != "l"].copy()
    fire["cell"] = ((fire.latitude / 0.02).round().astype(int).astype(str) + "_" +
                    (fire.longitude / 0.02).round().astype(int).astype(str))
    typed_static = set(fire.loc[fire["type"].fillna(0) > 0, "cell"]) if "type" in fire else set()
    if "type" in fire:
        fire = fire[fire["type"].fillna(0) == 0]            # vegetation fires only
    t = (pd.to_datetime(fire["acq_date"]) + pd.to_timedelta(fire["acq_time"] // 100, "h")
         + pd.to_timedelta(fire["acq_time"] % 100, "m"))
    fire["date"] = (t + pd.Timedelta(hours=8)).dt.normalize()   # FIRMS is UTC -> Malaysia time
    fire = fire.drop_duplicates(["latitude", "longitude", "date"])
    raw_days = set(fire.date.unique())
    return fire, typed_static, raw_days

def locate_fires(fire):
    """Add distance/bearing from Seri Iskandar and region; keep Sumatra, Peninsula, Borneo within 1500 km."""
    fire = fire.copy()
    lat0, lon0 = np.deg2rad(UTP[0]), np.deg2rad(UTP[1])
    la, lo = np.deg2rad(fire.latitude.values), np.deg2rad(fire.longitude.values)
    dlon = lo - lon0
    fire["dist"] = 6371 * np.arccos(np.clip(np.sin(lat0)*np.sin(la) + np.cos(lat0)*np.cos(la)*np.cos(dlon), -1, 1))
    fire["bearing"] = (np.rad2deg(np.arctan2(np.sin(dlon)*np.cos(la),
                        np.cos(lat0)*np.sin(la) - np.sin(lat0)*np.cos(la)*np.cos(dlon))) + 360) % 360
    lon, lat = fire.longitude, fire.latitude
    strait_lon = np.interp(lat, [1.3, 2.2, 3.0, 4.0, 6.0], [103.0, 101.6, 100.8, 100.2, 99.0])
    peninsula = (lat > 1.2) & (lat < 7) & (lon > strait_lon) & (lon < 105)
    fire["region"] = np.select(
        [peninsula,
         (lon < 106.5) & (lat < 6) & (lon <= strait_lon) | ((lon < 106.5) & (lat <= 1.2)),
         (lon >= 108)],
        ["peninsula", "sumatra", "borneo"], "other")
    return fire[(fire.dist <= 1500) & (fire.region != "other") & (fire.latitude < 6.7)]

def mark_upwind(fire, wd):
    fire = fire.copy()
    fire["wind_from"] = fire["date"].map(wind_from_strait(wd))
    diff = np.abs(((fire.bearing - fire.wind_from) + 180) % 360 - 180)
    fire["upwind"] = diff <= 45
    return fire

def fire_features(fire, index, raw_days):
    """Daily fire counts/FRP by distance band, region and upwind; plus lag1, 3-day sums and logs."""
    ff = pd.DataFrame(index=index)
    def agg(mask, name):
        s = fire[mask].groupby("date")
        ff[f"n_{name}"] = s.size(); ff[f"frp_{name}"] = s["frp"].sum()
    for lo_, hi_ in [(0, 300), (300, 600), (600, 1500)]:
        agg((fire.dist >= lo_) & (fire.dist < hi_), f"r{lo_}_{hi_}")
    for r in ["sumatra", "peninsula", "borneo"]:
        agg(fire.region == r, r)
    agg(fire.upwind, "upwind")
    agg(fire.upwind & (fire.dist < 800), "upwind800")
    ff = ff.fillna(0)
    ff.loc[~ff.index.isin(list(raw_days)), :] = np.nan   # no satellite data that day = missing, not zero
    for c in list(ff.columns):
        ff[c + "_lag1"] = ff[c].shift(1)
        ff[c + "_3d"] = ff[c].rolling(3).sum()
        ff["log_" + c] = np.log1p(ff[c])
    return ff


# ---------------- feature table ----------------
def build_X(daily, wd, ff, h=1):
    """One row per day D: conditions on D (+ weather on D+h). y = PM2.5 on D+h."""
    df = daily.join(wd, how="outer").asfreq("D")
    X = pd.DataFrame(index=df.index)
    X["pm"] = df.pm; X["pm_max"] = df.pm_max; X["aod"] = df.aod
    X["pm_lag1"] = df.pm.shift(1); X["pm_d1"] = df.pm - df.pm.shift(1); X["pm_3d"] = df.pm.rolling(3).mean()
    for c in wd.columns:
        X[c] = df[c]; X[c + "_tgt"] = df[c].shift(-h)
    doy = df.index.dayofyear
    X["sin"] = np.sin(2*np.pi*doy/365.25); X["cos"] = np.cos(2*np.pi*doy/365.25)
    if ff is not None:
        X = X.join(ff)
    y = df.pm.shift(-h)
    return X, y
