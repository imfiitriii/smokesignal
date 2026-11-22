"""Quick accuracy check: CAMS PM2.5 + ERA5 weather, Random Forest, train 2022-08..2025, test 2026."""
import requests, pandas as pd, numpy as np, os, sys
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error

OUT = os.path.dirname(os.path.abspath(__file__))
TZ = "Asia/Kuala_Lumpur"; S, E = "2022-08-04", "2026-11-20"

def cached(name, fn):
    p = os.path.join(OUT, name)
    if os.path.exists(p): return pd.read_pickle(p)
    df = fn(); df.to_pickle(p); return df

def get_aq():
    import json; j = json.load(open(os.path.join(OUT, "aq.json")))
    return pd.DataFrame(j["hourly"])

PTS = {"local": (4.385, 100.979), "strait": (3.0, 100.5), "riau": (0.5, 101.5)}
def get_wx():
    import json; j = json.load(open(os.path.join(OUT, "wx.json")))
    frames = []
    for name, loc in zip(PTS, j):
        h = pd.DataFrame(loc["hourly"]); h["pt"] = name; frames.append(h)
    return pd.concat(frames)

aq = cached("aq.pkl", get_aq); wx = cached("wx.pkl", get_wx)

aq["date"] = pd.to_datetime(aq["time"]).dt.normalize()
g = aq.groupby("date")
daily = pd.DataFrame({"pm": g["pm2_5"].mean(), "pm_max": g["pm2_5"].max(), "aod": g["aerosol_optical_depth"].mean(),
                      "n": g["pm2_5"].count()})
daily = daily[daily.n >= 18].drop(columns="n")

wx["date"] = pd.to_datetime(wx["time"]).dt.normalize()
rad = np.deg2rad(wx["wind_direction_10m"])
wx["u"] = -wx["wind_speed_10m"] * np.sin(rad); wx["v"] = -wx["wind_speed_10m"] * np.cos(rad)
wd = wx.groupby(["pt", "date"]).agg(rain=("precipitation", "sum"), rh=("relative_humidity_2m", "mean"),
                                    u=("u", "mean"), v=("v", "mean"), blh=("boundary_layer_height", "mean")).unstack(0)
wd.columns = [f"{a}_{b}" for a, b in wd.columns]
for p in PTS:  # consecutive dry days
    dry = (wd[f"rain_{p}"] < 1).astype(int)
    wd[f"drydays_{p}"] = dry.groupby((dry == 0).cumsum()).cumsum()

def cat(x): return np.where(x <= 15, 0, np.where(x <= 35, 1, 2))
NAMES = ["Good", "Moderate", "Unhealthy"]

def build(h):
    df = daily.join(wd, how="inner").asfreq("D")
    X = pd.DataFrame(index=df.index)
    X["pm"] = df.pm; X["pm_max"] = df.pm_max; X["aod"] = df.aod
    X["pm_lag1"] = df.pm.shift(1); X["pm_d1"] = df.pm - df.pm.shift(1); X["pm_3d"] = df.pm.rolling(3).mean()
    wcols = [c for c in wd.columns]
    for c in wcols: X[c] = df[c]                      # today's weather
    for c in wcols: X[c + "_tgt"] = df[c].shift(-h)   # weather on target day (stands in for forecast)
    doy = df.index.dayofyear; X["sin"] = np.sin(2*np.pi*doy/365.25); X["cos"] = np.cos(2*np.pi*doy/365.25)
    y = df.pm.shift(-h)
    m = X.notna().all(1) & y.notna()
    return X[m], y[m]

rows = []
for h in [1, 2, 3]:
    X, y = build(h)
    tr = X.index < "2026-01-01"; te = ~tr
    reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1).fit(X[tr], cat(y[tr]))
    pc = clf.predict(X[te])
    yt = cat(y[te].values); pr = cat(pred); pers = cat(X.loc[te, "pm"].values)
    def stats(p):
        acc = (p == yt).mean(); u = yt == 2
        rec = (p[u] == 2).mean() if u.any() else np.nan
        prec = (yt[p == 2] == 2).mean() if (p == 2).any() else np.nan
        return acc, rec, prec, (p == 2).sum()
    for name, p in [("persistence", pers), ("RF regressor", pr), ("RF classifier", pc)]:
        a, r, pcs, n2 = stats(p)
        rows.append(dict(horizon=f"D+{h}", model=name, accuracy=round(a, 2), unhealthy_recall=round(r, 2),
                         unhealthy_precision=round(pcs, 2), unhealthy_predicted=int(n2), unhealthy_actual=int((yt == 2).sum())))
    rows.append(dict(horizon=f"D+{h}", model="RF regressor MAE ug/m3", accuracy=round(mean_absolute_error(y[te], pred), 1)))
    rows.append(dict(horizon=f"D+{h}", model="persistence MAE ug/m3", accuracy=round(mean_absolute_error(y[te], X.loc[te, "pm"]), 1)))
    if h == 1:
        print("D+1 RF regressor confusion (rows=actual G/M/U):\n", confusion_matrix(yt, pr, labels=[0, 1, 2]))
        print("D+1 persistence confusion:\n", confusion_matrix(yt, pers, labels=[0, 1, 2]))
        imp = pd.Series(reg.feature_importances_, X.columns).sort_values(ascending=False).head(10)
        print("Top features D+1:\n", imp.round(3))
        print("train unhealthy days:", int((cat(y[tr]) == 2).sum()), " test days:", int(te.sum()))
print(pd.DataFrame(rows).to_string(index=False))
