"""SmokeSignal experiment: does adding FIRMS fire features improve next-day haze prediction?

Inputs (all in this folder):
  aq.json  - Open-Meteo Air Quality (CAMS) hourly PM2.5 at Seri Iskandar
  wx.json  - Open-Meteo ERA5 hourly weather at 3 points (Seri Iskandar, Strait, Riau)
  firms/*.csv - NASA FIRMS VIIRS NOAA-20 yearly country files (Indonesia, Malaysia)
               + optional firms/area_2026.csv from the FIRMS area API (needs MAP_KEY)
"""
import glob, json, os, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
from features import (PTS, SELECTED, cat, daily_air, daily_weather, prep_fires, locate_fires,
                      mark_upwind, fire_features, build_X)

# ---------------- air quality (target + today's level) ----------------
daily = daily_air(pd.DataFrame(json.load(open(os.path.join(HERE, "aq.json")))["hourly"]))

# ---------------- weather ----------------
wd = daily_weather({name: pd.DataFrame(loc["hourly"])
                    for name, loc in zip(PTS, json.load(open(os.path.join(HERE, "wx.json"))))})

# ---------------- fires ----------------
fl = []
for f in sorted(glob.glob(os.path.join(HERE, "firms", "*.csv"))):
    fl.append(pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type"]
                          if "area_" not in os.path.basename(f) else None, low_memory=False))
fire, typed_static, raw_days = prep_fires(pd.concat(fl, ignore_index=True))
fire = locate_fires(fire)
# Remove static heat sources (gas flares, factories) the same way for every year:
# a 0.02-degree cell that shows a "fire" on more than 60 days in a year is not a vegetation fire.
fire["year"] = fire.date.dt.year
days_seen = fire.groupby(["year", "cell"])["date"].transform("nunique")
# the live dashboard only sees ~3 days of fires, so it cannot apply the 60-day rule itself:
# save the list of static cells found in history and filter live data with it
static_cells = sorted(set(fire.loc[days_seen > 60, "cell"]) | typed_static)
pd.Series(static_cells, name="cell").to_csv(os.path.join(HERE, "static_cells.csv"), index=False)
fire = fire[(days_seen <= 60) & ~fire.cell.isin(typed_static)]
print("fires per year after cleaning:", fire.groupby("year").size().to_dict())
fire = mark_upwind(fire, wd)
ff = fire_features(fire, pd.date_range(daily.index.min(), daily.index.max(), freq="D"), raw_days)
print("days with no fire data (treated as missing):",
      [str(d.date()) for d in ff.index[ff["n_upwind"].isna()] if d <= fire.date.max()])
print(f"fire records used: {len(fire):,}  dates {fire.date.min().date()} -> {fire.date.max().date()}")

# ---------------- feature table ----------------
def build(h, use_fire):
    X, y = build_X(daily, wd, ff if use_fire else None, h)
    m = X.notna().all(axis=1) & y.notna()
    return X[m], y[m]

def fit_predict(X, y, tr, te):
    d = y - X.pm   # predict the CHANGE from today (lets the forest go above values it has seen)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X[tr], d[tr])
    return X.pm[te] + rf.predict(X[te]), rf

def score(y_true, pred):
    yt, p = cat(np.asarray(y_true)), cat(np.asarray(pred)); u = yt == 2
    return dict(days=len(yt), accuracy=(p == yt).mean(), unhealthy=int(u.sum()),
                caught=int((p[u] == 2).sum()), false_alarms=int(((p == 2) & ~u).sum()),
                mae=np.abs(np.asarray(pred) - np.asarray(y_true)).mean())

def report(title, splits, h=1):
    print(f"\n=== {title} (D+{h}) ===")
    rows = []
    for use_fire in [False, True]:
        X, y = build(h, use_fire)
        allp, ally, allpers = [], [], []
        for tr_f, te_f in splits:
            tr, te = tr_f(X.index), te_f(X.index)
            if te.sum() == 0: continue
            pred, rf = fit_predict(X, y, tr, te)
            allp.append(pred); ally.append(y[te]); allpers.append(X.pm[te])
        P, Y, PE = pd.concat(allp), pd.concat(ally), pd.concat(allpers)
        if not use_fire:
            rows.append(dict(model="Same as today (baseline)", **score(Y, PE)))
        rows.append(dict(model="RF + fires (108)" if use_fire else "RF, no fires", **score(Y, P)))
        if use_fire:
            allp = []
            for tr_f, te_f in splits:
                tr, te = tr_f(X.index), te_f(X.index)
                if te.sum() == 0: continue
                pred, rf = fit_predict(X[SELECTED], y, tr, te); allp.append(pred)
            rows.append(dict(model="RF + fires, 28 selected (FINAL)", **score(Y, pd.concat(allp))))
            last_rf, last_X = rf, X[SELECTED]
    out = pd.DataFrame(rows)
    out["accuracy"] = (out.accuracy * 100).round(0).astype(int).astype(str) + "%"
    out["mae"] = out.mae.round(1)
    print(out.to_string(index=False))
    return last_rf, last_X

yr = lambda Y: (lambda idx: idx.year == Y)
notyr = lambda Y: (lambda idx: idx.year != Y)

has_2026 = ff.loc["2026-01-01":"2026-11-19"].notna().all(axis=1).mean() > 0.9 if ff.index.max() >= pd.Timestamp("2026-01-01") else False
if has_2026:
    rf, X = report("Train 2022-2025, test 2026", [(lambda i: i < "2026-01-01", lambda i: i >= "2026-01-01")])
else:
    print("\n(no 2026 fire data yet -> using leave-one-year-out on 2022-2025)")
    rf, X = report("Leave-one-year-out 2022-2025", [(notyr(Y), yr(Y)) for Y in [2022, 2023, 2024, 2025]])

imp = pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False)
print("\nTop 12 features (final 28-feature model):\n", imp.head(12).round(3).to_string())

# After testing, retrain the final models on ALL available days (incl. the 2026 haze) for the dashboard.
# D+1 is the main forecast; D+2 and D+3 are a lighter outlook (tested the same way, shown below).
import joblib
from sklearn.ensemble import RandomForestRegressor as RFR
models, outlook_scores = {}, {}
for h in [1, 2, 3]:
    Xh, yh = build(h, True)
    if h > 1:
        tr = Xh.index < "2026-01-01"
        pred, _ = fit_predict(Xh[SELECTED], yh, tr, ~tr)
        outlook_scores[h] = score(yh[~tr], pred)
        pers = score(yh[~tr], Xh.pm[~tr])
        print(f"D+{h} test 2026: model {outlook_scores[h]['accuracy']:.0%}, haze caught "
              f"{outlook_scores[h]['caught']}/{outlook_scores[h]['unhealthy']}, false alarms {outlook_scores[h]['false_alarms']}"
              f"  | same-as-today {pers['accuracy']:.0%}, {pers['caught']}/{pers['unhealthy']}, {pers['false_alarms']}")
    models[h] = RFR(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Xh[SELECTED], yh - Xh.pm)
Xa, _ = build(1, True)
joblib.dump({"models": models, "features": SELECTED,
             "target": "PM2.5 change from today, D+h (add to today's pm)",
             "trained_until": str(Xa.index.max().date()),
             # typical levels, so the dashboard can say whether today's fires/rain are high or low
             "reference": Xa[SELECTED].quantile([0.5, 0.75, 0.9]).to_dict(),
             "outlook_scores": outlook_scores}, os.path.join(HERE, "model.pkl"))
print(f"\nSaved model.pkl (D+1..D+3, trained on {len(Xa)} days up to {Xa.index.max().date()}) "
      f"and static_cells.csv ({len(static_cells)} cells)")

# Objective 1: simple relationship check
X1, y1 = build(1, True)
print("\nCorrelation of fire features with TOMORROW's PM2.5:")
for c in ["n_upwind", "frp_upwind", "frp_upwind800", "n_sumatra", "frp_sumatra", "n_peninsula", "frp_r0_300"]:
    print(f"  {c:16s} r = {np.corrcoef(np.log1p(X1[c]), y1)[0,1]:+.2f}")
print(f"  {'today PM2.5':16s} r = {np.corrcoef(X1.pm, y1)[0,1]:+.2f}")
