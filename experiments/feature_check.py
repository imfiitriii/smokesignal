"""Which features matter? Compares the full feature set with smaller sets.
Choice is made on TRAINING data only (leave-one-year-out 2022-2025, MAE); 2026 is reported afterwards."""
import io, contextlib, runpy, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(HERE, "model_fires.py"))
build, score = g["build"], g["score"]
from sklearn.ensemble import RandomForestRegressor
def fit_predict(Xs, y, tr, te, today_pm):
    d = y - today_pm   # same target as the main model: change from today
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Xs[tr], d[tr])
    return today_pm[te] + rf.predict(Xs[te]), rf

X, y = build(1, True)
cols = list(X.columns)
AIR = ["pm", "pm_max", "aod", "pm_lag1", "pm_d1", "pm_3d"]
SEASON = ["sin", "cos"]
FIRE = [c for c in cols if c.startswith(("n_", "frp_", "log_"))]
WX_TODAY = [c for c in cols if c not in AIR + SEASON + FIRE and not c.endswith("_tgt")]
WX_TGT = [c for c in cols if c.endswith("_tgt")]
print(f"TOTAL features: {len(cols)}  = air {len(AIR)} + fire {len(FIRE)} + weather today {len(WX_TODAY)}"
      f" + weather tomorrow {len(WX_TGT)} + season {len(SEASON)}")

COMPACT = (AIR + SEASON
           + ["rain_local_tgt", "rain_strait_tgt", "rain_riau_tgt", "rh_local_tgt", "blh_local_tgt",
              "u_strait_tgt", "v_strait_tgt", "u_riau_tgt", "v_riau_tgt", "drydays_riau", "drydays_local"]
           + ["n_upwind", "frp_upwind", "n_upwind_3d", "frp_upwind_3d", "n_upwind800",
              "n_sumatra", "frp_sumatra_3d", "n_peninsula", "frp_r0_300"])
sets = {
    "ALL features": cols,
    "no fire": [c for c in cols if c not in FIRE],
    "no weather (today+tomorrow)": [c for c in cols if c not in WX_TODAY + WX_TGT],
    "no tomorrow's weather": [c for c in cols if c not in WX_TGT],
    "no air (today's PM2.5 etc.)": [c for c in cols if c not in AIR],
    "COMPACT set": COMPACT,
}

rows = []
for name, fs in sets.items():
    # training-only check: leave-one-year-out on 2022-2025
    cv_p, cv_y = [], []
    for Y in [2022, 2023, 2024, 2025]:
        tr = (X.index.year != Y) & (X.index.year < 2026); te = X.index.year == Y
        p, _ = fit_predict(X[fs], y, tr, te, X.pm); cv_p.append(p); cv_y.append(y[te])
    cv = score(pd.concat(cv_y), pd.concat(cv_p))
    # final test 2026
    tr = X.index < "2026-01-01"; te = ~tr
    p, rf = fit_predict(X[fs], y, tr, te, X.pm); t = score(y[te], p)
    rows.append({"feature set": name, "n": len(fs), "train-CV error": round(cv["mae"], 2),
                 "2026 accuracy": f"{t['accuracy']*100:.0f}%", "2026 haze caught": f"{t['caught']}/{t['unhealthy']}",
                 "2026 false alarms": t["false_alarms"], "2026 error": round(t["mae"], 2)})
    if name == "COMPACT set":
        imp = pd.Series(rf.feature_importances_, fs).sort_values(ascending=False)
print(pd.DataFrame(rows).to_string(index=False))
print("\nCOMPACT set - importance ranking:\n" + imp.round(3).to_string())
