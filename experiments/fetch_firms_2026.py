"""Download 2026 VIIRS NOAA-20 fire detections from the NASA FIRMS area API.

1. Get a free MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Save it (just the key, one line) in experiments/firms_key.txt
3. Run:  python fetch_firms_2026.py
Writes firms/area_2026.csv. Uses curl (works with Windows certificates).
"""
import datetime as dt, io, os, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = open(os.path.join(HERE, "firms_key.txt")).read().strip()
BBOX = "95,-6,120,8"            # west,south,east,north: Sumatra + Peninsula + Borneo
START, END = dt.date(2026, 1, 1), dt.date.today()

def get(source, day, n):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{source}/{BBOX}/{n}/{day}"
    r = subprocess.run(["curl", "-s", "-f", url], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.startswith("latitude"):
        return None
    return pd.read_csv(io.StringIO(r.stdout))

parts, d = [], START
while d <= END:
    n = min(5, (END - d).days + 1)               # API allows at most 5 days per call
    # ask BOTH feeds: the archive (SP) and the near-real-time feed (NRT) meet mid-year,
    # so one 5-day request can straddle the switch-over date
    got = [x for x in (get("VIIRS_NOAA20_SP", d, n), get("VIIRS_NOAA20_NRT", d, n)) if x is not None and not x.empty]
    df = (pd.concat(got, ignore_index=True)
            .drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])) if got else None
    print(d, n, "days ->", 0 if df is None else len(df), "fires")
    if df is not None and not df.empty:
        parts.append(df)
    d += dt.timedelta(days=n)

out = pd.concat(parts, ignore_index=True)
out.to_csv(os.path.join(HERE, "firms", "area_2026.csv"), index=False)
print("saved", len(out), "rows to firms/area_2026.csv")
