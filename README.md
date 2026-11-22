# SmokeSignal: AI Early Warning System for Haze in Seri Iskandar

SmokeSignal predicts **tomorrow's air quality** in Seri Iskandar / UTP (Good, Moderate or Unhealthy) from satellite fire hotspots, weather and today's air quality.
On 2026, a year the model never saw, it chose the right category **76%** of the time and warned about **16 of 28 haze days** with **3 false alarms**.

Location: Seri Iskandar / UTP, 4.385 N, 100.979 E.

---

## How the solution works

The solution has two parts:

```
PART 1 · TRAINING (once)
  Clean data ──► 108 features ──► Select 28 ──► Train ──► Test on 2026
  (sources 1–4)   (1 row/day)     (CV 2022–25)  (Random   (76% correct)
                                                 Forest)
                                                   │
                                                   ▼
                                             model.pkl + static_cells.csv
                                                   │
PART 2 · DAILY PREDICTION (dashboard)              ▼
  Today's data ──► 28 features ──► Predict ──► Category ──► Dashboard
  (sources 1,3,5)  (same code)     PM2.5       + advice     (+ DOE readings, source 7)
```

Both parts build features with the **same file**, `experiments/features.py`, so the dashboard gives the model exactly the inputs it learned from.

---

## Data sources (all free and open)

| # | Data | Source | Used in |
|---|---|---|---|
| 1 | PM2.5 + haze thickness (AOD), hourly | [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (Copernicus CAMS), Aug 2022 → today | Training + dashboard |
| 2 | Fire hotspots 2022–2025 | [NASA FIRMS yearly country files](https://firms.modaps.eosdis.nasa.gov/country/) (VIIRS NOAA-20) | Training |
| 3 | Fire hotspots 2026 + live | [NASA FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/) (free key, max 5 days per request) | Testing + dashboard |
| 4 | Past weather at 3 points | [Open-Meteo Historical Weather](https://open-meteo.com/en/docs/historical-weather-api) (ERA5) | Training |
| 5 | Weather forecast at 3 points | [Open-Meteo Forecast API](https://open-meteo.com/en/docs) | Dashboard |
| 6 | Real station readings | [WAQI](https://aqicn.org/) (data from Malaysia DOE), manual CSV download | Checking CAMS (still to do) |
| 7 | Official Air Pollutant Index | [DOE APIMS](https://eqms.doe.gov.my/APIMS/main) | Dashboard, display only |

Weather points along the smoke path:
- **Seri Iskandar** (4.385, 100.979)
- **Strait of Malacca** (3.0, 100.5)
- **Riau, Sumatra** (0.5, 101.5)

---

## Part 1: Training (`experiments/model_fires.py`)

1. **Collect.** Download PM2.5 (1), fires (2, 3) and weather (4) for Aug 2022 – Nov 2026.
2. **Clean the fires.**
   - Drop low-confidence detections.
   - Keep Sumatra, Peninsular Malaysia and Borneo within 1500 km.
   - Remove gas flares and factories: NASA-labelled static sources, plus any spot "burning" on more than 60 days in a year.
   - Convert UTC to Malaysia time.
   - Days with no satellite data are treated as **missing**, not "zero fires".
3. **Clean the air data.** Hourly PM2.5 → daily average (days with at least 18 hours only).
4. **Build 108 candidate features**, one row per day: 6 air, 64 fire, 18 weather today, 18 weather tomorrow, 2 season.
5. **Add the answer.** Each row gets the **next day's** PM2.5.
6. **Split the years.** 2022–2025 for training and feature selection; **2026 is locked away** as the final test.
7. **Select features (108 → 28)**, using 2022–2025 only (see `experiments/feature_check.py`):
   - Group features by idea and keep one or two per idea by hand.
   - Check with leave-one-year-out cross-validation: the 28 give the same error as all 108 (3.09 vs 3.10 µg/m³).
   - Removing any group makes the error worse.
8. **Train** a Random Forest on 2022–2025. It predicts the *change* in PM2.5 from today, then adds it to today's value, so it can predict levels higher than it has seen.
9. **Test** once on 2026 and compare with the baseline "tomorrow = today".
10. **Retrain on all years** (including the 2026 haze) and save:
    - `model.pkl`: models for 1, 2 and 3 days ahead, plus reference levels
    - `static_cells.csv`: known gas-flare and factory locations, used to filter live fires

### The 28 features

| Group | Features |
|---|---|
| Air today (6) | PM2.5 average and highest hour, yesterday's PM2.5, change since yesterday, 3-day average, AOD |
| Tomorrow's weather (9) | rain at the 3 points; humidity and mixing height at Seri Iskandar; wind (u, v) at the Strait and Riau |
| Dryness (2) | days without rain at Seri Iskandar and Riau |
| Fires (9) | upwind count and strength (today and 3-day total), upwind within 800 km, Sumatra count and 3-day strength, Peninsula count, strength within 300 km |
| Season (2) | time of year (sin/cos) |

"Upwind" means within 45° of the direction the wind over the Strait comes from.

### Categories (daily average PM2.5)

| Category | PM2.5 | Basis |
|---|---|---|
| Good | ≤ 15 µg/m³ | WHO 2021 24-hour guideline |
| Moderate | 15–35 µg/m³ | above WHO, within Malaysia's standard |
| Unhealthy | > 35 µg/m³ | above Malaysia's 2020 24-hour standard |

### Results (test year 2026, 323 days, 28 haze days)

| Method | Right category | Haze days caught | False alarms | Avg error |
|---|---|---|---|---|
| "Same as today" baseline | 72% | 19 | 9 | 5.4 µg/m³ |
| Model without fires | 74% | 15 | 3 | 4.7 µg/m³ |
| **Final model (28 features)** | **76%** | **16** | **3** | **4.6 µg/m³** |
| 2 days ahead | 65% | 4 | 4 | – |
| 3 days ahead | 65% | 1 | 2 | – |

Upwind fires are clearly linked to next-day PM2.5 (correlation +0.45), more than Sumatra fires overall (+0.33).

---

## Part 2: Daily prediction (`dashboard/`)

Each time the page loads (data cached for 1 hour):

1. **Get live data:**
   - PM2.5 for the last 7 days + forecast (1)
   - fires for the last 5 days (3)
   - weather for the last 60 days + 4-day forecast (5)
   - DOE readings (7)
2. **Choose the base day.**
   - After 4 pm: today.
   - Before 4 pm: yesterday. Today's afternoon satellite pass (~1:30 pm) reaches FIRMS only a few hours later.
3. **Build the 28 features** with `features.py`, after removing known gas flares (`static_cells.csv`).
4. **Predict** PM2.5 for 1, 2 and 3 days ahead and convert it to a category.
5. **Show:**
   - forecast card with advice
   - reasons (today's PM2.5, upwind fires, wind, rain)
   - 2–3 day outlook (marked less certain)
   - fire map with the upwind zone and wind arrow
   - PM2.5 trend chart
   - official DOE readings

---

## How to run

Tested on Windows 11 with Python 3.14, scikit-learn 1.9, pandas 3.0, Streamlit 1.58 and Plotly 6.7.
All commands are for PowerShell.

### Quick start (data and model already in `experiments/`)

If `experiments/model.pkl` and `experiments/static_cells.csv` already exist, you only need steps 1, 2 and 6 below:

```powershell
pip install pandas numpy scikit-learn requests streamlit plotly truststore
python -m streamlit run dashboard/app.py
```

Then open http://localhost:8501.

### Full run from scratch

**Step 1. Install the Python packages.**

```powershell
pip install pandas numpy scikit-learn requests streamlit plotly truststore
```

`truststore` makes Python use the Windows certificate store, which avoids SSL errors on some PCs.

**Step 2. Get a free NASA FIRMS key.**

1. Request a MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/ (it arrives by email).
2. Save only the key, on one line, in `experiments/firms_key.txt`:

```powershell
Set-Content -Encoding ascii experiments/firms_key.txt "YOUR_KEY_HERE"
```

Keep this file private; do not share or commit it.

**Step 3. Download the training data.** Run these from the `experiments` folder:

```powershell
cd experiments
```

PM2.5 and AOD (source 1) → `aq.json`:

```powershell
curl.exe -o aq.json "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=4.385&longitude=100.979&hourly=pm2_5,aerosol_optical_depth&start_date=2022-08-04&end_date=2026-11-20&timezone=Asia/Kuala_Lumpur"
```

Past weather at the 3 points (source 4) → `wx.json`:

```powershell
curl.exe -o wx.json "https://archive-api.open-meteo.com/v1/archive?latitude=4.385,3.0,0.5&longitude=100.979,100.5,101.5&start_date=2022-08-04&end_date=2026-11-20&hourly=precipitation,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height&timezone=Asia/Kuala_Lumpur"
```

Fires 2022–2025 (source 2) → `firms/` (8 files, no key needed):

```powershell
New-Item -ItemType Directory -Force firms; foreach ($y in 2022..2025) { foreach ($c in "Indonesia","Malaysia") { curl.exe -o "firms/viirs-jpss1_${y}_$c.csv" "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-jpss1/$y/viirs-jpss1_${y}_$c.csv" } }
```

Fires 2026 (source 3) → `firms/area_2026.csv` (uses your key; takes a few minutes):

```powershell
python fetch_firms_2026.py
```

Then go back to the project folder:

```powershell
cd ..
```

To train on newer data, change `end_date` in the two Open-Meteo links above.

**Step 4. Train, test and save the model.**

```powershell
python experiments/model_fires.py
```

This prints the 2026 test results (the Results table above) and saves `experiments/model.pkl` and `experiments/static_cells.csv`.

**Step 5 (optional). Re-check the feature selection.**

```powershell
python experiments/feature_check.py
```

This prints the cross-validation error for all 108 features, the 28 selected features and the sets with each group removed.

**Step 6. Start the dashboard.**

```powershell
python -m streamlit run dashboard/app.py
```

Open http://localhost:8501. The first load takes about 10–20 seconds while it downloads live data. After that, the data is cached for 1 hour.

### Common problems

| Problem | Fix |
|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED` | `pip install truststore` (the dashboard uses it automatically) |
| "NASA FIRMS key missing" or "FIRMS did not return data" | Check that `experiments/firms_key.txt` holds only the key |
| `FileNotFoundError` for `model.pkl` or `static_cells.csv` | Run Step 4 first |
| "Could not download live data" | Check your internet connection and reload; one of the free APIs may be briefly down |
| `streamlit` is not recognised | Use `python -m streamlit run ...` instead of `streamlit run ...` |
| Port 8501 already in use | Add `--server.port 8502` and open http://localhost:8502 |

## Project files

| File | Purpose |
|---|---|
| `experiments/features.py` | Shared feature code (training + dashboard) |
| `experiments/model_fires.py` | Clean data, train, test, save `model.pkl` and `static_cells.csv` |
| `experiments/feature_check.py` | Feature-selection check (108 vs 28) |
| `experiments/fetch_firms_2026.py` | Download 2026 fires from FIRMS |
| `experiments/aq.json`, `wx.json`, `firms/` | Downloaded data |
| `dashboard/app.py`, `dashboard/live.py` | Dashboard page and live data |
| `data_input_plan.md` | Data plan and decisions |

---

## Limitations

- **PM2.5 comes from CAMS, a model (~45 km grid), not a ground sensor.** Checking it against the DOE stations (source 6) is still to do.
  - On 22 Nov 2026, DOE showed Moderate (API 48–65) while CAMS gave about 12 µg/m³ (Good).
  - Part of this is the different scale; part may be CAMS reading low.
- **Few haze days to learn from before 2026**, so the model under-predicts sudden or extreme haze.
- **Training used actual past weather; the dashboard uses forecasts**, so live accuracy may be a little lower.
- **Satellites miss fires under cloud or thick smoke.**
- **This is an estimate**, not an official warning and not medical advice. Always follow DOE announcements.
