# SmokeSignal – Final Data Sources & Solution Flow (updated 21 Nov 2026)

Location: Seri Iskandar / UTP ≈ **4.385 N, 100.979 E**
Prediction: **tomorrow's air quality** (Good / Moderate / Unhealthy), with a light 2–3 day outlook.
All data is free, open and was downloaded and tested in `experiments/`.

---

## 1. Data sources

| # | Data | Source | Used for | Period | Access |
|---|---|---|---|---|---|
| 1 | **PM2.5 + aerosol optical depth (hourly)** | Open-Meteo Air Quality API (Copernicus CAMS) – https://open-meteo.com/en/docs/air-quality-api | Target (what we predict) + today's air level | 4 Aug 2022 → today | Free, no key |
| 2 | **Fire hotspots (VIIRS NOAA-20)**, 2022–2025 | NASA FIRMS country yearly CSVs – https://firms.modaps.eosdis.nasa.gov/country/ | Fire features | 2022–2025 | Free, no login |
| 3 | **Fire hotspots (VIIRS NOAA-20)**, 2026 + live | NASA FIRMS Area API – https://firms.modaps.eosdis.nasa.gov/api/ | Fire features (test year + dashboard) | 2026 → today | Free MAP_KEY, **max 5 days per request** |
| 4 | **Weather (hourly)**: rain, humidity, wind, boundary-layer height at 3 points | Open-Meteo Historical Weather API (ERA5) – https://open-meteo.com/en/docs/historical-weather-api | Weather features (training) | 2022 → today | Free, no key |
| 5 | **Weather forecast** (same variables, same 3 points) | Open-Meteo Forecast API – https://open-meteo.com/en/docs | Weather features (live dashboard) | next 1–16 days | Free, no key |
| 6 | **Real station readings** (Pegoh Ipoh, Jalan Tasek Ipoh, Seri Manjung) | WAQI (data from Malaysia DOE) – station pages on aqicn.org, "Download this data (CSV)" | Check that CAMS matches reality | Jul 2025 → today | Free, manual download |
| 7 | **Official DOE API (live)** | DOE APIMS – https://eqms.doe.gov.my/APIMS/main | Display only on dashboard | last 24–48 h | Free, public |

Weather points: **Seri Iskandar** (4.385, 100.979), **Strait of Malacca** (3.0, 100.5), **Riau, Sumatra** (0.5, 101.5).

Dropped from the original proposal:
- WAQI 2015–2024: not available (only ~Jul 2025 onward).
- DOE historical data: needs a formal request letter, too slow.
- VIIRS S-NPP: stopped 1 Nov 2026.

---

## 2. Solution flow

### A. Training (once)
1. **Download** sources 1, 2, 3 and 4.
2. **Clean fires**:
   - Drop low-confidence detections.
   - Keep Indonesia and Malaysia only (Sumatra, Peninsula, Borneo).
   - Remove static heat sources: any spot "burning" on more than 60 days a year.
   - Convert UTC times to Malaysia time.
3. **Build one row per day**:
   - **Air:** today's PM2.5 (mean and max), yesterday's, 3-day mean, AOD.
   - **Fires:** count and intensity (FRP) by distance (0–300, 300–600, 600–1500 km), by region (Sumatra, Peninsula, Borneo), and **upwind** (fires in the direction the Strait wind comes from, ±45°). Include yesterday's values and 3-day totals.
   - **Weather:** today's and tomorrow's rain, humidity, wind (u/v), boundary-layer height, and dry-day streak at the 3 points.
   - **Season:** day-of-year (sin/cos).
4. **Label**: the answer for each row = **tomorrow's** average PM2.5.
5. **Train a Random Forest** to predict *how much PM2.5 will change from today*, then add that to today's value.
6. **Convert to category**:
   - Good ≤ 15 µg/m³ (WHO 24-h guideline)
   - Moderate 15–35
   - Unhealthy > 35 (Malaysia 24-h standard)
7. **Test** on 2026 (unseen year, Aug–Oct haze) and compare with the "same as today" baseline. Optionally also test against the real stations (source 6).

### B. Daily prediction (dashboard, every evening)
1. Fetch today's PM2.5 (source 1), the fires for the last 3 days (source 3), the weather forecast (source 5) and the DOE API (source 7).
2. Build today's row **with the same code** as in training.
3. Run the model to get tomorrow's PM2.5, then convert it to a category with advice text.
4. Dashboard shows:
   - a map of fires and the wind arrow
   - tomorrow's category, estimated PM2.5 and advice
   - the D+2/D+3 outlook
   - today's official DOE API
   - the label "estimate only"

---

## 2b. Final feature set (28 features)
How they were selected:
1. Build 108 candidates.
2. Group them by idea (air, rain, wind, mixing, dryness, upwind fires, nearby fires, season).
3. Keep 1–2 per idea **by hand**, dropping near-duplicates.
4. Check the set with leave-one-year-out cross-validation on 2022–2025 only. It must match the error of all 108, and removing any group must hurt.
5. 2026 is kept out of the selection and used once, for the final test.

- **Air (6):** today's PM2.5 mean, max, yesterday's PM2.5, change since yesterday, 3-day mean, AOD
- **Weather for tomorrow (9):**
  - rain at Seri Iskandar, Strait and Riau
  - humidity at Seri Iskandar
  - boundary-layer height at Seri Iskandar
  - wind (u, v) at the Strait and at Riau
- **Dryness (2):** dry-day streak at Seri Iskandar and at Riau
- **Fires (9):**
  - upwind count and FRP (today and 3-day totals)
  - upwind count within 800 km
  - Sumatra count, Sumatra FRP over 3 days
  - Peninsula count
  - FRP within 300 km
- **Season (2):** sin and cos of day-of-year

Comparison (`experiments/feature_check.py`):
- All 108 features: train-CV error 3.10; 2026: 75%, 17/28 haze caught, 3 false alarms
- Compact 28 features: train-CV error 3.09; 2026: 76%, 16/28 caught, 3 false alarms (FINAL)

The two sets perform the same, so use the compact 28: simpler, faster and easier to explain. Removing any whole group (fire, weather or air) makes the error worse.

## 3. Test results (train 2022–2025, test 2026, next day)

| Model | Haze days caught (of 28) | False alarms | Category accuracy | Avg error |
|---|---|---|---|---|
| "Same as today" baseline | 19 | 9 | 72% | 5.4 µg/m³ |
| Random Forest, no fires | 15 | 3 | 74% | 4.7 |
| Random Forest + fires, all 108 features | 17 | 3 | 75% | 4.6 |
| **Random Forest + fires, 28 selected (FINAL)** | **16** | **3** | **76%** | **4.6** |

Link between features and tomorrow's PM2.5 (correlation r):
- upwind fires **+0.45**
- all Sumatra fires +0.33
- fires within 300 km +0.34
- today's PM2.5 +0.75

D+2 and D+3 forecasts are much weaker (with no fire data, they caught 3 or fewer of the 28 haze days). Use them only as an outlook.

After testing, the final model is retrained on all days up to Nov 2026 (including the 2026 haze) and saved as `experiments/model.pkl`. The list of known gas-flare and factory locations is saved as `experiments/static_cells.csv`; the dashboard uses it to filter live fires.

## 4. Limitations to state
- The target is CAMS (model output, ~45 km grid), not a ground sensor. Check it against the DOE stations.
- 2022–2025 had only ~10 haze days to learn from, so the model under-predicts extreme haze.
- Training uses actual past weather; the live dashboard uses forecast weather, so live accuracy will be slightly lower.
- Satellites miss fires under cloud or thick smoke.

## 5. Files
- `experiments/aq.json`, `experiments/wx.json`: CAMS and ERA5 downloads
- `experiments/firms/`: fire CSVs (2022–2025 country files + `area_2026.csv`)
- `experiments/fetch_firms_2026.py`: downloads 2026 fires (needs `firms_key.txt`)
- `experiments/model_fires.py`: builds features, trains the model, prints results
