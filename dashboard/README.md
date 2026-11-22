# SmokeSignal dashboard

Next-day haze forecast for Seri Iskandar / UTP.
For the full solution flow (data, training, feature selection, results), see the [project README](../README.md).

## Run

```
python -m streamlit run dashboard/app.py
```

Run it from the project folder, then open http://localhost:8501.

## Needs (all in `experiments/`)

- `model.pkl` and `static_cells.csv`: created by `python experiments/model_fires.py`
- `firms_key.txt`: your free NASA FIRMS MAP_KEY

## What it does each time it loads

1. Downloads, cached for 1 hour:
   - PM2.5 from Open-Meteo/CAMS
   - weather forecast for 3 points from Open-Meteo
   - the last 5 days of fires from NASA FIRMS
   - current DOE readings
2. Builds the 28 features with `experiments/features.py`, the same code used in training.
3. Predicts PM2.5 for the next 3 days and shows the category, advice, reasons, a fire map and a trend chart.

Before 4 pm, the forecast is made from **yesterday's** conditions (so day 1 = today). This is because the afternoon satellite pass reaches NASA FIRMS only a few hours after ~1:30 pm.
