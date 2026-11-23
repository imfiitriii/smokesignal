"""SmokeSignal dashboard - next-day haze forecast for Seri Iskandar / UTP.

Run:  python -m streamlit run dashboard/app.py
Needs: experiments/model.pkl, experiments/static_cells.csv (from model_fires.py)
       experiments/firms_key.txt (free NASA FIRMS MAP_KEY)
"""
import os
import joblib, numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live
from features import POINTS, UTP, CATEGORIES, wind_from_strait

st.set_page_config(page_title="SmokeSignal | Seri Iskandar haze forecast", layout="wide")

COLORS = {0: "#2f8f4e", 1: "#c98a00", 2: "#c0392b"}
ADVICE = {
    0: "No precautions needed.",
    1: "People with asthma, heart or lung conditions, children and older adults should cut down on long "
       "or strenuous outdoor activity.",
    2: "Limit time outdoors. Wear an N95 mask outside, keep windows closed and check on vulnerable family members.",
}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
def compass(deg): return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1180px;}
  [data-testid="stHeader"] {display: none;}
  h1, h2, h3 {font-weight: 600; letter-spacing: 0;}
  .top {display:flex; justify-content:space-between; align-items:baseline; border-bottom:2px solid #1f2328;
        padding-bottom:6px; margin-bottom:14px;}
  .top .name {font-size:24px; font-weight:700;}
  .top .meta {font-size:13px; color:#57606a; text-align:right;}
  .strip {display:grid; grid-template-columns: 1.6fr 1fr 1fr; border:1px solid #d0d7de;}
  .day {padding:14px 18px; border-right:1px solid #d0d7de;}
  .day:last-child {border-right:none;}
  .day .when {font-size:13px; color:#57606a;}
  .day .cat {font-size:22px; font-weight:600; margin-top:4px;}
  .day.main .cat {font-size:34px;}
  .day .sw {display:inline-block; width:12px; height:12px; margin-right:8px; vertical-align:middle;}
  .day.main .sw {width:18px; height:18px;}
  .day .pm {font-size:14px; margin-top:2px;}
  .day .conf {font-size:12px; color:#57606a; margin-top:6px;}
  .day.low .cat, .day.low .pm {color:#57606a;}
  .advice {border-left:4px solid; padding:6px 12px; margin:12px 0 4px 0; font-size:15px;}
  table.inp {width:100%; border-collapse:collapse; font-size:14px;}
  table.inp th {text-align:left; font-weight:600; color:#57606a; border-bottom:1px solid #1f2328; padding:4px 6px;}
  table.inp td {border-bottom:1px solid #eaeef2; padding:5px 6px;}
  table.inp td.n {text-align:right; font-variant-numeric: tabular-nums;}
  table.inp th.n {text-align:right;}
  .note {font-size:12px; color:#57606a;}
</style>
""", unsafe_allow_html=True)


# ---------------- data (cached for 1 hour) ----------------
@st.cache_resource
def load_model():
    b = joblib.load(os.path.join(live.EXP, "model.pkl"))
    static = set(pd.read_csv(os.path.join(live.EXP, "static_cells.csv")).cell)
    return b, static

@st.cache_data(ttl=3600, show_spinner="Loading air quality data")
def air(): return live.fetch_air()

@st.cache_data(ttl=3600, show_spinner="Loading weather forecast")
def weather(): return live.fetch_weather()

@st.cache_data(ttl=3600, show_spinner="Loading fire hotspots")
def fires(key): return live.fetch_fires(key)

@st.cache_data(ttl=1800, show_spinner="Loading DOE readings")
def doe(): return live.fetch_doe()


bundle, static_cells = load_model()
key_path = os.path.join(live.EXP, "firms_key.txt")
if not os.path.exists(key_path):
    st.error("NASA FIRMS key missing: save it in experiments/firms_key.txt "
             "(free at https://firms.modaps.eosdis.nasa.gov/api/map_key/).")
    st.stop()
key = open(key_path).read().strip()

try:
    air_h, wx, fire_raw = air(), weather(), fires(key)
except Exception as e:
    st.error(f"Could not download live data: {e}")
    st.stop()

now = pd.Timestamp.now(tz=live.TZ)
today = now.normalize().tz_localize(None)
# The afternoon satellite pass (~1:30 pm) reaches FIRMS a few hours later, so before 4 pm today's
# fires are incomplete: then the forecast is made from yesterday (D+1 = today).
base = today if now.hour >= 16 else today - pd.Timedelta(days=1)

out, feats, daily, wd, fire = live.forecast(bundle, static_cells, air_h, wx, fire_raw, base)


# ---------------- header ----------------
st.markdown(
    f"""<div class="top"><div class="name">SmokeSignal <span style="font-weight:400;color:#57606a">
    Haze forecast, Seri Iskandar &amp; UTP</span></div>
    <div class="meta">Issued {now:%d %b %Y, %H:%M}<br>Based on data up to {base:%d %b}</div></div>""",
    unsafe_allow_html=True)

main = out.get(1, {})
if "error" in main:
    st.error(f"Cannot make the forecast: {main['error']}")
    st.stop()


def day_name(d):
    if d == today: return "Today"
    if d == today + pd.Timedelta(days=1): return "Tomorrow"
    return f"{d:%A}"


# ---------------- forecast strip ----------------
cells = []
for h in (1, 2, 3):
    o = out.get(h, {})
    cls = "day main" if h == 1 else "day low"
    if "pm" not in o:
        cells.append(f'<div class="{cls}"><div class="when">Day {h}</div><div class="cat">n/a</div></div>')
        continue
    conf = "Main forecast" if h == 1 else "Low confidence"
    cells.append(
        f"""<div class="{cls}"><div class="when">{day_name(o['date'])}, {o['date']:%d %b}</div>
        <div class="cat"><span class="sw" style="background:{COLORS[o['cat']]}"></span>{CATEGORIES[o['cat']]}</div>
        <div class="pm">PM2.5 {o['pm']:.0f} µg/m³ (daily average)</div><div class="conf">{conf}</div></div>""")
st.markdown(f'<div class="strip">{"".join(cells)}</div>', unsafe_allow_html=True)

c = main["cat"]
st.markdown(f'<div class="advice" style="border-color:{COLORS[c]}">{ADVICE[c]}</div>', unsafe_allow_html=True)
s2 = bundle.get("outlook_scores", {}).get(2)
if s2:
    st.markdown(f'<div class="note">Days 2 and 3 are rough. In the 2026 test, the 2-day forecast caught '
                f'{s2["caught"]} of {s2["unhealthy"]} unhealthy days.</div>', unsafe_allow_html=True)

st.write("")

# ---------------- inputs table + map ----------------
ref = bundle["reference"]
wfrom = float(wind_from_strait(wd).get(base, np.nan))
tgt = main["date"]

def row(label, value, name=None, unit="", fmt="{:.0f}"):
    usual = fmt.format(ref[name][0.5]) + unit if name in ref else ""
    return f'<tr><td>{label}</td><td class="n">{value}</td><td class="n">{usual}</td></tr>'

rows = [
    row(f"PM2.5, {base:%d %b}", f"{feats.pm:.0f} µg/m³", "pm", " µg/m³"),
    row("Fire hotspots upwind, that day", f"{int(feats.n_upwind)}", "n_upwind"),
    row("Fire hotspots upwind, last 3 days", f"{int(feats.n_upwind_3d)}", "n_upwind_3d"),
    row("Fire hotspots in Sumatra", f"{int(feats.n_sumatra)}", "n_sumatra"),
    row("Fire hotspots in Peninsular Malaysia", f"{int(feats.n_peninsula)}", "n_peninsula"),
    row("Wind over the Strait, from", f"{compass(wfrom)} ({wfrom:.0f}°)" if not np.isnan(wfrom) else "n/a"),
    row(f"Rain over the Strait, {tgt:%d %b} (forecast)", f"{feats.rain_strait_tgt:.1f} mm", "rain_strait_tgt", " mm", "{:.1f}"),
    row(f"Rain at Seri Iskandar, {tgt:%d %b} (forecast)", f"{feats.rain_local_tgt:.1f} mm", "rain_local_tgt", " mm", "{:.1f}"),
]

left, right = st.columns([1, 1.05], gap="large")
with left:
    st.subheader("Inputs")
    st.markdown(f'<table class="inp"><tr><th>Measure</th><th class="n">Value</th><th class="n">Usual</th></tr>'
                f'{"".join(rows)}</table>', unsafe_allow_html=True)
    st.markdown('<div class="note" style="margin-top:6px">Usual = median day, Aug 2022 to Nov 2026. '
                'Upwind = within 45° of the direction the wind over the Strait comes from. '
                'Rain washes smoke out of the air.</div>', unsafe_allow_html=True)

with right:
    st.subheader(f"Fire hotspots, {base:%d %b}")
    today_f = fire[fire.date == base]
    fig = go.Figure()
    if not np.isnan(wfrom):          # shaded upwind sector: fires here can blow towards Seri Iskandar
        angs = np.linspace(wfrom - 45, wfrom + 45, 30)
        R = 1500 / 111.0
        lat = [UTP[0]] + [UTP[0] + R * np.cos(np.deg2rad(a)) for a in angs] + [UTP[0]]
        lon = [UTP[1]] + [UTP[1] + R * np.sin(np.deg2rad(a)) / np.cos(np.deg2rad(UTP[0])) for a in angs] + [UTP[1]]
        fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", fill="toself", name="Upwind zone",
                                    fillcolor="rgba(192,57,43,0.07)", line=dict(color="rgba(192,57,43,0.35)", width=1),
                                    hoverinfo="skip"))
    n_up = int(today_f.upwind.sum())
    for up, colr, name in [(False, "#8c959f", f"Other hotspots ({len(today_f) - n_up})"),
                           (True, "#c0392b", f"Upwind hotspots ({n_up})")]:
        d = today_f[today_f.upwind == up]
        fig.add_trace(go.Scattermap(lat=d.latitude, lon=d.longitude, mode="markers", name=name,
                                    marker=dict(size=6, color=colr, opacity=0.85),
                                    hovertext=[f"{r:.0f} km away, FRP {f:.0f} MW" for r, f in zip(d.dist, d.frp)],
                                    hoverinfo="text"))
    if not np.isnan(wfrom):          # wind over the Strait, dot marks the downwind end
        s = POINTS["strait"]; L = 2.2
        a0, a1 = np.deg2rad(wfrom), np.deg2rad(wfrom + 180)
        fig.add_trace(go.Scattermap(lat=[s[0] + L*np.cos(a0), s[0] + L*np.cos(a1)],
                                    lon=[s[1] + L*np.sin(a0), s[1] + L*np.sin(a1)],
                                    mode="lines+markers", name="Wind (dot = blowing towards)",
                                    line=dict(color="#1d4e89", width=3), marker=dict(size=[0, 11], color="#1d4e89"),
                                    hoverinfo="skip"))
    fig.add_trace(go.Scattermap(lat=[UTP[0]], lon=[UTP[1]], mode="markers+text", name="Seri Iskandar",
                                marker=dict(size=12, color="#1f2328"), text=["Seri Iskandar"],
                                textposition="top right", hoverinfo="skip"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=2.0, lon=102.0), zoom=4.3),
                      margin=dict(l=0, r=0, t=0, b=0), height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    def key(color, text, shape="border-radius:50%"):
        return (f'<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:9px;'
                f'height:9px;background:{color};{shape};margin-right:5px;vertical-align:middle"></span>{text}</span>')
    st.markdown('<div class="note">'
                + key("#c0392b", f"Upwind hotspots ({n_up})")
                + key("#8c959f", f"Other hotspots ({len(today_f) - n_up})")
                + key("rgba(192,57,43,0.25)", "Upwind zone", "")
                + key("#1d4e89", "Wind (dot = blowing towards)", "height:3px")
                + '</div><div class="note">Source: NASA FIRMS, VIIRS NOAA-20. '
                  'Gas flares and other fixed heat sources removed.</div>', unsafe_allow_html=True)

# ---------------- PM2.5 trend ----------------
st.subheader("PM2.5, past week and forecast")
hist = daily.loc[base - pd.Timedelta(days=6): base, "pm"]
fc = pd.Series({o["date"]: o["pm"] for o in out.values() if "pm" in o})
f2 = go.Figure()
top = max(45, float(max(hist.max(), fc.max() if len(fc) else 0)) + 10)
for lo_, hi_, k in [(0, 15, 0), (15, 35, 1), (35, top, 2)]:
    f2.add_hrect(y0=lo_, y1=hi_, fillcolor=COLORS[k], opacity=0.06, line_width=0,
                 annotation_text=CATEGORIES[k], annotation_position="top right",
                 annotation_font=dict(size=11, color="#57606a"))
f2.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Measured (CAMS)",
                        line=dict(color="#1f2328", width=2)))
f2.add_trace(go.Scatter(x=[hist.index[-1]] + list(fc.index), y=[hist.values[-1]] + list(fc.values),
                        mode="lines+markers", name="Forecast",
                        line=dict(color="#1d4e89", width=2, dash="dash"), marker=dict(size=8)))
f2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="µg/m³, daily average",
                 yaxis_range=[0, top], legend=dict(orientation="h", y=1.1, x=0),
                 plot_bgcolor="white", xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#eaeef2"))
st.plotly_chart(f2, use_container_width=True)

# ---------------- official readings ----------------
st.subheader("DOE readings, nearest stations")
try:
    d = doe()
    if len(d):
        st.dataframe(d[["Station", "API", "Category", "Updated"]], hide_index=True, use_container_width=True)
        st.markdown('<div class="note">Air Pollutant Index from DOE APIMS. The API scale is not the same as '
                    'PM2.5 in µg/m³, so the two cannot be compared directly. DOE announcements take priority '
                    'over this forecast.</div>', unsafe_allow_html=True)
    else:
        st.write("DOE readings are not available right now.")
except Exception:
    st.write("DOE readings are not available right now.")

# ---------------- about ----------------
with st.expander("Method and data"):
    st.markdown(f"""
A random forest model predicts the next day's average PM2.5 from 28 inputs: recent PM2.5, fire hotspots
(mainly upwind ones), the forecast rain, wind and humidity for that day, and the time of year.

Categories, daily average PM2.5: Good up to 15 µg/m³ (WHO 2021 guideline), Moderate 15 to 35,
Unhealthy above 35 (Malaysian 24-hour standard, 2020).

Tested on 2026 data the model had not seen: correct category on 76% of days, 16 of 28 unhealthy days
caught, 3 false alarms. It tends to miss the first day of a sudden haze episode.

Data: Open-Meteo / Copernicus CAMS (air quality, about 45 km grid), NASA FIRMS VIIRS NOAA-20 (fires),
Open-Meteo (weather), DOE APIMS (official readings). Model trained on data up to {bundle['trained_until']}.

This is an estimate, not an official warning or medical advice.
""")
