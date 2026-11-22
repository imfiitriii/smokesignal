"""SmokeSignal dashboard - next-day haze forecast for Seri Iskandar / UTP.

Run:  streamlit run dashboard/app.py
Needs: experiments/model.pkl, experiments/static_cells.csv (from model_fires.py)
       experiments/firms_key.txt (free NASA FIRMS MAP_KEY)
"""
import os
import joblib, numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live
from features import POINTS, UTP, CATEGORIES, wind_from_strait

st.set_page_config(page_title="SmokeSignal – Seri Iskandar haze forecast", page_icon="🌫️", layout="wide")

COLORS = {0: "#2e9e5b", 1: "#e0a800", 2: "#d64545"}
ADVICE = {
    0: "Air is expected to be good. Normal outdoor activities are fine.",
    1: "Children, the elderly and people with asthma or heart conditions should limit long or heavy outdoor activity.",
    2: "Move outdoor activities indoors where possible. Wear a mask (N95) outside, keep windows closed, "
       "and check on vulnerable family members.",
}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
def compass(deg): return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


# ---------------- data (cached for 1 hour) ----------------
@st.cache_resource
def load_model():
    b = joblib.load(os.path.join(live.EXP, "model.pkl"))
    static = set(pd.read_csv(os.path.join(live.EXP, "static_cells.csv")).cell)
    return b, static

@st.cache_data(ttl=3600, show_spinner="Getting air quality (Open-Meteo / CAMS)…")
def air(): return live.fetch_air()

@st.cache_data(ttl=3600, show_spinner="Getting weather forecast (Open-Meteo)…")
def weather(): return live.fetch_weather()

@st.cache_data(ttl=3600, show_spinner="Getting fire hotspots (NASA FIRMS)…")
def fires(key): return live.fetch_fires(key)

@st.cache_data(ttl=1800, show_spinner="Getting official DOE readings…")
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
st.title("SmokeSignal")
st.caption(f"Haze early warning for **Seri Iskandar & UTP** · updated {now:%d %b %Y, %H:%M} "
           f"· forecast made from conditions on {base:%d %b}")

main = out.get(1, {})
if "error" in main:
    st.error(f"Cannot make tomorrow's forecast: {main['error']}")
    st.stop()

# ---------------- main forecast card ----------------
c = main["cat"]
day_word = "Today" if main["date"] == today else ("Tomorrow" if main["date"] == today + pd.Timedelta(days=1)
                                                  else f"{main['date']:%A}")
st.markdown(
    f"""<div style="border-radius:14px;padding:22px 26px;background:{COLORS[c]}1f;border:2px solid {COLORS[c]}">
    <div style="font-size:15px;opacity:.8">{day_word}, {main['date']:%A %d %b}</div>
    <div style="font-size:40px;font-weight:700;color:{COLORS[c]};line-height:1.2">{CATEGORIES[c]}</div>
    <div style="font-size:17px;margin-top:2px">Estimated PM2.5: <b>{main['pm']:.0f} µg/m³</b> (daily average)</div>
    <div style="font-size:17px;margin-top:12px">{ADVICE[c]}</div></div>""",
    unsafe_allow_html=True)

# ---------------- why ----------------
ref = bundle["reference"]
def level(name, v):
    q50, q90 = ref[name][0.5], ref[name][0.9]
    return "very high" if v >= q90 else ("above normal" if v > q50 else "low")

wfrom = float(wind_from_strait(wd).get(base, np.nan))
reasons = [
    f"Today's PM2.5 is about **{feats.pm:.0f} µg/m³** ({CATEGORIES[int(feats.pm > 15) + int(feats.pm > 35)]}).",
    f"**{int(feats.n_upwind)} fire hotspots upwind** today ({int(feats.n_upwind_3d)} over 3 days) – "
    f"{level('n_upwind_3d', feats.n_upwind_3d)} for this area.",
    f"Wind over the Strait of Malacca is coming from the **{compass(wfrom)}** ({wfrom:.0f}°)."
    if not np.isnan(wfrom) else "Wind direction over the Strait is unavailable.",
    (f"**{feats.rain_strait_tgt:.0f} mm of rain** expected over the Strait – rain washes smoke out of the air."
     if feats.rain_strait_tgt >= 5 else
     f"Little or no rain expected over the Strait ({feats.rain_strait_tgt:.1f} mm), so smoke is not washed out."),
    f"{int(feats.n_sumatra)} hotspots in Sumatra and {int(feats.n_peninsula)} in Peninsular Malaysia today.",
]
left, right = st.columns([1.1, 1])
with left:
    st.subheader("Why")
    st.markdown("\n".join(f"- {r}" for r in reasons))

    st.subheader("Outlook (less certain)")
    cols = st.columns(2)
    scores = bundle.get("outlook_scores", {})
    for col, h in zip(cols, [2, 3]):
        o = out.get(h, {})
        with col:
            if "pm" not in o:
                st.info(f"Day {h}: not available")
                continue
            st.markdown(
                f"""<div style="border-radius:10px;padding:12px 14px;border:1px dashed {COLORS[o['cat']]}">
                <div style="font-size:13px;opacity:.8">{o['date']:%A %d %b}</div>
                <div style="font-size:22px;font-weight:600;color:{COLORS[o['cat']]}">{CATEGORIES[o['cat']]}?</div>
                <div style="font-size:13px">est. {o['pm']:.0f} µg/m³</div></div>""", unsafe_allow_html=True)
    s2 = scores.get(2)
    if s2:
        st.caption(f"Beyond tomorrow the model rarely spots haze (2026 test: it caught {s2['caught']} of "
                   f"{s2['unhealthy']} haze days two days ahead). Check again tomorrow evening.")

# ---------------- map ----------------
with right:
    st.subheader("Fire hotspots today and wind")
    today_f = fire[fire.date == base]
    fig = go.Figure()
    if not np.isnan(wfrom):          # shaded "upwind" sector: fires here can blow towards Seri Iskandar
        angs = np.linspace(wfrom - 45, wfrom + 45, 30)
        R = 1500 / 111.0
        lat = [UTP[0]] + [UTP[0] + R * np.cos(np.deg2rad(a)) for a in angs] + [UTP[0]]
        lon = [UTP[1]] + [UTP[1] + R * np.sin(np.deg2rad(a)) / np.cos(np.deg2rad(UTP[0])) for a in angs] + [UTP[1]]
        fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", fill="toself", name="Upwind zone",
                                    fillcolor="rgba(214,69,69,0.08)", line=dict(color="rgba(214,69,69,0.4)")))
    for up, colr, name in [(False, "#999999", "Other hotspots"), (True, "#d64545", "Upwind hotspots")]:
        d = today_f[today_f.upwind == up]
        fig.add_trace(go.Scattermap(lat=d.latitude, lon=d.longitude, mode="markers", name=f"{name} ({len(d)})",
                                    marker=dict(size=6, color=colr, opacity=0.8),
                                    hovertext=[f"{r:.0f} km away, FRP {f:.0f} MW" for r, f in zip(d.dist, d.frp)]))
    if not np.isnan(wfrom):          # wind arrow over the Strait (blowing towards wfrom+180)
        s = POINTS["strait"]; L = 2.2
        a0, a1 = np.deg2rad(wfrom), np.deg2rad(wfrom + 180)
        fig.add_trace(go.Scattermap(lat=[s[0] + L*np.cos(a0), s[0] + L*np.cos(a1)],
                                    lon=[s[1] + L*np.sin(a0), s[1] + L*np.sin(a1)],
                                    mode="lines+markers+text", name="Wind (blows →)",
                                    line=dict(color="#1f6feb", width=4), marker=dict(size=[0, 14], color="#1f6feb"),
                                    text=["", "wind"], textposition="top right"))
    fig.add_trace(go.Scattermap(lat=[UTP[0]], lon=[UTP[1]], mode="markers+text", name="Seri Iskandar / UTP",
                                marker=dict(size=16, color="#111111"), text=["Seri Iskandar"],
                                textposition="top right"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=2.0, lon=102.0), zoom=4.3),
                      margin=dict(l=0, r=0, t=0, b=0), height=430, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    n_up = int(today_f.upwind.sum())
    st.caption(f"**Red dots** = {n_up} upwind hotspots · **grey dots** = {len(today_f) - n_up} other hotspots · "
               "**red shading** = upwind zone (smoke from here can drift to Seri Iskandar) · "
               "**blue line** = wind over the Strait (the dot shows where it blows to) · **black dot** = Seri Iskandar. "
               "Source: NASA FIRMS VIIRS NOAA-20.")

# ---------------- PM2.5 trend ----------------
st.subheader("PM2.5 – last week and forecast")
hist = daily.loc[base - pd.Timedelta(days=6): base, "pm"]
fc = pd.Series({o["date"]: o["pm"] for o in out.values() if "pm" in o})
f2 = go.Figure()
top = max(45, float(max(hist.max(), fc.max() if len(fc) else 0)) + 10)
for lo_, hi_, k in [(0, 15, 0), (15, 35, 1), (35, top, 2)]:
    f2.add_hrect(y0=lo_, y1=hi_, fillcolor=COLORS[k], opacity=0.08, line_width=0,
                 annotation_text=CATEGORIES[k], annotation_position="top left")
f2.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Past days (CAMS)",
                        line=dict(color="#555", width=3)))
f2.add_trace(go.Scatter(x=[hist.index[-1]] + list(fc.index), y=[hist.values[-1]] + list(fc.values),
                        mode="lines+markers", name="SmokeSignal forecast",
                        line=dict(color="#1f6feb", width=3, dash="dash"), marker=dict(size=10)))
f2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="PM2.5 daily average (µg/m³)",
                 yaxis_range=[0, top], legend=dict(orientation="h", y=1.08))
st.plotly_chart(f2, use_container_width=True)

# ---------------- official readings ----------------
st.subheader("Official DOE readings now (nearest stations)")
try:
    d = doe()
    if len(d):
        st.dataframe(d[["Station", "API", "Category", "Updated"]], hide_index=True, use_container_width=True)
        st.caption("Air Pollutant Index (API) from the Department of Environment (APIMS). It uses a different "
                   "scale from PM2.5 above and is shown for comparison only. Always follow official DOE warnings.")
    else:
        st.info("DOE readings unavailable right now.")
except Exception:
    st.info("DOE readings unavailable right now.")

# ---------------- about ----------------
with st.expander("About this forecast"):
    st.markdown(f"""
**How it works.** A Random Forest model looks at 28 inputs: today's air quality, fire hotspots (especially
those upwind), tomorrow's rain, wind and humidity, and the time of year. It predicts tomorrow's average PM2.5.

**Categories (daily average PM2.5).** Good ≤ 15 µg/m³ (WHO 2021 guideline) · Moderate 15–35 ·
Unhealthy > 35 (Malaysia's 2020 24-hour standard).

**How accurate is it?** Tested on 2026 (a year it never saw): right category on 76% of days, warned
about 16 of 28 haze days, with 3 false alarms. It can miss very sudden haze.

**Data.** Air quality: Open-Meteo / Copernicus CAMS (model, ~45 km grid). Fires: NASA FIRMS (VIIRS NOAA-20).
Weather: Open-Meteo. Official readings: DOE APIMS. Model trained up to {bundle['trained_until']}.

⚠️ **This is an estimate, not an official warning, and not medical advice.**
""")
