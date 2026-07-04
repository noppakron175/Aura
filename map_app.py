import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime

# ============================================================
# CONFIG — EDIT THIS
# ============================================================
# Publish your Google Sheet to the web as CSV:
# File -> Share -> Publish to web -> choose the sheet -> CSV -> Publish
# Paste the resulting URL below.
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/gviz/tq?tqx=out:csv&sheet=Sheet1"

REFRESH_SECONDS = 60  # auto-refresh interval

st.set_page_config(
    page_title="Air Quality Map",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# zero-dependency auto refresh (same trick as Aurafarm)
st.markdown(f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>", unsafe_allow_html=True)

# ============================================================
# DARK THEME / MOBILE-FIRST CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background: #1a1d24; border-radius: 12px; padding: 16px;
        text-align: center; border: 1px solid #2a2d34;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #9aa0a6; font-weight: 500; }
    .metric-card p { margin: 4px 0 0 0; font-size: 1.8rem; font-weight: 700; }
    .aqi-badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; color: #000;
    }
    @media (max-width: 640px) {
        .metric-card p { font-size: 1.3rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=REFRESH_SECONDS)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in ["LAT", "LON", "TVOC", "HP0", "HP3", "MQ135", "MQ7", "PM2.5", "PM10"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["LAT", "LON", "Timestamp"])
    return df


def load_demo_data() -> pd.DataFrame:
    """Fallback sample data so the app is viewable before the sheet is connected."""
    rng = np.random.default_rng(42)
    stations = [
        (18.7883, 98.9853, "Chiang Mai City"),
        (18.5747, 98.9339, "Hang Dong"),
        (18.5023, 98.8853, "Pa Sang"),
        (18.9088, 99.0044, "San Sai"),
    ]
    rows = []
    now = pd.Timestamp.now()
    for lat, lon, _name in stations:
        for i in range(5):
            rows.append({
                "Timestamp": now - pd.Timedelta(minutes=10 * i),
                "LAT": lat + rng.normal(0, 0.001),
                "LON": lon + rng.normal(0, 0.001),
                "TVOC": rng.uniform(50, 800),
                "HP0": rng.uniform(0.8, 1.5),
                "HP3": rng.uniform(0.8, 1.5),
                "MQ135": rng.uniform(200, 900),
                "MQ7": rng.uniform(100, 700),
                "PM2.5": rng.uniform(5, 180),
                "PM10": rng.uniform(10, 250),
            })
    return pd.DataFrame(rows)


try:
    df = load_data(SHEET_CSV_URL)
    using_demo = df.empty
except Exception:
    df = pd.DataFrame()
    using_demo = True

if using_demo:
    df = load_demo_data()

# ============================================================
# AQI SCORING (PM2.5-driven, US EPA-style breakpoints)
# ============================================================
def pm25_to_aqi(pm):
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500),
    ]
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        if c_lo <= pm <= c_hi:
            return round((a_hi - a_lo) / (c_hi - c_lo) * (pm - c_lo) + a_lo)
    return 500


def aqi_bucket(aqi):
    if aqi <= 50:
        return "Good", [0, 200, 83]
    elif aqi <= 100:
        return "Moderate", [255, 214, 0]
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", [255, 140, 0]
    elif aqi <= 200:
        return "Unhealthy", [230, 57, 70]
    elif aqi <= 300:
        return "Very Unhealthy", [142, 36, 170]
    else:
        return "Hazardous", [126, 0, 35]


# latest reading per station (unique LAT/LON rounded to group same physical station)
df["station_key"] = df["LAT"].round(4).astype(str) + "_" + df["LON"].round(4).astype(str)
latest = df.sort_values("Timestamp").groupby("station_key").tail(1).copy()
latest["AQI"] = latest["PM2.5"].apply(pm25_to_aqi)
latest[["AQI_Label", "AQI_Color"]] = latest["AQI"].apply(lambda a: pd.Series(aqi_bucket(a)))

# ============================================================
# HEADER + STATUS
# ============================================================
st.markdown("## 🌫️ Air Quality Map")
if using_demo:
    st.warning("Showing demo data — paste your published Google Sheet CSV URL into `SHEET_CSV_URL` in app.py to go live.")
else:
    st.caption(f"Live from Google Sheets · last row at {df['Timestamp'].max()}")

# ============================================================
# METRIC CARDS
# ============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><h3>STATIONS</h3><p>{latest.shape[0]}</p></div>""", unsafe_allow_html=True)
with c2:
    avg_aqi = round(latest["AQI"].mean()) if not latest.empty else 0
    st.markdown(f"""<div class="metric-card"><h3>AVG AQI</h3><p>{avg_aqi}</p></div>""", unsafe_allow_html=True)
with c3:
    worst = latest.loc[latest["AQI"].idxmax()] if not latest.empty else None
    worst_val = int(worst["AQI"]) if worst is not None else 0
    st.markdown(f"""<div class="metric-card"><h3>WORST AQI</h3><p>{worst_val}</p></div>""", unsafe_allow_html=True)
with c4:
    avg_pm25 = round(latest["PM2.5"].mean(), 1) if not latest.empty else 0
    st.markdown(f"""<div class="metric-card"><h3>AVG PM2.5</h3><p>{avg_pm25}</p></div>""", unsafe_allow_html=True)

st.write("")

# ============================================================
# MAP
# ============================================================
if not latest.empty:
    view_state = pdk.ViewState(
        latitude=latest["LAT"].mean(),
        longitude=latest["LON"].mean(),
        zoom=10,
        pitch=0,
    )

    latest["radius"] = 200 + latest["AQI"] * 4

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=latest,
        get_position=["LON", "LAT"],
        get_fill_color="AQI_Color",
        get_radius="radius",
        pickable=True,
        opacity=0.8,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    tooltip = {
        "html": """
        <b>AQI: {AQI} ({AQI_Label})</b><br/>
        PM2.5: {PM2.5} µg/m³<br/>
        PM10: {PM10} µg/m³<br/>
        TVOC: {TVOC} ppb<br/>
        MQ135: {MQ135}<br/>
        MQ7: {MQ7}<br/>
        Updated: {Timestamp}
        """,
        "style": {"backgroundColor": "#1a1d24", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(
        map_style="dark",
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip,
    ))
else:
    st.info("No station data available yet.")

# ============================================================
# TREND CHARTS
# ============================================================
st.markdown("### 📈 Trends")
tab1, tab2, tab3 = st.tabs(["PM2.5 / PM10", "TVOC", "MQ135 / MQ7"])

station_options = latest["station_key"].tolist() if not latest.empty else []
station_names = {row.station_key: f"{row.LAT:.4f}, {row.LON:.4f}" for row in latest.itertuples()}

with tab1:
    chart_df = df.pivot_table(index="Timestamp", columns="station_key", values="PM2.5")
    st.line_chart(chart_df)
with tab2:
    chart_df = df.pivot_table(index="Timestamp", columns="station_key", values="TVOC")
    st.line_chart(chart_df)
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("MQ135 (general air quality)")
        st.line_chart(df.pivot_table(index="Timestamp", columns="station_key", values="MQ135"))
    with c2:
        st.caption("MQ7 (CO)")
        st.line_chart(df.pivot_table(index="Timestamp", columns="station_key", values="MQ7"))

# ============================================================
# STATION TABLE
# ============================================================
st.markdown("### 📋 Station Details")
display_cols = ["Timestamp", "LAT", "LON", "PM2.5", "PM10", "TVOC", "MQ135", "MQ7", "HP0", "HP3", "AQI", "AQI_Label"]
st.dataframe(latest[display_cols].sort_values("AQI", ascending=False), use_container_width=True, hide_index=True)
