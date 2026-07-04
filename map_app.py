import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# ============================================================
# CONFIG — EDIT THIS
# ============================================================
# Publish your Google Sheet to the web as CSV:
# File -> Share -> Publish to web -> choose the sheet -> CSV -> Publish
# Paste the resulting URL below.
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/gviz/tq?tqx=out:csv&sheet=Sheet1"

REFRESH_SECONDS = 60  # auto-refresh interval
ALL_SENSOR_COLS = ["PM2.5", "PM10", "TVOC", "MQ135", "MQ7", "HP0", "HP3"]

st.set_page_config(
    page_title="Air Quality Map",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>", unsafe_allow_html=True)

# ============================================================
# DARK THEME / MOBILE-FIRST CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background: #1a1d24; border-radius: 12px; padding: 12px;
        text-align: center; border: 1px solid #2a2d34;
    }
    .metric-card h3 { margin: 0; font-size: 0.75rem; color: #9aa0a6; font-weight: 500; }
    .metric-card p { margin: 4px 0 0 0; font-size: 1.4rem; font-weight: 700; }
    .station-card {
        background: #1a1d24; border-radius: 10px; padding: 10px 12px;
        border: 1px solid #2a2d34; margin-bottom: 8px;
    }
    section[data-testid="stSidebar"] { background-color: #12151c; }
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button {
        width: 100%; text-align: left; border-radius: 8px;
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
    for col in ["LAT", "LON"] + ALL_SENSOR_COLS:
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
    for lat, lon, name in stations:
        for i in range(5):
            rows.append({
                "Timestamp": now - pd.Timedelta(minutes=10 * i),
                "LAT": lat + rng.normal(0, 0.001),
                "LON": lon + rng.normal(0, 0.001),
                "StationName": name,
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

if "StationName" not in df.columns:
    df["StationName"] = None

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


df["station_key"] = df["LAT"].round(4).astype(str) + "_" + df["LON"].round(4).astype(str)
latest = df.sort_values("Timestamp").groupby("station_key").tail(1).copy()
latest["AQI"] = latest["PM2.5"].apply(pm25_to_aqi)
latest[["AQI_Label", "AQI_Color"]] = latest["AQI"].apply(lambda a: pd.Series(aqi_bucket(a)))
latest["radius"] = 200 + latest["AQI"] * 4
latest["DisplayName"] = latest.apply(
    lambda r: r["StationName"] if pd.notna(r["StationName"]) and str(r["StationName"]).strip() else f"{r['LAT']:.4f}, {r['LON']:.4f}",
    axis=1,
)

DEFAULT_ZOOM = 10.5
STATION_ZOOM = 15

# ============================================================
# SESSION STATE FOR MAP VIEW
# ============================================================
if "map_lat" not in st.session_state:
    st.session_state.map_lat = latest["LAT"].mean() if not latest.empty else 18.7
    st.session_state.map_lon = latest["LON"].mean() if not latest.empty else 98.9
    st.session_state.map_zoom = DEFAULT_ZOOM
    st.session_state.selected_station = None


def fly_to(station_key):
    row = latest[latest["station_key"] == station_key].iloc[0]
    st.session_state.map_lat = row["LAT"]
    st.session_state.map_lon = row["LON"]
    st.session_state.map_zoom = STATION_ZOOM
    st.session_state.selected_station = station_key


def reset_view():
    st.session_state.map_lat = latest["LAT"].mean() if not latest.empty else 18.7
    st.session_state.map_lon = latest["LON"].mean() if not latest.empty else 98.9
    st.session_state.map_zoom = DEFAULT_ZOOM
    st.session_state.selected_station = None

# ============================================================
# HEADER
# ============================================================
st.markdown("## 🌫️ Air Quality Map")
if using_demo:
    st.warning("Showing demo data — paste your published Google Sheet CSV URL into `SHEET_CSV_URL` in app.py to go live.")
else:
    st.caption(f"Live from Google Sheets · last row at {df['Timestamp'].max()}")

# ============================================================
# SIDEBAR: COLUMN SELECTOR + STATION LIST
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Columns to show")
    selected_cols = st.multiselect(
        "Sensor readings",
        options=ALL_SENSOR_COLS,
        default=["PM2.5", "PM10", "TVOC"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📍 Stations")
    if st.button("🌍 Reset view (show all)", use_container_width=True):
        reset_view()

    for row in latest.sort_values("AQI", ascending=False).itertuples():
        label, color = row.AQI_Label, row.AQI_Color
        badge_color = f"rgb({color[0]},{color[1]},{color[2]})"
        is_selected = st.session_state.selected_station == row.station_key
        btn_label = f"{'📌 ' if is_selected else ''}{row.DisplayName} · AQI {row.AQI}"
        if st.button(btn_label, key=f"btn_{row.station_key}", use_container_width=True):
            fly_to(row.station_key)
        st.markdown(
            f"<div style='margin:-8px 0 10px 4px;font-size:0.75rem;'>"
            f"<span style='color:{badge_color};'>●</span> {label}</div>",
            unsafe_allow_html=True,
        )

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
    worst_val = int(latest["AQI"].max()) if not latest.empty else 0
    st.markdown(f"""<div class="metric-card"><h3>WORST AQI</h3><p>{worst_val}</p></div>""", unsafe_allow_html=True)
with c4:
    avg_pm25 = round(latest["PM2.5"].mean(), 1) if not latest.empty else 0
    st.markdown(f"""<div class="metric-card"><h3>AVG PM2.5</h3><p>{avg_pm25}</p></div>""", unsafe_allow_html=True)

st.write("")

# ============================================================
# MAP (main focus — large)
# ============================================================
if not latest.empty:
    view_state = pdk.ViewState(
        latitude=st.session_state.map_lat,
        longitude=st.session_state.map_lon,
        zoom=st.session_state.map_zoom,
        pitch=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=latest,
        get_position=["LON", "LAT"],
        get_fill_color="AQI_Color",
        get_radius="radius",
        pickable=True,
        opacity=0.85,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    label_layer = pdk.Layer(
        "TextLayer",
        data=latest,
        get_position=["LON", "LAT"],
        get_text="DisplayName",
        get_size=14,
        get_color=[255, 255, 255],
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -18],
    )

    # build tooltip dynamically from selected columns only
    tooltip_lines = "<b>{DisplayName}</b><br/><b>AQI: {AQI} ({AQI_Label})</b><br/>"
    for col in selected_cols:
        safe = col.replace(".", "\\.")
        tooltip_lines += f"{col}: {{{col}}}<br/>"
    tooltip_lines += "Updated: {Timestamp}"

    tooltip = {
        "html": tooltip_lines,
        "style": {"backgroundColor": "#1a1d24", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style="dark",
            initial_view_state=view_state,
            layers=[layer, label_layer],
            tooltip=tooltip,
        ),
        use_container_width=True,
        height=560,
    )
else:
    st.info("No station data available yet.")

# ============================================================
# SELECTED SENSOR DATA (only chosen columns)
# ============================================================
st.markdown("### 📋 Station Readings")
display_cols = ["DisplayName", "Timestamp"] + selected_cols + ["AQI", "AQI_Label"]
display_cols = [c for c in display_cols if c in latest.columns]
st.dataframe(
    latest[display_cols].sort_values("AQI", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# TRENDS (only for selected columns)
# ============================================================
if selected_cols:
    st.markdown("### 📈 Trends")
    tabs = st.tabs(selected_cols)
    for tab, col in zip(tabs, selected_cols):
        with tab:
            chart_df = df.pivot_table(index="Timestamp", columns="station_key", values=col)
            chart_df.columns = [
                latest.loc[latest["station_key"] == k, "DisplayName"].values[0]
                if k in latest["station_key"].values else k
                for k in chart_df.columns
            ]
            st.line_chart(chart_df)
