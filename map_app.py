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
MARKER_COLOR = [66, 165, 245]  # fixed blue dot, no AQI scoring
MARKER_RADIUS = 40  # small, fixed size, just marks the exact location

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

df["station_key"] = df["LAT"].round(4).astype(str) + "_" + df["LON"].round(4).astype(str)
latest = df.sort_values("Timestamp").groupby("station_key").tail(1).copy()
latest["DisplayName"] = latest.apply(
    lambda r: r["StationName"] if pd.notna(r["StationName"]) and str(r["StationName"]).strip() else f"{r['LAT']:.4f}, {r['LON']:.4f}",
    axis=1,
)
latest["marker_color"] = [MARKER_COLOR] * len(latest)
latest["radius"] = MARKER_RADIUS

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

    preview_col = selected_cols[0] if selected_cols else None
    for _, row in latest.iterrows():
        is_selected = st.session_state.selected_station == row["station_key"]
        preview = f" · {preview_col} {row[preview_col]:.1f}" if preview_col and pd.notna(row.get(preview_col)) else ""
        btn_label = f"{'📌 ' if is_selected else ''}{row['DisplayName']}{preview}"
        if st.button(btn_label, key=f"btn_{row['station_key']}", use_container_width=True):
            fly_to(row["station_key"])

# ============================================================
# METRIC CARDS (raw data, no AQI)
# ============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><h3>STATIONS</h3><p>{latest.shape[0]}</p></div>""", unsafe_allow_html=True)
with c2:
    avg_pm25 = round(latest["PM2.5"].mean(), 1) if "PM2.5" in latest.columns and not latest.empty else "-"
    st.markdown(f"""<div class="metric-card"><h3>AVG PM2.5</h3><p>{avg_pm25}</p></div>""", unsafe_allow_html=True)
with c3:
    avg_pm10 = round(latest["PM10"].mean(), 1) if "PM10" in latest.columns and not latest.empty else "-"
    st.markdown(f"""<div class="metric-card"><h3>AVG PM10</h3><p>{avg_pm10}</p></div>""", unsafe_allow_html=True)
with c4:
    avg_tvoc = round(latest["TVOC"].mean(), 1) if "TVOC" in latest.columns and not latest.empty else "-"
    st.markdown(f"""<div class="metric-card"><h3>AVG TVOC</h3><p>{avg_tvoc}</p></div>""", unsafe_allow_html=True)

st.write("")

# ============================================================
# MAP (main focus — small fixed-size markers, raw data only)
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
        get_fill_color="marker_color",
        get_radius="radius",
        radius_min_pixels=4,
        radius_max_pixels=10,
        pickable=True,
        opacity=0.9,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    label_layer = pdk.Layer(
        "TextLayer",
        data=latest,
        get_position=["LON", "LAT"],
        get_text="DisplayName",
        get_size=13,
        get_color=[255, 255, 255],
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -12],
    )

    # tooltip shows only the raw columns selected in the sidebar
    tooltip_lines = "<b>{DisplayName}</b><br/>"
    for col in selected_cols:
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
# SELECTED SENSOR DATA (only chosen columns, raw values)
# ============================================================
st.markdown("### 📋 Station Readings")
display_cols = ["DisplayName", "Timestamp"] + selected_cols
display_cols = [c for c in display_cols if c in latest.columns]
st.dataframe(
    latest[display_cols],
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
