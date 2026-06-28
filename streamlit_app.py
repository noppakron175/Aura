import os
import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH  = "models/rf_model_unified.joblib"
MODEL_NAME  = "Aurafarm AI"
SHEET_URL   = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ"
    "/pub?output=csv"
)
BASE_LAT    = 18.5847
BASE_LON    = 99.0256
HOURS_BACK  = 24

FEATURE_COLS = ["TVOC", "eCO2", "Temp", "Humidity", "PM2.5", "CH0", "CH3", "MQ135"]
MAPPING_DICT = {
    "TVOC": "col_2", "eCO2": "col_3", "Temp": "col_4", "Humidity": "col_5",
    "PM2.5": "col_6", "CH0": "col_7", "CH3": "col_8", "MQ135": "col_9",
}
COLUMN_RENAME = {
    "Unnamed: 0": "Timestamp",
    "tvoc": "TVOC", "eco2": "eCO2", "temp": "Temp", "humidity": "Humidity",
    "ch0": "CH0", "ch3": "CH3", "mq135": "MQ135", "2.5": "PM2.5", "10": "PM10",
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Aurafarm", layout="wide")

st.markdown("""
<style>
/* ══════════════════════════════════════════
   RESET & BASE
══════════════════════════════════════════ */
html {
    -webkit-text-size-adjust: 100%;
    touch-action: manipulation;
}
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #c9d1d9;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: #161b22 !important; }

/* ══════════════════════════════════════════
   BLOCK CONTAINER — mobile-first
══════════════════════════════════════════ */
.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 3rem !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
    max-width: 100% !important;
}
@media (min-width: 480px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
@media (min-width: 768px) {
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
}

/* ══════════════════════════════════════════
   TYPOGRAPHY
══════════════════════════════════════════ */
* { font-family: 'Inter', 'Segoe UI', sans-serif; }
h1, h2, h3, h4 { color: #e6edf3; letter-spacing: -0.3px; }

/* ══════════════════════════════════════════
   EYEBROW LABEL
══════════════════════════════════════════ */
.eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6e7681;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.eyebrow::before { content: "●"; color: #3fb950; font-size: 0.5rem; }

/* ══════════════════════════════════════════
   CARDS
══════════════════════════════════════════ */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
    padding: 14px 12px !important;
    margin-bottom: 12px !important;
    margin-top: 4px !important;
    box-sizing: border-box !important;
}
@media (min-width: 480px) {
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 18px 16px !important;
        border-radius: 14px !important;
    }
}
@media (min-width: 768px) {
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 24px 26px !important;
        margin-bottom: 20px !important;
    }
}
/* nested — no double shadow */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: none !important;
    margin-bottom: 8px !important;
    margin-top: 0 !important;
}

/* ── inner padding wrapper ── */
.card-inner { padding: 4px 2px 8px 2px; }
@media (min-width: 480px) { .card-inner { padding: 6px 6px 10px 6px; } }
@media (min-width: 768px) { .card-inner { padding: 10px 12px 14px 12px; } }

/* ══════════════════════════════════════════
   HERO
══════════════════════════════════════════ */
.hero-title {
    font-size: clamp(1.7rem, 9vw, 3rem);
    font-weight: 800;
    line-height: 1.1;
    color: #e6edf3;
    letter-spacing: -1px;
    text-align: center;
}
.hero-sub {
    font-size: clamp(0.75rem, 3.5vw, 0.95rem);
    color: #6e7681;
    text-align: center;
    margin-top: 4px;
    margin-bottom: 14px;
}

/* ── Status pill — fully fluid ── */
.pill-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 0 4px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 20px;
    border-radius: 999px;
    font-weight: 700;
    font-size: clamp(0.8rem, 4vw, 1.05rem);
    white-space: nowrap;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    box-sizing: border-box;
}
.status-vape    { background:rgba(248,81,73,0.12);  border:1.5px solid rgba(248,81,73,0.5);  color:#ffa198; box-shadow:0 0 18px rgba(248,81,73,0.12); }
.status-clean   { background:rgba(63,185,80,0.10);  border:1.5px solid rgba(63,185,80,0.4);  color:#56d364; box-shadow:0 0 18px rgba(63,185,80,0.10); }
.status-offline { background:rgba(110,118,129,0.10); border:1.5px solid #30363d; color:#6e7681; }
.status-time    { color:#484f58; font-size:clamp(0.68rem,2.5vw,0.75rem); text-align:center; }

/* ══════════════════════════════════════════
   METRIC GRID — 2→4→7 columns
══════════════════════════════════════════ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px;
}
@media (min-width: 480px) { .metric-grid { grid-template-columns: repeat(4, 1fr); gap: 7px; } }
@media (min-width: 900px) { .metric-grid { grid-template-columns: repeat(7, 1fr); gap: 8px; } }

.metric-cell {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 9px;
    padding: 10px 10px 8px 10px;
    min-width: 0;
}
.metric-label {
    font-size: clamp(0.58rem, 2vw, 0.68rem);
    font-weight: 600;
    color: #6e7681;
    margin-bottom: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.metric-value {
    font-size: clamp(0.95rem, 3.5vw, 1.15rem);
    font-weight: 700;
    color: #e6edf3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.metric-delta-pos { font-size: 0.67rem; color: #56d364; }
.metric-delta-neg { font-size: 0.67rem; color: #ffa198; }
.metric-delta-neu { font-size: 0.67rem; color: #6e7681; }

/* ══════════════════════════════════════════
   QUICK STATS — 2→4 columns
══════════════════════════════════════════ */
.qs-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px;
    margin-bottom: 4px;
}
@media (min-width: 480px) { .qs-grid { grid-template-columns: repeat(4, 1fr); } }

.qs-cell {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 9px;
    padding: 10px 10px 8px 10px;
    min-width: 0;
}
.qs-label {
    font-size: clamp(0.58rem, 2vw, 0.65rem);
    font-weight: 600;
    color: #6e7681;
    margin-bottom: 3px;
}
.qs-value {
    font-size: clamp(0.9rem, 3.5vw, 1.05rem);
    font-weight: 700;
    color: #e6edf3;
}

/* ══════════════════════════════════════════
   TABS — scrollable on narrow screens
══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 10px;
    gap: 3px;
    padding: 3px;
    border: 1px solid #21262d;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6e7681;
    border-radius: 8px;
    font-size: clamp(0.72rem, 2.5vw, 0.82rem);
    white-space: nowrap;
    flex-shrink: 0;
    padding: 6px 10px !important;
}
.stTabs [data-baseweb="tab"] p { color: inherit !important; }
.stTabs [aria-selected="true"]  { background: #21262d !important; color: #e6edf3 !important; }

/* ══════════════════════════════════════════
   NODE CARDS
══════════════════════════════════════════ */
.node-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 7px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    font-size: clamp(0.75rem, 2.5vw, 0.83rem);
    color: #8b949e;
    min-width: 0;
}
.node-info { min-width: 0; flex: 1; }
.node-name {
    color: #c9d1d9;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: clamp(0.78rem, 2.8vw, 0.85rem);
}
.node-loc {
    color: #484f58;
    font-size: clamp(0.65rem, 2.2vw, 0.75rem);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pill-red   { color:#ffa198; font-weight:700; white-space:nowrap; }
.pill-green { color:#56d364; font-weight:700; white-space:nowrap; }
.pill-dot-red   { display:inline-block; width:6px; height:6px; border-radius:50%; background:#f85149; margin-right:4px; flex-shrink:0; }
.pill-dot-green { display:inline-block; width:6px; height:6px; border-radius:50%; background:#3fb950; margin-right:4px; flex-shrink:0; }

/* ══════════════════════════════════════════
   MAP LEGEND
══════════════════════════════════════════ */
.map-legend {
    display: flex;
    gap: 14px;
    margin-top: 8px;
    font-size: clamp(0.68rem, 2.5vw, 0.78rem);
    color: #6e7681;
    flex-wrap: wrap;
}
.legend-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
}

/* ══════════════════════════════════════════
   HEATMAP — horizontally scrollable
══════════════════════════════════════════ */
.heatmap-wrap {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    padding-bottom: 4px;
    /* hide scrollbar on iOS but keep scrollable */
    scrollbar-width: none;
}
.heatmap-wrap::-webkit-scrollbar { display: none; }
.heatmap-inner {
    display: flex;
    gap: 3px;
    min-width: 520px; /* ensures all 24 cols visible on scroll */
}

/* ══════════════════════════════════════════
   DATAFRAME
══════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    font-size: clamp(0.7rem, 2vw, 0.85rem) !important;
}

/* ══════════════════════════════════════════
   MISC
══════════════════════════════════════════ */
hr { border-color: #21262d; }
p, li { color: #8b949e; font-size: clamp(0.8rem, 2.5vw, 0.9rem); }
[data-testid="stCaptionContainer"] { color: #484f58 !important; font-size: 0.7rem !important; }
.stAlert { border-radius: 10px; }

/* ── touch target: make buttons finger-friendly ── */
button, [role="button"] { min-height: 40px !important; }

/* ── AI insight box ── */
.insight-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #3fb950;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: clamp(0.78rem, 2.5vw, 0.87rem);
    color: #8b949e;
    line-height: 1.75;
}

/* ── section divider inside cards ── */
.card-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 16px 0 14px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

my_model = load_model()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        df = df.rename(columns=COLUMN_RENAME)
        if "Unnamed: 1" in df.columns:
            df = df.drop(columns=["Unnamed: 1"])
        if "Timestamp" in df.columns:
            df["Display_Time"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
            df["Sort_Time"]    = df["Display_Time"] + pd.Timedelta(hours=7)
            df = df.dropna(subset=["Display_Time"]).sort_values("Sort_Time", ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_sensor_data()

if df.empty:
    st.warning("No data available. Check your Google Sheets connection.")
    st.stop()

latest   = df.iloc[0]
previous = df.iloc[1] if len(df) > 1 else latest

# ─────────────────────────────────────────────
# RUN MODEL
# ─────────────────────────────────────────────
prediction   = None
confidence   = None
all_features = df[FEATURE_COLS].rename(columns=MAPPING_DICT)

if my_model:
    try:
        latest_features = latest[FEATURE_COLS].to_frame().T.rename(columns=MAPPING_DICT)
        prediction = int(my_model.predict(latest_features)[0])
        if hasattr(my_model, "predict_proba"):
            proba      = my_model.predict_proba(latest_features)[0]
            confidence = float(proba[prediction]) * 100
        df["is_vape"] = my_model.predict(all_features)
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────
# SENSOR DATA
# ─────────────────────────────────────────────
live_state = 1 if prediction == 1 else 0

mock_sensors = pd.DataFrame({
    "sensor_id":    ["SN-01", "SN-02", "SN-03", "SN-04"],
    "location":     ["Main Lobby", "East Restroom", "Breakroom", "Stairwell B"],
    "lat":          [BASE_LAT + 0.0004, BASE_LAT + 0.0004, BASE_LAT - 0.0005, BASE_LAT + 0.0002],
    "lon":          [BASE_LON,          BASE_LON - 0.0006,  BASE_LON - 0.0002, BASE_LON + 0.0005],
    "vape_detected":[live_state, 0, 0, 0],
})
mock_sensors["status_text"] = mock_sensors["vape_detected"].map({1: "Vape Detected", 0: "Clean"})
mock_sensors["fill_r"] = mock_sensors["vape_detected"].map({1: 248, 0: 63})
mock_sensors["fill_g"] = mock_sensors["vape_detected"].map({1: 81,  0: 185})
mock_sensors["fill_b"] = mock_sensors["vape_detected"].map({1: 73,  0: 80})

# ─────────────────────────────────────────────
# HERO CARD
# ─────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div class='hero-title'>VAPONOWAY</div>"
        "<div class='hero-sub'>Facility Air Quality Monitor</div>",
        unsafe_allow_html=True,
    )

    if prediction is None:
        st.markdown(
            "<div class='pill-wrap'>"
            "<span class='status-pill status-offline'>⚠️ Detection Offline</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif prediction == 1:
        conf_str = f" · {confidence:.0f}%" if confidence else ""
        ts = latest['Display_Time'].strftime('%H:%M:%S')
        st.markdown(
            f"<div class='pill-wrap'>"
            f"<span class='status-pill status-vape'>🚨 Vape Detected{conf_str}</span>"
            f"<span class='status-time'>as of {ts}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        conf_str = f" · {confidence:.0f}%" if confidence else ""
        ts = latest['Display_Time'].strftime('%H:%M:%S')
        st.markdown(
            f"<div class='pill-wrap'>"
            f"<span class='status-pill status-clean'>✅ Air Quality Clean{conf_str}</span>"
            f"<span class='status-time'>as of {ts}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
def fmt_delta(col, inverse=False):
    try:
        d = round(float(latest[col]) - float(previous[col]), 2)
        if d == 0:
            return f"<span class='metric-delta-neu'>→ {d:+.2f}</span>"
        up = d > 0
        good = not up if inverse else up
        cls = "metric-delta-pos" if good else "metric-delta-neg"
        arrow = "↑" if up else "↓"
        return f"<span class='{cls}'>{arrow} {abs(d)}</span>"
    except Exception:
        return ""

metrics = [
    ("TVOC",     f"{latest['TVOC']} ppb",    fmt_delta("TVOC",     inverse=True)),
    ("PM 2.5",   f"{latest['PM2.5']} μg/m³", fmt_delta("PM2.5",    inverse=True)),
    ("eCO₂",     f"{latest['eCO2']} ppm",    fmt_delta("eCO2",     inverse=True)),
    ("MQ7",      f"{latest['CH0']}",         fmt_delta("CH0",      inverse=True)),
    ("MQ135",    f"{latest['MQ135']}",       fmt_delta("MQ135",    inverse=True)),
    ("Temp",     f"{latest['Temp']} °C",     fmt_delta("Temp")),
    ("Humidity", f"{latest['Humidity']} %",  fmt_delta("Humidity")),
]

cells_html = "".join(
    f"<div class='metric-cell'>"
    f"<div class='metric-label'>{label}</div>"
    f"<div class='metric-value'>{val}</div>"
    f"{delta}"
    f"</div>"
    for label, val, delta in metrics
)

with st.container(border=True):
    st.markdown(
        "<div class='eyebrow'>Live Readings</div>"
        f"<div class='card-inner'><div class='metric-grid'>{cells_html}</div></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# DETECTION HISTORY
# ─────────────────────────────────────────────
with st.container(border=True):
    st.markdown("<div class='eyebrow'>Detection History</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-inner'>", unsafe_allow_html=True)

    if my_model and "is_vape" in df.columns:
        vape_rows = df[df["is_vape"] == 1].copy()
        if not vape_rows.empty:
            vape_rows = vape_rows.sort_values("Display_Time")
            vape_rows["block"] = (
                vape_rows["Display_Time"].diff() > pd.Timedelta(minutes=5)
            ).cumsum()
            event_rows = []
            for _, grp in vape_rows.groupby("block"):
                event_rows.append({
                    "Date":       grp["Display_Time"].min().strftime("%Y-%m-%d"),
                    "Start":      grp["Display_Time"].min().strftime("%H:%M"),
                    "End":        grp["Display_Time"].max().strftime("%H:%M"),
                    "Duration":   str(grp["Display_Time"].max() - grp["Display_Time"].min()).split(".")[0],
                    "Peak TVOC":  f"{grp['TVOC'].max():.0f} ppb",
                    "Peak PM2.5": f"{grp['PM2.5'].max():.1f} μg/m³",
                })
            events_df = pd.DataFrame(event_rows[::-1])
            st.dataframe(
                events_df, use_container_width=True, hide_index=True,
                column_config={
                    "Date":       st.column_config.TextColumn("Date"),
                    "Start":      st.column_config.TextColumn("Start"),
                    "End":        st.column_config.TextColumn("End"),
                    "Duration":   st.column_config.TextColumn("Duration"),
                    "Peak TVOC":  st.column_config.TextColumn("Peak TVOC"),
                    "Peak PM2.5": st.column_config.TextColumn("Peak PM2.5"),
                },
            )
            st.caption(f"{len(events_df)} detection event(s) in available data.")
        else:
            st.markdown(
                "<div style='color:#484f58;padding:20px 0;text-align:center;font-size:0.83rem'>"
                "No vape events detected in the available data.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='color:#484f58;padding:20px 0;text-align:center;font-size:0.83rem'>"
            "Detection system offline — history unavailable.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Quick Stats ──
    st.markdown("<hr class='card-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>Quick Stats</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-inner'>", unsafe_allow_html=True)

    if my_model and "is_vape" in df.columns:
        vape_qs = df[df["is_vape"] == 1].copy()
        if not vape_qs.empty:
            vape_qs2 = vape_qs.sort_values("Display_Time")
            vape_qs2["block"] = (vape_qs2["Display_Time"].diff() > pd.Timedelta(minutes=5)).cumsum()
            total_events = vape_qs2["block"].nunique()
            vape_qs2["hour"] = vape_qs2["Display_Time"].dt.hour
            worst_hour = int(vape_qs2["hour"].value_counts().idxmax())
            worst_hour_str = f"{worst_hour:02d}:00–{worst_hour+1:02d}:00"
            clean_rows = df[df["is_vape"] == 0].sort_values("Sort_Time")
            if len(clean_rows) > 1:
                gaps = clean_rows["Sort_Time"].diff().dropna()
                longest = gaps.max()
                h, rem = divmod(int(longest.total_seconds()), 3600)
                m = rem // 60
                longest_clean_str = f"{h}h {m}m" if h else f"{m}m"
            else:
                longest_clean_str = "N/A"
            peak_tvoc_str = f"{vape_qs['TVOC'].max():.0f} ppb"
        else:
            total_events, worst_hour_str, longest_clean_str, peak_tvoc_str = 0, "—", "—", "—"

        qs_cells = "".join(
            f"<div class='qs-cell'><div class='qs-label'>{lbl}</div><div class='qs-value'>{val}</div></div>"
            for lbl, val in [
                ("Total Events", str(total_events)),
                ("Worst Hour",   worst_hour_str),
                ("Clean Run",    longest_clean_str),
                ("Peak TVOC",    peak_tvoc_str),
            ]
        )
        st.markdown(f"<div class='qs-grid'>{qs_cells}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#484f58;font-size:0.83rem'>Model offline — stats unavailable.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Hourly Heatmap ──
    st.markdown("<hr class='card-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>Hourly Detection Heatmap</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-inner'>", unsafe_allow_html=True)

    if my_model and "is_vape" in df.columns and not df[df["is_vape"] == 1].empty:
        heat_df = df.copy()
        heat_df["hour"] = heat_df["Display_Time"].dt.hour
        hourly_counts = heat_df[heat_df["is_vape"] == 1].groupby("hour").size()
        all_hours = pd.Series(0, index=range(24))
        all_hours.update(hourly_counts)
        max_count = max(all_hours.max(), 1)

        cells = ""
        for h in range(24):
            count     = int(all_hours[h])
            intensity = count / max_count
            if intensity == 0:
                bg, border, txt_color = "#161b22", "#21262d", "#484f58"
            elif intensity < 0.4:
                g = int(185 - intensity * 100)
                bg = f"rgba(63,{g},80,{0.3 + intensity * 0.4:.2f})"
                border, txt_color = "#3fb950", "#56d364"
            elif intensity < 0.75:
                bg = f"rgba(210,153,34,{0.3 + intensity * 0.3:.2f})"
                border, txt_color = "#d29922", "#e3b341"
            else:
                bg = f"rgba(248,81,73,{0.3 + intensity * 0.4:.2f})"
                border, txt_color = "#f85149", "#ffa198"

            count_disp = str(count) if count > 0 else "·"
            cells += (
                f"<div style='display:flex;flex-direction:column;align-items:center;"
                f"background:{bg};border:1px solid {border};border-radius:5px;"
                f"padding:5px 3px;flex:1;min-width:14px'>"
                f"<span style='font-size:0.5rem;color:#6e7681;line-height:1'>{h:02d}</span>"
                f"<span style='font-size:0.65rem;font-weight:700;color:{txt_color};line-height:1.4'>{count_disp}</span>"
                f"</div>"
            )

        st.markdown(
            f"<div class='heatmap-wrap'><div class='heatmap-inner'>{cells}</div></div>"
            f"<div style='font-size:0.65rem;color:#484f58;margin-top:5px'>"
            f"Swipe to see all hours · colour = intensity · number = readings</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#484f58;font-size:0.83rem;padding:6px 0'>No detection data to build heatmap.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── AI Insight ──
    st.markdown("<hr class='card-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>AI Insight</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-inner'>", unsafe_allow_html=True)

    if my_model and "is_vape" in df.columns:
        vape_ai = df[df["is_vape"] == 1].copy()
        if not vape_ai.empty:
            vape_ai2 = vape_ai.sort_values("Display_Time")
            vape_ai2["block"] = (vape_ai2["Display_Time"].diff() > pd.Timedelta(minutes=5)).cumsum()
            n_events      = vape_ai2["block"].nunique()
            peak_tvoc_val = vape_ai["TVOC"].max()
            peak_pm_val   = vape_ai["PM2.5"].max()
            vape_ai["hour"] = vape_ai["Display_Time"].dt.hour
            top_hour      = int(vape_ai["hour"].value_counts().idxmax())
            top_hour_str  = f"{top_hour:02d}:00–{top_hour+1:02d}:00"
            avg_tvoc_clean = df[df["is_vape"] == 0]["TVOC"].mean()
            tvoc_spike_pct = ((peak_tvoc_val - avg_tvoc_clean) / max(avg_tvoc_clean, 1)) * 100
            freq_note = (
                "A single vaping event has been recorded." if n_events == 1
                else f"{n_events} vaping events detected — low but worth monitoring." if n_events <= 3
                else f"{n_events} events detected — recurring pattern emerging." if n_events <= 8
                else f"{n_events} events detected — high frequency, action recommended."
            )
            tvoc_note = (
                f"TVOC spiked {tvoc_spike_pct:.0f}% above the clean-air baseline — significant chemical load." if tvoc_spike_pct > 150
                else f"TVOC rose {tvoc_spike_pct:.0f}% above baseline — moderate chemical signature." if tvoc_spike_pct > 50
                else f"TVOC elevation was mild ({tvoc_spike_pct:.0f}% above baseline)."
            )
            insight = (
                f"{freq_note} Most activity concentrates around <b style='color:#e6edf3'>{top_hour_str}</b>. "
                f"{tvoc_note} Peak PM 2.5 reached <b style='color:#e6edf3'>{peak_pm_val:.1f} μg/m³</b>. "
                f"Consider scheduling checks or improving ventilation during that window."
            )
        else:
            insight = "✅ No vaping events detected in available data. Air quality has remained within normal parameters across all sensors."

        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#484f58;font-size:0.83rem'>Model offline — insight unavailable.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FACILITY MAP
# ─────────────────────────────────────────────
with st.container(border=True):
    st.markdown("<div class='eyebrow'>Facility Sensor Network</div>", unsafe_allow_html=True)

    halo_layer = pdk.Layer(
        "ScatterplotLayer", data=mock_sensors,
        get_position=["lon", "lat"],
        get_fill_color=["fill_r", "fill_g", "fill_b", 50],
        get_radius=14, radius_units="meters",
        radius_min_pixels=12, radius_max_pixels=24, pickable=False,
    )
    dot_layer = pdk.Layer(
        "ScatterplotLayer", data=mock_sensors,
        get_position=["lon", "lat"],
        get_fill_color=["fill_r", "fill_g", "fill_b", 230],
        get_radius=5, radius_units="meters",
        radius_min_pixels=5, radius_max_pixels=12,
        pickable=True, stroked=True,
        get_line_color=[255, 255, 255, 80], line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(latitude=BASE_LAT, longitude=BASE_LON, zoom=17, pitch=0)

    st.pydeck_chart(
        pdk.Deck(
            layers=[halo_layer, dot_layer],
            initial_view_state=view_state,
            map_style="dark",
            tooltip={"text": "{sensor_id} — {location}\nStatus: {status_text}"},
        ),
        height=260,
        use_container_width=True,
    )

    st.markdown(
        "<div class='map-legend'>"
        "<span><span class='legend-dot' style='background:#3fb950'></span>Clean</span>"
        "<span><span class='legend-dot' style='background:#f85149'></span>Vape Detected</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='card-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>Live Node Status</div>", unsafe_allow_html=True)

    # single-column on phone, two-column on tablet+
    node_cols = st.columns([1, 1])
    for i, (_, row) in enumerate(mock_sensors.iterrows()):
        pill = (
            "<span class='pill-red'><span class='pill-dot-red'></span>Vape Detected</span>"
            if row["vape_detected"]
            else "<span class='pill-green'><span class='pill-dot-green'></span>Clean</span>"
        )
        node_cols[i % 2].markdown(
            f"<div class='node-card'>"
            f"<div class='node-info'>"
            f"<div class='node-name'>{row['sensor_id']}</div>"
            f"<div class='node-loc'>{row['location']}</div>"
            f"</div>"
            f"{pill}</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# TREND CHARTS — tabbed, single card
# ─────────────────────────────────────────────
chart_data = df.sort_values("Sort_Time", ascending=True).copy()
cutoff     = chart_data["Sort_Time"].max() - pd.Timedelta(hours=HOURS_BACK)
chart_data = chart_data[chart_data["Sort_Time"] >= cutoff].set_index("Sort_Time")
numeric_cols = chart_data.select_dtypes(include="number").columns
chart_data   = chart_data[numeric_cols].resample("1min").mean().interpolate(method="time")

if my_model and "is_vape" in df.columns:
    vape_overlay = (
        df[df["Sort_Time"] >= cutoff].sort_values("Sort_Time")
        .set_index("Sort_Time")[["is_vape"]]
        .resample("1min").max()
        .rename(columns={"is_vape": "⚠ Vape Event"})
    )
    chart_data = chart_data.join(vape_overlay, how="left")

CHART_H = 220  # slightly shorter on mobile

with st.container(border=True):
    st.markdown(
        f"<div class='eyebrow'>Sensor Trends · Last {HOURS_BACK} Hours</div>",
        unsafe_allow_html=True,
    )

    tab_particles, tab_air, tab_climate, tab_all = st.tabs(
        ["🟤 Particles", "🌫 Air Quality", "🌡 Climate", "📊 All"]
    )

    with tab_particles:
        cols_p = [c for c in ["PM2.5", "PM10", "MQ135"] if c in chart_data.columns]
        if "⚠ Vape Event" in chart_data.columns:
            cols_p.append("⚠ Vape Event")
        st.line_chart(chart_data[cols_p], height=CHART_H, use_container_width=True)

    with tab_air:
        cols_a = [c for c in ["TVOC", "eCO2"] if c in chart_data.columns]
        if "⚠ Vape Event" in chart_data.columns:
            cols_a.append("⚠ Vape Event")
        st.line_chart(chart_data[cols_a], height=CHART_H, use_container_width=True)

    with tab_climate:
        cols_c = [c for c in ["Temp", "Humidity"] if c in chart_data.columns]
        st.line_chart(chart_data[cols_c], height=CHART_H, use_container_width=True)

    with tab_all:
        display_cols = [c for c in FEATURE_COLS if c in chart_data.columns]
        st.line_chart(chart_data[display_cols], height=CHART_H, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#484f58;font-size:0.68rem;padding:6px 0 2px 0'>"
    "Aurafarm · Auto-refreshes every 30 s · Sensor data via Google Sheets"
    "</div>",
    unsafe_allow_html=True,
)
