import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from predict import SoilPredictor
from preprocess import load_and_clean

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3D Soil Profile Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_ROOT, "data", "bangkok_boring_logs.csv")
MODEL_DIR = os.path.join(_ROOT, "models")

LAYER_SEQUENCE = ["MG", "VSC", "SOC", "SC", "SS", "MSC", "FS"]

LAYER_COLORS = {
    "MG":  "#8D6E63",
    "VSC": "#4FC3F7",
    "SOC": "#1E88E5",
    "SC":  "#1565C0",
    "MSC": "#0D2B6B",
    "FS":  "#FFB74D",
    "SS":  "#F9A825",
}
LAYER_LABELS = {
    "MG":  "Made Ground / Fill",
    "VSC": "Very Soft Clay",
    "SOC": "Soft Clay",
    "SC":  "Stiff Clay",
    "MSC": "Medium-Hard Clay",
    "FS":  "Firm Sand (transition)",
    "SS":  "Sand",
}
METHOD_MAP = {
    "Distance-Weighted Average": "dwa",
    "Random Forest":             "rf",
    "XGBoost":                   "xgb",
}

# ── UTM Zone 47N ↔ WGS84 helpers ─────────────────────────────────────────────
_WGS84_A    = 6_378_137.0
_WGS84_F    = 1 / 298.257223563
_WGS84_E2   = 2 * _WGS84_F - _WGS84_F ** 2
_UTM47_LON0 = np.radians(99.0)
_UTM_K0     = 0.9996
_UTM_E0     = 500_000.0


def _utm47n_to_latlon_arr(eastings, northings):
    a, e2, k0 = _WGS84_A, _WGS84_E2, _UTM_K0
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    x  = np.asarray(eastings,  dtype=float) - _UTM_E0
    y  = np.asarray(northings, dtype=float)
    M  = y / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    phi1 = (mu
            + (3*e1/2   - 27*e1**3/32)         * np.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32)       * np.sin(4*mu)
            + (151*e1**3/96)                     * np.sin(6*mu)
            + (1097*e1**4/512)                   * np.sin(8*mu))
    sp = np.sin(phi1)
    cp = np.cos(phi1)
    tp = np.tan(phi1)
    N1 = a / np.sqrt(1 - e2 * sp**2)
    T1 = tp ** 2
    C1 = e2 * cp**2 / (1 - e2)
    R1 = a * (1 - e2) / (1 - e2 * sp**2) ** 1.5
    D  = x / (N1 * k0)
    lat = phi1 - (N1 * tp / R1) * (
        D**2/2
        - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e2/(1-e2)) * D**4/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e2/(1-e2) - 3*C1**2) * D**6/720
    )
    lon = _UTM47_LON0 + (
        D
        - (1 + 2*T1 + C1)                                          * D**3/6
        + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e2/(1-e2) + 24*T1**2) * D**5/120
    ) / cp
    return np.degrees(lat), np.degrees(lon)


def _utm47n_to_latlon(easting: float, northing: float) -> tuple[float, float]:
    la, lo = _utm47n_to_latlon_arr(np.array([easting]), np.array([northing]))
    return float(la[0]), float(lo[0])


def _latlon_to_utm47n(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    a, e2, k0 = _WGS84_A, _WGS84_E2, _UTM_K0
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl, tl = np.sin(lat), np.cos(lat), np.tan(lat)
    N = a / np.sqrt(1 - e2 * sl**2)
    T = tl ** 2
    C = e2 * cl**2 / (1 - e2)
    A = cl * (lon - _UTM47_LON0)
    M = a * (
        (1 - e2/4 - 3*e2**2/64   - 5*e2**3/256)   * lat
        - (3*e2/8 + 3*e2**2/32   + 45*e2**3/1024)  * np.sin(2*lat)
        + (15*e2**2/256           + 45*e2**3/1024)  * np.sin(4*lat)
        - (35*e2**3/3072)                            * np.sin(6*lat)
    )
    east = _UTM_E0 + k0 * N * (
        A
        + (1 - T + C)                                  * A**3/6
        + (5 - 18*T + T**2 + 72*C - 58*e2/(1-e2))    * A**5/120
    )
    north = k0 * (M + N * tl * (
        A**2/2
        + (5 - T + 9*C + 4*C**2)                      * A**4/24
        + (61 - 58*T + T**2 + 600*C - 330*e2/(1-e2)) * A**6/720
    ))
    return float(east), float(north)

# ── Password protection ───────────────────────────────────────────────────────
def _render_login() -> None:
    try:
        correct = st.secrets["password"]
    except Exception:
        st.error(
            "Password not configured. "
            "Add `password = \"...\"` to `.streamlit/secrets.toml`."
        )
        return

    st.markdown(
        "<h2 style='margin-bottom:0'>Login Required</h2>"
        "<p style='color:#666;margin-top:4px;'>Bangkok Soil Profile Predictor</p>",
        unsafe_allow_html=True,
    )
    col, _ = st.columns([1, 2])
    with col:
        pwd = st.text_input(
            "Password", type="password",
            label_visibility="collapsed", placeholder="Enter password..."
        )
        if st.button("Login", type="primary", use_container_width=True):
            if pwd == correct:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")


if not st.session_state.get("authenticated"):
    _render_login()
    st.stop()


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def _get_predictor():
    return SoilPredictor(model_dir=MODEL_DIR, data_path=DATA_PATH)


@st.cache_data(show_spinner=False)
def _get_df():
    return load_and_clean(DATA_PATH)


@st.cache_data(show_spinner=False)
def _get_bh_pos(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("borehole_id")
          .agg(easting=("easting", "first"), northing=("northing", "first"))
          .reset_index()
          .sort_values("borehole_id")
          .reset_index(drop=True)
    )


predictor = _get_predictor()
df        = _get_df()
bh_pos    = _get_bh_pos(df)


# ── Shared helpers ────────────────────────────────────────────────────────────
def _hex_rgba(hex_color: str, alpha: float = 0.85) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _hover(row) -> str:
    su  = f"{row['su_kpa']:.1f} kPa"       if pd.notna(row.get("su_kpa"))      else "--"
    spt = f"{row['spt_n']:.0f}"             if pd.notna(row.get("spt_n"))       else "--"
    uw  = f"{row['unit_weight']:.2f} kN/m3" if pd.notna(row.get("unit_weight")) else "--"
    return (
        f"<b>{row['borehole_id']}</b><br>"
        f"Layer: {row['soil_layer']} - {row['soil_desc']}<br>"
        f"Depth: {row['depth_m']:.1f} m "
        f"({row['depth_top_m']:.1f}-{row['depth_bot_m']:.1f} m)<br>"
        f"Consistency: {row.get('consistency') or '--'}<br>"
        f"su: {su}    SPT-N: {spt}<br>"
        f"Unit wt: {uw}"
    )


# ── View 1: 3D Borehole View ──────────────────────────────────────────────────
def build_figure(
    df: pd.DataFrame,
    pred_point: tuple | None = None,
    pred_layer: str | None = None,
    virtual_bh: dict | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Real borehole path lines
    for bh_id, bh in df.groupby("borehole_id"):
        s = bh.sort_values("depth_m")
        fig.add_trace(go.Scatter3d(
            x=s["easting"], y=s["northing"], z=-s["depth_m"],
            mode="lines",
            line=dict(color="#bdbdbd", width=4),
            name=bh_id, showlegend=True, hoverinfo="skip",
        ))

    # Sample markers coloured by layer
    for layer, color in LAYER_COLORS.items():
        sub = df[df["soil_layer"] == layer]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=sub["easting"], y=sub["northing"], z=-sub["depth_m"],
            mode="markers",
            marker=dict(size=9, color=color, opacity=0.88,
                        line=dict(color="white", width=0.6)),
            name=f"{layer}  {LAYER_LABELS.get(layer, '')}",
            text=sub.apply(_hover, axis=1).tolist(),
            hovertemplate="%{text}<extra></extra>",
        ))

    # Single-point prediction diamond
    if pred_point is not None:
        e, n, d = pred_point
        color = LAYER_COLORS.get(pred_layer, "#e53935") if pred_layer else "#e53935"
        fig.add_trace(go.Scatter3d(
            x=[e], y=[n], z=[-d], mode="markers",
            marker=dict(size=16, color=color, symbol="diamond",
                        opacity=1.0, line=dict(color="white", width=2)),
            name="Prediction point",
            hovertemplate=(
                f"<b>Prediction Point</b><br>E: {e:.1f}  N: {n:.1f}<br>"
                f"Depth: {d:.1f} m<br>Layer: {pred_layer or '--'}<extra></extra>"
            ),
        ))

    # Virtual borehole (cross markers + dashed stick)
    if virtual_bh is not None:
        ve   = virtual_bh["easting"]
        vn   = virtual_bh["northing"]
        rows = virtual_bh["rows"]
        all_depths = [r["depth_m"] for r in rows]

        fig.add_trace(go.Scatter3d(
            x=[ve] * len(all_depths), y=[vn] * len(all_depths),
            z=[-d for d in all_depths],
            mode="lines",
            line=dict(color="#444444", width=3, dash="dash"),
            name=f"Virtual BH  E:{ve:.0f} N:{vn:.0f}",
            showlegend=True, hoverinfo="skip",
        ))

        by_layer: dict = defaultdict(list)
        for r in rows:
            by_layer[r.get("layer", "?")].append(r)

        for layer, layer_rows in by_layer.items():
            color = LAYER_COLORS.get(layer, "#e53935")
            texts = []
            for r in layer_rows:
                su_s  = f"{r['su_kpa']:.1f} kPa" if r.get("su_kpa")  is not None else "--"
                spt_s = f"{r['spt_n']:.0f}"       if r.get("spt_n")   is not None else "--"
                texts.append(
                    f"<b>Virtual BH</b><br>Depth: {r['depth_m']:.0f} m<br>"
                    f"Layer: {layer}<br>su: {su_s}  SPT-N: {spt_s}<br>"
                    f"Confidence: {r.get('layer_confidence', 0)*100:.0f}%"
                )
            fig.add_trace(go.Scatter3d(
                x=[ve] * len(layer_rows), y=[vn] * len(layer_rows),
                z=[-r["depth_m"] for r in layer_rows],
                mode="markers",
                marker=dict(size=11, color=color, symbol="cross",
                            opacity=0.95, line=dict(color="white", width=0.8)),
                name=f"Virtual · {layer}",
                text=texts, hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Easting (m)", yaxis_title="Northing (m)",
            zaxis_title="- Depth (m)", bgcolor="#f0f4f8",
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=2.5),
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.7)),
        ),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#ddd", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, b=0, t=0),
        height=640, paper_bgcolor="white",
    )
    return fig


# ── Mini plan view (sidebar — always visible, Feature 3) ──────────────────────
def build_mini_planview(query_e: float, query_n: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bh_pos["easting"], y=bh_pos["northing"],
        mode="markers+text",
        marker=dict(size=7, color="#1E88E5", line=dict(color="white", width=0.8)),
        text=bh_pos["borehole_id"], textposition="top center",
        textfont=dict(size=7), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[query_e], y=[query_n], mode="markers",
        marker=dict(size=11, color="#e53935", symbol="diamond",
                    line=dict(color="white", width=1.5)),
        hovertemplate=f"Query<br>E: {query_e:.0f}<br>N: {query_n:.0f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=185, margin=dict(l=4, r=4, t=22, b=4),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"),
        paper_bgcolor="white", plot_bgcolor="#f5f7fa",
        title=dict(text="Query Location", x=0.5, font=dict(size=10)),
    )
    return fig


# ── View 2: 3D Solid Model ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_layer_bounds(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (bh_id, layer), grp in df.groupby(["borehole_id", "soil_layer"]):
        row0 = grp.iloc[0]
        records.append({
            "borehole_id": bh_id,
            "easting":     row0["easting"],
            "northing":    row0["northing"],
            "soil_layer":  layer,
            "top":         grp["depth_top_m"].min(),
            "bot":         grp["depth_bot_m"].max(),
        })
    return pd.DataFrame(records)


def build_solid_figure(df: pd.DataFrame) -> go.Figure:
    bounds = get_layer_bounds(df)
    fig    = go.Figure()

    for layer in reversed(LAYER_SEQUENCE):
        sub = bounds[bounds["soil_layer"] == layer]
        if len(sub) < 3:
            continue
        color_rgb = _hex_rgba(LAYER_COLORS.get(layer, "#aaaaaa"), 0.75)

        e_min, e_max = bh_pos["easting"].min(),  bh_pos["easting"].max()
        n_min, n_max = bh_pos["northing"].min(), bh_pos["northing"].max()
        me = (e_max - e_min) * 0.05
        mn = (n_max - n_min) * 0.05
        GE, GN = np.meshgrid(
            np.linspace(e_min - me, e_max + me, 20),
            np.linspace(n_min - mn, n_max + mn, 20),
        )
        pts = sub[["easting", "northing"]].values
        try:
            it = LinearNDInterpolator(pts, sub["top"].values, fill_value=np.nan)
            ib = LinearNDInterpolator(pts, sub["bot"].values, fill_value=np.nan)
        except Exception:
            continue
        Zt, Zb = it(GE, GN), ib(GE, GN)
        if (~(np.isnan(Zt) | np.isnan(Zb))).sum() < 4:
            continue
        for Z, lbl in [(Zt, "top"), (Zb, "bot")]:
            fig.add_trace(go.Surface(
                x=GE, y=GN, z=-Z,
                colorscale=[[0, color_rgb], [1, color_rgb]],
                showscale=False, opacity=0.7,
                name=f"{layer} {lbl}", showlegend=False,
                hovertemplate=(
                    f"<b>{layer} — {LAYER_LABELS.get(layer,'')}</b><br>{lbl}<br>"
                    "E: %{x:.0f}  N: %{y:.0f}<br>Depth: %{z:.1f} m<extra></extra>"
                ),
                lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1),
            ))

    for bh_id, bh in df.groupby("borehole_id"):
        s = bh.sort_values("depth_m")
        fig.add_trace(go.Scatter3d(
            x=s["easting"], y=s["northing"], z=-s["depth_m"],
            mode="lines+text", line=dict(color="#333333", width=3),
            text=[bh_id] + [""] * (len(s) - 1),
            textposition="top center", textfont=dict(size=10, color="#333333"),
            name=bh_id, showlegend=False, hoverinfo="skip",
        ))

    for layer, color in LAYER_COLORS.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=10, color=color, opacity=0.85),
            name=f"{layer}  {LAYER_LABELS.get(layer, '')}",
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Easting (m)", yaxis_title="Northing (m)",
            zaxis_title="- Depth (m)", bgcolor="#e8edf2",
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=2.5),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#ddd", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, b=0, t=30), height=680, paper_bgcolor="white",
        title=dict(text="3D Solid Soil Model — Interpolated Layer Surfaces",
                   x=0.5, font=dict(size=14)),
    )
    return fig


# ── View 3: Plan View ─────────────────────────────────────────────────────────
# Trace index contract (used by click-event handler):
#   0 → ghost grid  (click-anywhere target, invisible)
#   1 → boreholes   (visible, customdata = borehole_id)
#   2 → section line (mode=lines, not selectable as points)
#   3 → query diamond (always orange, not processed as coord update)

def build_planview_figure(
    selected: list[str],
    query_e: float,
    query_n: float,
) -> go.Figure:
    selected_set = set(selected)
    order_map    = {bh: i + 1 for i, bh in enumerate(selected)}

    # Invisible ghost grid — catches clicks anywhere on the map
    e_lo, e_hi = df["easting"].min()  - 3000, df["easting"].max()  + 3000
    n_lo, n_hi = df["northing"].min() - 3000, df["northing"].max() + 3000
    GE, GN = np.meshgrid(np.linspace(e_lo, e_hi, 30),
                         np.linspace(n_lo, n_hi, 30))

    fig = go.Figure()

    # Trace 0: ghost grid
    fig.add_trace(go.Scatter(
        x=GE.ravel(), y=GN.ravel(), mode="markers",
        marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=0)),
        selected=dict(marker=dict(color="rgba(0,0,0,0)")),
        unselected=dict(marker=dict(color="rgba(0,0,0,0)")),
        hoverinfo="skip", showlegend=False, name="_ghost",
    ))

    # Trace 1: boreholes
    colors = ["#e53935" if b in selected_set else "#1E88E5"
              for b in bh_pos["borehole_id"]]
    sizes  = [15 if b in selected_set else 10 for b in bh_pos["borehole_id"]]
    fig.add_trace(go.Scatter(
        x=bh_pos["easting"], y=bh_pos["northing"],
        mode="markers+text",
        marker=dict(size=sizes, color=colors,
                    line=dict(color="white", width=1.5), opacity=1.0),
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=1.0)),
        text=bh_pos["borehole_id"], textposition="top center",
        textfont=dict(size=9, color="#222"),
        customdata=bh_pos["borehole_id"].tolist(),
        hovertemplate=(
            "<b>%{customdata}</b><br>E: %{x:.0f}  N: %{y:.0f}"
            "<br><i>Click: set coords + add to section</i><extra></extra>"
        ),
        showlegend=False, name="_boreholes",
    ))

    # Trace 2: section path line (mode=lines → not selectable as points)
    if len(selected) >= 2:
        idx = bh_pos.set_index("borehole_id")
        path_e = [idx.loc[b, "easting"]  for b in selected if b in idx.index]
        path_n = [idx.loc[b, "northing"] for b in selected if b in idx.index]
        fig.add_trace(go.Scatter(
            x=path_e, y=path_n, mode="lines",
            line=dict(color="#e53935", width=2, dash="dot"),
            hoverinfo="skip", showlegend=False, name="_section_line",
        ))
    else:
        # Keep trace index consistent: always add trace 2
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 hoverinfo="skip", showlegend=False,
                                 name="_section_line"))

    # Trace 3: query-point diamond (orange, always shown)
    fig.add_trace(go.Scatter(
        x=[query_e], y=[query_n], mode="markers",
        marker=dict(size=13, color="#ff6f00", symbol="diamond",
                    line=dict(color="white", width=2), opacity=1.0),
        selected=dict(marker=dict(size=13, color="#ff6f00", opacity=1.0)),
        unselected=dict(marker=dict(size=13, color="#ff6f00", opacity=1.0)),
        hovertemplate=(
            f"<b>Query Point</b><br>E: {query_e:.1f}  N: {query_n:.1f}<extra></extra>"
        ),
        showlegend=False, name="_query",
    ))

    # Order-number annotations on selected boreholes
    idx2 = bh_pos.set_index("borehole_id")
    for bh, order in order_map.items():
        if bh not in idx2.index:
            continue
        fig.add_annotation(
            x=float(idx2.loc[bh, "easting"]),
            y=float(idx2.loc[bh, "northing"]),
            text=f"<b>{order}</b>", showarrow=False, yshift=-18,
            font=dict(size=10, color="#e53935", family="Arial Black"),
            bgcolor="rgba(255,255,255,0.7)", borderpad=1,
        )

    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(title="Easting (m)", tickformat=".0f"),
        yaxis=dict(title="Northing (m)", tickformat=".0f", scaleanchor="x"),
        showlegend=False, paper_bgcolor="white", plot_bgcolor="#f5f7fa",
        title=dict(
            text="Plan View — click anywhere to set coords; click borehole to add to section",
            x=0.5, font=dict(size=11),
        ),
        dragmode="pan",
    )
    return fig


# ── View 3: Mapbox Plan View (Street Map / Satellite) ────────────────────────
# Same trace-index contract as build_planview_figure:
#   0 → ghost grid    1 → boreholes    2 → section line    3 → query diamond

def build_mapbox_figure(
    selected: list[str],
    query_e: float,
    query_n: float,
    satellite: bool = False,
) -> go.Figure:
    selected_set = set(selected)

    lats_bh, lons_bh = _utm47n_to_latlon_arr(
        bh_pos["easting"].values, bh_pos["northing"].values
    )
    query_lat, query_lon = _utm47n_to_latlon(query_e, query_n)

    lat_c = float(np.mean(lats_bh))
    lon_c = float(np.mean(lons_bh))

    # Ghost grid in lat/lon — captures clicks anywhere on the map
    lat_lo = float(lats_bh.min()) - 0.06
    lat_hi = float(lats_bh.max()) + 0.06
    lon_lo = float(lons_bh.min()) - 0.06
    lon_hi = float(lons_bh.max()) + 0.06
    GLAT, GLON = np.meshgrid(
        np.linspace(lat_lo, lat_hi, 25),
        np.linspace(lon_lo, lon_hi, 25),
    )

    fig = go.Figure()

    # Trace 0: ghost grid
    fig.add_trace(go.Scattermapbox(
        lat=GLAT.ravel().tolist(), lon=GLON.ravel().tolist(),
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)", opacity=0),
        selected=dict(marker=dict(opacity=0)),
        unselected=dict(marker=dict(opacity=0)),
        hoverinfo="skip", showlegend=False, name="_ghost",
    ))

    # Trace 1: boreholes
    colors = ["#e53935" if b in selected_set else "#1E88E5"
              for b in bh_pos["borehole_id"]]
    sizes  = [14 if b in selected_set else 10 for b in bh_pos["borehole_id"]]
    fig.add_trace(go.Scattermapbox(
        lat=lats_bh.tolist(), lon=lons_bh.tolist(),
        mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=1.0),
        text=bh_pos["borehole_id"].tolist(),
        textposition="top right",
        customdata=bh_pos["borehole_id"].tolist(),
        hovertemplate=(
            "<b>%{customdata}</b><br>Lat: %{lat:.5f}  Lon: %{lon:.5f}"
            "<br><i>Click: set coords + add to section</i><extra></extra>"
        ),
        showlegend=False, name="_boreholes",
    ))

    # Trace 2: section path
    if len(selected) >= 2:
        idx = bh_pos.set_index("borehole_id")
        sl, sl_lon = [], []
        for b in selected:
            if b in idx.index:
                la, lo = _utm47n_to_latlon(
                    float(idx.loc[b, "easting"]), float(idx.loc[b, "northing"])
                )
                sl.append(la); sl_lon.append(lo)
        fig.add_trace(go.Scattermapbox(
            lat=sl, lon=sl_lon, mode="lines",
            line=dict(color="#e53935", width=2),
            hoverinfo="skip", showlegend=False, name="_section_line",
        ))
    else:
        fig.add_trace(go.Scattermapbox(
            lat=[], lon=[], mode="lines",
            hoverinfo="skip", showlegend=False, name="_section_line",
        ))

    # Trace 3: query diamond
    fig.add_trace(go.Scattermapbox(
        lat=[query_lat], lon=[query_lon],
        mode="markers",
        marker=dict(size=14, color="#ff6f00", symbol="diamond",
                    opacity=1.0, allowoverlap=True),
        selected=dict(marker=dict(size=14, color="#ff6f00", opacity=1.0)),
        unselected=dict(marker=dict(size=14, color="#ff6f00", opacity=1.0)),
        hovertemplate=(
            f"<b>Query Point</b><br>E: {query_e:.1f}  N: {query_n:.1f}<extra></extra>"
        ),
        showlegend=False, name="_query",
    ))

    # Compute auto-zoom from borehole spread
    lat_span = float(lats_bh.max() - lats_bh.min())
    lon_span = float(lons_bh.max() - lons_bh.min())
    span_deg = max(lat_span, lon_span, 0.01)
    zoom = int(np.clip(np.log2(0.7 / span_deg) + 12, 9, 15))

    mapbox_cfg: dict = dict(center=dict(lat=lat_c, lon=lon_c), zoom=zoom)
    if satellite:
        mapbox_cfg["style"]  = "white-bg"
        mapbox_cfg["layers"] = [{
            "sourcetype": "raster",
            "source": [
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ],
            "below": "traces",
        }]
    else:
        mapbox_cfg["style"] = "open-street-map"

    fig.update_layout(
        mapbox=mapbox_cfg,
        height=380, margin=dict(l=0, r=0, t=36, b=0),
        showlegend=False,
        clickmode="event+select",
        dragmode="zoom",
        title=dict(
            text="Plan View — click anywhere to set coords; click borehole to add to section",
            x=0.5, font=dict(size=11),
        ),
    )
    return fig


# ── View 3: Cross-Section ─────────────────────────────────────────────────────
def _borehole_interfaces(layer_dict: dict, layer_sequence: list[str]) -> list[float]:
    interfaces, current = [], 0.0
    for layer in layer_sequence:
        interfaces.append(current)
        if layer in layer_dict:
            current = float(layer_dict[layer][1])
    interfaces.append(current)
    return interfaces


def build_crosssection_figure(df: pd.DataFrame, selected: list[str]) -> go.Figure:
    bounds   = get_layer_bounds(df)
    bpos_idx = bh_pos.set_index("borehole_id")

    valid_sel = [b for b in selected if b in bpos_idx.index]
    if len(valid_sel) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least 2 boreholes to draw a cross-section.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#888"),
        )
        return fig

    cum_dist = [0.0]
    for i in range(1, len(valid_sel)):
        b0, b1 = valid_sel[i - 1], valid_sel[i]
        de = float(bpos_idx.loc[b1, "easting"])  - float(bpos_idx.loc[b0, "easting"])
        dn = float(bpos_idx.loc[b1, "northing"]) - float(bpos_idx.loc[b0, "northing"])
        cum_dist.append(cum_dist[-1] + np.sqrt(de**2 + dn**2))
    bh_x = {bh: cum_dist[i] for i, bh in enumerate(valid_sel)}

    bh_layer = {}
    for bh in valid_sel:
        sub = bounds[bounds["borehole_id"] == bh]
        bh_layer[bh] = {r["soil_layer"]: (r["top"], r["bot"]) for _, r in sub.iterrows()}

    ifaces    = {bh: _borehole_interfaces(bh_layer[bh], LAYER_SEQUENCE) for bh in valid_sel}
    max_depth = max(max(v) for v in ifaces.values())
    max_depth = max(max_depth * 1.05, 10.0)

    fig = go.Figure()
    xs  = [bh_x[bh] for bh in valid_sel]

    # Filled layer bands
    for li, layer in enumerate(LAYER_SEQUENCE):
        tops = [ifaces[bh][li]     for bh in valid_sel]
        bots = [ifaces[bh][li + 1] for bh in valid_sel]
        if all(abs(t - b) < 1e-6 for t, b in zip(tops, bots)):
            continue
        color_hex = LAYER_COLORS.get(layer, "#aaaaaa")
        poly_x = xs + list(reversed(xs)) + [xs[0]]
        poly_y = tops + list(reversed(bots)) + [tops[0]]
        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y, fill="toself",
            fillcolor=_hex_rgba(color_hex, 0.82),
            line=dict(color=_hex_rgba(color_hex, 1.0), width=1.2),
            mode="lines",
            name=f"{layer}  {LAYER_LABELS.get(layer, '')}",
            hoverinfo="skip",
        ))

    # Borehole sticks + labels + depth ticks
    for bh in valid_sel:
        x_pos   = bh_x[bh]
        col_bot = ifaces[bh][-1]
        fig.add_shape(type="line", x0=x_pos, x1=x_pos, y0=0, y1=col_bot,
                      line=dict(color="#111111", width=1.8), layer="above")
        fig.add_annotation(x=x_pos, y=0, text=f"<b>{bh}</b>",
                           showarrow=False, yshift=14,
                           font=dict(size=10, color="#111"),
                           bgcolor="rgba(255,255,255,0.85)", borderpad=2)
        for d_tick in np.arange(10, col_bot + 1, 10):
            fig.add_annotation(x=x_pos, y=d_tick, text=f"{int(d_tick)}m",
                               showarrow=False, xshift=7,
                               font=dict(size=7, color="#444"))

    # Distance labels between adjacent boreholes
    for i in range(1, len(valid_sel)):
        b0, b1 = valid_sel[i - 1], valid_sel[i]
        fig.add_annotation(
            x=(bh_x[b0] + bh_x[b1]) / 2, y=-3,
            text=f"{bh_x[b1] - bh_x[b0]:.0f} m",
            showarrow=False, font=dict(size=9, color="#444"),
            bgcolor="rgba(255,255,255,0.7)",
        )

    # Invisible ghost grid — click target for depth/coord picking (Feature 2)
    ghost_xs = np.linspace(0, cum_dist[-1], 50)
    ghost_ys = np.linspace(0, max_depth, 35)
    GX, GY   = np.meshgrid(ghost_xs, ghost_ys)
    fig.add_trace(go.Scatter(
        x=GX.ravel(), y=GY.ravel(), mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=0)),
        selected=dict(marker=dict(color="rgba(0,0,0,0)")),
        unselected=dict(marker=dict(color="rgba(0,0,0,0)")),
        hoverinfo="skip", showlegend=False, name="_cs_ghost",
    ))

    fig.update_layout(
        height=580, margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(
            title="Horizontal Distance (m)",
            showgrid=True, gridcolor="#dde", zeroline=False,
            range=[-cum_dist[-1] * 0.02, cum_dist[-1] * 1.04],
        ),
        yaxis=dict(
            title="Depth (m)", autorange="reversed",
            showgrid=True, gridcolor="#dde",
            range=[-6, max_depth],
            zeroline=True, zerolinecolor="#888", zerolinewidth=1.5,
        ),
        legend=dict(x=1.01, y=1, bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#ccc", borderwidth=1,
                    font=dict(size=10), xanchor="left"),
        paper_bgcolor="white", plot_bgcolor="#f9fafb",
        title=dict(
            text="Geological Cross-Section  —  " + "  →  ".join(valid_sel),
            x=0.5, font=dict(size=13),
        ),
    )
    return fig


# ── UI helpers ────────────────────────────────────────────────────────────────
def _prop_card(label, value_str, unit, std_str=None, color="#1565C0"):
    unc = (f'<span style="font-size:.78rem;color:#757575;">+/- {std_str}</span>'
           if std_str else
           '<span style="font-size:.78rem;color:#bdbdbd;">uncertainty N/A</span>')
    st.markdown(
        f'<div style="background:#fff;border-left:5px solid {color};'
        f'padding:11px 16px;margin:5px 0;border-radius:6px;'
        f'box-shadow:0 1px 4px rgba(0,0,0,.07);">'
        f'<div style="font-size:.7rem;color:#9e9e9e;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:.08em;">{label}</div>'
        f'<div style="display:flex;align-items:baseline;gap:6px;margin-top:2px;">'
        f'<span style="font-size:1.45rem;font-weight:700;color:#212121;">{value_str}</span>'
        f'<span style="font-size:.82rem;color:#757575;">{unit}</span></div>'
        f'<div style="margin-top:3px;">{unc}</div></div>',
        unsafe_allow_html=True,
    )


def _confidence_bar(conf: float) -> None:
    pct = int(conf * 100)
    bar_color = "#43a047" if conf >= 0.70 else "#fb8c00" if conf >= 0.40 else "#e53935"
    label = ("High confidence"   if conf >= 0.70 else
             "Moderate confidence" if conf >= 0.40 else
             "Low confidence - add nearby boreholes")
    st.markdown(
        f'<div style="margin:4px 0 12px;">'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:.78rem;color:#757575;margin-bottom:4px;">'
        f'<span>{label}</span><span>{pct}%</span></div>'
        f'<div style="background:#e0e0e0;border-radius:4px;height:7px;">'
        f'<div style="width:{pct}%;background:{bar_color};'
        f'height:7px;border-radius:4px;"></div></div></div>',
        unsafe_allow_html=True,
    )


# ── Session state initialisation ──────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "_query_easting":   658871.0,   # internal — NOT bound to any widget key
    "_query_northing":  1522280.0,
    "_query_depth":     10.0,
    "result":           None,
    "pred_coords":      None,
    "cs_ordered":       [],
    "virtual_borehole": None,
    "_plan_sel_id":     None,
    "_cs_sel_id":       None,
    "_map_style":       "Abstract",
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:0;'>Soil Predictor</h2>"
        "<p style='color:#888;font-size:.85rem;margin-top:2px;'>"
        "Bangkok Subsoil - MRT Orange Line</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### Query Point")
    # No key= on these widgets — value= is set from the internal _query_* state.
    # Returning the widget value and writing it back to the internal key lets
    # programmatic updates (map clicks) set _query_* without triggering the
    # StreamlitAPIException that occurs when writing a widget-bound key.
    _e = st.number_input("Easting (m)",  step=1.0, format="%.1f",
                         value=float(st.session_state._query_easting))
    st.session_state._query_easting = _e

    _n = st.number_input("Northing (m)", step=1.0, format="%.1f",
                         value=float(st.session_state._query_northing))
    st.session_state._query_northing = _n

    _d = st.number_input("Depth (m)", min_value=0.1, max_value=200.0, step=0.5,
                         value=float(st.session_state._query_depth))
    st.session_state._query_depth = _d

    st.divider()
    st.markdown("#### Prediction Method")
    method_label = st.radio(
        "method", list(METHOD_MAP.keys()), index=1, label_visibility="collapsed"
    )
    method = METHOD_MAP[method_label]
    if method == "xgb":
        st.caption("XGBoost does not provide per-prediction uncertainty estimates.")

    st.divider()
    run        = st.button("Run Prediction",           type="primary", use_container_width=True)
    run_column = st.button("Predict New Borehole Here", use_container_width=True)

    st.divider()
    st.caption(
        f"**Dataset** — {len(df)} samples — "
        f"{df['borehole_id'].nunique()} borehole(s)\n\n"
        f"**Layers**: {', '.join(sorted(df['soil_layer'].unique()))}"
    )

    # Mini plan view — always visible regardless of active tab (Feature 3)
    st.plotly_chart(
        build_mini_planview(
            st.session_state._query_easting,
            st.session_state._query_northing,
        ),
        use_container_width=True,
        config={"displayModeBar": False, "staticPlot": True},
        key="sidebar_mini_plan",
    )

    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# ── Button actions ────────────────────────────────────────────────────────────
if run:
    with st.spinner("Running prediction..."):
        st.session_state.result = predictor.predict(
            st.session_state._query_easting,
            st.session_state._query_northing,
            st.session_state._query_depth,
            method,
        )
        st.session_state.pred_coords = (
            st.session_state._query_easting,
            st.session_state._query_northing,
            st.session_state._query_depth,
        )

if run_column:
    with st.spinner("Predicting full soil column 0–60 m..."):
        vb_rows = predictor.predict_column(
            st.session_state._query_easting,
            st.session_state._query_northing,
            list(range(0, 62, 2)),
            method,
        )
        st.session_state.virtual_borehole = {
            "easting": st.session_state._query_easting,
            "northing": st.session_state._query_northing,
            "method": method_label,
            "rows": vb_rows,
        }

result      = st.session_state.result
pred_coords = st.session_state.pred_coords
vbh         = st.session_state.virtual_borehole


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0;'>3D Soil Profile Viewer</h1>"
    "<p style='color:#666;margin-top:2px;'>Bangkok subsoil — MRT Orange Line dataset</p>",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "3D Borehole View",
    "3D Solid Model",
    "2D Cross-Section",
])


# ══ Tab 1 — 3D Borehole View ══════════════════════════════════════════════════
with tab1:
    col_viewer, col_results = st.columns([3, 2], gap="large")

    with col_viewer:
        pred_layer = result["layer"] if result else None
        fig3d = build_figure(df, pred_point=pred_coords,
                             pred_layer=pred_layer, virtual_bh=vbh)
        st.plotly_chart(fig3d, use_container_width=True,
                        config={"displayModeBar": True, "scrollZoom": True})

        # Virtual borehole table
        if vbh is not None:
            with st.expander(
                f"Virtual Borehole Profile — E {vbh['easting']:.0f}  "
                f"N {vbh['northing']:.0f}  [{vbh['method']}]",
                expanded=True,
            ):
                _, c_clr = st.columns([6, 1])
                with c_clr:
                    if st.button("Clear", key="clr_vbh"):
                        st.session_state.virtual_borehole = None
                        st.rerun()
                tbl = []
                for r in vbh["rows"]:
                    tbl.append({
                        "Depth (m)":       r["depth_m"],
                        "Layer":           r.get("layer", ""),
                        "Conf.":           f"{r.get('layer_confidence', 0)*100:.0f}%",
                        "su (kPa)":        f"{r['su_kpa']:.1f}"      if r.get("su_kpa")        is not None else "--",
                        "SPT-N":           f"{r['spt_n']:.0f}"       if r.get("spt_n")         is not None else "--",
                        "Unit Wt (kN/m³)": f"{r['unit_weight']:.2f}" if r.get("unit_weight")   is not None else "--",
                        "PI (%)":          f"{r['plasticity_idx']:.1f}" if r.get("plasticity_idx") is not None else "--",
                    })
                st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    with col_results:
        st.markdown("#### Prediction Results")
        if result is None:
            st.info(
                "Set coordinates in the sidebar, choose a method, "
                "then click **Run Prediction**."
            )
            st.markdown("**Soil Layer Legend**")
            for lyr, color in LAYER_COLORS.items():
                st.markdown(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{color};border-radius:2px;margin-right:6px;'
                    f'vertical-align:middle;"></span>'
                    f"**{lyr}** — {LAYER_LABELS.get(lyr, '')}",
                    unsafe_allow_html=True,
                )
        else:
            layer       = result["layer"]
            conf        = result["layer_confidence"]
            layer_color = LAYER_COLORS.get(layer, "#607d8b")
            st.markdown(
                f'<div style="background:{layer_color};color:#fff;'
                f'padding:16px 20px;border-radius:8px;margin-bottom:6px;'
                f'box-shadow:0 2px 8px rgba(0,0,0,.18);">'
                f'<div style="font-size:.72rem;opacity:.85;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.1em;">Predicted Soil Layer</div>'
                f'<div style="font-size:2.1rem;font-weight:800;margin:4px 0 2px;">{layer}</div>'
                f'<div style="font-size:.92rem;opacity:.9;">{LAYER_LABELS.get(layer, "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _confidence_bar(conf)
            st.markdown("**Predicted Properties**")
            su,  su_std  = result.get("su_kpa"),        result.get("su_kpa_std")
            spt, spt_std = result.get("spt_n"),         result.get("spt_n_std")
            uw,  uw_std  = result.get("unit_weight"),   result.get("unit_weight_std")
            pi,  pi_std  = result.get("plasticity_idx"),result.get("plasticity_idx_std")
            _prop_card("Undrained Shear Strength su",
                       f"{su:.1f}"  if su  is not None else "--", "kPa",
                       f"{su_std:.1f} kPa" if su_std  is not None else None, "#1565C0")
            _prop_card("SPT Blow Count N",
                       f"{spt:.0f}" if spt is not None else "--", "blows / 300 mm",
                       f"{spt_std:.1f}" if spt_std is not None else None, "#F9A825")
            _prop_card("Unit Weight",
                       f"{uw:.2f}"  if uw  is not None else "--", "kN/m³",
                       f"{uw_std:.2f} kN/m³" if uw_std is not None else None, "#43a047")
            _prop_card("Plasticity Index PI",
                       f"{pi:.1f}"  if pi  is not None else "--", "%",
                       f"{pi_std:.1f}%" if pi_std is not None else None, "#8e24aa")
            st.caption(
                f"Method: **{method_label}**  |  "
                f"E {st.session_state._query_easting:.1f}  "
                f"N {st.session_state._query_northing:.1f}  |  "
                f"Depth {st.session_state._query_depth:.1f} m"
            )

    with st.expander("Borehole Dataset", expanded=False):
        show_cols = [
            "borehole_id", "depth_m", "depth_top_m", "depth_bot_m",
            "soil_layer", "soil_desc", "consistency",
            "su_kpa", "su_method", "spt_n",
            "unit_weight", "plasticity_idx", "liquid_limit", "plastic_limit",
        ]
        st.dataframe(
            df[[c for c in show_cols if c in df.columns]].reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )


# ══ Tab 2 — 3D Solid Model ════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<p style='color:#666;font-size:.9rem;margin-bottom:8px;'>"
        "Interpolated layer surfaces rendered as a 3D solid model. "
        "Drag to rotate, scroll to zoom.</p>",
        unsafe_allow_html=True,
    )
    with st.spinner("Building 3D solid model..."):
        solid_fig = build_solid_figure(df)
    st.plotly_chart(solid_fig, use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True})
    st.caption(
        "Layer surfaces interpolated using linear triangulation. "
        "Areas outside the borehole convex hull are not extrapolated."
    )


# ══ Tab 3 — 2D Cross-Section ══════════════════════════════════════════════════
with tab3:
    all_bh_ids = sorted(df["borehole_id"].unique().tolist())
    col_plan, col_cs = st.columns([2, 3], gap="large")

    # ── Left column: interactive plan view ────────────────────────────────────
    with col_plan:
        st.markdown("#### Plan View")

        # Map-style toggle
        _prev_style = st.session_state.get("_map_style", "Abstract")
        _map_style  = st.radio(
            "Map style",
            ["Abstract", "Street Map", "Satellite"],
            horizontal=True,
            index=["Abstract", "Street Map", "Satellite"].index(_prev_style),
        )
        if _map_style != _prev_style:
            st.session_state._plan_sel_id = None   # reset fingerprint on style change
        st.session_state._map_style = _map_style

        st.markdown(
            "<p style='font-size:.85rem;color:#666;margin-top:-6px;margin-bottom:4px;'>"
            "Click anywhere to set E/N coords. Click a borehole to also add it to the section.</p>",
            unsafe_allow_html=True,
        )

        _plan_pts  = None   # selection points from whichever chart fires
        _is_mapbox = False

        if _map_style == "Abstract":
            plan_fig   = build_planview_figure(
                st.session_state.cs_ordered,
                st.session_state._query_easting,
                st.session_state._query_northing,
            )
            plan_event = st.plotly_chart(
                plan_fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["points"],
                config={"displayModeBar": False, "scrollZoom": False},
                key="plan_view_chart",
            )
            if plan_event and plan_event.selection and plan_event.selection.points:
                _plan_pts  = plan_event.selection.points
                _is_mapbox = False
        else:
            mapbox_fig = build_mapbox_figure(
                st.session_state.cs_ordered,
                st.session_state._query_easting,
                st.session_state._query_northing,
                satellite=(_map_style == "Satellite"),
            )
            mapbox_event = st.plotly_chart(
                mapbox_fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["points"],
                config={"displayModeBar": False, "scrollZoom": True},
                key="mapbox_view_chart",
            )
            if mapbox_event and mapbox_event.selection and mapbox_event.selection.points:
                _plan_pts  = mapbox_event.selection.points
                _is_mapbox = True

        # Process plan-view click — shared for Abstract and Mapbox (Feature 1)
        if _plan_pts is not None:
            sel_id = str([(p.get("curve_number"), p.get("point_index")) for p in _plan_pts])

            if sel_id != st.session_state._plan_sel_id:
                st.session_state._plan_sel_id = sel_id
                pt    = _plan_pts[0]
                curve = pt.get("curve_number", -1)

                if _is_mapbox:
                    lat_c = float(pt.get("lat", 13.75))
                    lon_c = float(pt.get("lon", 100.5))
                    new_e, new_n = _latlon_to_utm47n(lat_c, lon_c)
                else:
                    new_e = float(pt["x"])
                    new_n = float(pt["y"])

                changed = (
                    abs(new_e - st.session_state._query_easting)  > 1.0
                    or abs(new_n - st.session_state._query_northing) > 1.0
                )

                # Borehole click (curve 1): also add to section selection
                if curve == 1:
                    raw   = pt.get("customdata")
                    bh_id = raw[0] if isinstance(raw, list) else raw
                    if bh_id and bh_id not in st.session_state.cs_ordered:
                        st.session_state.cs_ordered.append(bh_id)
                        changed = True

                # curve 3 = query-point diamond: ignore coordinate update
                if curve != 3 and changed:
                    st.session_state._query_easting  = new_e
                    st.session_state._query_northing = new_n
                    st.rerun()
                elif curve == 1 and changed:
                    st.rerun()

        # Ordered selection list with inline X buttons
        if st.session_state.cs_ordered:
            st.markdown(
                "<p style='font-size:.82rem;font-weight:600;margin-bottom:4px;'>"
                "Selected (in order):</p>",
                unsafe_allow_html=True,
            )
            for i, bh in enumerate(list(st.session_state.cs_ordered)):
                c_lbl, c_btn = st.columns([5, 1])
                with c_lbl:
                    st.markdown(
                        f'<div style="padding:3px 0;font-size:.9rem;">'
                        f'<span style="background:#e53935;color:#fff;border-radius:50%;'
                        f'padding:1px 6px;font-size:.75rem;font-weight:700;margin-right:6px;">'
                        f'{i+1}</span>{bh}</div>',
                        unsafe_allow_html=True,
                    )
                with c_btn:
                    if st.button("✕", key=f"rm_{bh}_{i}", help=f"Remove {bh}"):
                        st.session_state.cs_ordered.remove(bh)
                        st.rerun()
        else:
            st.caption("No boreholes selected yet.")

        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear selection", use_container_width=True):
                st.session_state.cs_ordered = []
                st.rerun()
        with col_b:
            if st.button("Select all", use_container_width=True):
                st.session_state.cs_ordered = all_bh_ids[:]
                st.rerun()

        st.markdown("**Legend**")
        for lyr, color in LAYER_COLORS.items():
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{color};border-radius:2px;margin-right:6px;'
                f'vertical-align:middle;"></span>'
                f"**{lyr}** — {LAYER_LABELS.get(lyr, '')}",
                unsafe_allow_html=True,
            )

    # ── Right column: cross-section ───────────────────────────────────────────
    with col_cs:
        st.markdown("#### Geological Cross-Section")
        st.markdown(
            "<p style='font-size:.85rem;color:#666;margin-top:-6px;margin-bottom:4px;'>"
            "Click on the section to set E/N and depth for prediction.</p>",
            unsafe_allow_html=True,
        )

        if len(st.session_state.cs_ordered) < 2:
            st.info(
                "Click 2 or more boreholes on the plan view to draw a cross-section. "
                "Order matters — section is drawn in the order you select them."
            )
        else:
            cs_fig   = build_crosssection_figure(df, st.session_state.cs_ordered)
            cs_event = st.plotly_chart(
                cs_fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["points"],
                config={"displayModeBar": True, "scrollZoom": False},
                key="cs_chart",
            )

            # Process cross-section click (Feature 2)
            if cs_event and cs_event.selection and cs_event.selection.points:
                pts    = cs_event.selection.points
                sel_id = str([(p.get("curve_number"), p.get("point_index")) for p in pts])

                if sel_id != st.session_state._cs_sel_id:
                    st.session_state._cs_sel_id = sel_id
                    pt      = pts[0]
                    x_click = float(pt["x"])
                    d_click = max(0.0, float(pt["y"]))

                    # Interpolate E/N from section line geometry
                    bpos_idx = bh_pos.set_index("borehole_id")
                    valid    = [b for b in st.session_state.cs_ordered
                                if b in bpos_idx.index]

                    if len(valid) >= 2:
                        cum_d = [0.0]
                        for i in range(1, len(valid)):
                            b0, b1 = valid[i - 1], valid[i]
                            de = float(bpos_idx.loc[b1, "easting"])  - float(bpos_idx.loc[b0, "easting"])
                            dn = float(bpos_idx.loc[b1, "northing"]) - float(bpos_idx.loc[b0, "northing"])
                            cum_d.append(cum_d[-1] + np.sqrt(de**2 + dn**2))

                        x_c   = max(0.0, min(x_click, cum_d[-1]))
                        new_e = float(bpos_idx.loc[valid[0], "easting"])
                        new_n = float(bpos_idx.loc[valid[0], "northing"])
                        for i in range(len(valid) - 1):
                            if cum_d[i] <= x_c <= cum_d[i + 1]:
                                span  = cum_d[i + 1] - cum_d[i]
                                t     = (x_c - cum_d[i]) / span if span > 0 else 0.0
                                b0, b1 = valid[i], valid[i + 1]
                                new_e = (float(bpos_idx.loc[b0, "easting"])
                                         + t * (float(bpos_idx.loc[b1, "easting"])
                                                - float(bpos_idx.loc[b0, "easting"])))
                                new_n = (float(bpos_idx.loc[b0, "northing"])
                                         + t * (float(bpos_idx.loc[b1, "northing"])
                                                - float(bpos_idx.loc[b0, "northing"])))
                                break

                        st.session_state._query_easting  = new_e
                        st.session_state._query_northing = new_n
                        st.session_state._query_depth    = d_click
                        st.rerun()

            st.caption(
                "Layer boundaries interpolated between boreholes — "
                "click anywhere on the section to update the query coordinates and depth."
            )
