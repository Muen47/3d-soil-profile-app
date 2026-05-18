import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

import streamlit as st
import streamlit.components.v1 as components
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
DATA_PATH = os.path.join(_ROOT, "data", "bangkok_boring_logs_real.csv")
MODEL_DIR = os.path.join(_ROOT, "models")

LAYER_SEQUENCE = ["MG", "VSC", "SOC", "SC", "SS", "MSC", "FS"]

LAYER_COLORS = {
    "MG":  "#8D6E63",
    "VSC": "#4FC3F7",   # Soft Clay (merged VSC + SOC)
    "SOC": "#4FC3F7",   # Soft Clay (merged VSC + SOC)
    "SC":  "#1565C0",   # Medium Stiff Clay
    "MSC": "#0D2B6B",   # Stiff Clay
    "FS":  "#F9A825",   # Sand (merged FS + SS)
    "SS":  "#F9A825",   # Sand (merged FS + SS)
}
LAYER_LABELS = {
    "MG":  "Made Ground / Fill",
    "VSC": "Soft Clay",           # merged display: VSC + SOC → Soft Clay
    "SOC": "Soft Clay",           # merged display: VSC + SOC → Soft Clay
    "SC":  "Medium Stiff Clay",
    "MSC": "Stiff Clay",
    "FS":  "Sand",                # merged display: FS + SS → Sand
    "SS":  "Sand",                # merged display: FS + SS → Sand
}
# Unique (color, display_label) pairs in LAYER_SEQUENCE order — use for HTML legends
_LEGEND_ITEMS: list[tuple[str, str]] = []
_seen_lbl: set[str] = set()
for _lyr in LAYER_SEQUENCE:
    _lbl = LAYER_LABELS.get(_lyr, _lyr)
    if _lbl not in _seen_lbl:
        _seen_lbl.add(_lbl)
        _LEGEND_ITEMS.append((LAYER_COLORS[_lyr], _lbl))
del _seen_lbl, _lyr, _lbl

# ── Change 1: Bridge from Excel soil names → existing layer codes ─────────────
SOIL_LABEL_MAP = {
    "Fill":              "MG",
    "Fill Material":     "MG",
    "Soft clay":         "VSC",
    "Medium stiff clay": "SC",
    "1st stiff clay":    "MSC",
    "2nd stiff clay":    "MSC",
    "3rd stiff clay":    "MSC",
    "1st sand":          "FS",
    "2nd sand":          "SS",
    "3rd sand":          "SS",
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


# ── Change 3: Cached Excel loader for lab data ────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_soil_props(file_bytes: bytes) -> dict:
    import io

    # Property-only sheets carry ONE value column. Their header text is
    # unreliable ("Su" and "SPT" sheets mislabel it "Total Unit Weight
    # (kN/m3)"), so the value column is named from the SHEET name instead.
    sheet_prop = {
        "unit weight": "unit_weight",
        "su":          "su_kpa",
        "spt":         "spt_n",
        "wn":          "wn_pct",
        "ll":          "ll_pct",
        "pl":          "pl_pct",
        "pi":          "pi_pct",
    }

    def _canon(header: str) -> str:
        # Canonical name for a value column header on the wide soil sheets,
        # tolerant of unit suffixes / wording variants.
        low = str(header).strip().lower()
        if "unit weight" in low:
            return "unit_weight"
        if "spt" in low:
            return "spt_n"
        if "undrained" in low or low == "su" or low.startswith("su"):
            return "su_kpa"
        if "water content" in low or low == "wn":
            return "wn_pct"
        if "liquid limit" in low or low == "ll":
            return "ll_pct"
        if "plasticity" in low or low == "pi":
            return "pi_pct"
        if "plastic limit" in low or low == "pl":
            return "pl_pct"
        return str(header).strip()

    xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    result = {}
    for sheet in xl.sheet_names:
        raw = pd.read_excel(
            io.BytesIO(file_bytes), sheet_name=sheet,
            header=None, engine="openpyxl",
        )
        # Strip phantom all-empty leading/trailing rows & columns so the
        # layout is clean regardless of openpyxl's dimension behavior.
        raw = (raw.dropna(axis=1, how="all")
                  .dropna(axis=0, how="all")
                  .reset_index(drop=True))
        raw.columns = range(raw.shape[1])
        if raw.shape[0] < 3 or raw.shape[1] < 2:
            continue

        # Locate the label row dynamically (immune to leading blank
        # rows/columns): the first row containing a "Soil" cell and a
        # "Depth" cell.
        hdr = soil_col = depth_col = None
        for r in range(min(15, raw.shape[0])):
            row_vals = [str(v).strip().lower() for v in raw.iloc[r].tolist()]
            s_idx = next((i for i, v in enumerate(row_vals) if v == "soil"), None)
            d_idx = next((i for i, v in enumerate(row_vals) if "depth" in v), None)
            if s_idx is not None and d_idx is not None:
                hdr, soil_col, depth_col = r, s_idx, d_idx
                break
        if hdr is None:
            continue

        label_row = raw.iloc[hdr].tolist()
        prop_name = sheet_prop.get(str(sheet).strip().lower())

        keep_idx, final_cols, seen = [], [], set()
        for c in range(raw.shape[1]):
            if c == soil_col:
                name = "soil_type"
            elif c == depth_col:
                name = "depth_m"
            elif prop_name is not None:
                name = prop_name          # property-only sheet → name by sheet
            else:
                name = _canon(label_row[c])
            if name in seen:
                continue
            seen.add(name)
            keep_idx.append(c)
            final_cols.append(name)

        data = raw.iloc[hdr + 1:, :].copy().reset_index(drop=True)
        data = data.iloc[:, keep_idx]
        data.columns = final_cols

        for col in final_cols:
            if col != "soil_type":
                data[col] = pd.to_numeric(data[col], errors="coerce")
        if "soil_type" in data.columns:
            data["soil_type"] = data["soil_type"].astype(str).str.strip()

        if "depth_m" not in data.columns:
            continue
        data = data.dropna(subset=["depth_m"]).reset_index(drop=True)
        if data.empty:
            continue
        result[sheet] = data
    return result


predictor = _get_predictor()
df        = _get_df()
bh_pos    = _get_bh_pos(df)

# Default Mapbox viewport — computed once so _SS_DEFAULTS can reference them
_MAP_LATS, _MAP_LONS = _utm47n_to_latlon_arr(
    bh_pos["easting"].values, bh_pos["northing"].values
)
_MAP_LAT_C = float(np.mean(_MAP_LATS))
_MAP_LON_C = float(np.mean(_MAP_LONS))
_MAP_SPAN  = max(float(_MAP_LATS.max()) - float(_MAP_LATS.min()),
                 float(_MAP_LONS.max()) - float(_MAP_LONS.min()), 0.01)
_MAP_ZOOM  = int(np.clip(np.log2(0.7 / _MAP_SPAN) + 12, 9, 15))


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
        f"Layer: {LAYER_LABELS.get(row['soil_layer'], row['soil_layer'])}<br>"
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
    virtual_bhs: list | None = None,
    depth_limit: float = 80.0,
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
    _lbl_seen: set[str] = set()
    for layer, color in LAYER_COLORS.items():
        sub = df[df["soil_layer"] == layer]
        if sub.empty:
            continue
        _lbl = LAYER_LABELS.get(layer, layer)
        fig.add_trace(go.Scatter3d(
            x=sub["easting"], y=sub["northing"], z=-sub["depth_m"],
            mode="markers",
            marker=dict(size=9, color=color, opacity=0.88,
                        line=dict(color="white", width=0.6)),
            name=_lbl,
            showlegend=_lbl not in _lbl_seen,
            text=sub.apply(_hover, axis=1).tolist(),
            hovertemplate="%{text}<extra></extra>",
        ))
        _lbl_seen.add(_lbl)

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
                f"Depth: {d:.1f} m<br>Layer: {LAYER_LABELS.get(pred_layer, pred_layer) if pred_layer else '--'}<extra></extra>"
            ),
        ))

    # Virtual boreholes (cross markers + dashed stick, one per entry)
    for vbh in (virtual_bhs or []):
        ve   = vbh["easting"]
        vn   = vbh["northing"]
        vname = vbh.get("name", f"VBH-{int(ve)}-{int(vn)}")
        rows = [r for r in vbh["rows"] if r["depth_m"] <= depth_limit]
        all_depths = [r["depth_m"] for r in rows]

        fig.add_trace(go.Scatter3d(
            x=[ve] * len(all_depths), y=[vn] * len(all_depths),
            z=[-d for d in all_depths],
            mode="lines",
            line=dict(color="#444444", width=3, dash="dash"),
            name=vname,
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
                    f"<b>{vname}</b><br>Depth: {r['depth_m']:.0f} m<br>"
                    f"Layer: {LAYER_LABELS.get(layer, layer)}<br>su: {su_s}  SPT-N: {spt_s}<br>"
                    f"Confidence: {r.get('layer_confidence', 0)*100:.0f}%"
                )
            fig.add_trace(go.Scatter3d(
                x=[ve] * len(layer_rows), y=[vn] * len(layer_rows),
                z=[-r["depth_m"] for r in layer_rows],
                mode="markers",
                marker=dict(size=11, color=color, symbol="cross",
                            opacity=0.95, line=dict(color="white", width=0.8)),
                name=f"{vname} · {layer}",
                text=texts, hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Easting (m)", yaxis_title="Northing (m)",
            zaxis_title="- Depth (m)", bgcolor="#f0f4f8",
            zaxis=dict(range=[-depth_limit, 0]),
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


# ── Change 5: Property vs Depth comparison chart ──────────────────────────────
def build_property_depth_chart(
    all_props: dict,
    prop_col: str,
    prop_label: str,
    pred_depth: float | None = None,
    pred_value: float | None = None,
) -> go.Figure:
    frames = []
    for df in all_props.values():
        if prop_col in df.columns and "soil_type" in df.columns:
            sub = df[["soil_type", "depth_m", prop_col]].dropna()
            if not sub.empty:
                frames.append(sub)
    fig = go.Figure()
    if not frames:
        fig.add_annotation(text=f"No {prop_label} data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=12, color="#888"))
    else:
        merged = pd.concat(frames, ignore_index=True)
        _seen: set[str] = set()
        for soil_name, grp in merged.groupby("soil_type"):
            layer_code = SOIL_LABEL_MAP.get(str(soil_name).strip(), "MG")
            color      = LAYER_COLORS.get(layer_code, "#888888")
            disp_label = LAYER_LABELS.get(layer_code, str(soil_name))
            valid = grp.dropna(subset=[prop_col, "depth_m"])
            if valid.empty:
                continue
            fig.add_trace(go.Scatter(
                x=valid[prop_col], y=valid["depth_m"],
                mode="markers",
                marker=dict(size=7, color=color, opacity=0.7,
                            line=dict(color="white", width=0.5)),
                name=disp_label,
                showlegend=disp_label not in _seen,
                legendgroup=disp_label,
                hovertemplate=(
                    f"<b>{disp_label}</b><br>"
                    f"Depth: %{{y:.1f}} m<br>{prop_label}: %{{x:.2f}}<extra></extra>"
                ),
            ))
            _seen.add(disp_label)
    # Predicted point — red diamond, on top of everything
    _pd = pred_depth if pred_depth is not None else float("nan")
    _pv = pred_value if pred_value is not None else float("nan")
    if not (np.isnan(_pd) or np.isnan(_pv)):
        fig.add_trace(go.Scatter(
            x=[_pv], y=[_pd], mode="markers",
            marker=dict(size=20, color="#e53935", symbol="diamond",
                        opacity=1.0, line=dict(color="white", width=2.5)),
            name="Predicted (this point)",
            hovertemplate=(
                f"<b>Predicted</b><br>"
                f"Depth: {_pd:.1f} m<br>{prop_label}: {_pv:.2f}<extra></extra>"
            ),
        ))
    fig.update_layout(
        height=620,
        margin=dict(l=60, r=25, t=20, b=90),
        xaxis_title=prop_label,
        yaxis=dict(title="Depth (m)", autorange="reversed",
                   showgrid=True, gridcolor="#e0e0e0", zeroline=False),
        xaxis=dict(showgrid=True, gridcolor="#e0e0e0", zeroline=False),
        legend=dict(orientation="h", yanchor="top", y=-0.10,
                    xanchor="center", x=0.5, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ddd", borderwidth=1),
        paper_bgcolor="white", plot_bgcolor="#f9fafb",
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


def build_solid_figure(df: pd.DataFrame, depth_limit: float = 80.0) -> go.Figure:
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
        _disp_lbl = LAYER_LABELS.get(layer, layer)
        for Z, lbl in [(Zt, "top"), (Zb, "bot")]:
            fig.add_trace(go.Surface(
                x=GE, y=GN, z=-Z,
                colorscale=[[0, color_rgb], [1, color_rgb]],
                showscale=False, opacity=0.7,
                name=f"{_disp_lbl} {lbl}", showlegend=False,
                hovertemplate=(
                    f"<b>{_disp_lbl}</b><br>{lbl}<br>"
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

    _lbl_seen_solid: set[str] = set()
    for layer, color in LAYER_COLORS.items():
        _lbl = LAYER_LABELS.get(layer, layer)
        if _lbl in _lbl_seen_solid:
            continue
        _lbl_seen_solid.add(_lbl)
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=10, color=color, opacity=0.85),
            name=_lbl,
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Easting (m)", yaxis_title="Northing (m)",
            zaxis_title="- Depth (m)", bgcolor="#e8edf2",
            zaxis=dict(range=[-depth_limit, 0]),
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
    virtual_bhs: list | None = None,
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
        marker=dict(size=13, color="#43a047", symbol="diamond",
                    line=dict(color="white", width=2), opacity=1.0),
        selected=dict(marker=dict(size=13, color="#43a047", opacity=1.0)),
        unselected=dict(marker=dict(size=13, color="#43a047", opacity=1.0)),
        hovertemplate=(
            f"<b>Query Point</b><br>E: {query_e:.1f}  N: {query_n:.1f}<extra></extra>"
        ),
        showlegend=False, name="_query",
    ))

    # Virtual borehole star markers (appended after trace 3 — not in click-handler contract)
    if virtual_bhs:
        fig.add_trace(go.Scatter(
            x=[v["easting"]  for v in virtual_bhs],
            y=[v["northing"] for v in virtual_bhs],
            mode="markers+text",
            marker=dict(size=14, color="#43a047", symbol="star",
                        line=dict(color="white", width=1.5), opacity=1.0),
            text=[v.get("name", "") for v in virtual_bhs],
            textposition="top center",
            textfont=dict(size=8, color="#43a047"),
            customdata=[v.get("name", "") for v in virtual_bhs],
            hovertemplate="<b>%{customdata}</b><br>E: %{x:.0f}  N: %{y:.0f}<extra></extra>",
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=1.0)),
            showlegend=False, name="_vbh_markers",
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

    # Auto-fit range — include boreholes + query point + any virtual BH locations
    all_e = list(bh_pos["easting"]) + [query_e]
    all_n = list(bh_pos["northing"]) + [query_n]
    if virtual_bhs:
        all_e += [v["easting"]  for v in virtual_bhs]
        all_n += [v["northing"] for v in virtual_bhs]
    e_span = max(float(max(all_e)) - float(min(all_e)), 500.0)
    n_span = max(float(max(all_n)) - float(min(all_n)), 500.0)
    e_margin = e_span * 0.12
    n_margin = n_span * 0.12
    x_range = [float(min(all_e)) - e_margin, float(max(all_e)) + e_margin]
    y_range = [float(min(all_n)) - n_margin, float(max(all_n)) + n_margin]

    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(title="Easting (m)", tickformat=".0f", range=x_range),
        yaxis=dict(title="Northing (m)", tickformat=".0f", scaleanchor="x",
                   range=y_range),
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
    center_lat: float | None = None,
    center_lon: float | None = None,
    zoom: int | None = None,
    virtual_bhs: list | None = None,
) -> go.Figure:
    selected_set = set(selected)

    lats_bh, lons_bh = _utm47n_to_latlon_arr(
        bh_pos["easting"].values, bh_pos["northing"].values
    )
    query_lat, query_lon = _utm47n_to_latlon(query_e, query_n)

    lat_c = center_lat if center_lat is not None else float(np.mean(lats_bh))
    lon_c = center_lon if center_lon is not None else float(np.mean(lons_bh))

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

    # Virtual borehole star markers
    if virtual_bhs:
        vbh_e_arr = np.array([v["easting"]  for v in virtual_bhs], dtype=float)
        vbh_n_arr = np.array([v["northing"] for v in virtual_bhs], dtype=float)
        vbh_lats, vbh_lons = _utm47n_to_latlon_arr(vbh_e_arr, vbh_n_arr)
        vbh_names = [v.get("name", "") for v in virtual_bhs]
        fig.add_trace(go.Scattermapbox(
            lat=vbh_lats.tolist(), lon=vbh_lons.tolist(),
            mode="markers+text",
            marker=dict(size=14, color="#43a047", symbol="star", opacity=1.0),
            text=vbh_names,
            textposition="top right",
            customdata=vbh_names,
            hovertemplate="<b>%{customdata}</b><br>Lat: %{lat:.5f}  Lon: %{lon:.5f}<extra></extra>",
            showlegend=False, name="_vbh_markers",
        ))

    # Trace 3: query point — red marker (circle is reliably supported on all mapbox versions)
    fig.add_trace(go.Scattermapbox(
        lat=[query_lat], lon=[query_lon],
        mode="markers",
        marker=dict(size=16, color="#43a047", symbol="circle", opacity=1.0),
        hovertemplate=(
            f"<b>Query Point</b><br>E: {query_e:.1f}  N: {query_n:.1f}<extra></extra>"
        ),
        showlegend=False, name="_query",
    ))

    # Zoom: use passed-in value or compute from borehole spread
    if zoom is None:
        lat_span = float(lats_bh.max()) - float(lats_bh.min())
        lon_span = float(lons_bh.max()) - float(lons_bh.min())
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


def build_crosssection_figure(df: pd.DataFrame, selected: list[str], depth_limit: float = 80.0) -> go.Figure:
    bounds   = get_layer_bounds(df)   # borehole_id, soil_layer, top, bot
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

    # ── Cumulative horizontal distance along the section line ─────────────────
    cum_dist = [0.0]
    for i in range(1, len(valid_sel)):
        b0, b1 = valid_sel[i - 1], valid_sel[i]
        de = float(bpos_idx.loc[b1, "easting"])  - float(bpos_idx.loc[b0, "easting"])
        dn = float(bpos_idx.loc[b1, "northing"]) - float(bpos_idx.loc[b0, "northing"])
        cum_dist.append(cum_dist[-1] + np.sqrt(de**2 + dn**2))
    bh_x = {bh: cum_dist[i] for i, bh in enumerate(valid_sel)}
    xs   = [bh_x[bh] for bh in valid_sel]

    # ── Layer boundary lookup: actual top/bot from data at each borehole ──────
    # bh_data[bh][layer] = (top_m, bot_m)  or absent
    bh_data: dict[str, dict[str, tuple[float, float]]] = {}
    for bh in valid_sel:
        sub = bounds[bounds["borehole_id"] == bh]
        bh_data[bh] = {
            row["soil_layer"]: (float(row["top"]), float(row["bot"]))
            for _, row in sub.iterrows()
        }

    # ── Fix 1: remove isolated blobs ─────────────────────────────────────────
    # A layer at borehole i is only drawn if at least one neighbour (i-1 or
    # i+1) also has that layer with non-zero thickness.  Isolated single-
    # borehole presences are treated as absent to avoid lens/blob artefacts.
    def _has_nonzero(bh, layer):
        if layer not in bh_data[bh]:
            return False
        t, b = bh_data[bh][layer]
        return (b - t) > 0.01

    effective: dict[str, dict[str, bool]] = {}
    for idx, bh in enumerate(valid_sel):
        effective[bh] = {}
        for layer in LAYER_SEQUENCE:
            present = _has_nonzero(bh, layer)
            if present:
                left  = idx > 0               and _has_nonzero(valid_sel[idx - 1], layer)
                right = idx < len(valid_sel)-1 and _has_nonzero(valid_sel[idx + 1], layer)
                if not left and not right:
                    present = False   # isolated → treat as absent
            effective[bh][layer] = present

    # ── Fix 2: shared boundary stack — guarantees zero gaps ──────────────────
    # Compute N+1 interface depths per borehole (N = len(LAYER_SEQUENCE)).
    # boundary[i] is the depth of the interface ABOVE layer i.
    # Adjacent layers share the same boundary array index, so their
    # interpolated edge curves are numerically identical → no white gaps.
    bh_bdry: dict[str, list[float]] = {}
    for bh in valid_sel:
        bdry = [0.0]
        for layer in LAYER_SEQUENCE:
            if effective[bh][layer]:
                _, bot = bh_data[bh][layer]
                bdry.append(max(float(bot), bdry[-1]))  # monotone downward
            else:
                bdry.append(bdry[-1])                   # pinch-out
        bh_bdry[bh] = bdry

    # ── Axis limits ───────────────────────────────────────────────────────────
    max_depth = min(max(bh_bdry[bh][-1] for bh in valid_sel) * 1.05, depth_limit)
    max_depth = max(max_depth, 10.0)

    # Dense x-grid — 200 pts per segment for professional smooth curves
    n_pts   = max(2, (len(valid_sel) - 1) * 200 + 1)
    x_dense = np.linspace(0, cum_dist[-1], n_pts)

    # Pre-interpolate all N+1 boundary curves with PCHIP monotone spline.
    # PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is preferred
    # over CubicSpline because it never overshoots/oscillates between nodes,
    # which keeps boundary curves well-behaved and gap-free.
    from scipy.interpolate import PchipInterpolator

    bdry_curves: list[np.ndarray] = []
    for bi in range(len(LAYER_SEQUENCE) + 1):
        vals = [min(bh_bdry[bh][bi], depth_limit) for bh in valid_sel]
        # Cap MG bottom boundary (index 1 = bottom of first layer = MG) at 5 m
        if bi == 1:
            vals = [min(v, 5.0) for v in vals]
        if len(xs) >= 2:
            curve = PchipInterpolator(xs, vals)(x_dense)
            curve = np.clip(curve, 0.0, depth_limit)   # hard clamp to valid range
        else:
            curve = np.full(len(x_dense), vals[0])
        bdry_curves.append(curve)

    # Enforce strictly non-decreasing depth order across all boundaries.
    # PCHIP can produce tiny undershoots (~1e-10) near pinch-out zones where
    # adjacent boundary values are equal.  This clamp collapses them to zero
    # thickness without creating any visible gap.
    for bi in range(1, len(bdry_curves)):
        bdry_curves[bi] = np.maximum(bdry_curves[bi], bdry_curves[bi - 1])

    fig = go.Figure()

    # ── Filled layer bands — no border lines, so no colour seams ─────────────
    _lbl_seen_cs: set[str] = set()
    for li, layer in enumerate(LAYER_SEQUENCE):
        top_curve = bdry_curves[li]
        bot_curve = bdry_curves[li + 1]

        if np.all(np.abs(bot_curve - top_curve) < 1e-6):
            continue   # zero thickness everywhere — skip

        poly_x = list(x_dense) + list(x_dense[::-1]) + [x_dense[0]]
        poly_y = list(top_curve) + list(bot_curve[::-1]) + [float(top_curve[0])]

        color_hex = LAYER_COLORS.get(layer, "#aaaaaa")
        _lbl = LAYER_LABELS.get(layer, layer)
        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y,
            fill="toself",
            fillcolor=_hex_rgba(color_hex, 0.82),
            line=dict(width=0),          # no border — prevents seam artefacts
            mode="lines",
            name=_lbl,
            showlegend=_lbl not in _lbl_seen_cs,
            hoverinfo="skip",
        ))
        _lbl_seen_cs.add(_lbl)

    # Draw layer-interface lines on top as explicit separate traces so
    # boundaries are still visible without double-border colour bleeding
    for li in range(1, len(LAYER_SEQUENCE)):
        curve = bdry_curves[li]
        if np.all(np.abs(curve - bdry_curves[li - 1]) < 1e-6) and \
           np.all(np.abs(curve - bdry_curves[li + 1]) < 1e-6):
            continue   # flat (absent on both sides) — nothing to draw
        fig.add_trace(go.Scatter(
            x=list(x_dense), y=list(curve),
            mode="lines",
            line=dict(color="rgba(80,80,80,0.45)", width=0.8),
            hoverinfo="skip", showlegend=False, name="_iface",
        ))

    # ── Borehole sticks + labels + depth ticks ────────────────────────────────
    for bh in valid_sel:
        x_pos     = bh_x[bh]
        stick_bot = min(bh_bdry[bh][-1], depth_limit)
        if bh_data[bh]:
            stick_bot = min(max(b for _, b in bh_data[bh].values()), depth_limit)
        fig.add_shape(type="line", x0=x_pos, x1=x_pos, y0=0, y1=stick_bot,
                      line=dict(color="#111111", width=1.8), layer="above")
        fig.add_annotation(x=x_pos, y=0, text=f"<b>{bh}</b>",
                           showarrow=False, yshift=14,
                           font=dict(size=10, color="#111"),
                           bgcolor="rgba(255,255,255,0.85)", borderpad=2)
        for d_tick in np.arange(10, stick_bot + 1, 10):
            fig.add_annotation(x=x_pos, y=d_tick, text=f"{int(d_tick)}m",
                               showarrow=False, xshift=7,
                               font=dict(size=7, color="#444"))

    # ── Distance labels between adjacent boreholes ────────────────────────────
    for i in range(1, len(valid_sel)):
        b0, b1 = valid_sel[i - 1], valid_sel[i]
        fig.add_annotation(
            x=(bh_x[b0] + bh_x[b1]) / 2, y=-3,
            text=f"{bh_x[b1] - bh_x[b0]:.0f} m",
            showarrow=False, font=dict(size=9, color="#444"),
            bgcolor="rgba(255,255,255,0.7)",
        )

    # ── Invisible ghost grid — click target for depth/coord picking ───────────
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
    "virtual_boreholes": [],
    "_plan_sel_id":     None,
    "_cs_sel_id":       None,
    "_map_style":       "Abstract",
    "_map_center_lat":  _MAP_LAT_C,
    "_map_center_lon":  _MAP_LON_C,
    "_map_zoom":        _MAP_ZOOM,
    "_max_display_depth": 80,
    "_cs_text_raw":     "",   # raw text in the borehole-selection text box
    "_cs_ordered_fp":   "",   # fingerprint of cs_ordered last reflected in the text box
    "uploaded_props_data":    None,   # dict[sheet_name -> DataFrame] from _load_soil_props()
    "_user_uploaded_xl_file": None,   # raw bytes of user's uploaded xlsx (None = using default)
    "_user_uploaded_json":    None,   # raw text of user's uploaded validation_results.json
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
    st.markdown("#### Display Settings")
    st.number_input(
        "Max Display Depth (m)",
        min_value=20, max_value=300,
        value=st.session_state._max_display_depth,
        step=5,
        key="_max_display_depth",
        help="Hides data deeper than this value and clamps the Z / Y axis on all views.",
    )

    st.divider()
    run        = st.button("Run Prediction",           type="primary", use_container_width=True)
    run_column = st.button("Predict New Borehole Here", use_container_width=True)

    # ── Change 4: Lab data uploader ───────────────────────────────────────────
    st.divider()
    st.subheader("Lab Data (optional)", help="The default data shown is a preset provided by the developers. To use your own data, download the template below, fill in your measurements, and upload it here.")
    _xl_upload = st.file_uploader(
        "Upload Soil_Properties.xlsx",
        type=["xlsx"],
        help="Upload to compare predictions against lab data",
        label_visibility="collapsed",
    )
    if _xl_upload is not None:
        _xl_bytes = _xl_upload.read()
        st.session_state.uploaded_props_data   = _load_soil_props(_xl_bytes)
        st.session_state._user_uploaded_xl_file = _xl_bytes  # raw bytes for zip packaging
    # Auto-load bundled template as default when nothing has been uploaded yet
    if st.session_state.uploaded_props_data is None:
        _default_xlsx = os.path.join(_ROOT, "Soil_Properties.xlsx")
        if os.path.exists(_default_xlsx):
            with open(_default_xlsx, "rb") as _f:
                st.session_state.uploaded_props_data = _load_soil_props(_f.read())
    if st.session_state.uploaded_props_data is not None:
        st.caption("✅ Lab data loaded")
    # Download button for the data template
    _template_path = os.path.join(_ROOT, "Soil_Properties.xlsx")
    if os.path.exists(_template_path):
        with open(_template_path, "rb") as _f:
            st.download_button(
                "📥 Download data template",
                data=_f.read(),
                file_name="Soil_Properties.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    # Virtual borehole list — shown only when at least one exists
    if st.session_state.virtual_boreholes:
        st.markdown(
            "<p style='font-size:.82rem;font-weight:600;margin:8px 0 2px;'>"
            "Virtual Boreholes:</p>",
            unsafe_allow_html=True,
        )
        for _i, _vbh in enumerate(list(st.session_state.virtual_boreholes)):
            _c_lbl, _c_btn = st.columns([5, 1])
            with _c_lbl:
                st.caption(f"★ {_vbh['name']}  [{_vbh['method']}]")
            with _c_btn:
                if st.button("✕", key=f"del_vbh_{_i}", help=f"Remove {_vbh['name']}"):
                    st.session_state.virtual_boreholes.pop(_i)
                    st.rerun()
        if st.button("Clear All Virtual Boreholes", use_container_width=True,
                     key="clr_all_vbh"):
            st.session_state.virtual_boreholes = []
            st.rerun()

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
        _ve = st.session_state._query_easting
        _vn = st.session_state._query_northing
        vb_rows = []
        for _d in range(0, 62, 2):
            _r = predictor.predict(_ve, _vn, float(_d), method)
            _r["depth_m"] = float(_d)
            vb_rows.append(_r)
        # Build unique name; append suffix if coordinates already predicted
        _base = f"VBH-{int(_ve)}-{int(_vn)}"
        _existing = {v["name"] for v in st.session_state.virtual_boreholes}
        _name, _sfx = _base, 2
        while _name in _existing:
            _name = f"{_base}-{_sfx}"; _sfx += 1
        st.session_state.virtual_boreholes.append({
            "name":     _name,
            "easting":  _ve,
            "northing": _vn,
            "method":   method_label,
            "rows":     vb_rows,
        })

result      = st.session_state.result
pred_coords = st.session_state.pred_coords
vbhs        = st.session_state.virtual_boreholes

max_depth        = int(st.session_state._max_display_depth)
df_view          = df[df["depth_m"] <= max_depth].copy()
pred_coords_view = (
    pred_coords
    if pred_coords is None or pred_coords[2] <= max_depth
    else None
)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0;'>3D Soil Profile Viewer</h1>"
    "<p style='color:#666;margin-top:2px;'>Bangkok subsoil — MRT Orange Line dataset</p>",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "3D Borehole View",
    "3D Solid Model",
    "2D Cross-Section",
    "Dataset Overview",
    "Model Validation",
])


# ══ Tab 1 — 3D Borehole View ══════════════════════════════════════════════════
with tab1:
    pred_layer = result["layer"] if result else None
    fig3d = build_figure(df_view, pred_point=pred_coords_view,
                         pred_layer=pred_layer, virtual_bhs=vbhs,
                         depth_limit=max_depth)
    st.plotly_chart(fig3d, use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True})


# ══ Tab 2 — 3D Solid Model ════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<p style='color:#666;font-size:.9rem;margin-bottom:8px;'>"
        "Interpolated layer surfaces rendered as a 3D solid model. "
        "Drag to rotate, scroll to zoom.</p>",
        unsafe_allow_html=True,
    )
    with st.spinner("Building 3D solid model..."):
        solid_fig = build_solid_figure(df_view, depth_limit=max_depth)
    st.plotly_chart(solid_fig, use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True})
    st.caption(
        "Layer surfaces interpolated using linear triangulation. "
        "Areas outside the borehole convex hull are not extrapolated."
    )


# ══ Tab 3 — 2D Cross-Section ══════════════════════════════════════════════════
with tab3:
    all_bh_ids = sorted(df["borehole_id"].unique().tolist())
    # Case-insensitive lookup: "ow-01" → "OW-01" (the canonical ID in the dataset)
    _bh_lookup = {b.upper(): b for b in all_bh_ids}
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

        # ── Borehole selection text input (syncs both ways with map clicks) ──────
        # When cs_ordered changes via a map click the fingerprint becomes stale,
        # so we push the new value into the text-box widget before it renders.
        _current_fp = ",".join(st.session_state.cs_ordered)
        if _current_fp != st.session_state._cs_ordered_fp:
            st.session_state._cs_text_raw  = ", ".join(st.session_state.cs_ordered)
            st.session_state._cs_ordered_fp = _current_fp

        st.caption("Type borehole IDs separated by commas. Order determines the cross-section direction.")
        _col_txt, _col_btn = st.columns([4, 1])
        with _col_txt:
            _text_val = st.text_input(
                "Borehole selection",
                placeholder="e.g. OW-01, OW-05, OW-12, OW-20",
                key="_cs_text_raw",
                label_visibility="collapsed",
                help="Type borehole IDs separated by commas to select them in order. "
                     "Clicking boreholes on the map also updates this box.",
            )
        with _col_btn:
            st.write(" ")  # spacer to align button baseline with the text input
            _copy_clicked = st.button("📋", help="Copy selection to clipboard",
                                      use_container_width=True)
        st.caption("Your selection above updates automatically — copy it anytime to reload later.")

        if _copy_clicked:
            _copy_str = ", ".join(st.session_state.cs_ordered)
            components.html(
                f"<script>navigator.clipboard.writeText({repr(_copy_str)})"
                f".catch(function(e){{console.error('Clipboard error:', e)}});</script>",
                height=0,
            )
            st.toast("Copied to clipboard! 📋")

        # Parse text → resolve to canonical IDs → sync to cs_ordered
        _raw_tokens = [t.strip() for t in _text_val.split(",") if t.strip()]
        _parsed = list(dict.fromkeys(          # deduplicate, preserve order
            _bh_lookup[t.upper()]
            for t in _raw_tokens
            if t.upper() in _bh_lookup
        ))
        _invalid = [t for t in _raw_tokens if t.upper() not in _bh_lookup]
        if _invalid:
            st.caption(f"⚠ Unrecognized ID(s) ignored: {', '.join(_invalid)}")

        if _parsed != st.session_state.cs_ordered:
            st.session_state.cs_ordered     = _parsed
            st.session_state._cs_ordered_fp = ",".join(_parsed)
            st.rerun()
        # ── end text input ──────────────────────────────────────────────────────

        st.markdown(
            "<p style='font-size:.85rem;color:#666;margin-top:2px;margin-bottom:4px;'>"
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
                virtual_bhs=vbhs,
            )
            plan_event = st.plotly_chart(
                plan_fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["points"],
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "scrollZoom": True,
                    "modeBarButtons": [["pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"]],
                    "modeBarButtonsToAdd": ["toggleFullscreen"],
                },
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
                center_lat=st.session_state._map_center_lat,
                center_lon=st.session_state._map_center_lon,
                zoom=st.session_state._map_zoom,
                virtual_bhs=vbhs,
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
        for _color, _lbl in _LEGEND_ITEMS:
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{_color};border-radius:2px;margin-right:6px;'
                f'vertical-align:middle;"></span>{_lbl}',
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
            cs_fig   = build_crosssection_figure(df_view, st.session_state.cs_ordered,
                                                  depth_limit=max_depth)
            cs_event = st.plotly_chart(
                cs_fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["points"],
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "scrollZoom": False,
                    "modeBarButtonsToAdd": ["toggleFullscreen"],
                },
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
                "Layer boundaries are interpolated and smoothed between boreholes for visualization. "
                "Thin layers present at only one borehole may not be shown. "
                "Always verify with original boring log data. "
                "Click anywhere on the section to update the query coordinates and depth."
            )


# ══ Tab 4 — Dataset Overview ══════════════════════════════════════════════════
def _img(filename):
    """Return absolute path to an asset image."""
    return os.path.join(_ROOT, "assets", filename)


def _grid(items):
    """Render a list of (filename, caption) pairs in a 2-column grid."""
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(items):
                fname, cap = items[i + j]
                with col:
                    st.image(_img(fname), caption=cap, use_column_width=True)


with tab4:
    # ── Section 1: Property Distributions ────────────────────────────────────
    st.subheader("Property Distributions")
    _grid([
        ("Distribution - Unit weight.jpg", "Unit Weight"),
        ("Distribution - wn.png",          "Water Content"),
        ("Distribution - Su.jpg",           "Undrained Shear Strength"),
        ("Distribution - SPT.png",          "SPT-N"),
        ("Distribution - LL.png",           "Liquid Limit"),
        ("Distribution - PI.png",           "Plasticity Index (PI)"),
        ("Distribution - PL.png",           "Plastic Limit (PL)"),
    ])

    # ── Section 2: Properties vs Depth ───────────────────────────────────────
    st.subheader("Properties vs Depth")
    _grid([
        ("Curve - Unit weight.jpg", "Unit Weight vs Depth"),
        ("Curve - wn.jpg",          "Water Content vs Depth"),
        ("Curve - Su.png",          "Undrained Shear Strength vs Depth"),
        ("Curve - SPT.jpg",         "SPT-N vs Depth"),
        ("Curve - LL.jpg",          "Liquid Limit vs Depth"),
        ("Curve - PI.png",          "Plasticity Index vs Depth"),
        ("Curve - PL.jpg",          "Plastic Limit vs Depth"),
    ])

    # ── Section 3: Soil Profiles ─────────────────────────────────────────────
    st.subheader("Soil Profiles")
    _grid([
        ("Soil profile 1.png", "Soil Profile 1"),
        ("Soil profile 2.png", "Soil Profile 2"),
    ])


with tab5:
    # ── What is LOOCV? ────────────────────────────────────────────────────────
    st.subheader("What is Leave-One-Out Cross-Validation?")
    st.markdown(
        """
        **Leave-One-Out Cross-Validation (LOOCV) by borehole** is a way to test
        how well the prediction models generalise to locations they have never seen.

        The idea is simple:

        1. Pick one borehole and **hide it completely** from the models.
        2. Train (or parameterise) each method using **all the other boreholes**.
        3. Ask each method to predict the soil layer and properties at the hidden
           borehole's locations.
        4. Compare those predictions with the **real measurements** from the hidden
           borehole to compute errors.
        5. Repeat for every borehole in the dataset and average the results.

        This gives an honest estimate of how accurate the app would be at a
        **brand-new borehole location** — the most realistic test for practical use.

        **Metrics reported**
        - *Soil Layer Classification* → **Accuracy** (% of depth intervals
          where the predicted soil layer matches the recorded one)
        - *Unit Weight, Su, SPT-N* → **RMSE** (Root Mean Squared Error) and
          **MAE** (Mean Absolute Error) in the original units
        """
    )

    st.divider()

    # ── Download validation package ───────────────────────────────────────────
    st.subheader("Run the Validation Yourself")
    st.markdown(
        "Download the validation package below. It includes everything you need "
        "to run the full LOO-CV locally or on Google Colab and produce a "
        "`validation_results.json` file you can upload here."
    )

    # Data-source status message
    _user_uploaded_props = st.session_state.get("uploaded_props_data") is not None
    _user_used_own_file  = st.session_state.get("_user_uploaded_xl_file") is not None
    if _user_used_own_file:
        st.success("✅ Using your uploaded data for the validation package.")
    else:
        st.warning(
            "⚠️ You are using the default preset data. To validate against your "
            "own data, upload your own file in the **Lab Data** section in the "
            "sidebar first."
        )

    # Build the zip in memory
    import zipfile as _zipfile
    import io as _io

    _val_script_path = os.path.join(_ROOT, "validation.py")
    _default_xl_path = os.path.join(_ROOT, "Soil_Properties.xlsx")

    def _build_validation_zip() -> bytes:
        buf = _io.BytesIO()
        with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
            # validation.py
            if os.path.exists(_val_script_path):
                with open(_val_script_path, "rb") as f:
                    zf.writestr("validation.py", f.read())

            # Lab data — user upload takes priority over default
            _xl_file = st.session_state.get("_user_uploaded_xl_file")
            if _xl_file is not None:
                zf.writestr("Soil_Properties.xlsx", _xl_file)
            elif os.path.exists(_default_xl_path):
                with open(_default_xl_path, "rb") as f:
                    zf.writestr("Soil_Properties.xlsx", f.read())

            # All .joblib model files
            _models_dir = os.path.join(_ROOT, "models")
            if os.path.isdir(_models_dir):
                for _jf in sorted(os.listdir(_models_dir)):
                    if _jf.endswith(".joblib"):
                        with open(os.path.join(_models_dir, _jf), "rb") as f:
                            zf.writestr(f"models/{_jf}", f.read())

            # README
            zf.writestr(
                "README.txt",
                "Soil Profile App — Validation Package\n"
                "======================================\n\n"
                "Requirements:\n"
                "  pip install scikit-learn xgboost pandas numpy openpyxl\n\n"
                "Steps:\n"
                "  1. Run:   python validation.py\n"
                "  2. Upload the output  validation_results.json  back to the\n"
                "     app's 'Model Validation' tab to display your results.\n\n"
                "Notes:\n"
                "  - Soil_Properties.xlsx contains the lab data used for validation.\n"
                "  - The models/ folder contains the pre-trained model files.\n"
                "  - validation.py will look for data and models in these relative paths.\n",
            )
        return buf.getvalue()

    st.download_button(
        "📥 Download Validation Package",
        data=_build_validation_zip(),
        file_name="validation_package.zip",
        mime="application/zip",
        use_container_width=False,
    )

    st.info(
        "💡 **Recommended:** Upload the package to "
        "[Google Colab](https://colab.research.google.com) for a free cloud "
        "environment with all required packages pre-installed. Alternatively, "
        "run it on your own machine if you have Python installed "
        "(`pip install scikit-learn xgboost pandas numpy openpyxl`).",
        icon=None,
    )

    st.divider()

    # ── View validation results ───────────────────────────────────────────────
    st.subheader("View Validation Results")

    import json as _json

    _default_results_path = os.path.join(_ROOT, "validation_results.json")

    def _render_validation_results(vres: dict, is_default: bool):
        if is_default:
            st.caption("Showing default Bangkok MRT dataset results.")

        # ── Classification ────────────────────────────────────────────────────
        st.markdown("#### Soil Layer Classification Accuracy")
        _cls = vres.get("classification", {})
        _cls_rows = []
        for _m, _mlabel in [("dwa", "Distance-Weighted Average"),
                             ("rf",  "Random Forest"),
                             ("xgb", "XGBoost")]:
            _acc = _cls.get(_m, {}).get("accuracy")
            if _acc is not None:
                _cls_rows.append({
                    "Method":       _mlabel,
                    "Accuracy (%)": round(_acc * 100, 1),
                })
        if _cls_rows:
            _cls_df = pd.DataFrame(_cls_rows).set_index("Method")
            st.dataframe(_cls_df, use_container_width=True)
            _cls_fig = go.Figure(go.Bar(
                x=_cls_df.index,
                y=_cls_df["Accuracy (%)"],
                marker_color=["#78909C", "#1565C0", "#F9A825"],
                text=[f"{v:.1f}%" for v in _cls_df["Accuracy (%)"]],
                textposition="outside",
            ))
            _cls_fig.update_layout(
                title="Soil Layer Classification Accuracy (%)",
                yaxis=dict(title="Accuracy (%)", range=[0, 105]),
                height=340,
                margin=dict(t=50, b=30, l=40, r=20),
                plot_bgcolor="white",
            )
            st.plotly_chart(_cls_fig, use_container_width=True,
                            config={"displaylogo": False})

        # ── Regression ────────────────────────────────────────────────────────
        _reg_targets = [
            ("su_kpa",  "Undrained Shear Strength (Su)", "kPa"),
            ("spt_n",   "SPT-N",                         "blows/0.3 m"),
        ]
        _reg = vres.get("regression", {})
        for _col, _rlabel, _unit in _reg_targets:
            st.markdown(f"#### {_rlabel}")
            _prop = _reg.get(_col, {})
            _reg_rows = []
            for _m, _mlabel in [("dwa", "Distance-Weighted Average"),
                                 ("rf",  "Random Forest"),
                                 ("xgb", "XGBoost")]:
                _d = _prop.get(_m, {})
                if _d:
                    _reg_rows.append({
                        "Method":          _mlabel,
                        f"RMSE ({_unit})": round(_d.get("rmse", float("nan")), 3),
                        f"MAE ({_unit})":  round(_d.get("mae",  float("nan")), 3),
                        "N samples":       _d.get("n", "—"),
                    })
            if _reg_rows:
                _reg_df = pd.DataFrame(_reg_rows).set_index("Method")
                st.dataframe(_reg_df, use_container_width=True)
                _methods_r   = [r["Method"]          for r in _reg_rows]
                _rmse_vals_r = [r[f"RMSE ({_unit})"] for r in _reg_rows]
                _mae_vals_r  = [r[f"MAE ({_unit})"]  for r in _reg_rows]
                _reg_fig = go.Figure()
                _reg_fig.add_trace(go.Bar(
                    name="RMSE", x=_methods_r, y=_rmse_vals_r,
                    marker_color="#1565C0",
                    text=[f"{v:.3f}" for v in _rmse_vals_r],
                    textposition="outside",
                ))
                _reg_fig.add_trace(go.Bar(
                    name="MAE", x=_methods_r, y=_mae_vals_r,
                    marker_color="#4FC3F7",
                    text=[f"{v:.3f}" for v in _mae_vals_r],
                    textposition="outside",
                ))
                _reg_fig.update_layout(
                    title=f"{_rlabel} — RMSE & MAE ({_unit})",
                    yaxis_title=_unit,
                    barmode="group",
                    height=340,
                    margin=dict(t=50, b=30, l=40, r=20),
                    plot_bgcolor="white",
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(_reg_fig, use_container_width=True,
                                config={"displaylogo": False})

    _json_upload = st.file_uploader(
        "Upload your own validation_results.json (optional)",
        type=["json"],
        help="JSON file produced by running validation.py on your own data.",
    )
    if _json_upload is not None:
        st.session_state._user_uploaded_json = _json_upload.read()

    if st.button("🔍 View Validation Results", use_container_width=False):
        try:
            _uploaded_json = st.session_state.get("_user_uploaded_json")
            if _uploaded_json is not None:
                _vres      = _json.loads(_uploaded_json)
                _is_default = False
            elif os.path.exists(_default_results_path):
                with open(_default_results_path, "r") as _f:
                    _vres = _json.load(_f)
                _is_default = True
            else:
                st.error("No validation results found. Run validation.py and upload the output JSON.")
                _vres = None

            if _vres is not None:
                _render_validation_results(_vres, _is_default)
        except Exception as _e:
            st.error(f"Could not load results: {_e}")


# ══ Permanent results panel (always visible below all tabs) ══════════════════
st.divider()
col_res, col_vbh = st.columns([2, 3], gap="large")

with col_res:
    st.markdown("#### Prediction Results")
    if result is None:
        st.info(
            "Set coordinates in the sidebar, choose a method, "
            "then click **Run Prediction**."
        )
        st.markdown("**Soil Layer Legend**")
        for _color, _lbl in _LEGEND_ITEMS:
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{_color};border-radius:2px;margin-right:6px;'
                f'vertical-align:middle;"></span>{_lbl}',
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
            f'<div style="font-size:2.1rem;font-weight:800;margin:4px 0 2px;">{LAYER_LABELS.get(layer, layer)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _confidence_bar(conf)
        st.markdown("**Predicted Properties**")
        su,  su_std  = result.get("su_kpa"),         result.get("su_kpa_std")
        spt, spt_std = result.get("spt_n"),          result.get("spt_n_std")
        uw,  uw_std  = result.get("unit_weight"),    result.get("unit_weight_std")
        pi,  pi_std  = result.get("plasticity_idx"), result.get("plasticity_idx_std")
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

with col_vbh:
    # Virtual borehole tables — one collapsible expander per VBH
    if vbhs:
        st.markdown("#### Virtual Borehole Profiles")
        for _vi, _vbh in enumerate(vbhs):
            with st.expander(
                f"{_vbh['name']}  [{_vbh['method']}]  "
                f"E {_vbh['easting']:.0f}  N {_vbh['northing']:.0f}",
                expanded=(_vi == len(vbhs) - 1),
            ):
                tbl = []
                for r in _vbh["rows"]:
                    tbl.append({
                        "Depth (m)":       r["depth_m"],
                        "Layer":           r.get("layer", ""),
                        "Conf.":           f"{r.get('layer_confidence', 0)*100:.0f}%",
                        "su (kPa)":        f"{r['su_kpa']:.1f}"        if r.get("su_kpa")        is not None else "--",
                        "SPT-N":           f"{r['spt_n']:.0f}"         if r.get("spt_n")         is not None else "--",
                        "Unit Wt (kN/m³)": f"{r['unit_weight']:.2f}"   if r.get("unit_weight")   is not None else "--",
                        "PI (%)":          f"{r['plasticity_idx']:.1f}" if r.get("plasticity_idx") is not None else "--",
                    })
                st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)


# ── Change 6: Lab data comparison charts (full-width, below results) ──────────
_props = st.session_state.get("uploaded_props_data")
if result is not None and _props is not None:
    st.divider()
    st.markdown("#### 📈 Comparison with Lab Data")
    st.caption(
        "Lab measurements coloured by soil type; the large red diamond is "
        "the model prediction for the current query point."
    )
    _res = st.session_state.result
    _d  = float(_res.get("depth_m", st.session_state._query_depth))

    def _flt(v):
        try:
            return float(v)
        except Exception:
            return float("nan")

    _su = _flt(_res.get("su_kpa"))
    _sn = _flt(_res.get("spt_n"))
    _uw = _flt(_res.get("unit_weight"))
    _cc1, _cc2, _cc3 = st.columns(3, gap="large")
    with _cc1:
        st.markdown("**Su (kPa) vs Depth**")
        st.plotly_chart(
            build_property_depth_chart(_props, "su_kpa", "Su (kPa)", _d, _su),
            use_container_width=True,
        )
    with _cc2:
        st.markdown("**SPT-N vs Depth**")
        st.plotly_chart(
            build_property_depth_chart(_props, "spt_n", "SPT-N (blows/30cm)", _d, _sn),
            use_container_width=True,
        )
    with _cc3:
        st.markdown("**Unit Weight vs Depth**")
        st.plotly_chart(
            build_property_depth_chart(_props, "unit_weight", "Unit Weight (kN/m³)", _d, _uw),
            use_container_width=True,
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
