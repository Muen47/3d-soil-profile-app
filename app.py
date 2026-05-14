import os
import sys

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

LAYER_ORDER = ["MG", "VSC", "SOC", "SC", "SS", "MSC", "FS", "SS"]
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


predictor = _get_predictor()
df        = _get_df()


# ── Shared helpers ────────────────────────────────────────────────────────────
def _hex_rgba(hex_color: str, alpha: float = 0.85) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _hover(row) -> str:
    su  = f"{row['su_kpa']:.1f} kPa" if pd.notna(row.get("su_kpa"))    else "--"
    spt = f"{row['spt_n']:.0f}"      if pd.notna(row.get("spt_n"))     else "--"
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
) -> go.Figure:
    fig = go.Figure()

    for bh_id, bh in df.groupby("borehole_id"):
        s = bh.sort_values("depth_m")
        fig.add_trace(go.Scatter3d(
            x=s["easting"], y=s["northing"], z=-s["depth_m"],
            mode="lines",
            line=dict(color="#bdbdbd", width=4),
            name=bh_id,
            showlegend=True,
            hoverinfo="skip",
        ))

    for layer, color in LAYER_COLORS.items():
        sub = df[df["soil_layer"] == layer]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=sub["easting"],
            y=sub["northing"],
            z=-sub["depth_m"],
            mode="markers",
            marker=dict(
                size=9, color=color, opacity=0.88,
                line=dict(color="white", width=0.6),
            ),
            name=f"{layer}  {LAYER_LABELS.get(layer, '')}",
            text=sub.apply(_hover, axis=1).tolist(),
            hovertemplate="%{text}<extra></extra>",
        ))

    if pred_point is not None:
        e, n, d = pred_point
        color = LAYER_COLORS.get(pred_layer, "#e53935") if pred_layer else "#e53935"
        fig.add_trace(go.Scatter3d(
            x=[e], y=[n], z=[-d],
            mode="markers",
            marker=dict(
                size=16, color=color, symbol="diamond",
                opacity=1.0, line=dict(color="white", width=2),
            ),
            name="Prediction point",
            hovertemplate=(
                f"<b>Prediction Point</b><br>"
                f"E: {e:.1f}  N: {n:.1f}<br>"
                f"Depth: {d:.1f} m<br>"
                f"Layer: {pred_layer or '--'}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="- Depth (m)",
            bgcolor="#f0f4f8",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=2.5),
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.7)),
        ),
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=640,
        paper_bgcolor="white",
    )
    return fig


# ── View 2: 3D Solid Model helpers ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_layer_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each borehole compute the min depth_top_m and max depth_bot_m per layer.
    Returns a DataFrame with columns: borehole_id, easting, northing, soil_layer, top, bot.
    """
    records = []
    for (bh_id, layer), grp in df.groupby(["borehole_id", "soil_layer"]):
        row0 = grp.iloc[0]
        top = grp["depth_top_m"].min()
        bot = grp["depth_bot_m"].max()
        records.append({
            "borehole_id": bh_id,
            "easting":     row0["easting"],
            "northing":    row0["northing"],
            "soil_layer":  layer,
            "top":         top,
            "bot":         bot,
        })
    return pd.DataFrame(records)


def build_solid_figure(df: pd.DataFrame) -> go.Figure:
    bounds = get_layer_bounds(df)
    fig = go.Figure()

    # Determine all unique borehole positions
    bh_pos = bounds[["borehole_id","easting","northing"]].drop_duplicates()

    for layer in reversed(LAYER_SEQUENCE):
        sub = bounds[bounds["soil_layer"] == layer]
        if len(sub) < 3:
            continue

        color_hex = LAYER_COLORS.get(layer, "#aaaaaa")
        color_rgb = _hex_rgba(color_hex, 0.75)

        # Build a regular grid over easting/northing
        e_min, e_max = bh_pos["easting"].min(), bh_pos["easting"].max()
        n_min, n_max = bh_pos["northing"].min(), bh_pos["northing"].max()

        # Margin to avoid edge artifacts
        margin_e = (e_max - e_min) * 0.05
        margin_n = (n_max - n_min) * 0.05

        grid_e = np.linspace(e_min - margin_e, e_max + margin_e, 20)
        grid_n = np.linspace(n_min - margin_n, n_max + margin_n, 20)
        GE, GN = np.meshgrid(grid_e, grid_n)

        pts    = sub[["easting","northing"]].values
        tops   = sub["top"].values
        bots   = sub["bot"].values

        try:
            interp_top = LinearNDInterpolator(pts, tops, fill_value=np.nan)
            interp_bot = LinearNDInterpolator(pts, bots, fill_value=np.nan)
        except Exception:
            continue

        Z_top = interp_top(GE, GN)
        Z_bot = interp_bot(GE, GN)

        # Only plot where we have valid interpolation (inside convex hull)
        valid = ~(np.isnan(Z_top) | np.isnan(Z_bot))
        if valid.sum() < 4:
            continue

        # Top surface
        fig.add_trace(go.Surface(
            x=GE, y=GN, z=-Z_top,
            colorscale=[[0, color_rgb], [1, color_rgb]],
            showscale=False,
            name=f"{layer} top",
            showlegend=False,
            opacity=0.7,
            hovertemplate=(
                f"<b>{layer} — {LAYER_LABELS.get(layer,'')}</b><br>"
                "Top surface<br>"
                "E: %{x:.0f}  N: %{y:.0f}<br>"
                "Depth: %{z:.1f} m<extra></extra>"
            ),
            lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1),
        ))

        # Bottom surface
        fig.add_trace(go.Surface(
            x=GE, y=GN, z=-Z_bot,
            colorscale=[[0, color_rgb], [1, color_rgb]],
            showscale=False,
            name=f"{layer} bot",
            showlegend=False,
            opacity=0.7,
            hovertemplate=(
                f"<b>{layer} — {LAYER_LABELS.get(layer,'')}</b><br>"
                "Bottom surface<br>"
                "E: %{x:.0f}  N: %{y:.0f}<br>"
                "Depth: %{z:.1f} m<extra></extra>"
            ),
            lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1),
        ))

    # Borehole sticks
    for bh_id, bh in df.groupby("borehole_id"):
        s = bh.sort_values("depth_m")
        fig.add_trace(go.Scatter3d(
            x=s["easting"], y=s["northing"], z=-s["depth_m"],
            mode="lines+text",
            line=dict(color="#333333", width=3),
            text=[bh_id] + [""] * (len(s) - 1),
            textposition="top center",
            textfont=dict(size=10, color="#333333"),
            name=bh_id,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Invisible legend traces
    for layer, color in LAYER_COLORS.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=10, color=color, opacity=0.85),
            name=f"{layer}  {LAYER_LABELS.get(layer, '')}",
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="- Depth (m)",
            bgcolor="#e8edf2",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=2.5),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=680,
        paper_bgcolor="white",
        title=dict(text="3D Solid Soil Model — Interpolated Layer Surfaces", x=0.5, font=dict(size=14)),
    )
    return fig


# ── View 3: 2D Cross-Section helpers ─────────────────────────────────────────
def build_planview_figure(df: pd.DataFrame, selected: list[str]) -> go.Figure:
    bh_pos = (
        df.groupby("borehole_id")
          .agg(easting=("easting","first"), northing=("northing","first"))
          .reset_index()
          .sort_values("borehole_id")
          .reset_index(drop=True)
    )

    selected_set = set(selected)
    order_map    = {bh: i + 1 for i, bh in enumerate(selected)}

    colors = ["#e53935" if b in selected_set else "#1E88E5"
              for b in bh_pos["borehole_id"]]
    sizes  = [15 if b in selected_set else 10
              for b in bh_pos["borehole_id"]]

    fig = go.Figure()

    # Single trace — all boreholes clickable via on_select
    fig.add_trace(go.Scatter(
        x=bh_pos["easting"],
        y=bh_pos["northing"],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color="white", width=1.5),
            opacity=1.0,
        ),
        # Keep full opacity on all points regardless of plotly selection state
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=1.0)),
        text=bh_pos["borehole_id"],
        textposition="top center",
        textfont=dict(size=9, color="#222"),
        customdata=bh_pos["borehole_id"].tolist(),
        hovertemplate="<b>%{customdata}</b><br>E: %{x:.0f}  N: %{y:.0f}"
                      "<br><i>Click to add to section</i><extra></extra>",
        name="Boreholes",
        showlegend=False,
    ))

    # Section path line
    if len(selected) >= 2:
        sel_df = bh_pos[bh_pos["borehole_id"].isin(selected)].set_index("borehole_id")
        path_e = [sel_df.loc[b, "easting"]  for b in selected if b in sel_df.index]
        path_n = [sel_df.loc[b, "northing"] for b in selected if b in sel_df.index]
        fig.add_trace(go.Scatter(
            x=path_e, y=path_n,
            mode="lines",
            line=dict(color="#e53935", width=2, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Order number annotations on selected boreholes
    for bh, order in order_map.items():
        row = bh_pos[bh_pos["borehole_id"] == bh]
        if row.empty:
            continue
        fig.add_annotation(
            x=float(row["easting"].iloc[0]),
            y=float(row["northing"].iloc[0]),
            text=f"<b>{order}</b>",
            showarrow=False,
            yshift=-18,
            font=dict(size=10, color="#e53935", family="Arial Black"),
            bgcolor="rgba(255,255,255,0.7)",
            borderpad=1,
        )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(title="Easting (m)", tickformat=".0f"),
        yaxis=dict(title="Northing (m)", tickformat=".0f", scaleanchor="x"),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="#f5f7fa",
        title=dict(text="Plan View — click boreholes to build section", x=0.5,
                   font=dict(size=12)),
        dragmode="pan",
    )
    return fig


def _borehole_interfaces(layer_dict: dict, layer_sequence: list[str]) -> list[float]:
    """
    Convert a per-borehole {layer: (top, bot)} dict into an ordered list of
    n_layers+1 interface depths that are guaranteed to be contiguous (no gaps).

    Missing layers get zero thickness — they are pinched at whatever depth
    the stack has reached, so adjacent layers always share an exact boundary.
    """
    interfaces = []
    current = 0.0
    for layer in layer_sequence:
        interfaces.append(current)           # top of this layer slot
        if layer in layer_dict:
            current = float(layer_dict[layer][1])   # advance to actual bottom
        # if absent: current stays the same → zero thickness
    interfaces.append(current)               # final bottom of column
    return interfaces


def build_crosssection_figure(df: pd.DataFrame, selected: list[str]) -> go.Figure:
    """
    Draw a 2D cross-section along the ordered list of borehole IDs.
    X-axis = cumulative distance along section (m), Y-axis = depth (positive down).

    Every layer is drawn as a continuous polygon across ALL boreholes.
    At boreholes where a layer is absent its thickness is forced to zero
    (pinch-out), so there are no white gaps between bands.
    """
    bounds = get_layer_bounds(df)
    bh_pos = (
        df.groupby("borehole_id")
          .agg(easting=("easting","first"), northing=("northing","first"))
          .reset_index()
          .set_index("borehole_id")
    )

    valid_sel = [b for b in selected if b in bh_pos.index]
    if len(valid_sel) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least 2 boreholes to draw a cross-section.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#888"),
        )
        return fig

    # Cumulative horizontal distances along section
    cum_dist = [0.0]
    for i in range(1, len(valid_sel)):
        b0, b1 = valid_sel[i-1], valid_sel[i]
        de = bh_pos.loc[b1, "easting"]  - bh_pos.loc[b0, "easting"]
        dn = bh_pos.loc[b1, "northing"] - bh_pos.loc[b0, "northing"]
        cum_dist.append(cum_dist[-1] + np.sqrt(de**2 + dn**2))
    bh_x = {bh: cum_dist[i] for i, bh in enumerate(valid_sel)}

    # Build {layer: (top, bot)} lookup per borehole
    bh_layer = {}
    for bh in valid_sel:
        sub = bounds[bounds["borehole_id"] == bh]
        bh_layer[bh] = {r["soil_layer"]: (r["top"], r["bot"]) for _, r in sub.iterrows()}

    # Compute contiguous interface stacks for every borehole
    # ifaces[bh][i] = top of LAYER_SEQUENCE[i]; ifaces[bh][-1] = column bottom
    ifaces = {bh: _borehole_interfaces(bh_layer[bh], LAYER_SEQUENCE) for bh in valid_sel}

    max_depth = max(v[-1] for v in ifaces.values())
    max_depth = max(max_depth * 1.05, 10.0)

    fig = go.Figure()

    xs = [bh_x[bh] for bh in valid_sel]

    # One filled polygon per layer — spans ALL boreholes with no gaps
    for li, layer in enumerate(LAYER_SEQUENCE):
        tops = [ifaces[bh][li]     for bh in valid_sel]
        bots = [ifaces[bh][li + 1] for bh in valid_sel]

        # Skip layer if it has zero thickness everywhere (absent from all boreholes)
        if all(abs(t - b) < 1e-6 for t, b in zip(tops, bots)):
            continue

        color_hex = LAYER_COLORS.get(layer, "#aaaaaa")
        fill_rgba = _hex_rgba(color_hex, 0.82)
        line_rgba = _hex_rgba(color_hex, 1.0)

        # Polygon: forward across tops, backward across bottoms, close
        poly_x = xs + list(reversed(xs)) + [xs[0]]
        poly_y = tops + list(reversed(bots)) + [tops[0]]

        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y,
            fill="toself",
            fillcolor=fill_rgba,
            line=dict(color=line_rgba, width=1.2),
            mode="lines",
            name=f"{layer}  {LAYER_LABELS.get(layer, '')}",
            hovertemplate=f"<b>{layer}</b> — {LAYER_LABELS.get(layer,'')}<extra></extra>",
        ))

    # Borehole sticks + labels + depth ticks
    for bh in valid_sel:
        x_pos  = bh_x[bh]
        col_bot = ifaces[bh][-1]
        fig.add_shape(
            type="line",
            x0=x_pos, x1=x_pos, y0=0, y1=col_bot,
            line=dict(color="#111111", width=1.8),
            layer="above",
        )
        fig.add_annotation(
            x=x_pos, y=0,
            text=f"<b>{bh}</b>",
            showarrow=False, yshift=14,
            font=dict(size=10, color="#111"),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
        )
        for d_tick in np.arange(10, col_bot + 1, 10):
            fig.add_annotation(
                x=x_pos, y=d_tick,
                text=f"{int(d_tick)}m",
                showarrow=False, xshift=7,
                font=dict(size=7, color="#444"),
                bgcolor="rgba(255,255,255,0)",
            )

    # Horizontal distance labels between adjacent boreholes
    for i in range(1, len(valid_sel)):
        b0, b1  = valid_sel[i-1], valid_sel[i]
        mid_x   = (bh_x[b0] + bh_x[b1]) / 2
        dist    = bh_x[b1] - bh_x[b0]
        fig.add_annotation(
            x=mid_x, y=-3,
            text=f"{dist:.0f} m",
            showarrow=False,
            font=dict(size=9, color="#444"),
            bgcolor="rgba(255,255,255,0.7)",
        )

    fig.update_layout(
        height=580,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(
            title="Horizontal Distance (m)",
            showgrid=True, gridcolor="#dde",
            zeroline=False,
            range=[-cum_dist[-1] * 0.02, cum_dist[-1] * 1.04],
        ),
        yaxis=dict(
            title="Depth (m)",
            autorange="reversed",
            showgrid=True, gridcolor="#dde",
            range=[-6, max_depth],
            zeroline=True, zerolinecolor="#888", zerolinewidth=1.5,
        ),
        legend=dict(
            x=1.01, y=1,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#ccc", borderwidth=1,
            font=dict(size=10),
            xanchor="left",
        ),
        paper_bgcolor="white",
        plot_bgcolor="#f9fafb",
        title=dict(
            text="Geological Cross-Section  —  " + "  →  ".join(valid_sel),
            x=0.5, font=dict(size=13),
        ),
    )
    return fig


# ── UI helpers ────────────────────────────────────────────────────────────────
def _prop_card(
    label: str,
    value_str: str,
    unit: str,
    std_str: str | None = None,
    color: str = "#1565C0",
) -> None:
    unc = (
        f'<span style="font-size:.78rem;color:#757575;">+/- {std_str}</span>'
        if std_str
        else '<span style="font-size:.78rem;color:#bdbdbd;">uncertainty N/A</span>'
    )
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
    bar_color = (
        "#43a047" if conf >= 0.70 else
        "#fb8c00" if conf >= 0.40 else
        "#e53935"
    )
    label = (
        "High confidence"   if conf >= 0.70 else
        "Moderate confidence" if conf >= 0.40 else
        "Low confidence - add nearby boreholes"
    )
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
    easting  = st.number_input("Easting (m)",  value=658871.0, step=1.0, format="%.1f")
    northing = st.number_input("Northing (m)", value=1522280.0, step=1.0, format="%.1f")
    depth    = st.number_input(
        "Depth (m)", min_value=0.1, max_value=200.0, value=10.0, step=0.5
    )

    st.divider()
    st.markdown("#### Prediction Method")
    method_label = st.radio(
        "method", list(METHOD_MAP.keys()), index=1, label_visibility="collapsed"
    )
    method = METHOD_MAP[method_label]
    if method == "xgb":
        st.caption("XGBoost does not provide per-prediction uncertainty estimates.")

    st.divider()
    run = st.button("Run Prediction", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        f"**Dataset** - {len(df)} samples - "
        f"{df['borehole_id'].nunique()} borehole(s)\n\n"
        f"**Layers**: {', '.join(sorted(df['soil_layer'].unique()))}"
    )
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result      = None
    st.session_state.pred_coords = None
if "cs_ordered" not in st.session_state:
    st.session_state.cs_ordered  = []

if run:
    with st.spinner("Running prediction..."):
        st.session_state.result      = predictor.predict(easting, northing, depth, method)
        st.session_state.pred_coords = (easting, northing, depth)

result      = st.session_state.result
pred_coords = st.session_state.pred_coords


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0;'>3D Soil Profile Viewer</h1>"
    "<p style='color:#666;margin-top:2px;'>Bangkok subsoil - MRT Orange Line dataset</p>",
    unsafe_allow_html=True,
)

# ── View tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "3D Borehole View",
    "3D Solid Model",
    "2D Cross-Section",
])


# ── Tab 1: 3D Borehole View ───────────────────────────────────────────────────
with tab1:
    col_viewer, col_results = st.columns([3, 2], gap="large")

    with col_viewer:
        pred_layer = result["layer"] if result else None
        fig = build_figure(df, pred_point=pred_coords, pred_layer=pred_layer)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
        )

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
                    f"**{lyr}** - {LAYER_LABELS.get(lyr, '')}",
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

            su, su_std = result.get("su_kpa"), result.get("su_kpa_std")
            _prop_card(
                "Undrained Shear Strength su",
                f"{su:.1f}" if su is not None else "--",
                "kPa",
                f"{su_std:.1f} kPa" if su_std is not None else None,
                color="#1565C0",
            )

            spt, spt_std = result.get("spt_n"), result.get("spt_n_std")
            _prop_card(
                "SPT Blow Count N",
                f"{spt:.0f}" if spt is not None else "--",
                "blows / 300 mm",
                f"{spt_std:.1f}" if spt_std is not None else None,
                color="#F9A825",
            )

            uw, uw_std = result.get("unit_weight"), result.get("unit_weight_std")
            _prop_card(
                "Unit Weight",
                f"{uw:.2f}" if uw is not None else "--",
                "kN/m3",
                f"{uw_std:.2f} kN/m3" if uw_std is not None else None,
                color="#43a047",
            )

            pi, pi_std = result.get("plasticity_idx"), result.get("plasticity_idx_std")
            _prop_card(
                "Plasticity Index PI",
                f"{pi:.1f}" if pi is not None else "--",
                "%",
                f"{pi_std:.1f}%" if pi_std is not None else None,
                color="#8e24aa",
            )

            st.caption(
                f"Method: **{method_label}**  |  "
                f"E {easting:.1f}  N {northing:.1f}  |  Depth {depth:.1f} m"
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
            use_container_width=True,
            hide_index=True,
        )


# ── Tab 2: 3D Solid Model ─────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "<p style='color:#666;font-size:.9rem;margin-bottom:8px;'>"
        "Interpolated layer surfaces rendered as a 3D solid model. "
        "Drag to rotate, scroll to zoom.</p>",
        unsafe_allow_html=True,
    )
    with st.spinner("Building 3D solid model..."):
        solid_fig = build_solid_figure(df)
    st.plotly_chart(
        solid_fig,
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    st.caption(
        "Layer surfaces interpolated from borehole data using linear triangulation. "
        "Areas beyond the borehole convex hull are not extrapolated."
    )


# ── Tab 3: 2D Cross-Section ───────────────────────────────────────────────────
with tab3:
    all_bh_ids = sorted(df["borehole_id"].unique().tolist())

    col_plan, col_cs = st.columns([2, 3], gap="large")

    with col_plan:
        st.markdown("#### Plan View")
        st.markdown(
            "<p style='font-size:.85rem;color:#666;margin-top:-6px;margin-bottom:4px;'>"
            "Click boreholes on the map to add them to the section in order.</p>",
            unsafe_allow_html=True,
        )

        plan_fig = build_planview_figure(df, st.session_state.cs_ordered)
        plan_event = st.plotly_chart(
            plan_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=["points"],
            config={"displayModeBar": False, "scrollZoom": False},
            key="plan_view_chart",
        )

        # Process click — add borehole to ordered selection
        if plan_event and plan_event.selection and plan_event.selection.points:
            for pt in plan_event.selection.points:
                raw = pt.get("customdata")
                # customdata may arrive as a string or a single-item list
                bh_clicked = raw[0] if isinstance(raw, list) else raw
                if bh_clicked and bh_clicked not in st.session_state.cs_ordered:
                    st.session_state.cs_ordered.append(bh_clicked)
                    st.rerun()

        # Selected borehole list with inline X buttons
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

    with col_cs:
        st.markdown("#### Geological Cross-Section")
        if len(st.session_state.cs_ordered) < 2:
            st.info(
                "Click 2 or more boreholes on the plan view map to draw a cross-section. "
                "The section is drawn in the order you click them."
            )
        else:
            cs_fig = build_crosssection_figure(df, st.session_state.cs_ordered)
            st.plotly_chart(
                cs_fig,
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": False},
            )
            st.caption(
                "Layer boundaries are linearly interpolated between boreholes. "
                "Layers absent from a borehole are omitted from that segment."
            )
