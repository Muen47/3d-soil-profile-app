import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

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
# st.stop() MUST be called at the top level of the script (not inside a
# function) to reliably halt execution in Streamlit >= 1.45.

def _render_login() -> None:
    """Draw the login form. Calls st.rerun() on success."""
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


# Top-level guard — st.stop() here, not inside _render_login()
if not st.session_state.get("authenticated"):
    _render_login()
    st.stop()


# ── Cached resources (only reached when authenticated) ────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def _get_predictor():
    return SoilPredictor(model_dir=MODEL_DIR, data_path=DATA_PATH)


@st.cache_data(show_spinner=False)
def _get_df():
    return load_and_clean(DATA_PATH)


predictor = _get_predictor()
df        = _get_df()


# ── 3-D figure ────────────────────────────────────────────────────────────────
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


def build_figure(
    df: pd.DataFrame,
    pred_point: tuple | None = None,
    pred_layer: str | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Borehole path lines
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

    # Coloured sample markers per soil layer
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

    # Prediction point
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

# ── Main columns ──────────────────────────────────────────────────────────────
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

        # Layer badge
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

# ── Borehole data table ───────────────────────────────────────────────────────
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
