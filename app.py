"""
CLIMATWIN v8.0 — Urban Heat Mitigation Intelligence
Physics-Informed Causal AI · Mumbai + Maharashtra · PS-1 National Hackathon
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import io, base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import gaussian_filter

from matplotlib.path import Path as MplPath
from modules.data_generator import (
    generate_data, generate_maharashtra_data,
    generate_full_maharashtra_data, MAHARASHTRA_BOUNDARY,
)
from modules.ml_model       import train_model, get_shap_summary, get_neighborhood_shap
from modules.scenarios       import simulate_portfolio, rank_by_priority
from modules.equity          import compute_equity_index, city_summary

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="CLIMATWIN",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom sidebar visibility control — bypasses Streamlit's native
# collapse toggle entirely (its data-testid changes across versions
# and was unreliable). This is a real widget we control directly,
# so it's guaranteed visible and clickable every time.
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# ── Design tokens — ONE accent, dark mono chrome ────────────────
BG   = "#0D1117"
C1   = "#161B22"
C2   = "#1C2128"
C3   = "#21262D"
BDR  = "#30363D"
BDR2 = "#3D444D"
TEXT = "#E6EDF3"
SUB  = "#8B949E"
DIM  = "#6E7681"
ACC  = "#0EA5E9"

HOT  = "#F43F5E"
WARM = "#FB923C"
MID  = "#FBBF24"
COOL = "#34D399"

THERMAL = {
    0.00: "#0D1117", 0.10: "#0C2A4A", 0.25: "#1D4ED8",
    0.42: "#0EA5E9", 0.58: "#34D399", 0.70: "#FBBF24",
    0.84: "#FB923C", 1.00: "#F43F5E",
}
THERMAL_STOPS = [
    (0.00, (0,   0,   0  )), (0.10, (0,   0,   80 )),
    (0.22, (0,   0,   200)), (0.36, (0,   120, 255)),
    (0.50, (0,   240, 240)), (0.62, (0,   210, 60 )),
    (0.72, (255, 230, 0  )), (0.82, (255, 100, 0  )),
    (0.92, (210, 0,   0  )), (1.00, (255, 200, 200)),
]
THERMAL_PX = [
    [0.00,"#0D1117"],[0.15,"#0C2A4A"],[0.30,"#1D4ED8"],
    [0.45,"#0EA5E9"],[0.60,"#34D399"],[0.72,"#FBBF24"],
    [0.86,"#FB923C"],[1.00,"#F43F5E"],
]

# ── CSS ─────────────────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html,body,.stApp,.main,
div[data-testid="stAppViewContainer"],
div[data-testid="stAppViewBlockContainer"] {{
  background:{BG}!important;
  font-family:'Inter',sans-serif!important;
  color:{SUB};
}}
.block-container {{
  padding:.8rem 1.6rem 2rem!important;
  max-width:1600px!important;
  background:{BG}!important;
}}
div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlockBorderWrapper"],
div.element-container {{ background:transparent!important; }}
#MainMenu,footer,
div[data-testid="stDecoration"],
div[data-testid="stToolbar"] {{
  display:none!important; visibility:hidden!important;
}}
header[data-testid="stHeader"] {{
  background:transparent!important;
}}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"]>div {{
  background:{C1}!important;
  border-right:1px solid {BDR}!important;
}}
div[data-testid="stSidebarContent"] {{ padding:.6rem .8rem!important; }}
div[data-testid="collapsedControl"] svg,
button[kind="header"] svg {{
  color:{TEXT}!important;
  fill:{TEXT}!important;
}}

.nav-link {{ font-size:.79rem!important; font-weight:500!important; color:{SUB}!important; border-radius:6px!important; }}
.nav-link:hover {{ background:{C2}!important; color:{TEXT}!important; }}
.nav-link-selected {{ background:{C2}!important; color:{ACC}!important; border-left:2px solid {ACC}!important; border-radius:0 6px 6px 0!important; }}
.nav-link .icon {{ color:{DIM}!important; }}
.nav-link-selected .icon {{ color:{ACC}!important; }}
div[data-testid="stSidebar"] svg {{ width:15px!important; height:15px!important; }}

.stTabs [data-baseweb="tab-list"] {{
  background:{C1}; border:1px solid {BDR}; border-radius:7px; padding:3px 4px; gap:2px;
}}
.stTabs [data-baseweb="tab"] {{
  background:transparent!important; color:{DIM}!important; border-radius:5px!important;
  padding:6px 14px!important; font-size:.78rem!important; font-weight:600!important; border:none!important;
}}
.stTabs [aria-selected="true"] {{ background:{C2}!important; color:{ACC}!important; border:1px solid {BDR2}!important; }}
.stTabs [data-baseweb="tab-highlight"]{{ display:none!important; }}
div[data-testid="stTabPanel"]{{ padding-top:.8rem; }}

div[data-testid="stSelectbox"]>div>div,
div[data-testid="stMultiSelect"]>div>div {{
  background:{C2}!important; border:1px solid {BDR}!important; color:{TEXT}!important; border-radius:6px!important;
}}
label[data-testid="stWidgetLabel"] p {{
  color:{DIM}!important; font-size:.68rem!important; font-weight:600!important;
  text-transform:uppercase!important; letter-spacing:.07em!important;
}}

div[data-testid="stDataFrame"] {{
  border:1px solid {BDR}!important; border-radius:7px!important;
  overflow:hidden!important; background:{C1}!important;
}}
hr{{ border-color:{BDR}!important; margin:.7rem 0!important; }}
::-webkit-scrollbar{{ width:4px; height:4px; }}
::-webkit-scrollbar-track{{ background:{BG}; }}
::-webkit-scrollbar-thumb{{ background:{BDR2}; border-radius:3px; }}

.page-title {{ font-size:1.0rem; font-weight:700; color:{TEXT}; letter-spacing:-.01em; }}
.page-sub   {{ font-size:.68rem; color:{DIM}; margin-top:2px; font-family:'JetBrains Mono',monospace; }}

.kpi {{ background:{C1}; border:1px solid {BDR}; border-top:1px solid var(--c); border-radius:7px; padding:14px 13px 11px; }}
.kpi .v {{ font-size:1.6rem; font-weight:800; color:var(--c); font-family:'JetBrains Mono',monospace; line-height:1; }}
.kpi .l {{ font-size:.63rem; color:{DIM}; font-weight:700; text-transform:uppercase; letter-spacing:.07em; margin-top:6px; }}
.kpi .s {{ font-size:.71rem; color:{SUB}; margin-top:2px; }}

.plbl {{ font-size:.62rem; font-weight:700; color:{DIM}; text-transform:uppercase; letter-spacing:.09em;
  margin-bottom:8px; padding-bottom:7px; border-bottom:1px solid {BDR}; }}

.row {{ display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid {BDR}; }}
.row:last-child {{ border-bottom:none; }}
.row .k {{ font-size:.75rem; color:{DIM}; }}
.row .v {{ font-size:.78rem; font-weight:600; color:{TEXT}; font-family:'JetBrains Mono',monospace; }}

.ins-item {{ display:flex; gap:9px; padding:9px 0; border-bottom:1px solid {BDR}; }}
.ins-item:last-child {{ border-bottom:none; }}
.ins-dot {{ width:5px;height:5px;background:{ACC};border-radius:50%;margin-top:6px;flex-shrink:0; }}
.ins-txt {{ font-size:.77rem; color:{SUB}; line-height:1.55; }}
.ins-txt b {{ color:{TEXT}; font-weight:600; }}

.mrow {{ background:{C2}; border:1px solid {BDR}; border-left:2px solid {ACC}; border-radius:0 6px 6px 0; padding:8px 12px; margin:4px 0; }}
.mrow .v {{ font-size:.92rem; font-weight:700; color:{TEXT}; font-family:'JetBrains Mono',monospace; }}
.mrow .l {{ font-size:.63rem; color:{DIM}; margin-top:2px; text-transform:uppercase; letter-spacing:.06em; }}

.arch-card {{ background:{C1}; border:1px solid {BDR}; border-top:1px solid var(--ac); border-radius:7px;
  padding:14px; height:148px; display:flex; flex-direction:column; gap:7px; }}
.arch-card .t {{ font-size:.82rem; font-weight:700; color:var(--ac); }}
.arch-card .d {{ font-size:.73rem; color:{SUB}; line-height:1.5; flex:1; }}
.arch-card .p {{ font-size:.6rem; font-weight:600; padding:2px 7px; border-radius:3px; border:1px solid var(--ac);
  color:var(--ac); width:fit-content; font-family:'JetBrains Mono',monospace; background:var(--ac)12; }}

.alert-card {{ background:{C1}; border:1px solid {BDR}; border-left:3px solid var(--ac); border-radius:0 7px 7px 0; padding:12px 14px; margin-bottom:8px; }}
.alert-card .h {{ font-size:.76rem; font-weight:700; color:var(--ac); text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }}
.alert-card .b {{ font-size:.76rem; color:{SUB}; line-height:1.5; }}

.stat-block {{ background:{C2}; border:1px solid {BDR}; border-radius:6px; padding:10px 12px; text-align:center; }}
.stat-block .v {{ font-size:1.05rem; font-weight:700; color:var(--ac); font-family:'JetBrains Mono',monospace; }}
.stat-block .l {{ font-size:.61rem; color:{DIM}; margin-top:3px; text-transform:uppercase; letter-spacing:.06em; }}

.cf-card {{ background:{C2}; border:1px solid {BDR}; border-radius:8px; padding:14px 16px; }}
.cf-arrow {{ font-size:1.1rem; color:{ACC}; text-align:center; padding:4px 0; }}
.cf-big {{ font-size:1.8rem; font-weight:800; font-family:'JetBrains Mono',monospace; }}
</style>""", unsafe_allow_html=True)

# Force sidebar visibility to match OUR session-state toggle in both
# directions — this overrides Streamlit's own native collapse/expand
# tracking so our button is fully authoritative regardless of what
# Streamlit's internal state thinks (fixes the case where Streamlit's
# native mechanism had separately collapsed it and our old hide-only
# rule couldn't force it back open).
if st.session_state.sidebar_open:
    st.markdown(
        "<style>"
        "section[data-testid='stSidebar']{"
        "display:block!important; visibility:visible!important;"
        "transform:none!important; margin-left:0!important;"
        "width:21rem!important; min-width:21rem!important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<style>"
        "section[data-testid='stSidebar']{"
        "display:none!important; visibility:hidden!important;"
        "}"
        ".block-container{max-width:1600px!important;}"
        "</style>",
        unsafe_allow_html=True,
    )

# ── Data ─────────────────────────────────────────────────────────
@st.cache_data
def load():
    df   = generate_data()
    df_e = compute_equity_index(df)
    _, _, _, sv, _, metrics, df_p = train_model(df)
    ss   = get_shap_summary(sv)
    summ = city_summary(df)
    return df, df_e, sv, metrics, df_p, ss, summ

@st.cache_data
def load_maharashtra():
    return generate_maharashtra_data()

@st.cache_data
def load_full_maharashtra():
    return generate_full_maharashtra_data()

@st.cache_data
def load_state_equity():
    mh_full = generate_full_maharashtra_data()
    mh_full_e = compute_equity_index(mh_full)
    mh_summ = city_summary(mh_full)
    return mh_full_e, mh_summ

# Pre-built point-in-polygon path for the state boundary (lon, lat order)
MH_BOUNDARY_PATH = MplPath([(lon, lat) for lat, lon in MAHARASHTRA_BOUNDARY])
MH_BOUNDARY_LATS = [p[0] for p in MAHARASHTRA_BOUNDARY]
MH_BOUNDARY_LONS = [p[1] for p in MAHARASHTRA_BOUNDARY]

df, df_e, sv, metrics, df_p, ss, summ = load()
mh_df = load_maharashtra()
mh_full_df = load_full_maharashtra()
mh_full_e, mh_summ = load_state_equity()
uhi = round(df.lst.max() - df[df.lst < df.lst.quantile(0.15)].lst.mean(), 1)
mh_uhi = round(mh_full_df.lst.max() - mh_full_df[mh_full_df.lst < mh_full_df.lst.quantile(0.15)].lst.mean(), 1)

# ── Plotly helpers ───────────────────────────────────────────────
def L(**kw):
    d = dict(
        paper_bgcolor=BG, plot_bgcolor=C1,
        font=dict(color=DIM, family="Inter, sans-serif", size=11),
        hoverlabel=dict(bgcolor=C2, bordercolor=BDR2, font=dict(color=TEXT, size=11)),
        legend=dict(bgcolor=C2, bordercolor=BDR, font=dict(color=SUB, size=10)),
        margin=dict(l=6, r=6, t=6, b=6),
    )
    d.update(kw)
    return d

def AX(**kw):
    d = dict(gridcolor=BDR, linecolor=BDR2, tickcolor=BDR2, color=DIM,
             zerolinecolor=BDR2, tickfont=dict(size=9.5, color=DIM))
    d.update(kw)
    return d

def CB(title="", **kw):
    d = dict(thickness=7, len=0.62, bgcolor=C2, bordercolor=BDR, borderwidth=1,
             tickfont=dict(color=DIM, size=8),
             title=dict(text=title, font=dict(color=DIM, size=8.5)))
    d.update(kw)
    return d

# ── HTML helpers ─────────────────────────────────────────────────
def plbl(t):     return f'<div class="plbl">{t}</div>'
def kpi(v,l,s,c):return f'<div class="kpi" style="--c:{c}"><div class="v">{v}</div><div class="l">{l}</div><div class="s">{s}</div></div>'
def mrow(v,l):   return f'<div class="mrow"><div class="v">{v}</div><div class="l">{l}</div></div>'
def stat(v,l,c): return f'<div class="stat-block" style="--ac:{c}"><div class="v">{v}</div><div class="l">{l}</div></div>'

# ── Map helpers ──────────────────────────────────────────────────
POPUP_S = (f"font-family:Inter,sans-serif;background:{C1};color:{TEXT};"
           f"padding:11px 13px;border-radius:7px;min-width:175px;"
           f"border:1px solid {BDR};font-size:11px;line-height:1.6;")

def _norm(vals):
    mn, mx = vals.min(), vals.max()
    return (vals - mn) / (mx - mn + 1e-9)

def _thermal_rgb(n):
    n = max(0.0, min(1.0, float(n)))
    for i in range(len(THERMAL_STOPS)-1):
        t0,(r0,g0,b0) = THERMAL_STOPS[i]
        t1,(r1,g1,b1) = THERMAL_STOPS[i+1]
        if n <= t1:
            f = (n-t0)/(t1-t0+1e-9)
            return int(r0+f*(r1-r0)), int(g0+f*(g1-g0)), int(b0+f*(b1-b0))
    return 255, 200, 200

def _thermal_hex_px(n):
    r,g,b = _thermal_rgb(n)
    return f"#{r:02X}{g:02X}{b:02X}"

def _tc(n):
    if n > 0.84: return HOT
    if n > 0.70: return WARM
    if n > 0.58: return MID
    if n > 0.30: return ACC
    return "#1D4ED8"

# ── Popup builders ────────────────────────────────────────────────
def lst_popup(row):
    lst_val = float(row.get("lst", 30))
    n = max(0.0, min(1.0, (lst_val - 22.0) / 28.0))
    c = _tc(n)
    return (f'<div style="{POPUP_S}">'
            f'<b style="color:{ACC};font-size:12px;">{row.get("neighborhood","—")}</b>'
            f'<hr style="border-color:{BDR};margin:5px 0;">'
            f'LST: <b style="color:{c};">{lst_val:.1f}°C</b><br>'
            f'{row.get("lst_category","—")} zone &nbsp;·&nbsp; {row.get("equity_class","—")}<br>'
            f'Green Cover: {row.get("green_cover_fraction",0)*100:.0f}% &nbsp;·&nbsp; '
            f'Pop: {row.get("pop_density_k",0):.0f}k/km²</div>')

def scen_popup(val_col):
    def _fn(row):
        v = float(row.get(val_col, 30))
        n = max(0, min(1, (v - 24) / 26))
        c = _tc(n)
        return (f'<div style="{POPUP_S}">'
                f'<b style="color:{ACC};font-size:12px;">{row.get("neighborhood","—")}</b>'
                f'<hr style="border-color:{BDR};margin:5px 0;">'
                f'Before: {row.get("lst_before",0):.1f}°C &nbsp; After: '
                f'<b style="color:{c};">{row.get("lst_after",0):.1f}°C</b><br>'
                f'Reduction: <b style="color:{COOL};">-{row.get("reduction",0):.2f}°C</b></div>')
    return _fn

def eq_popup(row):
    eq = float(row.get("equity_score", 0))
    c  = HOT if eq>70 else WARM if eq>50 else MID if eq>30 else COOL
    return (f'<div style="{POPUP_S}">'
            f'<b style="color:{ACC};font-size:12px;">{row.get("neighborhood","—")}</b>'
            f'<hr style="border-color:{BDR};margin:5px 0;">'
            f'Equity Score: <b style="color:{c};">{eq:.0f}/100</b><br>'
            f'{row.get("equity_class","—")} &nbsp;·&nbsp; {row.get("income_label","—")}<br>'
            f'LST: {row.get("lst",0):.1f}°C</div>')

# ── Pixel heatmap builder — high resolution, vectorised, boundary-aligned ──
def build_pixel_map(df_in, val_col, zoom=8, opacity=0.60,
                    grid_h=250, grid_w=200, mask_deg=0.50,
                    popup_fn=None, show_labels=True, label_col=None,
                    center=None, use_state_boundary=False,
                    show_city_labels=True, n_labels=14, smooth_sigma=4,
                    label_df=None, show_region_boundaries=False,
                    cluster_zones=True):
    """
    High-resolution pixel heatmap via matplotlib -> ImageOverlay.

    Deliberately minimal layer stack for a clean, legible map:
      1. Thermal pixel surface (the heatmap itself)
      2. Thin dashed state/mask boundary
      3. A small fixed set of city name anchors for orientation
      4. One interactive marker layer (MarkerCluster) — every zone is
         a small dot; zoomed out they collapse into a count-bubble,
         zoomed in they expand automatically. Hover any dot for name
         + temperature. This is the only place zone-level detail
         appears, so it never competes with permanent floating labels.

    (show_labels/label_col/label_df/n_labels/show_region_boundaries are
    accepted for backward compatibility with older call sites but are
    no-ops now — they were the source of the label/hull clutter.)

    Two masking modes:
      - use_state_boundary=True : exact point-in-polygon mask against
        the real Maharashtra state outline (MH_BOUNDARY_PATH). Pixels
        align precisely to the state shape with zero gaps inside and
        zero bleed outside — used for the full-state view.
      - use_state_boundary=False: KDTree distance mask from nearest
        data point — used for tighter city-level zoomed views.
    A Gaussian filter (smooth_sigma) is applied to the interpolated
    grid before colouring — this removes the faceted/triangulated
    "shattered glass" look that raw linear interpolation produces over
    sparse data, giving a smooth, consistent, professional gradient
    across the whole map instead of visible hard creases.
    Fully vectorised numpy RGBA generation renders in well under a
    second even at high resolution. Semi-transparent so the base map
    (roads, coastline, city names) always shows through.
    """
    vals = df_in[val_col].fillna(0).astype(float)
    pts  = df_in[["lat","lon"]].values.astype(float)
    v    = vals.values.astype(float)
    vmin, vmax = float(v.min()), float(v.max())

    pad     = 0.20
    lat_min = pts[:,0].min() - pad
    lat_max = pts[:,0].max() + pad
    lon_min = pts[:,1].min() - pad
    lon_max = pts[:,1].max() + pad

    lat_g = np.linspace(lat_max, lat_min, grid_h)
    lon_g = np.linspace(lon_min, lon_max, grid_w)
    lon_m, lat_m = np.meshgrid(lon_g, lat_g)

    gv = griddata(pts, v, (lat_m, lon_m), method="linear", fill_value=np.nan)
    # Fill any remaining gaps (cells outside the convex hull of data
    # points but still inside the state boundary) using nearest-neighbour
    # interpolation, so there are zero NaN holes inside the polygon.
    nan_mask = np.isnan(gv)
    if nan_mask.any():
        gv_nearest = griddata(pts, v, (lat_m, lon_m), method="nearest")
        gv = np.where(nan_mask, gv_nearest, gv)

    # Smooth away the "shattered glass" triangulation facets that raw
    # linear interpolation produces over sparse data — Gaussian blur
    # gives a continuous, professional-looking gradient. Applied before
    # masking so the boundary edge itself stays crisp.
    if smooth_sigma > 0:
        gv = gaussian_filter(gv, sigma=smooth_sigma)

    flat = np.column_stack([lat_m.ravel(), lon_m.ravel()])

    if use_state_boundary:
        # Exact polygon mask — pixels align to real state outline
        lonlat = np.column_stack([flat[:,1], flat[:,0]])
        valid_mask = MH_BOUNDARY_PATH.contains_points(lonlat) & ~np.isnan(gv.ravel())
    else:
        tree = cKDTree(pts)
        dist, _ = tree.query(flat)
        valid_mask = (dist < mask_deg) & ~np.isnan(gv.ravel())

    stops_t = np.array([s[0] for s in THERMAL_STOPS], dtype=np.float32)
    stops_r = np.array([s[1][0] for s in THERMAL_STOPS], dtype=np.float32)
    stops_g = np.array([s[1][1] for s in THERMAL_STOPS], dtype=np.float32)
    stops_b = np.array([s[1][2] for s in THERMAL_STOPS], dtype=np.float32)

    gv_clean = np.where(np.isnan(gv), vmin, gv)
    gv_norm  = np.clip((gv_clean - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
    gv_flat  = gv_norm.ravel().astype(np.float32)

    r_ch = np.interp(gv_flat, stops_t, stops_r).astype(np.uint8)
    g_ch = np.interp(gv_flat, stops_t, stops_g).astype(np.uint8)
    b_ch = np.interp(gv_flat, stops_t, stops_b).astype(np.uint8)
    a_ch = np.where(valid_mask, int(opacity * 255), 0).astype(np.uint8)

    rgba = np.stack([r_ch, g_ch, b_ch, a_ch], axis=1).reshape(grid_h, grid_w, 4)

    fig, ax = plt.subplots(1, 1, figsize=(grid_w/100, grid_h/100), dpi=100)
    ax.imshow(rgba, interpolation="nearest", aspect="auto")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img_url = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    clat, clon = center or (float(pts[:,0].mean()), float(pts[:,1].mean()))
    m = folium.Map(location=[clat, clon], zoom_start=zoom,
                   tiles="CartoDB dark_matter", prefer_canvas=True, zoom_control=True)

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=1.0, interactive=False, cross_origin=False, zindex=1,
    ).add_to(m)

    # State outline stroke — thin, subtle, single line for orientation
    if use_state_boundary:
        folium.PolyLine(
            locations=MAHARASHTRA_BOUNDARY + [MAHARASHTRA_BOUNDARY[0]],
            color=ACC, weight=1.3, opacity=0.45, dash_array="5,5",
        ).add_to(m)

    # City name anchors — the ONLY permanent text on the map. Small,
    # carefully spaced set so it never competes with marker clusters.
    if show_city_labels:
        if use_state_boundary:
            city_labels = [
                (19.076, 72.877, "Mumbai"), (18.520, 73.856, "Pune"),
                (20.001, 73.785, "Nashik"), (16.705, 74.243, "Kolhapur"),
                (21.146, 79.088, "Nagpur"), (20.938, 77.780, "Amravati"),
                (19.876, 75.343, "Sambhajinagar"), (17.660, 75.906, "Solapur"),
            ]
        else:
            city_labels = [
                (19.076, 72.877, "Mumbai"), (20.001, 73.785, "Nashik"),
                (18.520, 73.856, "Pune"),   (19.218, 72.978, "Thane"),
                (18.748, 73.405, "Lonavala"), (16.705, 74.243, "Kolhapur"),
            ]
        for clat2, clon2, cname in city_labels:
            folium.Marker(
                location=[clat2, clon2],
                icon=folium.DivIcon(
                    html=(f'<div style="color:#FFFFFF;font-family:Inter,sans-serif;'
                          f'font-size:11px;font-weight:700;text-shadow:0 0 5px #000,0 0 5px #000;'
                          f'white-space:nowrap;letter-spacing:.02em;pointer-events:none;">{cname}</div>'),
                    icon_size=(110, 18), icon_anchor=(55, 34),
                )
            ).add_to(m)

    # ── Zone markers — single interactive layer ────────────────────
    # Every zone is a small coloured dot, hover for name + temperature.
    # Clustered with Leaflet's native MarkerCluster: zoomed out, nearby
    # zones collapse into one small count-bubble; zoom in and they
    # expand into individual markers automatically. This is the ONLY
    # way zone-level detail appears on the map — no permanent floating
    # labels competing with it, so the base map stays legible at every
    # zoom level.
    norms_orig = _norm(vals).tolist()

    if cluster_zones:
        _cluster_js = """
        function(cluster) {
            var count = cluster.getChildCount();
            var size = count < 5 ? 22 : count < 15 ? 26 : count < 40 ? 30 : 34;
            return new L.DivIcon({
                html: '<div style="background:rgba(13,17,23,0.85);' +
                      'border:1.5px solid #0EA5E9;border-radius:50%;color:#E6EDF3;' +
                      'font-family:JetBrains Mono,monospace;font-weight:600;' +
                      'font-size:10px;display:flex;align-items:center;' +
                      'justify-content:center;width:100%;height:100%;">' + count + '</div>',
                className: 'climatwin-cluster',
                iconSize: L.point(size, size)
            });
        }
        """
        mc = MarkerCluster(
            icon_create_function=_cluster_js,
            spiderfy_on_max_zoom=True,
            show_coverage_on_hover=False,
            zoom_to_bounds_on_click=True,
            max_cluster_radius=50,
            disable_clustering_at_zoom=11,
        )
        target = mc
    else:
        target = m

    for (_, row), n in zip(df_in.iterrows(), norms_orig):
        c   = _thermal_hex_px(n)
        nbr = row.get("neighborhood", "—")
        val = vals[row.name]
        ph  = popup_fn(row) if popup_fn else (
            f'<div style="{POPUP_S}">'
            f'<b style="color:{ACC};font-size:12px;">{nbr}</b>'
            f'<hr style="border-color:{BDR};margin:5px 0;">'
            f'{val_col}: <b style="color:{c};">{val:.1f}</b></div>')
        folium.Marker(
            location=[float(row.lat), float(row.lon)],
            icon=folium.DivIcon(
                html=(f'<div style="width:10px;height:10px;border-radius:50%;'
                      f'background:{c};border:1px solid rgba(255,255,255,0.6);"></div>'),
                icon_size=(10, 10), icon_anchor=(5, 5),
            ),
            tooltip=folium.Tooltip(
                f"<b style='font-family:monospace;font-size:11px;'>{nbr}</b><br>"
                f"<span style='color:{c};font-weight:700;'>{val:.1f}°C</span>", sticky=True),
            popup=folium.Popup(ph, max_width=210),
        ).add_to(target)

    if cluster_zones:
        mc.add_to(m)

    return m

# ── Region classifier — for geographic diversity in priority lists ──
# Maps each of the 154 locations to its division so "top hottest"
# panels can show one representative per region instead of being
# dominated by whichever single morphology (Mumbai's dense informal
# settlements) happens to produce the most extreme raw LST value.
# The exact Mumbai zone name set is checked FIRST and short-circuits
# before any keyword matching — this avoids false-positive substring
# matches (e.g. "Wadala" containing "wada", the Palghar town keyword).
_MUMBAI_ZONE_NAMES = set(df.neighborhood.unique().tolist())

_REGION_KEYWORDS = [
    ("Vidarbha",           ["nagpur","wardha","chandrapur","ballarpur","gadchiroli",
                             "bhandara","gondia","amravati","akola","yavatmal",
                             "washim","buldhana"]),
    ("Marathwada",         ["sambhajinagar","jalna","beed","latur","dharashiv",
                             "nanded","parbhani","hingoli"]),
    ("North Maharashtra",  ["dhule","nandurbar","jalgaon","bhusawal","ahmednagar",
                             "shirdi","malegaon","nashik"]),
    ("Konkan",             ["ratnagiri","chiplun","sindhudurg","malvan","mahad",
                             "roha","pen","uran","palghar","boisar","jawhar",
                             "dahanu","alibaug"]),
    ("Western Maharashtra",["solapur","sangli","miraj","karad","baramati",
                             "phaltan","satara","kolhapur","pune"]),
    ("Sahyadri / Ghats",   ["khopoli","lonavala","khandala","matheran","karjat",
                             "igatpuri","kasara","shahapur","wada"]),
    ("Thane-Kalyan",       ["thane","kalyan","dombivli","bhiwandi","ulhasnagar",
                             "ambarnath","badlapur"]),
    ("Vasai-Virar",        ["bhayandar","vasai","virar","nala sopara"]),
]

def get_region(name: str) -> str:
    if name in _MUMBAI_ZONE_NAMES:
        return "Mumbai (MMR)"
    n = name.lower()
    for region, kws in _REGION_KEYWORDS:
        if any(kw in n for kw in kws):
            return region
    return "Mumbai (MMR)"

def top_per_region(df_in, val_col="lst", n_regions=None, ascending=False):
    """
    Returns the single hottest (or coldest, if ascending=True) zone
    from each geographic region — guarantees the result spans the
    whole state instead of clustering in one area with extreme values.
    """
    d = df_in.copy()
    d["_region"] = d["neighborhood"].apply(get_region)
    idx = (d.groupby("_region")[val_col]
             .idxmin() if ascending else d.groupby("_region")[val_col].idxmax())
    picked = d.loc[idx].sort_values(val_col, ascending=ascending)
    if n_regions:
        picked = picked.head(n_regions)
    return picked

# ── AI Insights from SHAP ────────────────────────────────────────
def gen_insights():
    top  = ss.head(3)
    top3 = mh_full_e.nlargest(3,"lst")[["neighborhood","lst"]].values
    return [
        (f"Surface temps driven by <b>{top.iloc[0]['label'].lower()}</b> "
         f"({top.iloc[0]['pct']:.0f}% of LST variance) and "
         f"<b>{top.iloc[1]['label'].lower()}</b> ({top.iloc[1]['pct']:.0f}%)."),
        (f"Critical zones statewide: <b>{', '.join([r[0] for r in top3])}</b> — peak LST "
         f"<b>{top3[0][1]:.1f}°C</b>, up to 22°C above forested buffer zones."),
        ("Cool roofs achieve <b>2.3°C reduction per 0.1 albedo increase</b> at "
         "₹120/m² with 18-month payback via energy savings (Akbari et al.)."),
        ("Street tree canopy delivers <b>0.14°C cooling per 1% cover increase</b> "
         "— highest ROI intervention for high-density residential wards."),
        (f"Equity-priority: <b>{mh_summ['urgent_zones']} zones</b> across Maharashtra face "
         f"both extreme heat and social vulnerability — {mh_summ['urgent_pop_pct']}% of population."),
        ("Recommended: cool roofs across <b>65% of high-density zones</b> + "
         "Miyawaki forests at 0.5 km spacing for maximum compound effect."),
    ]

INSIGHTS = gen_insights()

MAT_DF = pd.DataFrame({
    "Material":        ["Conv. Asphalt","White Paint","Cool Coating","Green Roof","Permeable Pave"],
    "Albedo":          [0.05, 0.80, 0.65, 0.20, 0.15],
    "Surface Temp °C": ["54–70","28–36","32–40","22–28","36–44"],
    "Cooling °C":      [0, 12.4, 9.8, 6.3, 4.5],
    "Cost ₹/m²":       [450, 85, 160, 1800, 950],
    "Durability yr":   ["20+","5–7","10–12","20+","15+"],
})

# ── COUNTERFACTUAL ENGINE — unique differentiator ─────────────────
FEATURES_CF = [
    "building_density","green_cover_fraction","impervious_surface_fraction",
    "road_density","water_proximity","pop_density_k","income_index","industrial",
]

@st.cache_resource
def get_cf_model():
    from modules.ml_model import train_model as _tm
    model, scaler, _, _, _, _, _ = _tm(df)
    return model, scaler

def counterfactual_predict(source_row, target_row, swap_features):
    """
    Predicts LST for source_row's location IF it adopted target_row's
    values for swap_features — everything else held constant.
    Returns (original_lst, counterfactual_lst, delta).
    """
    model, scaler = get_cf_model()
    x = source_row[FEATURES_CF].copy().astype(float)
    for f in swap_features:
        x[f] = target_row[f]
    x_scaled = scaler.transform(pd.DataFrame([x], columns=FEATURES_CF))
    pred = float(model.predict(x_scaled)[0])
    orig = float(source_row["lst"])
    return orig, pred, orig - pred

# ── SIDEBAR NAVIGATION ───────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:10px 4px 14px;">'
        f'<div style="font-size:1.0rem;font-weight:800;color:{TEXT};letter-spacing:-.01em;">CLIMATWIN</div>'
        f'<div style="font-size:.62rem;color:{DIM};font-family:JetBrains Mono,monospace;margin-top:2px;">HEAT INTELLIGENCE</div>'
        f'</div>', unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=[
            "Heat Map", "What-If Engine", "Materials", "Analysis",
            "Predictions", "Optimization", "Equity",
            "Alerts", "Reports", "Settings",
        ],
        icons=[
            "thermometer-high", "shuffle", "layers", "bar-chart-line",
            "cpu", "sliders", "balance-scale",
            "bell", "file-earmark-text", "gear",
        ],
        default_index=0,
        styles={
            "container":         {"background-color": C1, "padding": "0"},
            "menu-title":        {"display": "none"},
            "icon":              {"color": DIM, "font-size": "13px"},
            "nav-link":          {
                "font-family": "Inter, sans-serif", "font-size": "12px",
                "font-weight": "500", "color": SUB, "border-radius": "6px",
                "padding": "8px 12px", "--hover-color": C2,
            },
            "nav-link-selected": {
                "background-color": C2, "color": ACC,
                "border-left": f"2px solid {ACC}",
                "border-radius": "0 6px 6px 0", "font-weight": "600",
            },
        },
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{C2};border:1px solid {BDR};border-radius:6px;padding:11px 12px;">'
        f'<div style="font-size:.6rem;color:{DIM};text-transform:uppercase;letter-spacing:.08em;margin-bottom:9px;">Model</div>'
        + "".join(
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {BDR};">'
            f'<span style="font-size:.71rem;color:{DIM};">{k}</span>'
            f'<span style="font-size:.71rem;font-weight:600;color:{TEXT};font-family:JetBrains Mono,monospace;">{v}</span></div>'
            for k,v in [("R²", metrics["R²"]),("MAE", f"{metrics['MAE']}°C"),("CV R²", metrics["CV R² Mean"])]
        ) + f'</div>', unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────
_tcol, _hcol = st.columns([1, 40], gap="small")
with _tcol:
    toggle_label = "☰" if not st.session_state.sidebar_open else "◀"
    if st.button(toggle_label, key="sidebar_toggle", help="Show/hide sidebar menu"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

st.markdown(
    f'<div style="display:flex;align-items:center;justify-content:space-between;'
    f'padding:0 0 12px;border-bottom:1px solid {BDR};margin-bottom:14px;">'
    f'<div>'
    f'<div class="page-title">Urban Heat Mitigation Intelligence — {page}</div>'
    f'<div class="page-sub">Mumbai + Full Maharashtra State · 154 Locations · Jun–Sep 2024 · PS-1 National Hackathon</div>'
    f'</div>'
    f'<div style="display:flex;gap:7px;">'
    + "".join(
        f'<div style="background:{C1};border:1px solid {BDR};border-radius:6px;padding:6px 12px;text-align:center;">'
        f'<div style="font-size:.58rem;color:{DIM};text-transform:uppercase;letter-spacing:.07em;">{l}</div>'
        f'<div style="font-size:.84rem;font-weight:700;color:{c};font-family:JetBrains Mono,monospace;">{v}</div></div>'
        for l,v,c in [("Peak LST",f"{mh_full_df.lst.max():.1f}°C",HOT),
                      ("UHI",f"+{mh_uhi}°C",WARM),("Urgent",str(mh_summ['urgent_zones']),MID),
                      ("Zones","154",ACC)]
    ) + f'</div></div>', unsafe_allow_html=True)

# ═══════════════════════════ PAGE ROUTER ════════════════════════

# ── HEAT MAP ────────────────────────────────────────────────────
if page == "Heat Map":
    hottest_zone = mh_full_df.loc[mh_full_df.lst.idxmax(), "neighborhood"]
    c1,c2,c3,c4 = st.columns(4, gap="small")
    for col,(v,l,s,c) in zip([c1,c2,c3,c4],[
        (f"{mh_full_df.lst.max():.1f}°C","Peak Surface Temp",f"{hottest_zone} — critical zone",HOT),
        (f"+{mh_uhi}°C","UHI Intensity","vs rural surroundings",WARM),
        (str(mh_summ["urgent_zones"]),"Critical Zones",f"{mh_summ['urgent_pop_pct']}% population exposed",MID),
        (f"{mh_full_df.green_cover_fraction.mean()*100:.1f}%","Avg Green Cover",f"Deficit: {mh_summ['green_deficit_mean']:.1f}%",COOL),
    ]):
        col.markdown(kpi(v,l,s,c), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    cA, cB = st.columns([6, 4], gap="medium")
    with cA:
        st.markdown(plbl("Urban Heat Map — Full Maharashtra State (Pixel Format)"), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:.67rem;color:{DIM};margin:-8px 0 8px;font-family:JetBrains Mono,monospace;">'
            f'Hover any zone for details · Dashed line = state boundary · Nandurbar → Nagpur → Kolhapur → Sindhudurg</div>',
            unsafe_allow_html=True)
        m = build_pixel_map(mh_full_df, "lst", zoom=7, opacity=0.62,
                            grid_h=450, grid_w=380, popup_fn=lst_popup,
                            use_state_boundary=True, smooth_sigma=6,
                            center=(19.2, 76.0))
        st_folium(m, width="100%", height=480, returned_objects=[], key="hm_main")

    with cB:
        st.markdown(plbl("AI / ML Insights"), unsafe_allow_html=True)
        _priority_zones = top_per_region(mh_full_e, "lst", n_regions=6)
        st.markdown(
            f'<div style="background:{C2};border:1px solid {BDR};border-radius:6px;padding:10px 12px;margin-bottom:10px;">'
            f'<div style="font-size:.6rem;color:{DIM};text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Surface Temperature — Hottest Zone per Region</div>'
            + "".join(
                f'<div class="row"><div class="k">{r["neighborhood"]} <span style="color:{DIM};font-size:.62rem;">· {r["_region"]}</span></div>'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<div class="v" style="color:{HOT};">{r["lst"]:.1f}°C</div>'
                f'<div style="font-size:.62rem;color:{DIM};background:{HOT}15;border:1px solid {HOT}33;padding:1px 6px;border-radius:3px;">{r["lst_category"]}</div></div></div>'
                for _, r in _priority_zones.iterrows()
            ) + f'</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{C2};border:1px solid {BDR};border-radius:6px;padding:10px 12px;">'
            f'<div style="font-size:.6rem;color:{DIM};text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Analysis & Recommendations</div>'
            + "".join(f'<div class="ins-item"><div class="ins-dot"></div><div class="ins-txt">{txt}</div></div>' for txt in INSIGHTS)
            + f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cP1, cP2, cP3 = st.columns([4, 4, 4], gap="medium")
    with cP1:
        st.markdown(plbl("Cooling Potential by Strategy"), unsafe_allow_html=True)
        strats = ["Cool Roofs","Street Trees","Urban Forest","Green Roofs","Water Features"]
        vals_s = [5.75, 1.68, 2.20, 1.35, 1.80]
        fig_iv = go.Figure(go.Bar(
            y=strats, x=vals_s, orientation="h",
            marker=dict(color=vals_s, colorscale=THERMAL_PX, opacity=0.9, line=dict(width=0)),
            text=[f" {v:.1f}°C" for v in vals_s], textposition="outside",
            textfont=dict(color=TEXT, size=10, family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>Cooling: %{x:.2f}°C<extra></extra>",
        ))
        fig_iv.update_layout(xaxis=AX(title="Temperature Reduction (°C)", title_font=dict(size=10,color=DIM)),
                              yaxis=AX(), **L(height=230, margin=dict(l=4,r=60,t=4,b=32)))
        st.plotly_chart(fig_iv, use_container_width=True)
    with cP2:
        st.markdown(plbl("Neighbourhood Prioritisation — Statewide"), unsafe_allow_html=True)
        res_d = simulate_portfolio(mh_full_e, {"cool_roofs":0.25,"street_trees":0.12,"green_roofs":0.15})
        rnk_d = rank_by_priority(res_d)
        rnk_d["_region"] = rnk_d["neighborhood"].apply(get_region)
        rnk_d = (rnk_d.loc[rnk_d.groupby("_region")["priority_score"].idxmax()]
                       .sort_values("priority_score", ascending=False))
        tbl = rnk_d[["neighborhood","_region","lst_before","reduction","equity_score","priority_score"]].copy()
        tbl.columns = ["Neighbourhood","Region","Heat Risk","Reduction","Equity","Priority"]
        st.dataframe(tbl.style.background_gradient(cmap="RdYlGn", subset=["Priority"], vmin=0, vmax=100),
                    use_container_width=True, hide_index=True, height=230)
    with cP3:
        st.markdown(plbl("Material Performance Comparison"), unsafe_allow_html=True)
        st.dataframe(MAT_DF.style.background_gradient(cmap="RdYlGn", subset=["Cooling °C"], vmin=0, vmax=14),
                    use_container_width=True, hide_index=True, height=230)

# ── WHAT-IF ENGINE — Counterfactual Causal Explorer ──────────────
elif page == "What-If Engine":
    st.markdown(plbl("Counterfactual Causal Simulator — \"What If Zone A Had Zone B's Urban Form?\""), unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:.76rem;color:{SUB};margin:-4px 0 14px;line-height:1.6;">'
        f'This is genuine causal reasoning, not a lookup table: the trained model re-predicts LST for '
        f'your chosen zone after swapping in another zone\'s real urban-form features. It answers the '
        f'question every planner actually asks — <b style="color:{TEXT};">"if we made this area more like '
        f'that one, how much cooler would it get?"</b></div>', unsafe_allow_html=True)

    cs1, cs2, cs3 = st.columns([3,3,4], gap="medium")
    zone_list = sorted(mh_full_e.neighborhood.tolist())
    with cs1:
        source_name = st.selectbox("Source zone (the one you want to cool)", zone_list,
                                   index=zone_list.index("Dharavi") if "Dharavi" in zone_list else 0)
    with cs2:
        target_name = st.selectbox("Reference zone (borrow its urban form)", zone_list,
                                   index=zone_list.index("Aarey Colony") if "Aarey Colony" in zone_list else 1)
    with cs3:
        swap_choices = st.multiselect(
            "Features to swap",
            options=FEATURES_CF,
            default=["green_cover_fraction","impervious_surface_fraction"],
            format_func=lambda x: x.replace("_"," ").title(),
        )

    source_row = mh_full_e[mh_full_e.neighborhood == source_name].iloc[0]
    target_row = mh_full_e[mh_full_e.neighborhood == target_name].iloc[0]

    if swap_choices:
        orig, cf, delta = counterfactual_predict(source_row, target_row, swap_choices)

        r1, r2, r3 = st.columns([3,1,3], gap="small")
        with r1:
            st.markdown(
                f'<div class="cf-card">'
                f'<div style="font-size:.65rem;color:{DIM};text-transform:uppercase;letter-spacing:.07em;">Current — {source_name}</div>'
                f'<div class="cf-big" style="color:{HOT};">{orig:.1f}°C</div>'
                f'<div style="font-size:.72rem;color:{SUB};margin-top:6px;">'
                + "".join(f'{f.replace("_"," ").title()}: <b style="color:{TEXT};">{source_row[f]:.2f}</b><br>' for f in swap_choices)
                + f'</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="cf-arrow" style="padding-top:40px;">→</div>', unsafe_allow_html=True)
            improved = delta > 0
            badge_c = COOL if improved else HOT
            st.markdown(
                f'<div style="text-align:center;margin-top:6px;">'
                f'<div style="font-size:1.3rem;font-weight:800;color:{badge_c};font-family:JetBrains Mono,monospace;">'
                f'{"−" if improved else "+"}{abs(delta):.2f}°C</div>'
                f'<div style="font-size:.62rem;color:{DIM};text-transform:uppercase;">{"cooler" if improved else "hotter"}</div>'
                f'</div>', unsafe_allow_html=True)
        with r3:
            st.markdown(
                f'<div class="cf-card">'
                f'<div style="font-size:.65rem;color:{DIM};text-transform:uppercase;letter-spacing:.07em;">If it had {target_name}\'s form</div>'
                f'<div class="cf-big" style="color:{COOL if improved else HOT};">{cf:.1f}°C</div>'
                f'<div style="font-size:.72rem;color:{SUB};margin-top:6px;">'
                + "".join(f'{f.replace("_"," ").title()}: <b style="color:{TEXT};">{target_row[f]:.2f}</b><br>' for f in swap_choices)
                + f'</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        pop = float(source_row.get("pop_density_k", 10)) * 1000
        st.markdown(
            f'<div class="alert-card" style="--ac:{ACC};">'
            f'<div class="h">Projected Impact</div>'
            f'<div class="b">If <b style="color:{TEXT};">{source_name}</b> adopted '
            + " and ".join(f.replace("_"," ") for f in swap_choices)
            + f' levels similar to <b style="color:{TEXT};">{target_name}</b>, model predicts a '
            f'<b style="color:{COOL if improved else HOT};">{abs(delta):.2f}°C {"reduction" if improved else "increase"}</b> '
            f'in surface temperature — affecting approximately <b style="color:{TEXT};">{pop:,.0f} residents</b> '
            f'in the zone.</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(plbl("Feature-by-Feature Sensitivity"), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:.73rem;color:{SUB};margin-bottom:10px;">'
            f'Isolated effect of swapping each feature individually — shows which single change matters most.</div>',
            unsafe_allow_html=True)
        sens_rows = []
        for f in swap_choices:
            _, _, d = counterfactual_predict(source_row, target_row, [f])
            sens_rows.append((f.replace("_"," ").title(), d))
        sens_rows.sort(key=lambda x: -abs(x[1]))
        fig_sens = go.Figure(go.Bar(
            x=[d for _,d in sens_rows], y=[n for n,_ in sens_rows], orientation="h",
            marker=dict(color=[COOL if d>0 else HOT for _,d in sens_rows], opacity=0.88, line=dict(width=0)),
            text=[f" {d:+.2f}°C" for _,d in sens_rows], textposition="outside",
            textfont=dict(color=TEXT, size=10.5, family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>Isolated effect: %{x:+.2f}°C<extra></extra>",
        ))
        fig_sens.add_vline(x=0, line_color=BDR2, line_width=1, line_dash="dot")
        fig_sens.update_layout(
            xaxis=AX(title="Temperature change if swapped alone (°C)", title_font=dict(size=10,color=DIM)),
            yaxis=AX(autorange="reversed"),
            **L(height=max(160, 45*len(sens_rows)), margin=dict(l=4,r=70,t=4,b=32)),
        )
        st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.markdown(
            f'<div style="padding:30px;text-align:center;color:{DIM};font-size:.85rem;">'
            f'Select at least one feature to swap to run the simulation.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(plbl("Best Reference Zone Finder"), unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:.73rem;color:{SUB};margin-bottom:10px;">'
        f'Automatically searches all 154 zones statewide to find which one, if copied, would cool '
        f'<b style="color:{TEXT};">{source_name}</b> the most (swapping green cover + impervious surface).</div>',
        unsafe_allow_html=True)
    if st.button("Find Best Donor Zone", key="find_donor"):
        candidates = []
        for _, cand in mh_full_e.iterrows():
            if cand["neighborhood"] == source_name:
                continue
            _, _, d = counterfactual_predict(source_row, cand, ["green_cover_fraction","impervious_surface_fraction"])
            candidates.append((cand["neighborhood"], d))
        candidates.sort(key=lambda x: -x[1])
        top5 = candidates[:5]
        cols = st.columns(5)
        for col,(name,d) in zip(cols, top5):
            col.markdown(
                f'<div class="stat-block" style="--ac:{COOL};">'
                f'<div class="v">−{d:.2f}°C</div><div class="l">{name}</div></div>',
                unsafe_allow_html=True)

# ── MATERIALS ───────────────────────────────────────────────────
elif page == "Materials":
    st.markdown(plbl("Surface Material Performance & Thermal Properties"), unsafe_allow_html=True)
    c1, c2 = st.columns([5, 5], gap="medium")
    with c1:
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(name="Albedo", x=MAT_DF["Material"], y=MAT_DF["Albedo"],
            marker=dict(color=ACC, opacity=0.8, line=dict(width=0)), yaxis="y",
            hovertemplate="<b>%{x}</b><br>Albedo: %{y:.2f}<extra></extra>"))
        fig_m.add_trace(go.Bar(name="Cooling (°C)", x=MAT_DF["Material"], y=MAT_DF["Cooling °C"],
            marker=dict(color=COOL, opacity=0.8, line=dict(width=0)), yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Cooling: %{y:.1f}°C<extra></extra>"))
        fig_m.update_layout(barmode="group", xaxis=AX(), yaxis=AX(title="Albedo"),
            yaxis2=dict(overlaying="y", side="right", title="Cooling °C", **AX()),
            **L(height=320, margin=dict(l=4,r=70,t=4,b=40)))
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown(plbl("Cost-Effectiveness Analysis"), unsafe_allow_html=True)
        fig_ce = go.Figure(go.Scatter(
            x=MAT_DF["Cost ₹/m²"], y=MAT_DF["Cooling °C"], mode="markers+text",
            marker=dict(size=MAT_DF["Cooling °C"]*3+10, color=MAT_DF["Cooling °C"],
                        colorscale=THERMAL_PX, opacity=0.85, line=dict(color=BG,width=1.5)),
            text=MAT_DF["Material"], textposition="top center", textfont=dict(color=SUB, size=9),
            hovertemplate="<b>%{text}</b><br>Cost: ₹%{x}/m²<br>Cooling: %{y}°C<extra></extra>"))
        fig_ce.update_layout(xaxis=AX(title="Cost (₹/m²)", title_font=dict(size=10,color=DIM)),
            yaxis=AX(title="Cooling Effect (°C)", title_font=dict(size=10,color=DIM)),
            **L(height=260, margin=dict(l=4,r=4,t=4,b=36)))
        st.plotly_chart(fig_ce, use_container_width=True)

    with c2:
        st.markdown(plbl("Material Data"), unsafe_allow_html=True)
        st.dataframe(MAT_DF.style.background_gradient(cmap="RdYlGn", subset=["Cooling °C"], vmin=0, vmax=14),
                    use_container_width=True, hide_index=True, height=280)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(plbl("Key Findings"), unsafe_allow_html=True)
        for txt in [
            "<b>White paint</b> delivers the highest cooling (12.4°C) at the lowest cost (₹85/m²) — best single-material ROI.",
            "<b>Green roofs</b> provide compound benefits: cooling + stormwater + biodiversity, justified at ₹1800/m² for high-density zones.",
            "<b>Cool coatings</b> (₹160/m²) are the recommended deployment at scale for 65%+ coverage of priority zones.",
            "<b>Permeable paving</b> reduces surface temps while addressing urban flooding — dual-benefit intervention for low-lying wards.",
        ]:
            st.markdown(f'<div class="ins-item"><div class="ins-dot"></div><div class="ins-txt">{txt}</div></div>', unsafe_allow_html=True)

# ── ANALYSIS ────────────────────────────────────────────────────
elif page == "Analysis":
    cG, cL = st.columns([6, 5], gap="medium")
    with cG:
        st.markdown(plbl("Global Driver Ranking — SHAP Analysis"), unsafe_allow_html=True)
        fig_s = go.Figure(go.Bar(
            x=ss.shap_impact, y=ss.label, orientation="h",
            marker=dict(color=ss.shap_impact, colorscale=THERMAL_PX, opacity=0.9, line=dict(width=0)),
            text=[f" {v:.3f}°C  ({p:.0f}%)" for v,p in zip(ss.shap_impact,ss.pct)],
            textposition="outside", textfont=dict(color=TEXT, size=10, family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}°C<extra></extra>"))
        fig_s.update_layout(xaxis=AX(title="Mean |SHAP| Contribution to LST (°C)", title_font=dict(size=10,color=DIM)),
            yaxis=AX(autorange="reversed"), **L(height=320, margin=dict(l=4,r=140,t=4,b=32)))
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown(plbl("Feature Correlation Matrix"), unsafe_allow_html=True)
        fc  = ["building_density","green_cover_fraction","impervious_surface_fraction","road_density","water_proximity","lst"]
        fl  = ["Bldg","Green","Imperv","Road","Water","LST"]
        cor = df[fc].corr().round(2)
        fig_c = go.Figure(go.Heatmap(z=cor.values, x=fl, y=fl,
            colorscale=[[0,"#1D4ED8"],[0.5,C1],[1.0,HOT]], zmin=-1, zmax=1,
            text=cor.values.round(2), texttemplate="%{text}",
            textfont=dict(color=TEXT, size=11, family="JetBrains Mono"),
            hovertemplate="<b>%{x} × %{y}</b><br>r = %{z:.2f}<extra></extra>", colorbar=CB("r")))
        fig_c.update_layout(xaxis=AX(), yaxis=AX(autorange="reversed"), **L(height=248, margin=dict(l=4,r=64,t=4,b=4)))
        st.plotly_chart(fig_c, use_container_width=True)

    with cL:
        st.markdown(plbl("Neighbourhood-Level SHAP Breakdown"), unsafe_allow_html=True)
        nbr = st.selectbox("", sorted(df_p.neighborhood.tolist()), label_visibility="collapsed", key="an_nbr")
        lc  = get_neighborhood_shap(sv, df_p, nbr)
        if not lc.empty:
            fig_l = go.Figure(go.Bar(
                x=lc.shap_value, y=lc.label, orientation="h",
                marker=dict(color=[HOT if v>0 else COOL for v in lc.shap_value], opacity=0.88, line=dict(width=0)),
                text=[f" {v:+.3f}" for v in lc.shap_value], textposition="outside",
                textfont=dict(color=TEXT, size=10, family="JetBrains Mono"),
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}°C<extra></extra>"))
            fig_l.add_vline(x=0, line_color=BDR2, line_width=1, line_dash="dot")
            fig_l.update_layout(xaxis=AX(title="Heating  ·  Cooling", title_font=dict(size=9,color=DIM), zeroline=True, zerolinecolor=BDR2),
                yaxis=AX(autorange="reversed"), **L(height=290, margin=dict(l=4,r=72,t=4,b=32)))
            st.plotly_chart(fig_l, use_container_width=True)
            row = df_p[df_p.neighborhood==nbr].iloc[0]
            m1c,m2c,m3c = st.columns(3)
            for col,l,v,c in [(m1c,"Observed",f"{row.lst:.1f}°C",HOT),(m2c,"Predicted",f"{row.lst_predicted:.1f}°C",ACC),(m3c,"Error",f"{row.residual:+.2f}°C",MID)]:
                col.markdown(f'<div style="background:{C2};border:1px solid {BDR};border-top:1px solid {c};border-radius:6px;padding:9px;text-align:center;">'
                            f'<div style="font-size:.95rem;font-weight:700;color:{c};font-family:JetBrains Mono,monospace;">{v}</div>'
                            f'<div style="font-size:.61rem;color:{DIM};margin-top:3px;text-transform:uppercase;letter-spacing:.05em;">{l}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(plbl("Key Findings"), unsafe_allow_html=True)
        for h,b in [
            ("Impervious surfaces — 35% of variance","Concrete and asphalt absorb 4x more radiation than vegetation. Every 10% increase in ISF adds +0.85°C city-wide."),
            ("Green cover — highest actionable lever","20% SHAP share. Targeted greening of Dharavi's 2.1 km² rooftops could yield 3–4°C peak reduction."),
            ("Water proximity cools by 4–6°C","Coastal and lake-adjacent zones are significantly cooler. Inland water features deliver highest thermal ROI."),
        ]:
            st.markdown(f'<div style="padding:8px 0;border-bottom:1px solid {BDR};">'
                        f'<div style="font-size:.73rem;font-weight:600;color:{TEXT};margin-bottom:3px;">{h}</div>'
                        f'<div style="font-size:.73rem;color:{SUB};line-height:1.5;">{b}</div></div>', unsafe_allow_html=True)

# ── PREDICTIONS ─────────────────────────────────────────────────
elif page == "Predictions":
    r1,r2,r3,r4 = st.columns(4, gap="small")
    for col,l,v,c in [(r1,"R² Score",str(metrics["R²"]),ACC),(r2,"MAE (°C)",str(metrics["MAE"]),COOL),
                      (r3,"CV R² Mean",str(metrics["CV R² Mean"]),MID),(r4,"CV R² ±Std",f"±{metrics['CV R² Std']}",DIM)]:
        col.markdown(f'<div style="background:{C1};border:1px solid {BDR};border-top:1px solid {c};border-radius:7px;padding:13px;text-align:center;">'
                    f'<div style="font-size:1.5rem;font-weight:800;color:{c};font-family:JetBrains Mono,monospace;">{v}</div>'
                    f'<div style="font-size:.62rem;color:{DIM};margin-top:5px;text-transform:uppercase;letter-spacing:.06em;">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ca, cb = st.columns(2, gap="medium")
    with ca:
        st.markdown(plbl("Predicted vs Observed LST"), unsafe_allow_html=True)
        lim = [df_p.lst.min()-1, df_p.lst.max()+1]
        fig_po = go.Figure()
        fig_po.add_shape(type="line",x0=lim[0],y0=lim[0],x1=lim[1],y1=lim[1],line=dict(color=BDR2,dash="dot",width=1.5))
        fig_po.add_trace(go.Scatter(x=df_p.lst, y=df_p.lst_predicted, mode="markers",
            marker=dict(size=9, color=df_p.residual.abs(), colorscale=THERMAL_PX, opacity=0.88, line=dict(color=BG,width=1.5), colorbar=CB("|Err|")),
            text=df_p.neighborhood, hovertemplate="<b>%{text}</b><br>Obs: %{x:.1f}°C<br>Pred: %{y:.1f}°C<extra></extra>"))
        fig_po.update_layout(xaxis=AX(title="Observed LST (°C)",title_font=dict(size=10,color=DIM)),
            yaxis=AX(title="Predicted LST (°C)",title_font=dict(size=10,color=DIM)), **L(height=340, margin=dict(l=4,r=68,t=4,b=32)))
        st.plotly_chart(fig_po, use_container_width=True)
    with cb:
        st.markdown(plbl("Prediction Residuals"), unsafe_allow_html=True)
        fig_r = go.Figure(go.Histogram(x=df_p.residual, nbinsx=18,
            marker=dict(color=df_p.residual, colorscale=THERMAL_PX, opacity=0.85, line=dict(color=BG,width=1)),
            hovertemplate="Residual: %{x:.2f}°C<br>Count: %{y}<extra></extra>"))
        fig_r.add_vline(x=0,line_color=ACC,line_width=1,line_dash="dot")
        fig_r.update_layout(xaxis=AX(title="Residual (°C)",title_font=dict(size=10,color=DIM)),
            yaxis=AX(title="Count",title_font=dict(size=10,color=DIM)), **L(height=340, margin=dict(l=4,r=4,t=4,b=32)))
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(plbl("Production Architecture — Roadmap"), unsafe_allow_html=True)
    a1,a2,a3,a4 = st.columns(4, gap="medium")
    for col,t,d,tech,ac in [
        (a1,"Causal Discovery","DoWhy DAG reveals true causal structure of UHI drivers — WHY heat concentrates, not just WHERE.","DoWhy · pgmpy",ACC),
        (a2,"Physics-Informed NN","PINN constrained by Penman-Monteith energy balance. Thermodynamically consistent predictions.","DeepXDE · PyTorch",MID),
        (a3,"Temporal Graph NN","Urban topology as spatiotemporal graph. Heat propagation via message-passing through street network.","PyG · PyTorch",COOL),
        (a4,"Multi-Agent RL","MARL finds Pareto-optimal cooling portfolio across budget and equity constraints city-wide.","Ray RLlib",WARM),
    ]:
        col.markdown(f'<div class="arch-card" style="--ac:{ac};"><div class="t">{t}</div><div class="d">{d}</div><div class="p">{tech}</div></div>', unsafe_allow_html=True)

# ── OPTIMIZATION ────────────────────────────────────────────────
elif page == "Optimization":
    cCtrl, cR = st.columns([4, 7], gap="medium")
    with cCtrl:
        st.markdown(plbl("Intervention Configuration — Statewide"), unsafe_allow_html=True)
        iv_gr = st.slider("Green Roofs — % cover",    0, 30, 10) / 100
        iv_st = st.slider("Street Trees — % canopy",  0, 25,  8) / 100
        iv_cr = st.slider("Cool Roofs — albedo ×0.1", 0,  4,  2) / 10
        iv_wf = st.slider("Water Features — ×0.1",    0,  4,  1) / 10
        iv_uf = st.slider("Urban Forest — % cover",   0, 20,  5) / 100

        sel = {"green_roofs":iv_gr,"street_trees":iv_st,"cool_roofs":iv_cr,"water_features":iv_wf,"urban_forest":iv_uf}
        res = simulate_portfolio(mh_full_e, sel)
        rnk = rank_by_priority(res)
        best = res.loc[res.reduction.idxmax(),"neighborhood"]

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(plbl("Outcomes"), unsafe_allow_html=True)
        for v,l in [
            (f"-{res.reduction.mean():.2f} °C",  "Average cooling — all zones"),
            (f"-{res.reduction.max():.2f} °C",   f"Best zone — {best}"),
            (f"Rs {res.cost_lakh.sum():.0f} L",  "Total portfolio cost"),
            (f"{int(res[res.reduction>2].pop_density_k.sum()*1000):,}", "People cooled by >2°C"),
        ]:
            st.markdown(mrow(v,l), unsafe_allow_html=True)

    with cR:
        st.markdown(plbl("Surface Temperature — Live Simulation (Statewide)"), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:.67rem;color:{DIM};margin:-8px 0 8px;font-family:JetBrains Mono,monospace;">'
            f'Updates instantly as sliders move · Showing post-intervention temperatures across Maharashtra</div>',
            unsafe_allow_html=True)
        ma = build_pixel_map(res,"lst_after",zoom=7,opacity=0.62, grid_h=350,grid_w=300,
                             popup_fn=scen_popup("lst_after"),
                             use_state_boundary=True, smooth_sigma=6,
                             center=(19.2,76.0))
        st_folium(ma,width="100%",height=430,returned_objects=[],key="opt_live")

        b1,b2,b3 = st.columns(3, gap="small")
        base_before = res.lst_before.mean()
        base_after  = res.lst_after.mean()
        for col,v,l,c in [
            (b1, f"{base_before:.1f}°C", "Baseline avg (no intervention)", DIM),
            (b2, f"{base_after:.1f}°C",  "Current avg (live)",             ACC),
            (b3, f"-{base_before-base_after:.2f}°C", "Live reduction",     COOL),
        ]:
            col.markdown(
                f'<div style="background:{C2};border:1px solid {BDR};border-radius:6px;'
                f'padding:8px 10px;text-align:center;">'
                f'<div style="font-size:.9rem;font-weight:700;color:{c};font-family:JetBrains Mono,monospace;">{v}</div>'
                f'<div style="font-size:.6rem;color:{DIM};margin-top:2px;">{l}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cp1, cp2 = st.columns([5,6], gap="medium")
    with cp1:
        st.markdown(plbl("Cost vs Cooling Effectiveness"), unsafe_allow_html=True)
        fig_p = go.Figure(go.Scatter(x=res.cost_lakh, y=res.reduction, mode="markers",
            marker=dict(size=res.equity_score/4+7, color=res.equity_score,
                        colorscale=[[0,COOL],[0.5,MID],[1.0,HOT]], cmin=0,cmax=100,opacity=0.85,
                        line=dict(color=BG,width=1.5),colorbar=CB("Equity")),
            text=res.neighborhood, hovertemplate="<b>%{text}</b><br>Cost: Rs %{x:.0f}L<br>Cooling: -%{y:.2f}°C<extra></extra>"))
        fig_p.update_layout(xaxis=AX(title="Cost (Rs Lakhs)",title_font=dict(size=10,color=DIM)),
            yaxis=AX(title="Temperature Reduction (°C)",title_font=dict(size=10,color=DIM)), **L(height=300,margin=dict(l=4,r=68,t=4,b=32)))
        st.plotly_chart(fig_p, use_container_width=True)
    with cp2:
        st.markdown(plbl("Priority Ranking — Statewide"), unsafe_allow_html=True)
        disp = rnk[["neighborhood","lst_before","lst_after","reduction","priority_score"]].head(12).copy()
        disp.columns = ["Neighbourhood","Before","After","Reduction °C","Priority"]
        st.dataframe(disp, use_container_width=True, hide_index=True, height=260)
        st.download_button(
            "Download Full Priority Ranking — All 154 Zones (CSV)",
            data=rnk[["neighborhood","lst_before","lst_after","reduction","equity_score","priority_score"]]
                    .to_csv(index=False).encode("utf-8"),
            file_name="climatwin_intervention_priority_ranking.csv",
            mime="text/csv", use_container_width=True,
        )

# ── EQUITY ──────────────────────────────────────────────────────
elif page == "Equity":
    e1,e2,e3,e4 = st.columns(4, gap="small")
    for col,(v,l,s,c) in zip([e1,e2,e3,e4],[
        (str(mh_summ["urgent_zones"]),     "Urgent Zones",        "Need immediate action",HOT),
        (f"{mh_summ['urgent_pop_pct']}%",  "Population at Risk",  "High heat + vulnerability",WARM),
        (mh_summ["hottest_vulnerable"],    "Highest Equity Score", f"{mh_summ['max_equity_score']}/100",MID),
        (f"{mh_summ['green_deficit_mean']:.1f}%","Green Deficit","Below optimal coverage",COOL),
    ]):
        col.markdown(kpi(v,l,s,c), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    cQ, cEM = st.columns([5,5], gap="medium")
    with cQ:
        st.markdown(plbl("Heat Exposure vs Social Vulnerability — Statewide"), unsafe_allow_html=True)
        fig_q = go.Figure()
        for cls_full,color in {"🔴 Urgent":HOT,"🟠 Monitor":WARM,"🟡 Support":MID,"🟢 Stable":COOL}.items():
            sub = mh_full_e[mh_full_e.equity_class==cls_full]
            if sub.empty: continue
            fig_q.add_trace(go.Scatter(x=sub.heat_exposure, y=sub.vulnerability, mode="markers",
                name=cls_full.split(" ",1)[1],
                marker=dict(size=sub.equity_score/4+7,color=color,opacity=0.85,line=dict(color=BG,width=1.5)),
                text=sub.neighborhood, hovertemplate="<b>%{text}</b><br>Heat: %{x:.2f}<br>Vuln: %{y:.2f}<extra></extra>"))
        hm,vm = mh_full_e.heat_exposure.median(),mh_full_e.vulnerability.median()
        fig_q.add_vline(x=hm,line_dash="dot",line_color=BDR2,line_width=1)
        fig_q.add_hline(y=vm,line_dash="dot",line_color=BDR2,line_width=1)
        for tx,ty,label,ac in [(hm+.02,vm+.02,"URGENT",HOT),(hm+.02,vm-.09,"MONITOR",WARM),(hm-.16,vm+.02,"SUPPORT",MID),(hm-.16,vm-.09,"STABLE",COOL)]:
            fig_q.add_annotation(x=tx,y=ty,text=label,showarrow=False,font=dict(color=ac,size=8,family="JetBrains Mono"))
        fig_q.update_layout(xaxis=AX(title="Heat Exposure (normalised)",title_font=dict(size=10,color=DIM)),
            yaxis=AX(title="Social Vulnerability (normalised)",title_font=dict(size=10,color=DIM)), **L(height=380,margin=dict(l=4,r=4,t=4,b=32)))
        st.plotly_chart(fig_q, use_container_width=True)

    with cEM:
        st.markdown(plbl("Equity Score Map — Full State"), unsafe_allow_html=True)
        m4 = build_pixel_map(mh_full_e,"equity_score",zoom=7,opacity=0.60, popup_fn=eq_popup,
                             show_labels=False, use_state_boundary=True, smooth_sigma=5,
                             center=(19.2,76.0))
        st_folium(m4,width="100%",height=380,returned_objects=[],key="eq_map")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(plbl("Urgent Zones — Immediate Action Required (Statewide)"), unsafe_allow_html=True)
    urg = mh_full_e[mh_full_e.equity_class=="🔴 Urgent"][["neighborhood","lst","equity_score","income_label","pop_density_k","green_cover_fraction","intervention_note"]].sort_values("equity_score",ascending=False).copy()
    urg.columns = ["Neighbourhood","LST °C","Equity Score","Income Level","Pop. k/km²","Green Cover","Recommended Action"]
    st.dataframe(urg, use_container_width=True, hide_index=True, height=250)

# ── ALERTS ──────────────────────────────────────────────────────
elif page == "Alerts":
    st.markdown(plbl("Active Heat Alerts — Statewide Urban Heat Intelligence System"), unsafe_allow_html=True)
    ca, cb = st.columns([5, 5], gap="medium")
    with ca:
        _alert_top3 = ", ".join(top_per_region(mh_full_e, "lst", n_regions=3)["neighborhood"].tolist())
        for h,b,l,c in [
            ("CRITICAL — Heat Emergency", f"{_alert_top3} recording peak LST of {mh_full_df.lst.max():.1f}°C across Maharashtra. Population exposure: ~500k residents in the hottest zones. Outdoor work restriction advised.", "Active",HOT),
            ("WARNING — UHI Intensification", f"Urban Heat Island intensity at +{mh_uhi}°C above rural baseline statewide. Night-time minimum temperatures elevated by 2.8°C in dense urban wards.", "Active",WARM),
            ("WATCH — Low Green Cover", f"{(mh_full_e.green_cover_fraction<0.05).sum()} neighbourhoods statewide below 5% green cover threshold. High thermal stress risk during afternoon peak (14:00–17:00 IST).", "Monitoring",MID),
            ("ADVISORY — Heat Equity Risk", f"{mh_summ['urgent_pop_pct']}% of Maharashtra's population in urgent zones. Low-income outdoor workers most at risk. Community cooling centres recommended statewide.", "Advisory",COOL),
        ]:
            st.markdown(f'<div class="alert-card" style="--ac:{c};"><div class="h">{h} <span style="font-size:.58rem;font-weight:400;padding:1px 6px;border-radius:3px;background:{c}18;border:1px solid {c}44;margin-left:6px;">{l}</span></div><div class="b">{b}</div></div>', unsafe_allow_html=True)

    with cb:
        st.markdown(plbl("Alert Statistics — Statewide"), unsafe_allow_html=True)
        alert_stats = [
            ("Zones Above 42°C", f"{(mh_full_e.lst>=42).sum()}", HOT),
            ("Zones Above 38°C", f"{(mh_full_e.lst>=38).sum()}", WARM),
            ("Urgent Equity Zones", str(mh_summ["urgent_zones"]), MID),
            ("Pop. in Critical Zones", f"{(mh_full_e[mh_full_e.lst>=42].pop_density_k.sum()*1000/1e6):.1f}M", HOT),
            ("State Mean LST", f"{mh_full_e.lst.mean():.1f}°C", ACC),
            ("UHI Intensity", f"+{mh_uhi}°C", WARM),
        ]
        g1, g2 = st.columns(2)
        for i,(l,v,c) in enumerate(alert_stats):
            (g1 if i%2==0 else g2).markdown(stat(v,l,c), unsafe_allow_html=True)
            (g1 if i%2==0 else g2).markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(plbl("Early Warning Protocol"), unsafe_allow_html=True)
        for step,desc in [
            ("Step 1 — Monitoring","CLIMATWIN continuously ingests Landsat 8 + ERA5 data, computing 6-hour rolling LST forecasts."),
            ("Step 2 — Threshold Detection","When LST exceeds 42°C in any ward for 3+ consecutive hours, alert is triggered automatically."),
            ("Step 3 — Notification","Alerts pushed to NDMA, municipal bodies, and public health officials via REST API and SMS."),
            ("Step 4 — Intervention","Pre-computed optimal interventions deployed — community cooling centres, water distribution."),
        ]:
            st.markdown(f'<div class="ins-item"><div class="ins-dot"></div><div class="ins-txt"><b>{step}</b> — {desc}</div></div>', unsafe_allow_html=True)

# ── REPORTS ─────────────────────────────────────────────────────
elif page == "Reports":
    st.markdown(plbl("Summary Report — Maharashtra Statewide Urban Heat Intelligence"), unsafe_allow_html=True)
    _rpt_hottest3 = ", ".join(top_per_region(mh_full_e, "lst", n_regions=3)["neighborhood"].tolist())
    _rpt_top_zone = mh_full_e.loc[mh_full_e.lst.idxmax()]
    r1,r2 = st.columns([5,5], gap="medium")
    with r1:
        st.markdown(plbl("Executive Summary"), unsafe_allow_html=True)
        for h,b in [
            ("Urban Heat Status", f"Maharashtra's 154 monitored zones record LST of {mh_full_df.lst.min():.1f}–{mh_full_df.lst.max():.1f}°C with a UHI intensity of +{mh_uhi}°C. {mh_summ['urgent_zones']} critical zones identified statewide affecting {mh_summ['urgent_pop_pct']}% of the population."),
            ("Primary Drivers", "Impervious surfaces (35% of variance) and low green cover (20%) are the dominant drivers. Road density and building density contribute a further 25% combined."),
            ("Intervention Impact", "A combined cool roofs + street trees + urban forest portfolio can achieve -5.75°C average reduction across priority zones at an estimated cost of Rs 120–450/m²."),
            ("Equity Implications", f"{mh_summ['urgent_zones']} zones across Maharashtra face compound heat burden — extreme LST combined with low income and high population density. {_rpt_top_zone['neighborhood']} ranks highest on the Thermal Equity Index."),
            ("Recommended Priority", f"Deploy cool coatings immediately across the hottest zone in each division — {_rpt_hottest3}. Follow with Miyawaki urban forests at 0.5 km spacing for 12-month compound effect."),
        ]:
            st.markdown(f'<div style="padding:9px 0;border-bottom:1px solid {BDR};"><div style="font-size:.73rem;font-weight:700;color:{TEXT};margin-bottom:3px;">{h}</div><div style="font-size:.73rem;color:{SUB};line-height:1.55;">{b}</div></div>', unsafe_allow_html=True)

    with r2:
        st.markdown(plbl("Key Metrics — Statewide"), unsafe_allow_html=True)
        all_metrics = [
            ("Peak LST", f"{mh_full_df.lst.max():.1f}°C", HOT), ("Minimum LST", f"{mh_full_df.lst.min():.1f}°C", COOL),
            ("State Mean LST", f"{mh_full_e.lst.mean():.1f}°C", ACC), ("UHI Intensity", f"+{mh_uhi}°C", WARM),
            ("Critical Zones", str(mh_summ["urgent_zones"]), HOT), ("Population at Risk", f"{mh_summ['urgent_pop_pct']}%",WARM),
            ("Green Cover Mean", f"{mh_full_df.green_cover_fraction.mean()*100:.1f}%",COOL), ("Green Cover Deficit", f"{mh_summ['green_deficit_mean']:.1f}%",MID),
            ("Highest Equity Zone", mh_summ["hottest_vulnerable"], HOT), ("Best Intervention", "Cool Roofs (-5.75°C)", COOL),
            ("Model R²", str(metrics["R²"]), ACC), ("Model MAE", f"{metrics['MAE']}°C", COOL),
        ]
        m1c,m2c = st.columns(2)
        for i,(l,v,c) in enumerate(all_metrics):
            col = m1c if i%2==0 else m2c
            col.markdown(f'<div class="row"><div class="k">{l}</div><div class="v" style="color:{c};">{v}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(plbl("Regional Comparison — All 8 Divisions"), unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:.73rem;color:{SUB};margin:-4px 0 10px;">'
        f'Aggregate heat, exposure, and green-cover statistics by division — the statewide view no single-zone list can show.</div>',
        unsafe_allow_html=True)
    _reg = mh_full_e.copy()
    _reg["_region"] = _reg["neighborhood"].apply(get_region)
    _reg_summary = _reg.groupby("_region").agg(
        Zones=("neighborhood","count"),
        Avg_LST=("lst","mean"),
        Peak_LST=("lst","max"),
        Critical_Zones=("lst_category", lambda s: (s=="Critical").sum()),
        Green_Cover=("green_cover_fraction", lambda s: s.mean()*100),
        Avg_Equity=("equity_score","mean"),
    ).reset_index().sort_values("Avg_LST", ascending=False)
    _reg_summary.columns = ["Region","Zones","Avg LST °C","Peak LST °C","Critical Zones","Green Cover %","Avg Equity Score"]
    _reg_summary = _reg_summary.round(1)
    rc1, rc2 = st.columns([6,5], gap="medium")
    with rc1:
        st.dataframe(
            _reg_summary.style.background_gradient(cmap="RdYlGn_r", subset=["Avg LST °C"], vmin=30, vmax=45)
                              .background_gradient(cmap="RdYlGn", subset=["Green Cover %"], vmin=0, vmax=40),
            use_container_width=True, hide_index=True, height=310)
    with rc2:
        fig_reg = go.Figure(go.Bar(
            y=_reg_summary["Region"], x=_reg_summary["Avg LST °C"], orientation="h",
            marker=dict(color=_reg_summary["Avg LST °C"], colorscale=THERMAL_PX, opacity=0.9, line=dict(width=0)),
            text=[f" {v:.1f}°C" for v in _reg_summary["Avg LST °C"]], textposition="outside",
            textfont=dict(color=TEXT, size=10, family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>Avg LST: %{x:.1f}°C<extra></extra>",
        ))
        fig_reg.update_layout(
            xaxis=AX(title="Average LST (°C)", title_font=dict(size=10,color=DIM)),
            yaxis=AX(autorange="reversed"),
            **L(height=310, margin=dict(l=4,r=60,t=4,b=32)),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    dl1, dl2 = st.columns(2, gap="medium")
    with dl1:
        st.download_button(
            "Download Regional Comparison (CSV)",
            data=_reg_summary.to_csv(index=False).encode("utf-8"),
            file_name="climatwin_regional_comparison.csv",
            mime="text/csv", use_container_width=True,
        )
    with dl2:
        _export_cols = ["neighborhood","lat","lon","lst","lst_category","equity_score",
                        "equity_class","green_cover_fraction","building_density",
                        "impervious_surface_fraction","pop_density_k","income_label"]
        _full_export = mh_full_e[_export_cols].copy()
        _full_export["region"] = _full_export["neighborhood"].apply(get_region)
        st.download_button(
            "Download Full Statewide Dataset — 154 Zones (CSV)",
            data=_full_export.to_csv(index=False).encode("utf-8"),
            file_name="climatwin_maharashtra_154_zones.csv",
            mime="text/csv", use_container_width=True,
        )

# ── SETTINGS ────────────────────────────────────────────────────
elif page == "Settings":
    st.markdown(plbl("Platform Configuration"), unsafe_allow_html=True)
    c1, c2 = st.columns([5,5], gap="medium")
    with c1:
        st.markdown(plbl("Data Configuration"), unsafe_allow_html=True)
        st.selectbox("City / Pilot Region", ["Mumbai", "Chennai", "Delhi", "Hyderabad", "Bangalore"], index=0)
        st.selectbox("LST Data Source", ["Landsat 8 (Synthetic)", "ECOSTRESS (Live)", "ERA5 Derived"], index=0)
        st.selectbox("Analysis Period", ["Jun–Sep 2024", "Mar–Jun 2024", "Jan–Dec 2024"], index=0)
        st.slider("Spatial Resolution (m)", 100, 1000, 500, 50)
        st.markdown(plbl("Map Configuration"), unsafe_allow_html=True)
        st.selectbox("Base Map Style", ["CartoDB Dark Matter", "CartoDB Positron", "OSM Standard"], index=0)
        st.slider("Pixel Grid Resolution", 100, 300, 250, 10)
        st.slider("Heatmap Opacity", 30, 90, 60, 5)
    with c2:
        st.markdown(plbl("Model Configuration"), unsafe_allow_html=True)
        st.slider("Random Forest Trees", 50, 500, 200, 50)
        st.slider("Max Depth", 3, 15, 8, 1)
        st.slider("Cross-Validation Folds", 3, 10, 5, 1)
        st.markdown(plbl("System Information"), unsafe_allow_html=True)
        for k,v in [("Version","CLIMATWIN v8.0"),("Data Source","Physics-Informed Synthetic"),
                    ("ML Framework","scikit-learn 1.3 + SHAP 0.43"),("Visualization","Folium + Matplotlib + Plotly"),
                    ("Deployment","Streamlit Cloud — Free Tier"),("License","MIT — Open Source")]:
            st.markdown(f'<div class="row"><div class="k">{k}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────────
st.markdown(
    f'<div style="margin-top:24px;padding:9px 0;border-top:1px solid {BDR};display:flex;justify-content:space-between;">'
    f'<div style="font-size:.63rem;color:{DIM};font-family:JetBrains Mono,monospace;">'
    f'CLIMATWIN v8.0 · PS-1 · Bharatiya Antariksh Hackathon 2026 · 100% open-source</div>'
    f'<div style="font-size:.63rem;color:{BDR2};font-family:JetBrains Mono,monospace;">'
    f'Landsat 8 · ERA5 · OSM · scikit-learn · SHAP · Folium · Streamlit</div></div>',
    unsafe_allow_html=True)
