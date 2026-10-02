import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import datetime
import io
import time
import re
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import nnls

# ── Setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SG Macro Dashboard", layout="wide", page_icon="🇸🇬")

st.markdown("""
<style>
  .stApp { background-color: #0e1117; }
  .block-container { padding-top: 1rem; }
  .metric-card {
    background: #161b26; border: 1px solid #2a2f3e;
    border-radius: 8px; padding: 12px 16px; text-align: center;
  }
  .metric-label { font-size: 0.68rem; color: #8a94a6; letter-spacing: 0.08em;
                  text-transform: uppercase; margin-bottom: 3px; }
  .metric-value { font-size: 1.1rem; font-weight: 700; white-space: nowrap; }
  .metric-delta { font-size: 0.72rem; margin-top: 2px; }
  .metric-z { font-size: 0.64rem; margin-top: 3px; display: flex; justify-content: center; gap: 8px; }
  .metric-asof { font-size: 0.6rem; color: #5f6b7e; margin-top: 2px; }
  .band-wrap { background: #161b26; border: 1px solid #2a2f3e; border-radius: 8px; padding: 14px 18px; }
  .band-track { position: relative; height: 14px; border-radius: 7px; margin: 22px 0 6px;
                background: linear-gradient(90deg, #ef5350 0%, #3a3f4e 30%, #3a3f4e 70%, #26a69a 100%); }
  .band-mid { position: absolute; left: 50%; top: -4px; width: 2px; height: 22px; background: #e0e0e0; }
  .band-dot { position: absolute; top: -5px; width: 24px; height: 24px; margin-left: -12px; border-radius: 50%;
              background: #ff9800; border: 3px solid #0e1117; }
  .band-scale { display: flex; justify-content: space-between; font-size: 0.66rem; color: #8a94a6; }
  .positive { color: #26a69a; }
  .negative { color: #ef5350; }
  .neutral  { color: #e0e0e0; }
  .section-header {
    font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #8a94a6; margin: 1.2rem 0 0.5rem;
    border-bottom: 1px solid #2a2f3e; padding-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)

TEMPLATE   = "plotly_dark"
PAPER_BG   = "#0e1117"
PLOT_BG    = "#161b26"
GRID_COLOR = "#2a2f3e"

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ── Chart helpers (same visual language as usa_macro.py) ───────────────────────

def base_layout(title="", height=480):
    return dict(
        template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        height=height,
        margin=dict(l=50, r=50, t=45, b=30),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )

def dual_axis_layout(title, y1_title, y2_title, height=480):
    layout = base_layout(title, height)
    layout.update(
        yaxis =dict(title=y1_title, gridcolor=GRID_COLOR),
        yaxis2=dict(title=y2_title, overlaying="y", side="right", gridcolor=GRID_COLOR),
    )
    return layout

def csv_download(df: pd.DataFrame, label: str):
    buf = io.BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    st.download_button("⬇ CSV", buf, file_name=f"{label}.csv",
                       mime="text/csv", key=f"dl_{label}_{id(df)}")

def render_two_col(charts):
    """Render (title, fig [, df]) tuples in 2-column layout."""
    n, i = len(charts), 0
    while i < n:
        if i == n - 1 and n % 2 != 0:
            item = charts[i]
            st.plotly_chart(item[1], use_container_width=True, key=f"chart_{item[0]}")
            if len(item) > 2 and item[2] is not None:
                csv_download(item[2], item[0])
            i += 1
        else:
            c1, c2 = st.columns(2)
            for col, item in [(c1, charts[i]), (c2, charts[i+1])]:
                with col:
                    st.plotly_chart(item[1], use_container_width=True, key=f"chart_{item[0]}")
                    if len(item) > 2 and item[2] is not None:
                        csv_download(item[2], item[0])
            i += 2

# ── Z-scores (same conventions as usa_macro.py's summary bar) ──────────────────
# A 1-observation rolling window can't produce a std, so lower-frequency series use the
# closest honest equivalent windows instead of forcing the daily 1M/3M/1Y labels onto them.
Z_WINDOWS = {
    "D": {"1M": 21, "3M": 63, "1Y": 252},        # trading days
    "W": {"3M": 13, "6M": 26, "1Y": 52},         # weeks (MAS S$NEER is weekly)
    "M": {"3M": 3, "12M": 12, "36M": 36},        # months
    "Q": {"4Q": 4, "8Q": 8, "20Q": 20},          # quarters (GDP, unemployment)
}
Z_HOVER_WINDOW = {"D": 252, "W": 52, "M": 36, "Q": 20}
Z_HOVER_LABEL = {"D": "1Y", "W": "1Y", "M": "36M", "Q": "20Q"}

def _zscores(series, windows):
    out = {}
    series = series.dropna()
    for label, window in windows.items():
        if len(series) <= window + 1:
            out[label] = None
            continue
        z = (series - series.rolling(window).mean()) / series.rolling(window).std()
        v = z.iloc[-1]
        out[label] = float(v) if pd.notna(v) and np.isfinite(v) else None
    return out

def _zroll(series: pd.Series, window: int) -> pd.Series:
    """Full rolling z-score history (each point vs its own trailing window) - used as hover
    customdata so every point on a chart shows its own z-score, same as usa_macro's _z1y."""
    clean = series.dropna()
    return (clean - clean.rolling(window).mean()) / clean.rolling(window).std()

def _value_months_ago(s: pd.Series, months=1):
    """Value at-or-before exactly N calendar months before the latest date - a true 1M change
    for daily/weekly series (iloc[-2] on a daily series is just the previous day)."""
    v = s.asof(s.index[-1] - pd.DateOffset(months=months))
    return float(v) if pd.notna(v) else None

def _change_over(s: pd.Series, offset):
    s = s.dropna()
    if s.empty:
        return np.nan
    prev = s.asof(s.index[-1] - offset)
    return float(s.iloc[-1] - prev) if pd.notna(prev) else np.nan

def _pctile(s: pd.Series, years: int):
    s = s.dropna()
    if s.empty:
        return np.nan
    w = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    return float((w <= w.iloc[-1]).mean() * 100) if len(w) > 20 else np.nan

def zline(fig, s_full: pd.Series, name, freq="D", color=None, unit="", fmt=".2f", yaxis="y",
          dash=None, width=1.6, label_fn=None):
    """Add a line whose hover shows value + that point's own trailing z-score. Z is computed on
    the FULL history before clipping to the date slider, so the first visible point still has
    a full window behind it."""
    s_full = s_full.dropna()
    if s_full.empty:
        return
    z = _zroll(s_full, Z_HOVER_WINDOW[freq])
    s = clip(s_full)
    zc = z.reindex(s.index)
    xlab = label_fn(s.index) if label_fn else list(s.index.strftime("%Y-%m-%d"))
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, name=name, mode="lines", yaxis=yaxis,
        line=dict(color=color, width=width, dash=dash),
        customdata=np.column_stack([zc.values.astype(float), np.array(xlab, dtype=object)]),
        hovertemplate=f"%{{customdata[1]}}<br>{name}: %{{y:{fmt}}}{unit}<br>"
                      f"Z ({Z_HOVER_LABEL[freq]}): %{{customdata[0]:.2f}}<extra></extra>"))

def build_card(name, s, fmt, delta_kind, freq):
    """Returns the dict render_cards() needs. delta_kind: 'bps' (yield in %, delta in bps),
    'raw_bps' (series already in bps), 'pct' (% change of a level), 'pp' (pp change)."""
    s = s.dropna() if s is not None else pd.Series(dtype=float)
    if len(s) < 3:
        return {"name": name, "val": None}
    last = float(s.iloc[-1])
    prev = _value_months_ago(s) if freq in ("D", "W") else float(s.iloc[-2])
    d, dstr = None, ""
    if prev is not None:
        if delta_kind == "bps":
            d, dstr = (last - prev) * 100, f"{(last - prev) * 100:+.0f}bps"
        elif delta_kind == "raw_bps":
            d, dstr = last - prev, f"{last - prev:+.0f}bps"
        elif delta_kind == "pct":
            d, dstr = (last / prev - 1) * 100, f"{(last / prev - 1) * 100:+.2f}%"
        else:
            d, dstr = last - prev, f"{last - prev:+.2f}pp"
    dlabel = {"D": "1M", "W": "1M", "M": "MoM", "Q": "QoQ"}[freq]
    ts = s.index[-1]
    asof = (ts.strftime("%d %b %Y") if freq in ("D", "W") else
            f"{ts.year} Q{(ts.month - 1) // 3 + 1}" if freq == "Q" else ts.strftime("%b %Y"))
    return {"name": name, "val": fmt(last), "delta": f"{dstr} {dlabel}" if d is not None else "",
            "dcls": "positive" if (d or 0) > 0 else "negative" if (d or 0) < 0 else "neutral",
            "z": _zscores(s, Z_WINDOWS[freq]), "asof": f"as of {asof}"}

def render_cards(cards, row_size=6):
    for row_start in range(0, len(cards), row_size):
        cols = st.columns(row_size)
        for col, c in zip(cols, cards[row_start:row_start + row_size]):
            with col:
                if c.get("val") is None:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">{c["name"]}</div>'
                                f'<div class="metric-value neutral">N/A</div></div>', unsafe_allow_html=True)
                    continue
                z_spans = []
                for zl, zv in (c["z"] or {}).items():
                    if zv is None:
                        z_spans.append(f'<span class="neutral">{zl} n/a</span>')
                    else:
                        zcls = "positive" if zv > 0 else "negative" if zv < 0 else "neutral"
                        z_spans.append(f'<span class="{zcls}">{zl} {zv:+.1f}</span>')
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{c["name"]}</div>
                  <div class="metric-value neutral">{c["val"]}</div>
                  <div class="metric-delta {c["dcls"]}">{c["delta"]}</div>
                  <div class="metric-z">{"".join(z_spans)}</div>
                  <div class="metric-asof">{c["asof"]}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

def _zcolor(v):
    """Cell style for z-score columns in the monitor tables (green +, red -, darker = further out)."""
    if pd.isna(v):
        return "color: #5f6b7e"
    a = min(abs(v) / 2.5, 1.0)
    rgb = "38,166,154" if v > 0 else "239,83,80"
    return f"background-color: rgba({rgb},{0.10 + 0.5 * a:.2f}); color: #e0e0e0"

def _chgcolor(v):
    if pd.isna(v) or v == 0:
        return "color: #8a94a6"
    return "color: #26a69a" if v > 0 else "color: #ef5350"

def _pctcolor(v):
    if pd.isna(v):
        return "color: #5f6b7e"
    return _zcolor((v - 50) / 20)

def mom_yoy(df: pd.DataFrame, col: str, periods_per_year: int) -> pd.DataFrame:
    """periods_per_year=12 for monthly index series, 4 for quarterly."""
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[f"{col} MoM %", f"{col} YoY %"])
    out = pd.DataFrame(index=df.index)
    out[f"{col} MoM %"] = (df[col].pct_change() * 100).round(3)
    out[f"{col} YoY %"] = (df[col].pct_change(periods_per_year) * 100).round(3)
    return out

# ── SingStat Table Builder API (free, no key — verified live 2026-08-20) ───────
# https://tablebuilder.singstat.gov.sg/api/table/tabledata/{resourceId}
# Table (resourceId) vintages get periodically rebased by SingStat (e.g. "2024 as
# base year" CPI); if a resourceId below ever 404s, it's been retired on a rebase
# and needs re-finding via the /api/table/resourceid?keyword= search endpoint.
SINGSTAT_BASE = "https://tablebuilder.singstat.gov.sg/api/table/tabledata"

def _parse_singstat_period(key: str):
    key = key.strip()
    m = re.match(r"^(\d{4})\s+(\d)Q$", key)
    if m:
        return pd.Period(f"{m.group(1)}Q{m.group(2)}").to_timestamp()
    dt = pd.to_datetime(key, format="%Y %b", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(key, errors="coerce")
    return dt

def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        m = re.match(r"-?\d+\.?\d*", str(v))
        return float(m.group()) if m else np.nan

@st.cache_data(ttl=21600)
def fetch_singstat(resource_id: str, label: str, series_no="1", n_periods=200) -> pd.DataFrame:
    try:
        r = requests.get(f"{SINGSTAT_BASE}/{resource_id}", headers=HEADERS, timeout=20,
                          params={"seriesNoORrowNo": series_no, "limit": n_periods, "sortBy": "key desc"})
        r.raise_for_status()
        rows = r.json()["Data"]["row"]
        row = next((x for x in rows if x["seriesNo"] == str(series_no)), rows[0])
        data = {_parse_singstat_period(c["key"]): _to_float(c["value"]) for c in row["columns"]}
        s = pd.Series(data).sort_index()
        s = s[s.index.notna()]
        df = pd.DataFrame({label: s})
        df.index.name = "date"
        return df
    except Exception as e:
        st.warning(f"Could not load {label} ({resource_id}): {e}")
        return pd.DataFrame()

# ── CPI breakdown: SingStat index levels x SingStat official 2024-base weights ────────
# Same resourceId (M213751) already used for headline CPI carries ~200 series rows (divisions,
# groups, classes) - fetched here in ONE call. The table holds indices only, not basket weights,
# so the weights below (per 10,000) are hardcoded from SingStat's own "Rebasing of the Consumer
# Price Index (2024 as Base Year)" information paper, Appendix III (S-COICOP 2022). Verified
# 2026-09-20: the 10 division weights sum to exactly 10,000 and reproduce headline CPI to 0.00
# index points from Jan 2024 on, so contributions to YoY are exact from Jan 2025 (the first month
# whose year-ago value sits inside the 2024-base window; earlier history is chain-linked from
# the old 2019 basket and does not satisfy the identity). Refresh CPI_W_SG at the next rebasing.
CPI_W_SG = {
    "1.0": 2042,
    "1.02": 165,
    "1.03": 2938,
    "1.04": 547,
    "1.05": 1008,
    "1.06": 1307,
    "1.07": 381,
    "1.08": 595,
    "1.09": 579,
    "1.10": 438,
    "1.01": 651,
    "1.11.1": 585,
    "1.11.3": 707,
    "1.03.1.2": 2138,
    "1.03.1.1": 294,
    "1.03.1.3": 224,
    "1.03.2": 282,
    "1.04.1": 128,
    "1.04.3": 101,
    "1.04.6.2": 245,
    "1.02.1": 129,
    "1.05.2": 428,
    "1.05.3": 238,
    "1.05.5": 225,
    "1.06.1.1": 459,
    "1.06.1.5": 174,
    "1.06.2.1": 138,
    "1.06.2.2": 108,
    "1.06.3.1": 129,
    "1.07.3": 287,
    "1.08.7.3": 206,
    "1.08.7.1": 103,
    "1.08.4": 144,
    "1.09.1": 391,
    "1.09.2": 184,
    "1.10.1": 211,
    "1.03.2.3": 179,
    "1.01.1.1": 14,
    "1.01.7": 82,
    "1.01.2": 101,
    "1.06.1": 906,
    "1.03.1": 2656,
}
CPI_NAME_SG = {
    "1.0": "Food",
    "1.02": "Clothing & Footwear",
    "1.03": "Housing & Utilities",
    "1.04": "Household Durables & Services",
    "1.05": "Health",
    "1.06": "Transport",
    "1.07": "Info & Communication",
    "1.08": "Recreation, Sport & Culture",
    "1.09": "Education",
    "1.10": "Miscellaneous",
    "1.01": "Food excl. F&B Serving",
    "1.11.1": "Restaurants, Cafes & Pubs",
    "1.11.3": "Hawker Centres & Food Courts",
    "1.03.1.2": "Imputed Rentals",
    "1.03.1.1": "Actual Rentals",
    "1.03.1.3": "Housing Maintenance & Repairs",
    "1.03.2": "Utilities & Other Fuels",
    "1.04.1": "Furniture & Furnishings",
    "1.04.3": "Household Appliances",
    "1.04.6.2": "Domestic & Household Services",
    "1.02.1": "Clothing",
    "1.05.2": "Outpatient Care",
    "1.05.3": "Inpatient Care",
    "1.05.5": "Health Insurance",
    "1.06.1.1": "Motor Cars",
    "1.06.1.5": "Petrol",
    "1.06.2.1": "Bus & Train Fares",
    "1.06.2.2": "Point-to-Point Transport",
    "1.06.3.1": "Airfares",
    "1.07.3": "Info & Comm Services",
    "1.08.7.3": "Package Holidays",
    "1.08.7.1": "Hotels",
    "1.08.4": "Recreational Services",
    "1.09.1": "General, Vocational & Higher Ed.",
    "1.09.2": "Private Tuition & Courses",
    "1.10.1": "Personal Care",
    "1.03.2.3": "Electricity",
    "1.01.1.1": "Rice",
    "1.01.7": "Vegetables",
    "1.01.2": "Meat",
    "1.06.1": "Private Transport",
    "1.03.1": "Accommodation",
}
CPI_MAIN_SG = ["1.0", "1.02", "1.03", "1.04", "1.05", "1.06", "1.07", "1.08", "1.09", "1.10"]
# Direct classes of each division that are >=1% of the basket (not exhaustive per division).
CPI_KIDS_SG = {"1.0": ["1.01", "1.11.1", "1.11.3"], "1.02": ["1.02.1"], "1.03": ["1.03.1.2", "1.03.1.1", "1.03.1.3", "1.03.2"], "1.04": ["1.04.1", "1.04.3", "1.04.6.2"], "1.05": ["1.05.2", "1.05.3", "1.05.5"], "1.06": ["1.06.1.1", "1.06.1.5", "1.06.2.1", "1.06.2.2", "1.06.3.1"], "1.07": ["1.07.3"], "1.08": ["1.08.7.3", "1.08.7.1", "1.08.4"], "1.09": ["1.09.1", "1.09.2"], "1.10": ["1.10.1"]}
# Price-sensitive areas: volatile, policy-driven (COE, utility tariffs) or high-weight items.
# Some overlap each other (Private Transport / Motor Cars / Petrol, Accommodation / Rentals) -
# by design, not meant to be summed.
CPI_SENSITIVE_SG = ["1.01.1.1", "1.01.2", "1.01.7", "1.11.3", "1.11.1", "1.03.1.1", "1.03.1.2", "1.03.2.3", "1.06.1.1", "1.06.1.5", "1.06.3.1", "1.08.7.3", "1.08.7.1", "1.06.1", "1.03.1"]
CPI_GROUP_COLORS_SG = ["#ef5350", "#ff9800", "#ffd54f", "#9ccc65", "#26a69a", "#4fc3f7", "#5c9eff", "#ba68c8", "#f06292", "#8a94a6"]

@st.cache_data(ttl=21600)
def fetch_singstat_cpi_components(n_periods=180) -> pd.DataFrame:
    """Index levels for headline + every series in CPI_W_SG, one API call. Wide frame: date x series_no."""
    codes = ["1"] + list(CPI_W_SG)
    try:
        r = requests.get(f"{SINGSTAT_BASE}/M213751", headers=HEADERS, timeout=60,
                          params={"seriesNoORrowNo": ",".join(codes), "limit": n_periods * len(codes), "sortBy": "key desc"})
        r.raise_for_status()
        cols = {}
        for row in r.json()["Data"]["row"]:
            s = {_parse_singstat_period(c["key"]): _to_float(c["value"]) for c in row["columns"]}
            s = pd.Series(s).sort_index()
            cols[row["seriesNo"]] = s[s.index.notna()]
        df = pd.DataFrame(cols)
        df.index.name = "date"
        return df
    except Exception as e:
        st.warning(f"Could not load CPI components (SingStat M213751): {e}")
        return pd.DataFrame()

# ── MAS Bonds & Bills API (undocumented but public JSON, no key — verified live) ─
# This backs mas.gov.sg's own bonds-and-bills pages (found via their JS bundle),
# not an officially documented endpoint - could change without notice, but it's
# what MAS's own site calls, and it's a clean Solr-style filter/sort/rows API.
MAS_BONDS_BASE = "https://eservices.mas.gov.sg/statistics/api/v1/bondsandbills"

@st.cache_data(ttl=21600)
def fetch_sgs_yield(tenor: str, label: str, rows=1500) -> pd.DataFrame:
    try:
        r = requests.get(f"{MAS_BONDS_BASE}/m/pricesandyields", headers=HEADERS, timeout=20,
                          params={"filters": f"benchmark_tenor:{tenor}", "sort": "end_of_period desc", "rows": rows})
        r.raise_for_status()
        records = r.json()["result"]["records"]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["end_of_period"])
        df = df.set_index("date").sort_index()
        out = pd.DataFrame({label: pd.to_numeric(df["bid_yield"], errors="coerce")}).dropna()
        return out
    except Exception as e:
        st.warning(f"Could not load {label}: {e}")
        return pd.DataFrame()

def _interp_sg_yield_curve(row_vals, tenor_years, target_years):
    """Linearly interpolate a yield-curve snapshot onto arbitrary target maturities. SGS
    actually quotes a real benchmark all the way out to 50Y, so unlike the US dashboard's
    version of this helper, most ladder points here need no interpolation at all - only the
    5-year-spaced points between real benchmarks (e.g. 22.5Y, 35Y) do."""
    pairs = sorted((tenor_years[lbl], row_vals[lbl]) for lbl in row_vals.index
                   if lbl in tenor_years and pd.notna(row_vals[lbl]))
    if not pairs:
        return [None] * len(target_years)
    xs, ys = zip(*pairs)
    return list(np.interp(target_years, xs, ys))

@st.cache_data(ttl=21600)  # auction results only change a few times/week
def load_sgs_auctions() -> pd.DataFrame:
    r = requests.get(f"{MAS_BONDS_BASE}/m/listauctionbondsandbills", headers=HEADERS, timeout=30,
                      params={"sort": "auction_date desc", "rows": 6000})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["result"]["records"])
    for col in ["auction_date", "issue_date", "maturity_date", "ann_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["auction_amt", "total_amt_allot", "bid_to_cover", "cutoff_yield", "avg_yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Remaining-maturity ladder for outstanding SGS/T-Bills - same buckets as the US Treasury
# dashboard's MATURITY_LADDER, extended 5 years apart past 30Y (SGS issues out to 50Y, unlike
# US Treasury which caps at 30Y - no US-style "midpoint bucket" trick is needed there since
# real SGS benchmark tenors already exist at 35Y/40Y/45Y/50Y... in practice only 50Y is an
# actual benchmark, but keeping 5Y spacing out to 50Y matches how the US ladder extends past
# its last real benchmark (30Y) using evenly-spaced buckets).
SG_MATURITY_LADDER = {
    "3M": 0.25, "6M": 0.5, "12M": 1, "2Y": 2, "3Y": 3, "4Y": 4, "5Y": 5, "7Y": 7,
    "10Y": 10, "12Y": 12, "15Y": 15, "20Y": 20, "22.5Y": 22.5, "25Y": 25, "27.5Y": 27.5, "30Y": 30,
    "35Y": 35, "40Y": 40, "45Y": 45, "50Y": 50,
}
_SG_LADDER_ITEMS = list(SG_MATURITY_LADDER.items())
_SG_LADDER_ORDER = {label: i for i, label in enumerate(SG_MATURITY_LADDER)}

def _sg_nearest_maturity_bucket(years_left):
    return min(_SG_LADDER_ITEMS, key=lambda kv: abs(kv[1] - years_left))[0]

def _sg_bond_category(row):
    # bill_bond_ind + product_type distinguishes T-Bills / MAS Bills / Cash Management Bills;
    # for bonds, sgs_type carries MAS's own category - verified live against the real API
    # (listbondsandbills): "SGS (MD)" (Market Development), "SGS (Infra)", "Green SGS (Infra)",
    # plus a handful of older "U" (undefined/pre-classification) records.
    if row["bill_bond_ind"] == "bill":
        pt = row.get("product_type")
        if pt == "M":
            return "MAS Bills"
        if pt == "C":
            return "CMB"
        return "T-Bills"
    sgs_type = row.get("sgs_type")
    if sgs_type == "SGS (MD)":
        return "SGS (Market Dev.)"
    if sgs_type == "SGS (Infra)":
        return "SGS (Infra)"
    if sgs_type == "Green SGS (Infra)":
        return "SGS (Green Infra)"
    return "SGS (Other)"

SG_CATEGORY_COLORS = {
    "T-Bills": "#42a5f5", "MAS Bills": "#26c6da", "CMB": "#8d6e63",
    "SGS (Market Dev.)": "#26a69a", "SGS (Infra)": "#ff9800",
    "SGS (Green Infra)": "#66bb6a", "SGS (Other)": "#9e9e9e",
}

def get_sg_outstanding_by_remaining_maturity(auctions_df):
    today = pd.Timestamp.today().normalize()
    outstanding = auctions_df[(auctions_df["issue_date"] <= today) & (auctions_df["maturity_date"] > today)].copy()
    outstanding["years_to_maturity"] = (outstanding["maturity_date"] - today).dt.days / 365.25
    outstanding["amt_bil"] = outstanding["total_amt_allot"].fillna(outstanding["auction_amt"]) / 1000  # S$M -> S$B
    outstanding["maturity_bucket"] = outstanding["years_to_maturity"].apply(_sg_nearest_maturity_bucket)
    outstanding["category"] = outstanding.apply(_sg_bond_category, axis=1)
    summary = outstanding.groupby(["maturity_bucket", "category"])["amt_bil"].sum().reset_index()
    return outstanding, summary

def _sg_true_original_tenor_bucket(auctions_df):
    """Map each issue_code to its true original-tenor bucket, using the EARLIEST issue_date
    across all of that issue_code's auction events (initial issue + every reopening).
    MAS's own `first_issue_date` field is NOT reliable for this - verified live that it just
    repeats that specific auction event's own issue_date rather than the true original issue
    date. issue_code/ISIN and maturity_date stay constant across reopenings, though - live
    check on NX21100N: first issued 2021-07-01, reopened 2022-03-01 and again 2026-07-01, all
    maturing 2031-07-01 - true original tenor is 10Y, not the ~5Y a naive per-event
    (maturity - that event's issue_date) calc would give for the 2026 reopening. Same class of
    bug as the original US Treasury tenor-bucketing fix."""
    first_issue = auctions_df.groupby("issue_code")["issue_date"].min()
    maturity = auctions_df.groupby("issue_code")["maturity_date"].first()
    tenor_years = (maturity - first_issue).dt.days / 365.25
    return tenor_years.apply(_sg_nearest_maturity_bucket)

def get_sg_net_issuance(auctions_df, days_window):
    """Recently-issued (past days_window, real settled amounts) vs upcoming maturities (next
    days_window, also real/known amounts) by true original-tenor bucket. Not simply a
    forward-looking version of the US 'Issuance vs Maturity' chart - MAS doesn't disclose
    auction size until after the auction closes, so there's no free source for FUTURE issuance
    amounts (same limitation as the Upcoming SGS/T-Bill Issuance table). Using a trailing
    issuance window instead keeps every number on this chart real and settled."""
    today = pd.Timestamp.today().normalize()
    bucket_map = _sg_true_original_tenor_bucket(auctions_df)
    df = auctions_df.copy()
    df["tenor_bucket"] = df["issue_code"].map(bucket_map)
    df["amt_bil"] = df["total_amt_allot"].fillna(df["auction_amt"]) / 1000

    issued = df[(df["issue_date"] >= today - pd.Timedelta(days=days_window)) & (df["issue_date"] <= today)]
    issuance_summary = issued.groupby("tenor_bucket")["amt_bil"].sum()

    maturing = df[(df["maturity_date"] >= today) & (df["maturity_date"] <= today + pd.Timedelta(days=days_window))]
    maturity_summary = maturing.groupby("tenor_bucket")["amt_bil"].sum()

    labels = list(SG_MATURITY_LADDER.keys())
    combined = pd.DataFrame(index=labels)
    combined["Issuance"] = issuance_summary.reindex(labels).fillna(0)
    combined["Maturing"] = maturity_summary.reindex(labels).fillna(0)
    combined["Net"] = combined["Issuance"] - combined["Maturing"]
    combined = combined[(combined["Issuance"] != 0) | (combined["Maturing"] != 0)]
    return combined.reset_index().rename(columns={"index": "tenor_bucket"})

@st.cache_data(ttl=21600)
def load_sgs_issuance_calendar() -> pd.DataFrame:
    """Forward + historical auction/issue calendar. No offering-size field is
    exposed here (unlike US Treasury) - MAS only discloses auction size once the
    auction itself has closed (see load_sgs_auctions), not at announcement."""
    r = requests.get(f"{MAS_BONDS_BASE}/m/issuancecalendar", headers=HEADERS, timeout=30,
                      params={"sort": "auction_date desc", "rows": 800})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["result"]["records"])
    for col in ["ann_date", "auction_date", "issue_date", "maturity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ── S$NEER (MAS's own official weekly index) ────────────────────────────────────
# www.mas.gov.sg's own page (Statistics > Exchange Rates > S$NEER) publishes this
# directly - a real, official "Average for Week Ending" index, Jan 1999 = 100.
# www.mas.gov.sg sits behind bot-detection that blocks plain `requests` calls to
# most of its pages/APIs (returns a static "Maintenance" HTML shell) - the fix,
# found by inspecting the page's own network calls, is adding the
# X-Requested-With: XMLHttpRequest header (marks the request as the page's own
# in-page AJAX call rather than a bare bot request) alongside a matching Referer.
# This same header also unblocks the MPS statement search API - see get_mps_dates.
MAS_WWW_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
}
SNEER_URL = "https://www.mas.gov.sg/api/v1/MAS/chart/rev/sneer"

@st.cache_data(ttl=604800)  # MAS publishes this weekly
def fetch_sneer() -> pd.DataFrame:
    try:
        headers = {**MAS_WWW_HEADERS, "Referer": "https://www.mas.gov.sg/statistics/exchange-rates/sneer"}
        r = requests.get(SNEER_URL, headers=headers, timeout=20, params={"$start_index": 0, "$count": 2000})
        r.raise_for_status()
        els = r.json()["elements"]
        df = pd.DataFrame(els)
        df["date"] = pd.to_datetime(df["date"])
        out = pd.DataFrame({"S$NEER": pd.to_numeric(df["value"], errors="coerce").values}, index=df["date"])
        out.index.name = "date"
        return out.sort_index()
    except Exception as e:
        st.warning(f"Could not load S\\$NEER: {e}")
        return pd.DataFrame()

# ── SORA (MAS Domestic Interest Rates) ──────────────────────────────────────────
# eservices.mas.gov.sg is legacy ASP.NET WebForms (postback, not a REST GET) -
# verified live that a plain requests POST with the right hidden fields + the
# "SORA" checkbox works and returns a real HTML results table, no browser needed.
SORA_URL = "https://eservices.mas.gov.sg/Statistics/dir/DomesticInterestRates.aspx"

@st.cache_data(ttl=21600)
def fetch_sora(years_back=3) -> pd.DataFrame:
    try:
        s = requests.Session()
        r = s.get(SORA_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        def gv(name):
            el = soup.find(attrs={"name": name})
            return el.get("value", "") if el else ""
        today = datetime.date.today()
        data = {
            "__EVENTTARGET": "", "__EVENTARGUMENT": "",
            "__VIEWSTATE": gv("__VIEWSTATE"),
            "__VIEWSTATEGENERATOR": gv("__VIEWSTATEGENERATOR"),
            "__EVENTVALIDATION": gv("__EVENTVALIDATION"),
            "ctl00$ContentPlaceHolder1$StartYearDropDownList": str(today.year - years_back),
            "ctl00$ContentPlaceHolder1$EndYearDropDownList": str(today.year),
            "ctl00$ContentPlaceHolder1$StartMonthDropDownList": "1",
            "ctl00$ContentPlaceHolder1$EndMonthDropDownList": str(today.month),
            "ctl00$ContentPlaceHolder1$ColumnsCheckBoxList$13": "on",  # "SORA" column
            "ctl00$ContentPlaceHolder1$Button1": "Display",
        }
        r2 = s.post(SORA_URL, headers=HEADERS, data=data, timeout=25)
        soup2 = BeautifulSoup(r2.text, "html.parser")
        table = soup2.find("table")
        raw = pd.read_html(io.StringIO(str(table)))[0]
        pub_date, rate = raw.iloc[:, -2], raw.iloc[:, -1]
        out = pd.DataFrame({
            "date": pd.to_datetime(pub_date, format="%d %b %Y", errors="coerce"),
            "SORA": pd.to_numeric(rate, errors="coerce"),
        }).dropna().set_index("date").sort_index()
        return out
    except Exception as e:
        st.warning(f"Could not load SORA: {e}")
        return pd.DataFrame()

# ── MAS Monetary Policy Statement dates ─────────────────────────────────────────
# Uses the same www.mas.gov.sg search API the site's own News page calls, with
# the MAS_WWW_HEADERS fix above (X-Requested-With + matching Referer) - verified
# live, returns all 64 historical MPS releases with clean ISO dates.
# MPS_FALLBACK is a real, browser-confirmed publication-date history, kept only
# as a safety net if the live call ever breaks - MAS meets quarterly
# (Jan/Apr/Jul/Oct) but doesn't publish the exact day more than ~2 weeks ahead,
# so there's no free way to know the *next* exact date until MAS announces it.
MPS_FALLBACK = [
    "2024-04-12", "2024-07-26", "2024-10-14",
    "2025-01-24", "2025-04-14", "2025-07-30", "2025-10-14",
    "2026-01-29", "2026-04-14", "2026-07-27",
]

@st.cache_data(ttl=604800)
def get_mps_dates():
    try:
        headers = {**MAS_WWW_HEADERS, "Referer": "https://www.mas.gov.sg/news?content_type=Monetary%20Policy%20Statements"}
        r = requests.get("https://www.mas.gov.sg/api/v1/search", headers=headers, timeout=15,
                          params={"fq": '{!tag=mas_contenttype_s}mas_contenttype_s:("Monetary Policy Statements")',
                                  "q": "*:*", "sort": "mas_date_tdt desc", "rows": 100, "wt": "json"})
        r.raise_for_status()
        j = r.json()
        dates = sorted({d["mas_date_tdt"][:10] for d in j["response"]["docs"]})
        if len(dates) < 8:
            raise ValueError("live MPS fetch returned too few results")
        return dates
    except Exception:
        return MPS_FALLBACK

@st.cache_data(ttl=21600)
def fetch_singstat_multi(resource_id: str, series_nos: tuple, n_periods=200) -> pd.DataFrame:
    """Several rows of one SingStat table in a single call (same trick as the CPI components
    fetch). Note `limit` counts cells across all rows, not periods per row."""
    try:
        r = requests.get(f"{SINGSTAT_BASE}/{resource_id}", headers=HEADERS, timeout=60,
                         params={"seriesNoORrowNo": ",".join(series_nos), "limit": n_periods * len(series_nos),
                                 "sortBy": "key desc"})
        r.raise_for_status()
        cols = {}
        for row in r.json()["Data"]["row"]:
            s = pd.Series({_parse_singstat_period(c["key"]): _to_float(c["value"]) for c in row["columns"]}).sort_index()
            cols[row["seriesNo"]] = s[s.index.notna()]
        return pd.DataFrame(cols)
    except Exception:
        return pd.DataFrame()

# ── Long-history SGS benchmark yields (MAS legacy WebForms page) ─────────────────
# The bondsandbills JSON API above only keeps ~2 years per benchmark tenor (verified live
# 2026-09-30: 502 records per tenor, starting 2024-10-01) - too short for 1Y z-scores on
# spreads, 5Y percentiles or proper rolldown history. The legacy "SGS Prices and Yields -
# Benchmark Issues" page has daily history back to the 1990s via the same GET-then-POST
# WebForms pattern as SORA. It returns one table per calendar year and silently caps a
# single request at ~8 years (a 2016-2026 request came back starting 2019), so it's pulled
# in 6-year chunks concurrently.
MAS_BENCH_URL = "https://eservices.mas.gov.sg/statistics/fdanet/BenchmarkPricesAndYields.aspx"
SGS_BENCH_FIELDS = {
    "3M": "ThreeMonthTreasuryBillYield", "6M": "SixMonthTreasuryBillYield", "1Y": "OneYearTreasuryBillYield",
    "2Y": "TwoYearBondYield", "5Y": "FiveYearBondYield", "7Y": "SevenYearBondYield",
    "10Y": "TenYearBondYield", "15Y": "FifteenYearBondYield", "20Y": "TwentyYearBondYield",
    "30Y": "ThirtyYearBondYield", "50Y": "FiftyYearBondYield",
}
SGS_TENOR_YEARS = {"3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "5Y": 5, "7Y": 7, "10Y": 10,
                   "15Y": 15, "20Y": 20, "30Y": 30, "50Y": 50}

def _webforms_session(url):
    s = requests.Session()
    soup = BeautifulSoup(s.get(url, headers=HEADERS, timeout=20).text, "html.parser")
    gv = lambda n: (soup.find(attrs={"name": n}) or {}).get("value", "")
    base = {"__EVENTTARGET": "", "__EVENTARGUMENT": "", "__LASTFOCUS": "",
            "__VIEWSTATE": gv("__VIEWSTATE"), "__VIEWSTATEGENERATOR": gv("__VIEWSTATEGENERATOR"),
            "__EVENTVALIDATION": gv("__EVENTVALIDATION")}
    return s, base

def _parse_mas_year_tables(html):
    """MAS daily tables: first three columns are year / month / day with year+month only on
    the first row of each block. Returns a date-indexed frame of the remaining columns
    (publication-date columns dropped), with '-' / '--' coerced to NaN."""
    frames = []
    for tb in BeautifulSoup(html, "html.parser").find_all("table"):
        try:
            raw = pd.read_html(io.StringIO(str(tb)))[0]
        except ValueError:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [" ".join(dict.fromkeys(str(p) for p in c)) for c in raw.columns]
        if raw.shape[1] < 4:
            continue
        yr = pd.to_numeric(raw.iloc[:, 0], errors="coerce").ffill()
        mo = raw.iloc[:, 1].ffill().astype(str)
        dy = pd.to_numeric(raw.iloc[:, 2], errors="coerce")
        ok = yr.notna() & dy.notna()
        dates = pd.to_datetime(yr[ok].astype(int).astype(str) + " " + mo[ok] + " " + dy[ok].astype(int).astype(str),
                               format="%Y %b %d", errors="coerce")
        vals = raw.loc[ok, [c for c in raw.columns[3:] if "PUBLICATION" not in str(c).upper()]]
        vals = vals.apply(pd.to_numeric, errors="coerce")
        vals.index = dates
        frames.append(vals[vals.index.notna()])
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()

def _sgs_bench_chunk(y0, y1, m1):
    s, data = _webforms_session(MAS_BENCH_URL)
    P = "ctl00$ContentPlaceHolder1$"
    data.update({P + "StartYearDropDownList": str(y0), P + "StartMonthDropDownList": "1",
                 P + "EndYearDropDownList": str(y1), P + "EndMonthDropDownList": str(m1),
                 P + "FrequencyDropDownList": "D", P + "DisplayButton": "Display"})
    for f in SGS_BENCH_FIELDS.values():
        data[P + f + "CheckBox"] = "on"
    df = _parse_mas_year_tables(s.post(MAS_BENCH_URL, headers=HEADERS, data=data, timeout=90).text)
    # map columns by their "N-Month"/"N-Year" label rather than position
    rename = {}
    for c in df.columns:
        m = re.search(r"(\d+)-(Month|Year)", str(c))
        if m:
            rename[c] = f"{m.group(1)}{'M' if m.group(2) == 'Month' else 'Y'}"
    return df.rename(columns=rename)

@st.cache_data(ttl=21600)
def fetch_sgs_curve(start_year=2008) -> pd.DataFrame:
    """Daily SGS benchmark curve, 3M..50Y, columns ordered by tenor. Empty frame on failure."""
    today = datetime.date.today()
    chunks = []
    y = start_year
    while y <= today.year:
        y1 = min(y + 5, today.year)
        chunks.append((y, y1, today.month if y1 == today.year else 12))
        y = y1 + 1
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(lambda a: _sgs_bench_chunk(*a), chunks))
        df = pd.concat([p for p in parts if not p.empty]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
    except Exception:
        return pd.DataFrame()
    # Note: MAS stopped quoting 3M and 7Y benchmarks (blank on this page AND null in the
    # bondsandbills JSON API, checked 2026-09-30), so the live front end is 6M/1Y bills.
    return df.sort_index()[[t for t in SGS_TENOR_YEARS if t in df.columns]].dropna(how="all")

# ── SORA suite (same DomesticInterestRates.aspx page as fetch_sora) ──────────────
# Checkbox indices verified live 2026-09-30. Compounded SORA comes back in a different
# table layout when mixed with other columns, so it's requested on its own.
SORA_PANEL_COLS = {13: "SORA", 18: "SORA Volume", 19: "SORA High", 20: "SORA Low"}
SORA_COMP_COLS = {15: "1M Comp. SORA", 16: "3M Comp. SORA", 17: "6M Comp. SORA"}

def _sora_chunk(cols, y0, y1, m1):
    s, data = _webforms_session(SORA_URL)
    P = "ctl00$ContentPlaceHolder1$"
    data.pop("__LASTFOCUS")
    data.update({P + "StartYearDropDownList": str(y0), P + "EndYearDropDownList": str(y1),
                 P + "StartMonthDropDownList": "1", P + "EndMonthDropDownList": str(m1),
                 P + "Button1": "Display"})
    for i in cols:
        data[f"{P}ColumnsCheckBoxList${i}"] = "on"
    df = _parse_mas_year_tables(s.post(SORA_URL, headers=HEADERS, data=data, timeout=60).text)
    if df.shape[1] != len(cols):
        raise ValueError(f"unexpected SORA table layout: {list(df.columns)}")
    df.columns = [cols[i] for i in sorted(cols)]
    return df

@st.cache_data(ttl=21600)
def fetch_sora_suite(start_year=2016) -> pd.DataFrame:
    today = datetime.date.today()
    jobs = []
    for cols in (SORA_PANEL_COLS, SORA_COMP_COLS):
        y = start_year
        while y <= today.year:
            y1 = min(y + 5, today.year)
            jobs.append((cols, y, y1, today.month if y1 == today.year else 12))
            y = y1 + 1
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(lambda a: _sora_chunk(*a), jobs))
        panel = pd.concat([p for p, j in zip(parts, jobs) if j[0] is SORA_PANEL_COLS]).sort_index()
        comp = pd.concat([p for p, j in zip(parts, jobs) if j[0] is SORA_COMP_COLS]).sort_index()
        out = pd.concat([panel[~panel.index.duplicated()], comp[~comp.index.duplicated()]], axis=1)
        return out.dropna(how="all")
    except Exception:
        return pd.DataFrame()

# ── US comparables (keyless, so this app still deploys without a FRED key) ───────
UST_COLS = {"1 Mo": "1M", "3 Mo": "3M", "6 Mo": "6M", "1 Yr": "1Y", "2 Yr": "2Y", "5 Yr": "5Y",
            "7 Yr": "7Y", "10 Yr": "10Y", "20 Yr": "20Y", "30 Yr": "30Y"}

def _ust_year(y):
    u = (f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
         f"{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
    df = pd.read_csv(io.StringIO(requests.get(u, headers=HEADERS, timeout=30).text))
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df.set_index("Date")[[c for c in UST_COLS if c in df.columns]].rename(columns=UST_COLS)

@st.cache_data(ttl=21600)
def fetch_ust_curve(start_year=2008) -> pd.DataFrame:
    """US Treasury par curve from treasury.gov's per-year CSV (the all-years CSV is 403'd)."""
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            parts = list(pool.map(_ust_year, range(start_year, datetime.date.today().year + 1)))
        return pd.concat(parts).sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=21600)
def fetch_sofr() -> pd.Series:
    try:
        j = requests.get("https://markets.newyorkfed.org/api/rates/secured/sofr/search.json",
                         params={"startDate": "2018-04-02", "endDate": datetime.date.today().isoformat()},
                         timeout=40).json()
        return pd.Series({pd.Timestamp(r["effectiveDate"]): float(r["percentRate"]) for r in j["refRates"]},
                         name="SOFR").sort_index()
    except Exception:
        return pd.Series(dtype=float, name="SOFR")

# ── FX panel (yfinance) ─────────────────────────────────────────────────────────
# (ticker, True if quoted USD-per-FCY like EURUSD, False if FCY-per-USD like USDJPY)
# CNY (onshore) rather than CNH: Yahoo's CNH=X only returns a handful of recent rows.
FX_TICKERS = {
    "USD": ("SGD=X", None), "CNY": ("CNY=X", False), "MYR": ("MYR=X", False), "EUR": ("EURUSD=X", True),
    "JPY": ("JPY=X", False), "TWD": ("TWD=X", False), "KRW": ("KRW=X", False), "IDR": ("IDR=X", False),
    "HKD": ("HKD=X", False), "THB": ("THB=X", False), "INR": ("INR=X", False), "AUD": ("AUDUSD=X", True),
    "GBP": ("GBPUSD=X", True), "PHP": ("PHP=X", False), "VND": ("VND=X", False),
}
# Conventional market quote for each cross vs SGD (display only - analytics use FCY per SGD)
FX_DISPLAY = {"USD": "USD/SGD", "EUR": "EUR/SGD", "GBP": "GBP/SGD", "AUD": "AUD/SGD", "CNY": "SGD/CNY",
              "MYR": "SGD/MYR", "JPY": "SGD/JPY", "KRW": "SGD/KRW", "TWD": "SGD/TWD", "IDR": "SGD/IDR",
              "THB": "SGD/THB", "INR": "SGD/INR", "HKD": "SGD/HKD", "PHP": "SGD/PHP", "VND": "SGD/VND"}

@st.cache_data(ttl=3600)
def fetch_fx_panel(start="2008-01-01") -> pd.DataFrame:
    tickers = [t for t, _ in FX_TICKERS.values()] + ["DX-Y.NYB"]
    try:
        px = _yf_retry(lambda: yf.download(tickers, start=start, progress=False, auto_adjust=True)["Close"])
    except Exception:
        px = pd.DataFrame(columns=tickers)
    # yf.download doesn't raise when only SOME tickers fail (it just leaves NaN columns) - retry
    # those individually so one Yahoo hiccup doesn't silently drop a currency from the basket.
    for t in tickers:
        if t not in px.columns or px[t].dropna().empty:
            try:
                one = _yf_retry(lambda: yf.download(t, start=start, progress=False, auto_adjust=True)["Close"])
                one = one.iloc[:, 0] if isinstance(one, pd.DataFrame) else one
                px = px.reindex(px.index.union(one.index))
                px[t] = one
            except Exception:
                pass
    if px.empty:
        return px
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px.sort_index()

def fcy_per_sgd(px: pd.DataFrame) -> pd.DataFrame:
    """Units of each foreign currency per 1 SGD (up = SGD stronger), from USD crosses."""
    usdsgd = px["SGD=X"]
    out = {}
    for ccy, (tkr, usd_quote) in FX_TICKERS.items():
        if tkr not in px.columns:
            continue
        if ccy == "USD":
            out[ccy] = 1 / usdsgd
        elif usd_quote:
            out[ccy] = 1 / (px[tkr] * usdsgd)
        else:
            out[ccy] = px[tkr] / usdsgd
    return pd.DataFrame(out).ffill(limit=3)

def display_quote(fcy: pd.DataFrame, ccy: str) -> pd.Series:
    return 1 / fcy[ccy] if FX_DISPLAY[ccy].endswith("/SGD") else fcy[ccy]

# ── Daily S$NEER model ──────────────────────────────────────────────────────────
# MAS only publishes S$NEER weekly and with a lag (latest print was ~1 month old when this was
# built), and doesn't publish its trade weights. Standard sell-side workaround: recover the
# basket by regressing weekly log-changes of the official index on weekly log-changes of SGD
# crosses (non-negative least squares, weights normalised to sum to 1), then chain daily
# crosses with those weights and anchor the level to MAS's own latest print. Verified
# 2026-09-30 on a 3Y window: R-squared ~0.82 on weekly changes - good enough to fill the gap
# since MAS's last print, not a substitute for it.
@st.cache_data(ttl=3600)
def build_neer_model(fcy: pd.DataFrame, neer: pd.Series, calib_weeks=156):
    recent = fcy[fcy.index >= fcy.index[-1] - pd.DateOffset(weeks=calib_weeks + 8)]
    fcy = fcy[[c for c in fcy.columns if recent[c].notna().mean() > 0.9]]   # drop patchy tickers
    logd = np.log(fcy.ffill().dropna(how="any"))
    wk = logd.resample("W-FRI").mean().diff().dropna()      # MAS = "average for week ending" Friday
    y = np.log(neer).diff().dropna()
    idx = wk.index.intersection(y.index)[-calib_weeks:]
    X, Y = wk.loc[idx].values, y.loc[idx].values
    # HKD is pegged to USD, so those two columns are near-collinear - give NNLS room to converge
    w, _ = nnls(X, Y, maxiter=50 * X.shape[1])
    r2 = 1 - ((Y - X @ w) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    w = w / w.sum()
    model = np.exp(logd.values @ w)
    model = pd.Series(model, index=logd.index, name="Model S$NEER")
    last_wk = neer.index[-1]
    k = neer.iloc[-1] / model[(model.index > last_wk - pd.Timedelta(days=7)) & (model.index <= last_wk)].mean()
    model = model * k
    weights = pd.Series(w, index=fcy.columns).sort_values(ascending=False)
    return model, weights, float(r2), idx[0], idx[-1]

def estimate_band(level: pd.Series, anchor, half_width, slope_pa=None):
    """Estimated policy band: log-linear trend through the index since `anchor` (or a fixed
    user-supplied slope, intercept fitted), +/- half_width %. MAS doesn't publish the band's
    level, slope or width - this is the usual street-style estimate, not an official number."""
    s = level[level.index >= pd.Timestamp(anchor)].dropna()
    if len(s) < 10:
        return pd.DataFrame(), np.nan
    t = (s.index - s.index[0]).days.values / 365.25
    if slope_pa is None:
        b, a = np.polyfit(t, np.log(s.values), 1)
    else:
        b = np.log(1 + slope_pa / 100)
        a = float(np.mean(np.log(s.values) - b * t))
    mid = np.exp(a + b * t)
    band = pd.DataFrame({"Level": s.values, "Mid": mid, "Upper": mid * (1 + half_width / 100),
                         "Lower": mid * (1 - half_width / 100)}, index=s.index)
    band["Dev from Mid %"] = (band["Level"] / band["Mid"] - 1) * 100
    return band, (np.exp(b) - 1) * 100

# ── Curve analytics ─────────────────────────────────────────────────────────────
def curve_interp(row: pd.Series, targets):
    pairs = sorted((SGS_TENOR_YEARS[k], v) for k, v in row.items() if k in SGS_TENOR_YEARS and pd.notna(v))
    if len(pairs) < 2:
        return np.full(len(targets), np.nan)
    xs, ys = zip(*pairs)
    return np.interp(targets, xs, ys)

def rolldown_series(yc: pd.DataFrame, tenors, horizon=1.0) -> pd.DataFrame:
    """Rolldown(T) = y(T) - y(T - horizon) on each date's own curve, in bps (positive = the bond
    gains yield-pickup as it rolls down a positively sloped curve). Same idea as usa_macro's
    compute_rolldown_series."""
    yrs = [SGS_TENOR_YEARS[t] for t in tenors]
    vals = [curve_interp(row, yrs) - curve_interp(row, [y - horizon for y in yrs]) for _, row in yc.iterrows()]
    return pd.DataFrame(np.array(vals) * 100, index=yc.index, columns=tenors)

def auction_concessions(auctions: pd.DataFrame, yc: pd.DataFrame) -> pd.DataFrame:
    """For each SGS bond / T-bill auction: tail = cutoff - median yield, and concession = cutoff
    yield vs the previous business day's benchmark curve interpolated at the issue's remaining
    maturity (positive = auction cleared cheap to the secondary curve)."""
    df = auctions[auctions["product_type"].isin(["N", "B"]) & auctions["cutoff_yield"].notna()].copy()
    df = df[df["auction_date"] >= yc.index[0] + pd.Timedelta(days=5)]
    df["years"] = (df["maturity_date"] - df["issue_date"]).dt.days / 365.25
    df["Tail (bps)"] = (df["cutoff_yield"] - pd.to_numeric(df["median_yield"], errors="coerce")) * 100
    conc = []
    for _, r in df.iterrows():
        prev = yc[yc.index < r["auction_date"]]
        conc.append((r["cutoff_yield"] - curve_interp(prev.iloc[-1], [r["years"]])[0]) * 100 if len(prev) else np.nan)
    df["Concession (bps)"] = conc
    return df

# ── Dividend seasonality (SGX blue chips + major S-REITs, via yfinance) ─────────
# yfinance covers SGX tickers directly (".SI" suffix) - confirmed live for this basket.
# Not from SingStat/MAS; this is equities/corporate-actions data, a different domain from
# the rest of this dashboard.
SG_DIV_BASKET = {
    "D05.SI": "DBS Group", "O39.SI": "OCBC Bank", "U11.SI": "UOB", "Z74.SI": "Singtel",
    "C6L.SI": "Singapore Airlines", "C38U.SI": "CapitaLand Integrated Comm. Trust",
    "A17U.SI": "Ascendas REIT", "BUOU.SI": "Frasers Logistics & Comm. Trust",
    "ME8U.SI": "Mapletree Industrial Trust", "N2IU.SI": "Mapletree Pan Asia Comm. Trust",
    "M44U.SI": "Mapletree Logistics Trust", "C09.SI": "City Developments",
    "F34.SI": "Wilmar International", "G13.SI": "Genting Singapore", "BN4.SI": "Keppel Ltd",
    "S68.SI": "SGX", "Y92.SI": "Thai Beverage", "U96.SI": "Sembcorp Industries",
    "C52.SI": "ComfortDelGro", "V03.SI": "Venture Corp",
}

def _yf_retry(fn, retries=3, backoff=1.5):
    """yfinance/Yahoo Finance is well known to rate-limit or transiently block requests from
    cloud-hosted IPs (Streamlit Community Cloud, AWS, GCP etc.) far more readily than
    residential/dev IPs - a single 403 on a bad attempt is normal, not a sign the ticker is
    actually unavailable. Retry a few times with backoff before giving up."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("yfinance call failed")

# Short TTL is deliberate: a transient rate-limit/block (see _yf_retry) would otherwise get
# cached as "this ticker has no data" for the full TTL, silently dropping it from every chart
# until the cache expires - a 1hr window means a bad run self-heals within the hour instead of
# a full day.
@st.cache_data(ttl=3600)
def _fetch_dividends(ticker: str) -> pd.Series:
    try:
        div = _yf_retry(lambda: yf.Ticker(ticker).dividends)
        if div.empty:
            return pd.Series(dtype=float)
        div.index = pd.to_datetime(div.index).tz_localize(None)
        return div
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def _fetch_shares_history(ticker: str) -> pd.Series:
    """Historical shares-outstanding checkpoints, where yfinance has them. Coverage is very
    uneven - REITs (which issue new units often) tend to have dense multi-year histories;
    banks/industrials often only have a couple of recent points. Only trusted (see
    get_sg_dividend_payouts) when there are enough points to mean something."""
    try:
        shares = _yf_retry(lambda: yf.Ticker(ticker).get_shares_full(start="2010-01-01"))
        if shares is None or len(shares) == 0:
            return pd.Series(dtype=float)
        shares.index = pd.to_datetime(shares.index).tz_localize(None)
        return shares[~shares.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def _fetch_shares_current(ticker: str):
    try:
        return _yf_retry(lambda: yf.Ticker(ticker).info.get("sharesOutstanding"))
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_sg_dividend_payouts(years_back=11) -> tuple[pd.DataFrame, list[str]]:
    """Nominal S$ paid out per ex-dividend event = dividend/share x shares outstanding at the
    time. Uses the nearest known historical share-count checkpoint where enough of them exist
    (>=20 points - otherwise treat as too sparse to trust for point-in-time lookups) and falls
    back to today's share count otherwise. This is a real approximation, not a reconciliation
    against company filings - most accurate for recent years and for the REITs with dense
    share-count history, least accurate for older dividends from tickers whose share count has
    since moved a lot (buybacks or unit issuance).

    Returns (payouts_df, missing_tickers) - missing_tickers lists any basket member that came
    back with no usable data this run (e.g. DBS Group vanishing from a "top payer" ranking
    after a transient fetch failure), so the UI can surface it instead of silently omitting it."""
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years_back)
    rows = []
    missing = []
    for ticker, name in SG_DIV_BASKET.items():
        div = _fetch_dividends(ticker)
        if div.empty:
            missing.append(name)
            continue
        cur_shares = _fetch_shares_current(ticker)
        hist = _fetch_shares_history(ticker)
        use_hist = len(hist) >= 20
        got_any = False
        for date, amt in div.items():
            if date < cutoff:
                continue
            if use_hist:
                prior = hist[hist.index <= date]
                shares = prior.iloc[-1] if len(prior) > 0 else cur_shares
            else:
                shares = cur_shares
            if shares is None:
                continue
            got_any = True
            rows.append({"ticker": ticker, "name": name, "date": date, "amount": float(amt),
                         "shares": shares, "payout_sgd_m": float(amt) * shares / 1e6})
        if not got_any:
            missing.append(name)
    if not rows:
        return pd.DataFrame(columns=["ticker", "name", "date", "amount", "shares", "payout_sgd_m", "month", "year"]), missing
    df = pd.DataFrame(rows)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df, missing

# ── Date range ────────────────────────────────────────────────────────────────
st.title("🇸🇬 SG Macro Dashboard")
st.caption("Data: SingStat · MAS (S\\$NEER, SORA suite, SGS benchmarks, bonds & bills) · US Treasury · NY Fed · "
           "yfinance. Dotted verticals on rates/FX charts mark MAS Monetary Policy Statements (no NBER-style "
           "recession series exists for Singapore, so there's no recession shading).")

col_d1, col_d2 = st.columns([3, 1])
with col_d1:
    date_range = st.slider(
        "Date Range", min_value=datetime.date(1990, 1, 1),
        max_value=datetime.date.today(),
        value=(datetime.date.today().replace(year=datetime.date.today().year - 5), datetime.date.today()),
        format="YYYY-MM-DD"
    )
START = pd.Timestamp(date_range[0])
END   = pd.Timestamp(date_range[1])

def clip(df):
    return df[(df.index >= START) & (df.index <= END)] if not df.empty else df

def qlabels(index):
    """SingStat quarterly series are indexed at each quarter's START date (pandas Period
    convention - e.g. Q3 2023 = Jul-Sep is dated 2023-07-01). Plotly's default date-formatted
    x-axis/hover then shows that as "Jul 2023", which reads as "the Q2 result released around
    July" rather than "the Jul-Sep quarter" - a real misread the raw date alone invites. This
    gives hover text an unambiguous "20XX QN" label instead."""
    return [f"{ts.year} Q{(ts.month - 1) // 3 + 1}" for ts in index]

# ── Market data shared by the summary bar and the SGD / Rates / Funding tabs ─────
with st.spinner("Loading SGS curve, SORA, UST, FX & S\\$NEER…"):
    sgs = fetch_sgs_curve()
    sora_all = fetch_sora_suite()
    ust = fetch_ust_curve()
    sofr = fetch_sofr()
    fx_px = fetch_fx_panel()
    neer = fetch_sneer()
    mps_dates = [pd.Timestamp(d) for d in get_mps_dates()]

EMPTY = pd.Series(dtype=float)
def _col(df, c):
    return df[c].dropna() if (not df.empty and c in df.columns) else EMPTY

for _name, _df in [("SGS benchmark yields (MAS)", sgs), ("SORA / compounded SORA (MAS)", sora_all),
                   ("US Treasury curve (treasury.gov)", ust), ("FX (yfinance)", fx_px)]:
    if _df.empty:
        st.warning(f"Could not load {_name} - dependent charts and cards will show N/A until the next refresh.")

fcy = fcy_per_sgd(fx_px) if (not fx_px.empty and "SGD=X" in fx_px.columns) else pd.DataFrame()
usdsgd = _col(fx_px, "SGD=X")
neer_w = _col(neer, "S$NEER")
neer_model, neer_weights, neer_r2, neer_cal0, neer_cal1 = None, None, np.nan, None, None
if not fcy.empty and len(neer_w) > 200:
    try:
        neer_model, neer_weights, neer_r2, neer_cal0, neer_cal1 = build_neer_model(fcy, neer_w)
    except Exception as e:
        st.warning(f"S\\$NEER daily model could not be fitted: {e}")
neer_daily = neer_model if neer_model is not None else neer_w

# Policy-band assumptions live in session_state so the summary bar (rendered first) and the
# widgets in the SGD tab (rendered later) always agree - a widget change just reruns the script.
st.session_state.setdefault("band_anchor", (pd.Timestamp.today() - pd.DateOffset(years=1)).date())
st.session_state.setdefault("band_hw", 2.0)
st.session_state.setdefault("band_slope_mode", "Fit from data")
st.session_state.setdefault("band_slope", 1.0)
band_df, band_slope = estimate_band(
    neer_daily, st.session_state.band_anchor, st.session_state.band_hw,
    None if st.session_state.band_slope_mode == "Fit from data" else st.session_state.band_slope)

y2, y5, y10, y20, y30, y50 = (_col(sgs, t) for t in ("2Y", "5Y", "10Y", "20Y", "30Y", "50Y"))
bill6, bill12 = _col(sgs, "6M"), _col(sgs, "1Y")
sora = _col(sora_all, "SORA")
csora1, csora3, csora6 = (_col(sora_all, f"{m}M Comp. SORA") for m in (1, 3, 6))

def _spread(a, b, mult=100):
    return ((a - b) * mult).dropna()

SPREADS = {
    "2s5s": _spread(y5, y2), "2s10s": _spread(y10, y2), "5s30s": _spread(y30, y5),
    "10s30s": _spread(y30, y10), "30s50s": _spread(y50, y30),
    "2s5s10s Fly": ((2 * y5 - y2 - y10) * 100).dropna(),
}
sgs_ust_2 = _spread(y2, _col(ust, "2Y"))
sgs_ust_10 = _spread(y10, _col(ust, "10Y"))
sora_sofr = _spread(sora, sofr)

# ── Summary bar ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Latest Readings & Z-Scores</div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_macro_series():
    out = {}
    cpi_s = fetch_singstat("M213751", "CPI")
    core_s = fetch_singstat("M213891", "Core")
    nodx_s = fetch_singstat("M451301", "NODX")
    rs_s = fetch_singstat("M602122", "Retail")
    out["CPI YoY"] = (cpi_s["CPI"].pct_change(12) * 100) if not cpi_s.empty else EMPTY
    out["Core Infl. YoY"] = (core_s["Core"].pct_change(12) * 100) if not core_s.empty else EMPTY
    out["NODX YoY"] = (nodx_s["NODX"].pct_change(12) * 100) if not nodx_s.empty else EMPTY
    out["Retail Sales YoY"] = (rs_s["Retail"].pct_change(12) * 100) if not rs_s.empty else EMPTY
    # series_no="2" = real GDP (chained 2015 $) - series "1" is nominal and reads ~2x too high
    g = fetch_singstat("M015631", "GDP YoY", series_no="2")
    out["GDP YoY"] = g["GDP YoY"] if not g.empty else EMPTY
    u_s = fetch_singstat("M182342", "Unemployment")
    out["Unemp Rate"] = u_s["Unemployment"] if not u_s.empty else EMPTY
    return out

with st.spinner("Loading summary metrics…"):
    macro = get_macro_series()

pct2 = lambda v: f"{v:.2f}%"
bps0 = lambda v: f"{v:+.0f}bps"
cards = [
    build_card("USD/SGD", usdsgd, lambda v: f"{v:.4f}", "pct", "D"),
    build_card("S$NEER (daily model)", neer_daily, lambda v: f"{v:.2f}", "pct", "D"),
    build_card("S$NEER vs Est. Mid", band_df["Dev from Mid %"] if not band_df.empty else EMPTY,
               lambda v: f"{v:+.2f}%", "pp", "D"),
    build_card("SORA", sora, pct2, "bps", "D"),
    build_card("3M Comp. SORA", csora3, pct2, "bps", "D"),
    build_card("SORA − SOFR", sora_sofr, bps0, "raw_bps", "D"),
    build_card("2Y SGS", y2, pct2, "bps", "D"),
    build_card("10Y SGS", y10, pct2, "bps", "D"),
    build_card("2s10s", SPREADS["2s10s"], bps0, "raw_bps", "D"),
    build_card("5s30s", SPREADS["5s30s"], bps0, "raw_bps", "D"),
    build_card("2s5s10s Fly", SPREADS["2s5s10s Fly"], bps0, "raw_bps", "D"),
    build_card("10Y SGS − UST", sgs_ust_10, bps0, "raw_bps", "D"),
    build_card("CPI YoY", macro["CPI YoY"], pct2, "pp", "M"),
    build_card("Core Infl. YoY", macro["Core Infl. YoY"], pct2, "pp", "M"),
    build_card("GDP YoY", macro["GDP YoY"], pct2, "pp", "Q"),
    build_card("Unemp Rate", macro["Unemp Rate"], pct2, "pp", "Q"),
    build_card("NODX YoY", macro["NODX YoY"], pct2, "pp", "M"),
    build_card("Retail Sales YoY", macro["Retail Sales YoY"], pct2, "pp", "M"),
]
render_cards(cards)
st.caption("Z-scores: daily series use 1M/3M/1Y trailing windows, monthly 3M/12M/36M, quarterly 4Q/8Q/20Q "
           "(same convention as the US dashboard). \"S\\$NEER vs Est. Mid\" depends on the band assumptions set "
           "in the SGD & MAS Policy tab.")
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "SGD & MAS Policy",
    "SGS & Rates",
    "Funding & Liquidity",
    "Supply & Auctions",
    "Prices",
    "Growth & Labour",
    "Trade & Activity",
    "Dividends",
    "Economic Calendar",
])

def add_mps_lines(fig):
    for d in mps_dates:
        if START <= d <= END:
            fig.add_vline(x=d, line_dash="dot", line_color="rgba(255,255,255,0.18)", line_width=1)

def realized_vol(level: pd.Series, window: int) -> pd.Series:
    return np.log(level.dropna()).diff().rolling(window).std() * np.sqrt(252) * 100

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — SGD & MAS Policy
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("SGD & MAS Policy")
    st.caption("MAS targets the S\\$NEER inside an undisclosed band (slope, width and centre are never published), "
               "reviewed at each MPS - so the band below is a trend-fit estimate you control, not an official figure. "
               "MAS publishes S\\$NEER weekly with a lag; the daily line is a model (see weights at the bottom).")

    with st.expander("Policy band assumptions", expanded=False):
        b1, b2, b3, b4 = st.columns(4)
        b1.date_input("Trend-fit anchor (e.g. last re-centring / slope change)", key="band_anchor")
        b2.number_input("Band half-width (%)", min_value=0.25, max_value=5.0, step=0.25, key="band_hw")
        b3.radio("Slope", ["Fit from data", "Manual"], key="band_slope_mode", horizontal=True)
        b4.number_input("Manual slope (% p.a.)", min_value=-3.0, max_value=5.0, step=0.25, key="band_slope",
                        disabled=st.session_state.band_slope_mode != "Manual")

    hw = st.session_state.band_hw
    if not band_df.empty:
        dev = float(band_df["Dev from Mid %"].iloc[-1])
        pos = max(-1.0, min(1.0, dev / hw))
        g1, g2 = st.columns([2, 3])
        with g1:
            st.markdown(f"""
            <div class="band-wrap">
              <div class="metric-label">Position in estimated band ({band_df.index[-1]:%d %b %Y})</div>
              <div class="metric-value {'positive' if dev > 0 else 'negative'}">{dev:+.2f}% vs mid
                &nbsp;·&nbsp; {pos * 100:+.0f}% of half-width</div>
              <div class="band-track"><div class="band-mid"></div>
                <div class="band-dot" style="left:{50 + 50 * pos:.1f}%"></div></div>
              <div class="band-scale"><span>Weak edge −{hw:.2f}%</span><span>Mid</span><span>Strong edge +{hw:.2f}%</span></div>
            </div>""", unsafe_allow_html=True)
        with g2:
            k = st.columns(4)
            k[0].metric("Est. slope", f"{band_slope:+.2f}% p.a.")
            k[1].metric("Model S$NEER", f"{band_df['Level'].iloc[-1]:.2f}")
            k[2].metric("Last MAS print", f"{neer_w.iloc[-1]:.2f}" if not neer_w.empty else "N/A",
                        f"{neer_w.index[-1]:%d %b}" if not neer_w.empty else None, delta_color="off")
            k[3].metric("Model fit (R², weekly Δ)", f"{neer_r2:.2f}" if pd.notna(neer_r2) else "N/A")

    # S$NEER + estimated band
    fig_band = go.Figure()
    if not band_df.empty:
        bd = clip(band_df)
        fig_band.add_trace(go.Scatter(x=bd.index, y=bd["Upper"], line=dict(color="rgba(38,166,154,0.6)", dash="dot", width=1),
                                      name=f"Est. upper (+{hw:.2f}%)"))
        fig_band.add_trace(go.Scatter(x=bd.index, y=bd["Lower"], line=dict(color="rgba(239,83,80,0.6)", dash="dot", width=1),
                                      fill="tonexty", fillcolor="rgba(144,164,212,0.07)", name=f"Est. lower (−{hw:.2f}%)"))
        fig_band.add_trace(go.Scatter(x=bd.index, y=bd["Mid"], line=dict(color="#e0e0e0", dash="dash", width=1),
                                      name=f"Est. mid ({band_slope:+.2f}% p.a.)"))
    if neer_model is not None:
        fig_band.add_trace(go.Scatter(x=clip(neer_model).index, y=clip(neer_model).values, name="Daily model",
                                      line=dict(color="#ff9800", width=1.4)))
    if not neer_w.empty:
        nw = clip(neer_w)
        fig_band.add_trace(go.Scatter(x=nw.index, y=nw.values, name="MAS S$NEER (weekly)", mode="markers",
                                      marker=dict(color="#4fc3f7", size=4)))
    add_mps_lines(fig_band)
    fig_band.update_layout(**base_layout("S$NEER vs Estimated Policy Band (dotted verticals = MPS dates)", height=520))
    st.plotly_chart(fig_band, use_container_width=True, key="chart_neer_band")
    csv_download(band_df, "sneer_band_estimate")

    fig_dev = go.Figure()
    if not band_df.empty:
        zline(fig_dev, band_df["Dev from Mid %"], "Deviation from est. mid", color="#ff9800", unit="%")
        for lvl, c in [(hw, "#26a69a"), (-hw, "#ef5350"), (0, "#555")]:
            fig_dev.add_hline(y=lvl, line_dash="dot", line_color=c)
    fig_dev.update_layout(**base_layout("S$NEER — Deviation from Estimated Mid (%)"))
    fig_dev.update_yaxes(ticksuffix="%")

    fig_usdsgd = go.Figure()
    if not usdsgd.empty:
        zline(fig_usdsgd, usdsgd, "USD/SGD", color="#90a4d4", fmt=".4f")
        for w_, c in [(50, "#ff9800"), (200, "#ab47bc")]:
            ma = clip(usdsgd.rolling(w_).mean())
            fig_usdsgd.add_trace(go.Scatter(x=ma.index, y=ma.values, name=f"{w_}D MA", line=dict(color=c, width=1, dash="dot")))
    fig_usdsgd.update_layout(**base_layout("USD/SGD Spot with 50D / 200D Moving Averages"))

    fig_rvol = go.Figure()
    rvol_df = pd.DataFrame()
    if not usdsgd.empty:
        rvol_df = pd.DataFrame({"USD/SGD 1M": realized_vol(usdsgd, 21), "USD/SGD 3M": realized_vol(usdsgd, 63)})
        if neer_model is not None:
            rvol_df["S$NEER model 1M"] = realized_vol(neer_model, 21)
        for c_, col in zip(rvol_df.columns, ["#90a4d4", "#42a5f5", "#ff9800"]):
            zline(fig_rvol, rvol_df[c_], c_, color=col, unit="%")
    fig_rvol.update_layout(**base_layout("Realized Volatility (annualised, %)"))
    fig_rvol.update_yaxes(ticksuffix="%")

    # SGD performance heatmap (positive = SGD stronger vs that currency)
    fig_perf = go.Figure()
    perf = pd.DataFrame()
    if not fcy.empty:
        rows = {}
        for ccy in fcy.columns:
            s_ = fcy[ccy].dropna()
            if len(s_) < 260:
                continue
            last_ = s_.iloc[-1]
            ytd_base = s_[s_.index < pd.Timestamp(s_.index[-1].year, 1, 1)]
            rows[FX_DISPLAY[ccy]] = {
                "1W": (last_ / s_.asof(s_.index[-1] - pd.Timedelta(days=7)) - 1) * 100,
                "1M": (last_ / s_.asof(s_.index[-1] - pd.DateOffset(months=1)) - 1) * 100,
                "3M": (last_ / s_.asof(s_.index[-1] - pd.DateOffset(months=3)) - 1) * 100,
                "YTD": (last_ / ytd_base.iloc[-1] - 1) * 100 if not ytd_base.empty else np.nan,
                "1Y": (last_ / s_.asof(s_.index[-1] - pd.DateOffset(years=1)) - 1) * 100,
            }
        if neer_model is not None:
            nm = neer_model
            rows["S$NEER (model)"] = {
                "1W": (nm.iloc[-1] / nm.asof(nm.index[-1] - pd.Timedelta(days=7)) - 1) * 100,
                "1M": (nm.iloc[-1] / nm.asof(nm.index[-1] - pd.DateOffset(months=1)) - 1) * 100,
                "3M": (nm.iloc[-1] / nm.asof(nm.index[-1] - pd.DateOffset(months=3)) - 1) * 100,
                "YTD": (nm.iloc[-1] / nm[nm.index < pd.Timestamp(nm.index[-1].year, 1, 1)].iloc[-1] - 1) * 100,
                "1Y": (nm.iloc[-1] / nm.asof(nm.index[-1] - pd.DateOffset(years=1)) - 1) * 100,
            }
        perf = pd.DataFrame(rows).T.sort_values("3M")
        lim = float(np.nanmax(np.abs(perf.values))) if not perf.empty else 1
        fig_perf.add_trace(go.Heatmap(
            z=perf.values, x=perf.columns, y=perf.index, zmin=-lim, zmax=lim,
            colorscale=[[0, "#ef5350"], [0.5, PLOT_BG], [1, "#26a69a"]],
            text=perf.round(2).values, texttemplate="%{text:+.2f}%", textfont=dict(size=10),
            hovertemplate="SGD vs %{y}, %{x}: %{z:+.2f}%<extra></extra>", showscale=False))
    fig_perf.update_layout(**base_layout("SGD Performance vs Crosses (%, + = SGD stronger)", height=520))

    # Rebased crosses
    default_x = [c for c in ["USD", "CNY", "MYR", "JPY", "EUR"] if c in fcy.columns]
    pick = st.multiselect("Crosses to rebase (SGD strength, 100 = start of date range)",
                          [c for c in fcy.columns], default=default_x, key="sgd_rebase_pick",
                          format_func=lambda c: FX_DISPLAY[c])
    fig_rebase = go.Figure()
    for c_ in pick:
        s_ = clip(fcy[c_].dropna())
        if not s_.empty:
            fig_rebase.add_trace(go.Scatter(x=s_.index, y=s_ / s_.iloc[0] * 100, name=f"SGD vs {c_}", mode="lines"))
    if neer_model is not None:
        nm_c = clip(neer_model)
        fig_rebase.add_trace(go.Scatter(x=nm_c.index, y=nm_c / nm_c.iloc[0] * 100, name="S$NEER (model)",
                                        line=dict(color="white", width=2)))
    fig_rebase.add_hline(y=100, line_dash="dot", line_color="#555")
    fig_rebase.update_layout(**base_layout("SGD vs Selected Crosses — Rebased (up = SGD stronger)"))

    # USD-leg analytics: USD/X for Asia + DXY, daily log returns
    usd_x = pd.DataFrame()
    if not fx_px.empty:
        for ccy, (tkr, usd_quote) in FX_TICKERS.items():
            if tkr in fx_px.columns and ccy in ("USD", "CNY", "MYR", "KRW", "TWD", "JPY", "IDR", "THB", "INR", "EUR"):
                lab = "USD/SGD" if ccy == "USD" else f"USD/{ccy}"   # EUR/USD inverted so every pair is USD-base
                usd_x[lab] = 1 / fx_px[tkr] if usd_quote else fx_px[tkr]
        if "DX-Y.NYB" in fx_px.columns:
            usd_x["DXY"] = fx_px["DX-Y.NYB"]
    rets = np.log(usd_x.ffill(limit=3)).diff() if not usd_x.empty else pd.DataFrame()

    fig_beta = go.Figure()
    beta_df = pd.DataFrame()
    if not rets.empty and "USD/SGD" in rets:
        for drv, col in [("DXY", "#ff9800"), ("USD/CNY", "#ef5350"), ("USD/MYR", "#26a69a")]:
            if drv in rets:
                pair = rets[["USD/SGD", drv]].dropna()
                beta_df[f"β to {drv}"] = pair["USD/SGD"].rolling(60).cov(pair[drv]) / pair[drv].rolling(60).var()
        for c_, col in zip(beta_df.columns, ["#ff9800", "#ef5350", "#26a69a"]):
            zline(fig_beta, beta_df[c_], c_, color=col)
    fig_beta.update_layout(**base_layout("USD/SGD Rolling 60D Beta to DXY / USD-CNY / USD-MYR"))

    fig_corr = go.Figure()
    corr = pd.DataFrame()
    if not rets.empty:
        corr = rets[rets.index >= rets.index[-1] - pd.DateOffset(months=3)].corr()
        fig_corr.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                      colorscale=[[0, "#ef5350"], [0.5, PLOT_BG], [1, "#26a69a"]],
                                      text=corr.round(2).values, texttemplate="%{text:.2f}", textfont=dict(size=9),
                                      showscale=False))
    fig_corr.update_layout(**base_layout("3M Correlation of Daily Returns — USD vs Asia FX", height=520))

    # Covered-interest-parity forward points & carry (bill-implied, not dealer quotes)
    fig_fwd = go.Figure()
    fig_carry = go.Figure()
    fwd_df = pd.DataFrame()
    if not usdsgd.empty and not ust.empty and not sgs.empty:
        for lab, sg_c, us_c, days in [("6M", "6M", "6M", 182), ("12M", "1Y", "1Y", 365)]:
            f_ = pd.concat([usdsgd, _col(sgs, sg_c), _col(ust, us_c)], axis=1, keys=["S", "rs", "ru"]).ffill(limit=3).dropna()
            fwd = f_["S"] * (1 + f_["rs"] / 100 * days / 365) / (1 + f_["ru"] / 100 * days / 360)
            fwd_df[f"{lab} fwd pts (pips)"] = (fwd - f_["S"]) * 1e4
            fwd_df[f"{lab} USD−SGD rate diff (pp)"] = f_["ru"] - f_["rs"]
        for c_, col in [("6M fwd pts (pips)", "#42a5f5"), ("12M fwd pts (pips)", "#26a69a")]:
            zline(fig_fwd, fwd_df[c_], c_, color=col, fmt=".0f")
    fig_fwd.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_fwd.update_layout(**base_layout("USD/SGD Implied Forward Points (SGS bills vs UST bills, CIP)"))

    # MAS intervention proxy: FX reserves change in USD (removes USD/SGD translation effects)
    fig_resv = go.Figure()
    resv_df = pd.DataFrame()
    fx_res = fetch_singstat("M700031", "Official Foreign Reserves", n_periods=400)
    if not fx_res.empty and not usdsgd.empty:
        me_fx = usdsgd.resample("MS").last()
        resv_df = pd.DataFrame({"Reserves (S$M)": fx_res["Official Foreign Reserves"]}).join(me_fx.rename("USDSGD"), how="left")
        resv_df["Reserves (US$B)"] = resv_df["Reserves (S$M)"] / resv_df["USDSGD"] / 1000
        resv_df["MoM Δ (US$B)"] = resv_df["Reserves (US$B)"].diff()
        if not neer_w.empty:
            resv_df["S$NEER MoM %"] = neer_w.resample("MS").mean().pct_change() * 100
        rc = clip(resv_df)
        fig_resv.add_trace(go.Bar(x=rc.index, y=rc["MoM Δ (US$B)"], name="Reserves MoM Δ (US$B)",
                                  marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in rc["MoM Δ (US$B)"].fillna(0)]))
        if "S$NEER MoM %" in rc:
            fig_resv.add_trace(go.Scatter(x=rc.index, y=rc["S$NEER MoM %"], name="S$NEER MoM %", yaxis="y2",
                                          line=dict(color="#ff9800", width=1.6)))
    fig_resv.update_layout(**dual_axis_layout("MAS Intervention Proxy — FX Reserves Change (US$B) vs S$NEER",
                                              "US$ Billion", "S$NEER MoM %"))

    fig_w = go.Figure()
    if neer_weights is not None:
        nw_ = neer_weights[neer_weights > 0.001].sort_values()
        fig_w.add_trace(go.Bar(x=nw_.values * 100, y=[FX_DISPLAY[c] for c in nw_.index], orientation="h",
                               marker_color="#90a4d4", text=[f"{v * 100:.1f}%" for v in nw_.values], textposition="outside"))
    fig_w.update_layout(**base_layout(f"S$NEER Model — Implied Basket Weights (R² {neer_r2:.2f}, "
                                      f"{neer_cal0:%b %Y}–{neer_cal1:%b %Y})" if neer_weights is not None else
                                      "S$NEER Model — Implied Basket Weights", height=480))
    fig_w.update_xaxes(ticksuffix="%")

    render_two_col([
        ("S$NEER Deviation from Est Mid", fig_dev, band_df),
        ("USD-SGD Spot", fig_usdsgd, clip(usdsgd.to_frame("USD/SGD")) if not usdsgd.empty else None),
        ("SGD Performance Heatmap", fig_perf, perf),
        ("Realized Vol", fig_rvol, clip(rvol_df) if not rvol_df.empty else None),
        ("SGD Rebased Crosses", fig_rebase, None),
        ("USD-SGD Beta", fig_beta, clip(beta_df) if not beta_df.empty else None),
        ("USD-Asia Correlation", fig_corr, corr),
        ("S$NEER Model Weights", fig_w, neer_weights.to_frame("weight") if neer_weights is not None else None),
    ])

    # Forward points + carry get their own row so the vol-window selector sits right above the
    # carry chart it drives (render_two_col can't place widgets inside the grid).
    CARRY_VOL_WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    col_fwd, col_carry = st.columns(2)
    with col_fwd:
        st.plotly_chart(fig_fwd, use_container_width=True, key="chart_USD-SGD Forward Points")
        if not fwd_df.empty:
            csv_download(clip(fwd_df), "USD-SGD Forward Points")
    with col_carry:
        vol_win = st.radio("Realized vol window (carry-to-vol)", list(CARRY_VOL_WINDOWS), index=1,
                           horizontal=True, key="carry_vol_window")
        carry_df = pd.DataFrame()
        if not fwd_df.empty:
            rv = realized_vol(usdsgd, CARRY_VOL_WINDOWS[vol_win])
            carry_df = pd.concat([fwd_df["12M USD−SGD rate diff (pp)"], rv], axis=1,
                                 keys=["Carry (% p.a.)", f"{vol_win} realized vol (%)"]).dropna()
            carry_df["Carry / Vol"] = carry_df["Carry (% p.a.)"] / carry_df[f"{vol_win} realized vol (%)"]
            zline(fig_carry, carry_df["Carry (% p.a.)"], "Long USD/SGD carry (% p.a.)", color="#ff9800", unit="%")
            zline(fig_carry, carry_df["Carry / Vol"], f"Carry / {vol_win} realized vol", color="#90a4d4", yaxis="y2")
        fig_carry.update_layout(**dual_axis_layout(f"USD/SGD Carry (12M rate differential) & Carry-to-Vol ({vol_win} vol)",
                                                   "Carry (% p.a.)", "Carry / Vol"))
        st.plotly_chart(fig_carry, use_container_width=True, key="chart_USD-SGD Carry")
        if not carry_df.empty:
            csv_download(clip(carry_df), "USD-SGD Carry")

    st.plotly_chart(fig_resv, use_container_width=True, key="chart_MAS Intervention Proxy")
    if not resv_df.empty:
        csv_download(clip(resv_df), "MAS Intervention Proxy")
    st.caption("Forward points are theoretical covered-interest-parity values from SGS vs UST bill yields - real "
               "dealer forwards also carry a cross-currency basis. Reserves change is measured in US\\$ to strip "
               "USD/SGD translation, but still includes valuation moves on non-USD assets and investment returns, so "
               "treat it as a proxy for intervention direction, not size.")

    # FX monitor
    st.markdown('<div class="section-header">FX Monitor</div>', unsafe_allow_html=True)
    if not fcy.empty:
        mon = []
        series_map = {FX_DISPLAY[c]: display_quote(fcy, c) for c in fcy.columns}
        strength_map = {FX_DISPLAY[c]: fcy[c] for c in fcy.columns}
        if neer_model is not None:
            series_map["S$NEER (model)"] = neer_model
            strength_map["S$NEER (model)"] = neer_model
        for lab, s_ in series_map.items():
            s_, st_ = s_.dropna(), strength_map[lab].dropna()
            if len(s_) < 260:
                continue
            z_ = _zscores(s_, Z_WINDOWS["D"])
            mon.append({
                "Pair": lab, "Last": s_.iloc[-1],
                "SGD 1D %": (st_.iloc[-1] / st_.iloc[-2] - 1) * 100,
                "SGD 1W %": (st_.iloc[-1] / st_.asof(st_.index[-1] - pd.Timedelta(days=7)) - 1) * 100,
                "SGD 1M %": (st_.iloc[-1] / st_.asof(st_.index[-1] - pd.DateOffset(months=1)) - 1) * 100,
                "Z 1M": z_["1M"], "Z 3M": z_["3M"], "Z 1Y": z_["1Y"],
                "1Y %ile": _pctile(s_, 1), "1M RVol %": realized_vol(s_, 21).iloc[-1],
            })
        mon_df = pd.DataFrame(mon)
        sty = (mon_df.style
               .format({"Last": "{:,.4f}", "SGD 1D %": "{:+.2f}", "SGD 1W %": "{:+.2f}", "SGD 1M %": "{:+.2f}",
                        "Z 1M": "{:+.2f}", "Z 3M": "{:+.2f}", "Z 1Y": "{:+.2f}", "1Y %ile": "{:.0f}", "1M RVol %": "{:.1f}"},
                       na_rep="—")
               .map(_chgcolor, subset=["SGD 1D %", "SGD 1W %", "SGD 1M %"])
               .map(_zcolor, subset=["Z 1M", "Z 3M", "Z 1Y"])
               .map(_pctcolor, subset=["1Y %ile"]))
        st.dataframe(sty, hide_index=True, use_container_width=True, height=36 * (len(mon_df) + 1))
        st.caption("\"SGD x %\" columns are always SGD strength (+ = SGD appreciated vs that currency), regardless "
                   "of how the pair is quoted. Z-scores and percentile are on the quoted level.")
        csv_download(mon_df, "sg_fx_monitor")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — SGS & Rates
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("SGS & Rates")
    st.caption("SGS benchmark yields from MAS's daily benchmark-issue history (back to 2008), UST from treasury.gov, "
               "SOFR from the NY Fed. Hover any line for that day's trailing 1Y z-score. MAS no longer quotes a 3M or "
               "7Y benchmark, so the curve runs 6M → 50Y.")

    # Tenors MAS still quotes (3M/7Y went blank), and the curve built from them - used by the monitor
    # and the snapshot / rolldown / carry charts below.
    live_tenors = [t for t in sgs.columns if sgs[t].iloc[-60:].notna().any()] if not sgs.empty else []
    curve = sgs[live_tenors].dropna(how="all").ffill(limit=3) if live_tenors else pd.DataFrame()

    # Rates monitor: one collapsible panel holding three collapsible tables (outrights / spreads / flies),
    # same layout and carry conventions as usa_macro.py's monitor. Funding = 3M compounded SORA, matching
    # the carry + roll chart below. MAS no longer quotes 3M or 7Y, so 6M stands in for the US 3M legs.
    SG_RM_SPREADS = {   # spread = Σ weight × yield (bp); "long" = steepener (pay the positive-weight leg)
        "2s5s": {"5Y": 1, "2Y": -1}, "2s10s": {"10Y": 1, "2Y": -1}, "5s10s": {"10Y": 1, "5Y": -1},
        "5s30s": {"30Y": 1, "5Y": -1}, "10s30s": {"30Y": 1, "10Y": -1}, "6M10Y": {"10Y": 1, "6M": -1},
        "10s20s": {"20Y": 1, "10Y": -1}, "2s30s": {"30Y": 1, "2Y": -1}, "30s50s": {"50Y": 1, "30Y": -1},
    }
    SG_RM_FLIES = {     # long fly = pay the belly (positive weight), receive the wings
        "2s5s10s": {"5Y": 2, "2Y": -1, "10Y": -1}, "5s10s30s": {"10Y": 2, "5Y": -1, "30Y": -1},
        "10s20s30s": {"20Y": 2, "10Y": -1, "30Y": -1}, "1s10s20s": {"10Y": 2, "1Y": -1, "20Y": -1},
        "2s10s30s": {"10Y": 2, "2Y": -1, "30Y": -1}, "10s15s20s": {"15Y": 2, "10Y": -1, "20Y": -1},
        "20s30s50s": {"30Y": 2, "20Y": -1, "50Y": -1},
    }

    def build_sg_rates_monitor():
        fund_s = csora3.dropna()
        fund, fund_dt = float(fund_s.iloc[-1]), fund_s.index[-1]
        last = curve.iloc[-1].dropna()
        legs = {}
        for t in live_tenors:
            if t not in last:
                continue
            y_, T_ = float(last[t]), SGS_TENOR_YEARS[t]
            dur = (1 - (1 + y_ / 200) ** (-2 * T_)) / (y_ / 100)       # modified duration, par bond, semi-annual
            running = (y_ - fund) * 100                                # bp/yr, receiving the tenor, funded at 3M comp. SORA
            # ≤1Y matures inside the 1Y horizon: no yield risk to break even against, and rolling to "0Y" would
            # just land on the shortest quote, so breakeven carry & roll are left blank
            be = running / dur if T_ > 1 else np.nan
            roll = float(curve_interp(last, [T_])[0] - curve_interp(last, [T_ - 1])[0]) * 100 if T_ > 1 else np.nan
            legs[t] = (running, be, roll)

        def row(name, s_, unit, running=np.nan, be=np.nan, roll=np.nan):
            s_ = s_.dropna()
            if len(s_) < 30:
                return None
            m_ = 100 if unit == "%" else 1
            z_ = _zscores(s_, Z_WINDOWS["D"])
            vol = s_.diff().iloc[-252:].std() * m_ * np.sqrt(252)
            cr = be + roll
            return {"Instrument": name, "Level": float(s_.iloc[-1]), "_unit": unit,
                    "Δ1D (bp)": (s_.iloc[-1] - s_.iloc[-2]) * m_, "Δ1W (bp)": _change_over(s_, pd.Timedelta(days=7)) * m_,
                    "Δ1M (bp)": _change_over(s_, pd.DateOffset(months=1)) * m_, "Δ3M (bp)": _change_over(s_, pd.DateOffset(months=3)) * m_,
                    "Z 1M": z_["1M"], "Z 3M": z_["3M"], "Z 1Y": z_["1Y"], "1Y %ile": _pctile(s_, 1), "5Y %ile": _pctile(s_, 5),
                    "Carry (bp/yr)": running, "Breakeven carry (bp yld/yr)": be, "Roll 1Y (bp)": roll,
                    "Carry+Roll (bp yld/yr)": cr, "1Y vol (bp)": vol,
                    "C+R / Vol": cr / vol if pd.notna(cr) and vol else np.nan, "As of": s_.index[-1]}

        outr = [row("SORA", sora, "%"), row("1M Comp. SORA", csora1, "%"), row("3M Comp. SORA", csora3, "%"),
                row("6M Comp. SORA", csora6, "%")]
        outr += [row(f"{t} {'T-Bill' if SGS_TENOR_YEARS[t] <= 1 else 'SGS'}", _col(sgs, t), "%", *legs[t])
                 for t in live_tenors if t in legs]

        def combo(defs):
            out = []
            for n, w in defs.items():
                if not all(k in legs for k in w):
                    continue
                s_ = (sum(_col(sgs, k) * v for k, v in w.items()) * 100).dropna()
                be = -sum(v * legs[k][1] for k, v in w.items())       # DV01-neutral legs add in bp of yield
                roll = -sum(v * legs[k][2] for k, v in w.items())
                out.append(row(n, s_, "bp", np.nan, be, roll))
            return out
        spr = combo(SG_RM_SPREADS) + [row("SORA − SOFR", sora_sofr, "bp"), row("6M T-Bill − 6M Comp. SORA", _spread(bill6, csora6), "bp"),
                                      row("2Y SGS − UST", sgs_ust_2, "bp"), row("10Y SGS − UST", sgs_ust_10, "bp")]
        fly = combo(SG_RM_FLIES)
        clean = lambda rows: pd.DataFrame([r for r in rows if r is not None])
        return clean(outr), clean(spr), clean(fly), fund, fund_dt

    def render_sg_monitor_table(df, key, show_running):
        if df.empty:
            st.info("No data available for this table right now.")
            return
        if not show_running:
            df = df.drop(columns="Carry (bp/yr)")
        sortable = [c for c in df.columns if not c.startswith("_")]
        s1, s2 = st.columns([3, 2])
        sort_cols = s1.multiselect("Sort by (priority order — first pick sorts first)", sortable, default=[],
                                   key=f"sg_rm_sort_{key}", placeholder="Default order")
        asc = []
        if sort_cols:
            for dc, col in zip(s2.columns(len(sort_cols)), sort_cols):
                asc.append(dc.radio(col, ["↓ Desc", "↑ Asc"], key=f"sg_rm_dir_{key}_{col}") == "↑ Asc")
            df = df.sort_values(sort_cols, ascending=asc, na_position="last", kind="mergesort")
        else:
            s2.caption("Pick one or more columns to sort by, or click a column header for a quick single-column sort.")
        view = df.copy()
        view["Level"] = [f"{v:.3f}%" if u == "%" else f"{v:+.1f}bp" for v, u in zip(view["Level"], view["_unit"])]
        view["As of"] = view["As of"].dt.strftime("%d %b")
        view = view.drop(columns="_unit")
        chg = ["Δ1D (bp)", "Δ1W (bp)", "Δ1M (bp)", "Δ3M (bp)"]
        carry = [c for c in ["Carry (bp/yr)", "Breakeven carry (bp yld/yr)", "Roll 1Y (bp)", "Carry+Roll (bp yld/yr)"] if c in view]
        # callables rather than format strings + na_rep: st.dataframe renders a Styler's NaNs as "None" otherwise
        fmt = lambda f: (lambda v: "—" if v is None or pd.isna(v) else format(v, f))
        st.dataframe(view.style.format({**{c: fmt("+.1f") for c in chg + carry}, "Z 1M": fmt("+.2f"), "Z 3M": fmt("+.2f"),
                                        "Z 1Y": fmt("+.2f"), "1Y %ile": fmt(".0f"), "5Y %ile": fmt(".0f"), "1Y vol (bp)": fmt(".0f"),
                                        "C+R / Vol": fmt("+.2f")})
                     .map(_chgcolor, subset=chg + carry).map(_zcolor, subset=["Z 1M", "Z 3M", "Z 1Y", "C+R / Vol"])
                     .map(_pctcolor, subset=["1Y %ile", "5Y %ile"]),
                     hide_index=True, use_container_width=True, height=36 * (len(view) + 1))
        csv_download(df.drop(columns="_unit"), f"sg_rates_monitor_{key}")

    with st.expander("Rates Monitor — outrights, spreads & flies", expanded=True):
        try:
            rm_out, rm_spr, rm_fly, rm_fund, rm_fund_dt = build_sg_rates_monitor()
        except Exception as e:
            st.warning(f"Could not build the rates monitor: {e}")
            rm_out = None
        if rm_out is not None:
            st.caption(f"Carry is for receiving each tenor funded at 3M compounded SORA ({rm_fund:.2f}%, {rm_fund_dt:%d %b}). "
                       "Carry (bp/yr) = (yield − funding) × 100, annualised running carry. Breakeven carry = that ÷ modified "
                       "duration (the yield rise one year of carry offsets). Roll 1Y = rolldown on today's curve. "
                       "C+R / Vol = (breakeven carry + roll) ÷ 1Y realised vol. Tenors of 1Y and under (and flies using them) "
                       "are blank there, since they mature inside the 1Y horizon. Spreads & flies are DV01-neutral and shown "
                       "for being LONG the spread (steepener / long fly, i.e. paying the positive-weight legs); the SORA, bill "
                       "and SGS−UST spreads have no carry figures. Changes in bps; percentile = where today's level sits in "
                       "its own trailing 1Y / 5Y range.")
            with st.expander("Outright Rates", expanded=True):
                render_sg_monitor_table(rm_out, "outright", show_running=True)
            with st.expander("Spreads", expanded=True):
                render_sg_monitor_table(rm_spr, "spreads", show_running=False)
            with st.expander("Flies", expanded=True):
                render_sg_monitor_table(rm_fly, "flies", show_running=False)

    # SORA curve snapshots - same snapshot treatment as the SGS curve below (Latest/1D/1W/1M/3M
    # Ago), but across the SORA family's own tenor points: overnight SORA plus 1M/3M/6M
    # compounded SORA. That's the complete set MAS publishes here - there's no SORA equivalent
    # of SGS's 2Y-50Y benchmarks, so this curve only runs out to 6M.
    sora_curve = pd.concat([sora.rename("O/N"), csora1.rename("1M"), csora3.rename("3M"), csora6.rename("6M")], axis=1)
    sora_tenors = [t for t in sora_curve.columns if sora_curve[t].iloc[-60:].notna().any()] if not sora_curve.empty else []
    sora_curve_live = sora_curve[sora_tenors].dropna(how="all").ffill(limit=3) if sora_tenors else pd.DataFrame()
    fig_sora_curve = go.Figure()
    sora_snap_rows = {}
    if not sora_curve_live.empty:
        snaps_ = {"Latest": 0, "1D Ago": -1, "1W Ago": -5, "1M Ago": -21, "3M Ago": -63}
        colors_s = {"Latest": "cyan", "1D Ago": "magenta", "1W Ago": "orange", "1M Ago": "green", "3M Ago": "#90a4d4"}
        for lab, off in snaps_.items():
            row = sora_curve_live.iloc[max(0, len(sora_curve_live) - 1 + off)]
            sora_snap_rows[f"{lab} ({row.name:%Y-%m-%d})"] = row
            fig_sora_curve.add_trace(go.Scatter(x=sora_tenors, y=row.values, mode="lines+markers", name=f"{lab} ({row.name:%d %b})",
                                                line=dict(color=colors_s[lab], width=2 if lab == "Latest" else 1,
                                                          dash="solid" if lab == "Latest" else "dash")))
    fig_sora_curve.update_layout(**base_layout("SORA Curve — Snapshots"))
    fig_sora_curve.update_yaxes(ticksuffix="%")

    # Curve snapshots & changes
    fig_yc = go.Figure()
    fig_yc_chg = go.Figure()
    snap_rows = {}
    if not curve.empty:
        snaps = {"Latest": 0, "1D Ago": -1, "1W Ago": -5, "1M Ago": -21, "3M Ago": -63}
        colors_ = {"Latest": "cyan", "1D Ago": "magenta", "1W Ago": "orange", "1M Ago": "green", "3M Ago": "#90a4d4"}
        for lab, off in snaps.items():
            row = curve.iloc[max(0, len(curve) - 1 + off)]
            snap_rows[f"{lab} ({row.name:%Y-%m-%d})"] = row
            fig_yc.add_trace(go.Scatter(x=live_tenors, y=row.values, mode="lines+markers", name=f"{lab} ({row.name:%d %b})",
                                        line=dict(color=colors_[lab], width=2 if lab == "Latest" else 1,
                                                  dash="solid" if lab == "Latest" else "dash")))
            if lab != "Latest":
                if lab == "3M Ago":
                    continue
                fig_yc_chg.add_trace(go.Bar(x=live_tenors, y=(curve.iloc[-1] - row).values * 100, name=lab,
                                            marker_color=colors_[lab], opacity=0.85))
    fig_yc.update_layout(**base_layout("SGS Yield Curve — Snapshots"))
    fig_yc.update_yaxes(ticksuffix="%")
    fig_yc_chg.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_yc_chg.update_layout(**base_layout("SGS Curve Changes (bps)"), barmode="group")
    fig_yc_chg.update_yaxes(ticksuffix=" bps")

    fig_yields = go.Figure()
    for t, c_ in [("6M", "#8a94a6"), ("2Y", "#42a5f5"), ("5Y", "#26a69a"), ("10Y", "#ff9800"), ("20Y", "#ab47bc"), ("30Y", "#ef5350")]:
        zline(fig_yields, _col(sgs, t), f"{t} SGS", color=c_, unit="%")
    add_mps_lines(fig_yields)
    fig_yields.update_layout(**base_layout("SGS Benchmark Yields"))
    fig_yields.update_yaxes(ticksuffix="%")

    fig_spreads = go.Figure()
    for (k_, v_), c_ in zip(SPREADS.items(), ["#42a5f5", "#26a69a", "#ab47bc", "#ffd54f", "#8a94a6", "#ff9800"]):
        zline(fig_spreads, v_, k_, color=c_, unit=" bps", fmt=".1f")
    fig_spreads.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_spreads.update_layout(**base_layout("SGS Curve Spreads & 2s5s10s Butterfly (bps)"))
    fig_spreads.update_yaxes(ticksuffix=" bps")

    # 1Y rolldown history (on the clipped range only - one interpolation per day)
    fig_roll = go.Figure()
    roll_df = pd.DataFrame()
    roll_tenors = [t for t in ("2Y", "5Y", "10Y", "15Y", "20Y", "30Y") if t in live_tenors]
    if roll_tenors:
        roll_df = rolldown_series(clip(curve), roll_tenors, 1.0)
        for t in roll_tenors:
            fig_roll.add_trace(go.Scatter(x=roll_df.index, y=roll_df[t], name=t, mode="lines",
                                          hovertemplate=f"%{{x|%Y-%m-%d}}<br>{t} 1Y roll: %{{y:.1f}} bps<extra></extra>"))
    fig_roll.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_roll.update_layout(**base_layout("Outright SGS — 1Y Rolldown (bps)"))
    fig_roll.update_yaxes(ticksuffix=" bps")

    # Carry + roll snapshot (3M horizon, funded at 3M compounded SORA)
    fig_cr = go.Figure()
    cr_df = pd.DataFrame()
    if roll_tenors and not csora3.empty:
        last_curve = curve.iloc[-1]
        fund = float(csora3.iloc[-1])
        rows_ = []
        for t in roll_tenors:
            yT = float(last_curve[t])
            T_ = SGS_TENOR_YEARS[t]
            # Carry is a price return ((yield - funding) x 3/12, bp of notional); roll and vol are in bp of
            # YIELD. Divide by modified duration (par bond, semi-annual coupons) so all three share a unit -
            # without this, long-end carry was overstated by roughly its duration and the ranking inverted.
            dur = (1 - (1 + yT / 200) ** (-2 * T_)) / (yT / 100)
            carry_px = (yT - fund) * 100 * 0.25
            carry = carry_px / dur
            roll = float(curve_interp(last_curve, [SGS_TENOR_YEARS[t]])[0] - curve_interp(last_curve, [SGS_TENOR_YEARS[t] - 0.25])[0]) * 100
            vol = sgs[t].dropna().diff().iloc[-63:].std() * 100 * np.sqrt(63)
            rows_.append({"Tenor": t, "Yield %": yT, "Mod. duration": dur, "Carry (bps price, 3M)": carry_px,
                          "Carry (bps yield, 3M)": carry, "Roll (bps, 3M)": roll,
                          "Carry+Roll (bps)": carry + roll, "3M yield vol (bps)": vol,
                          "Breakeven ratio": (carry + roll) / vol if vol else np.nan})
        cr_df = pd.DataFrame(rows_)
        fig_cr.add_trace(go.Bar(x=cr_df["Tenor"], y=cr_df["Carry (bps yield, 3M)"], name="Carry (bp yield)", marker_color="#42a5f5"))
        fig_cr.add_trace(go.Bar(x=cr_df["Tenor"], y=cr_df["Roll (bps, 3M)"], name="Roll", marker_color="#26a69a"))
        fig_cr.add_trace(go.Scatter(x=cr_df["Tenor"], y=cr_df["Breakeven ratio"], name="(Carry+Roll) / 3M vol",
                                    yaxis="y2", mode="lines+markers", line=dict(color="#ff9800", width=2)))
        fig_cr.update_layout(**dual_axis_layout(f"SGS Carry + Roll, 3M Horizon (funded at 3M Comp. SORA {fund:.2f}%)",
                                                "bps of yield", "Carry+Roll / Vol"), barmode="relative")

    fig_vs_ust = go.Figure()
    zline(fig_vs_ust, sgs_ust_2, "2Y SGS − UST", color="#42a5f5", unit=" bps", fmt=".0f")
    zline(fig_vs_ust, sgs_ust_10, "10Y SGS − UST", color="#ff9800", unit=" bps", fmt=".0f")
    fig_vs_ust.update_layout(**base_layout("SGS − UST Yield Spreads (bps)"))
    fig_vs_ust.update_yaxes(ticksuffix=" bps")

    # Weekly-change beta to UST: Singapore closes ~12h before New York, so same-day daily
    # changes mis-align; weekly changes sidestep the time-zone lag.
    fig_ust_beta = go.Figure()
    ust_beta = pd.DataFrame()
    for t, c_ in [("2Y", "#42a5f5"), ("10Y", "#ff9800")]:
        a_, b_ = _col(sgs, t), _col(ust, t)
        if a_.empty or b_.empty:
            continue
        wk_ = pd.concat([a_, b_], axis=1, keys=["sg", "us"]).resample("W-FRI").last().diff().dropna()
        ust_beta[f"{t} β"] = wk_["sg"].rolling(26).cov(wk_["us"]) / wk_["us"].rolling(26).var()
        ust_beta[f"{t} corr"] = wk_["sg"].rolling(26).corr(wk_["us"])
        bc_ = clip(ust_beta[f"{t} β"].dropna())
        fig_ust_beta.add_trace(go.Scatter(x=bc_.index, y=bc_.values, name=f"{t} beta", line=dict(color=c_)))
        cc_ = clip(ust_beta[f"{t} corr"].dropna())
        fig_ust_beta.add_trace(go.Scatter(x=cc_.index, y=cc_.values, name=f"{t} corr", yaxis="y2",
                                          line=dict(color=c_, dash="dot", width=1)))
    fig_ust_beta.update_layout(**dual_axis_layout("SGS Beta & Correlation to UST (26W, weekly changes)", "Beta", "Correlation"))

    # Ex-post real 10Y
    fig_real = go.Figure()
    real_df = pd.DataFrame()
    if not y10.empty and not macro["Core Infl. YoY"].empty:
        y10m = y10.resample("MS").mean()
        real_df = pd.DataFrame({"10Y − Core YoY": y10m - macro["Core Infl. YoY"],
                                "10Y − Headline YoY": y10m - macro["CPI YoY"]}).dropna(how="all")
        zline(fig_real, real_df["10Y − Core YoY"], "10Y SGS − MAS Core YoY", freq="M", color="#26a69a", unit="%")
        zline(fig_real, real_df["10Y − Headline YoY"], "10Y SGS − Headline CPI YoY", freq="M", color="#ef5350", unit="%", dash="dot")
    fig_real.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_real.update_layout(**base_layout("Ex-Post Real 10Y SGS Yield (monthly avg)"))
    fig_real.update_yaxes(ticksuffix="%")

    render_two_col([
        ("SORA Curve Snapshots", fig_sora_curve, pd.DataFrame(sora_snap_rows).T if sora_snap_rows else None),
        ("SGS Yield Curve Snapshots", fig_yc, pd.DataFrame(snap_rows).T if snap_rows else None),
        ("SGS Curve Changes", fig_yc_chg, None),
        ("SGS Benchmark Yields", fig_yields, clip(sgs)),
        ("SGS Curve Spreads", fig_spreads, clip(pd.DataFrame(SPREADS))),
        ("SGS 1Y Rolldown", fig_roll, roll_df),
        ("SGS Carry and Roll", fig_cr, cr_df),
        ("SGS vs UST Spreads", fig_vs_ust, clip(pd.DataFrame({"2Y": sgs_ust_2, "10Y": sgs_ust_10}))),
        ("SGS Beta to UST", fig_ust_beta, clip(ust_beta) if not ust_beta.empty else None),
        ("Ex-Post Real 10Y", fig_real, clip(real_df) if not real_df.empty else None),
    ])
    if not cr_df.empty:
        st.dataframe(cr_df.style.format({c: "{:.2f}" for c in cr_df.columns if c != "Tenor"}),
                     hide_index=True, use_container_width=True)
        st.caption("All in bps of yield over 3 months. Carry = (yield − 3M compounded SORA) × 3/12, divided by modified "
                   "duration to turn that price return into the yield rise it offsets; roll = yield pickup from rolling 3 months down "
                   "today's curve; breakeven ratio = (carry + roll) / 3M realized yield vol - how many standard "
                   "deviations of adverse move the position can absorb over 3 months. Compounded SORA is "
                   "backward-looking, so this is a funding proxy rather than a traded term rate.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Funding & Liquidity
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Funding & Liquidity")

    fig_mm = go.Figure()
    for s_, n_, c_, d_ in [(sora, "SORA", "#90a4d4", None), (csora1, "1M Comp. SORA", "#42a5f5", "dot"),
                           (csora3, "3M Comp. SORA", "#26a69a", "dot"), (csora6, "6M Comp. SORA", "#9ccc65", "dot"),
                           (bill6, "6M T-Bill", "#ff9800", None), (bill12, "1Y T-Bill", "#ef5350", None)]:
        zline(fig_mm, s_, n_, color=c_, unit="%", dash=d_, width=1.3)
    add_mps_lines(fig_mm)
    fig_mm.update_layout(**base_layout("SGD Money Markets — SORA, Compounded SORA, T-Bills"))
    fig_mm.update_yaxes(ticksuffix="%")

    fig_ss = go.Figure()
    zline(fig_ss, sora, "SORA", color="#90a4d4", unit="%")
    zline(fig_ss, sofr, "SOFR", color="#ef5350", unit="%")
    zline(fig_ss, sora_sofr, "SORA − SOFR (bps)", color="#ff9800", unit=" bps", fmt=".0f", yaxis="y2", dash="dot")
    fig_ss.update_layout(**dual_axis_layout("SORA vs SOFR & Spread", "Rate (%)", "Spread (bps)"))

    fig_bill_ois = go.Figure()
    bill_ois = _spread(bill6, csora6)
    zline(fig_bill_ois, bill_ois, "6M T-Bill − 6M Comp. SORA", color="#26a69a", unit=" bps", fmt=".0f")
    fig_bill_ois.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_bill_ois.update_layout(**base_layout("6M T-Bill vs 6M Compounded SORA (bps) — forward vs realised funding"))
    fig_bill_ois.update_yaxes(ticksuffix=" bps")

    fig_sora_vol = go.Figure()
    sv = pd.DataFrame()
    if not sora_all.empty and "SORA Volume" in sora_all:
        sv = clip(sora_all[["SORA Volume", "SORA High", "SORA Low"]].dropna())
        fig_sora_vol.add_trace(go.Bar(x=sv.index, y=sv["SORA Volume"], name="Volume (S$M)", marker_color="rgba(144,164,212,0.5)"))
        fig_sora_vol.add_trace(go.Scatter(x=sv.index, y=(sv["SORA High"] - sv["SORA Low"]) * 100, name="High − Low (bps)",
                                          yaxis="y2", line=dict(color="#ff9800", width=1.2)))
    fig_sora_vol.update_layout(**dual_axis_layout("SORA Transaction Volume & Intraday Dispersion", "S$ Million", "bps"))

    with st.spinner("Loading money supply & bank loans…"):
        ms = {n_: fetch_singstat("M701111", n_, series_no=sn) for n_, sn in [("M1", "1.1.1"), ("M2", "1.1"), ("M3", "1")]}
        loans = {n_: fetch_singstat("M701091", n_, series_no=sn) for n_, sn in
                 [("Total Loans", "1"), ("Business Loans", "1.1"), ("Housing Loans", "1.2.1")]}
        gov_revenue = fetch_singstat("M130501", "Government Operating Revenue")

    fig_ms = go.Figure()
    ms_yoy = pd.DataFrame({k_: v_[k_].pct_change(12) * 100 for k_, v_ in ms.items() if not v_.empty})
    for c_, col in zip(ms_yoy.columns, ["#42a5f5", "#26a69a", "#ff9800"]):
        zline(fig_ms, ms_yoy[c_], f"{c_} YoY", freq="M", color=col, unit="%")
    fig_ms.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_ms.update_layout(**base_layout("Money Supply Growth — M1 / M2 / M3 YoY %"))
    fig_ms.update_yaxes(ticksuffix="%")

    fig_loans = go.Figure()
    loans_yoy = pd.DataFrame({k_: v_[k_].pct_change(12) * 100 for k_, v_ in loans.items() if not v_.empty})
    for c_, col in zip(loans_yoy.columns, ["#e0e0e0", "#42a5f5", "#ef5350"]):
        zline(fig_loans, loans_yoy[c_], f"{c_} YoY", freq="M", color=col, unit="%")
    fig_loans.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_loans.update_layout(**base_layout("Commercial Bank Loans to Residents — YoY %"))
    fig_loans.update_yaxes(ticksuffix="%")

    fig_reserves = go.Figure()
    if not resv_df.empty:
        rc = clip(resv_df)
        fig_reserves.add_trace(go.Scatter(x=rc.index, y=rc["Reserves (S$M)"] / 1000, name="S$ Billion",
                                          line=dict(color="#42a5f5"), fill="tozeroy", fillcolor="rgba(66,165,245,0.15)"))
        fig_reserves.add_trace(go.Scatter(x=rc.index, y=rc["Reserves (US$B)"], name="US$ Billion", yaxis="y2",
                                          line=dict(color="#ff9800", dash="dot")))
    fig_reserves.update_layout(**dual_axis_layout("Official Foreign Reserves", "S$ Billion", "US$ Billion"))

    fig_gov_rev = go.Figure()
    gov_c = clip(gov_revenue)
    if not gov_c.empty:
        fig_gov_rev.add_trace(go.Bar(x=gov_c.index, y=gov_c["Government Operating Revenue"], name="Monthly",
                                     marker_color="rgba(237,161,0,0.45)"))
        r12 = clip(gov_revenue["Government Operating Revenue"].rolling(12).sum() / 12)
        fig_gov_rev.add_trace(go.Scatter(x=r12.index, y=r12.values, name="12M avg", line=dict(color="#eda100", width=2)))
    fig_gov_rev.update_layout(**base_layout("Government Operating Revenue (S$M, monthly + 12M avg)"))

    render_two_col([
        ("SGD Money Markets", fig_mm, clip(sora_all)),
        ("SORA vs SOFR", fig_ss, clip(sora_sofr.to_frame("SORA-SOFR bps"))),
        ("Bill vs Comp SORA", fig_bill_ois, clip(bill_ois.to_frame("bps"))),
        ("SORA Volume", fig_sora_vol, sv),
        ("Money Supply Growth", fig_ms, clip(ms_yoy)),
        ("Bank Loans Growth", fig_loans, clip(loans_yoy)),
        ("Official Foreign Reserves", fig_reserves, clip(resv_df) if not resv_df.empty else None),
        ("Government Operating Revenue", fig_gov_rev, gov_c),
    ])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Supply & Auctions
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Supply & Auctions")
    with st.spinner("Loading SGS auction results…"):
        auctions = load_sgs_auctions()
        conc = auction_concessions(auctions, sgs) if not sgs.empty else pd.DataFrame()
    tenor_bucket = _sg_true_original_tenor_bucket(auctions)

    # Recent auctions table
    st.markdown('<div class="section-header">Recent SGS Bond & T-Bill Auctions</div>', unsafe_allow_html=True)
    if not conc.empty:
        c_ = conc.copy()
        c_["Tenor"] = c_["issue_code"].map(tenor_bucket)
        c_ = c_.sort_values("auction_date")
        c_["BTC z (vs last 12 same tenor)"] = c_.groupby("Tenor")["bid_to_cover"].transform(
            lambda s: (s - s.rolling(12, min_periods=4).mean().shift()) / s.rolling(12, min_periods=4).std().shift())
        recent = c_.sort_values("auction_date", ascending=False).head(25)
        tbl = pd.DataFrame({
            "Date": recent["auction_date"].dt.strftime("%Y-%m-%d"), "Issue": recent["issue_code"].str.strip(),
            "Type": recent["bill_bond_ind"].str.title(), "Tenor": recent["Tenor"],
            "Reopening": recent["reopened_issue"], "Size (S$B)": recent["total_amt_allot"].astype(float) / 1000,
            "BTC": recent["bid_to_cover"], "BTC z": recent["BTC z (vs last 12 same tenor)"],
            "Cutoff %": recent["cutoff_yield"], "Median %": pd.to_numeric(recent["median_yield"], errors="coerce"),
            "Tail (bps)": recent["Tail (bps)"], "% at Cutoff": recent["pct_cmpt_appls_cutoff"],
            "Concession (bps)": recent["Concession (bps)"],
        })
        sty = (tbl.style.format({"Size (S$B)": "{:.2f}", "BTC": "{:.2f}", "BTC z": "{:+.2f}", "Cutoff %": "{:.2f}",
                                 "Median %": "{:.2f}", "Tail (bps)": "{:.0f}", "% at Cutoff": "{:.1f}",
                                 "Concession (bps)": "{:+.1f}"}, na_rep="—")
               .map(_zcolor, subset=["BTC z"])
               .map(lambda v: _zcolor(v / 5) if pd.notna(v) else "color:#5f6b7e", subset=["Concession (bps)"]))
        st.dataframe(sty, hide_index=True, use_container_width=True, height=36 * 12)
        st.caption("Tail = cutoff − median yield. Concession = cutoff yield vs the previous day's benchmark curve "
                   "interpolated at the issue's maturity (+ = cleared cheap to secondary; rough for off-the-run "
                   "reopenings). BTC z compares each auction with the prior 12 auctions of the same original tenor.")
        csv_download(tbl, "sgs_recent_auctions")

    bonds_hist = clip(conc.set_index("auction_date")) if not conc.empty else pd.DataFrame()
    if not bonds_hist.empty:
        bonds_hist = bonds_hist[bonds_hist["bill_bond_ind"] == "bond"].copy()
        bonds_hist["Tenor"] = bonds_hist["issue_code"].map(tenor_bucket)
    fig_conc = go.Figure()
    fig_tail = go.Figure()
    if not bonds_hist.empty:
        for t in sorted(bonds_hist["Tenor"].dropna().unique(), key=lambda x: _SG_LADDER_ORDER.get(x, 999)):
            d_ = bonds_hist[bonds_hist["Tenor"] == t]
            fig_conc.add_trace(go.Bar(x=d_.index, y=d_["Concession (bps)"], name=t,
                                      customdata=d_["issue_code"], hovertemplate="%{x|%Y-%m-%d} %{customdata}: %{y:+.1f} bps<extra></extra>"))
            fig_tail.add_trace(go.Scatter(x=d_.index, y=d_["Tail (bps)"], name=t, mode="markers",
                                          marker=dict(size=7 + 3 * (d_["bid_to_cover"].fillna(1) - 1).clip(0, 3)),
                                          customdata=np.column_stack([d_["issue_code"], d_["bid_to_cover"]]),
                                          hovertemplate="%{x|%Y-%m-%d} %{customdata[0]}<br>Tail %{y:.0f} bps · BTC %{customdata[1]:.2f}<extra></extra>"))
    fig_conc.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_conc.update_layout(**base_layout("SGS Bond Auction Concession vs Secondary Curve (bps)"))
    fig_tail.update_layout(**base_layout("SGS Bond Auction Tails (bps, marker size = bid-to-cover)"))

    # Outstanding ladder with the live curve overlaid
    _, sg_outstanding_summary = get_sg_outstanding_by_remaining_maturity(auctions)
    sg_ladder_labels = list(SG_MATURITY_LADDER.keys())
    sg_pivot = sg_outstanding_summary.pivot_table(index="maturity_bucket", columns="category", values="amt_bil",
                                                  aggfunc="sum").reindex(sg_ladder_labels).fillna(0)
    fig_sg_outstanding = go.Figure()
    for cat in sg_pivot.columns:
        if sg_pivot[cat].sum() > 0:
            fig_sg_outstanding.add_trace(go.Bar(x=sg_ladder_labels, y=sg_pivot[cat], name=cat,
                                                marker_color=SG_CATEGORY_COLORS.get(cat, "#9e9e9e")))
    if not curve.empty:
        interp_ = curve_interp(curve.iloc[-1], list(SG_MATURITY_LADDER.values()))
        fig_sg_outstanding.add_trace(go.Scatter(x=sg_ladder_labels, y=interp_, name="Yield (latest)", yaxis="y2",
                                                mode="lines+markers", line=dict(color="cyan", width=2), marker=dict(size=4)))
    fig_sg_outstanding.update_layout(**dual_axis_layout(
        f"Outstanding SGS & T-Bills by Remaining Maturity (S${sg_pivot.values.sum():,.0f}B)", "S$ Billion", "Yield (%)"))
    fig_sg_outstanding.update_layout(barmode="stack", yaxis2=dict(ticksuffix="%"))

    net_issuance_days = st.select_slider("Net Issuance window (days)", options=[30, 60, 90, 180], value=90, key="sg_net_issuance_days")
    net_issuance_df = get_sg_net_issuance(auctions, net_issuance_days)
    fig_net_issuance = go.Figure()
    if not net_issuance_df.empty:
        fig_net_issuance.add_trace(go.Bar(x=net_issuance_df["tenor_bucket"], y=net_issuance_df["Issuance"], name="Issued", marker_color="#26a69a"))
        fig_net_issuance.add_trace(go.Bar(x=net_issuance_df["tenor_bucket"], y=-net_issuance_df["Maturing"], name="Maturing", marker_color="#ef5350"))
        fig_net_issuance.add_trace(go.Scatter(x=net_issuance_df["tenor_bucket"], y=net_issuance_df["Net"], name="Net",
                                              mode="lines+markers", line=dict(color="#90a4d4", width=2)))
    fig_net_issuance.update_layout(**base_layout(
        f"Net Issuance — Issued (past {net_issuance_days}d) vs Maturing (next {net_issuance_days}d), "
        f"Net S${net_issuance_df['Net'].sum() if not net_issuance_df.empty else 0:,.1f}B"), barmode="relative")

    # Gross bond issuance by year & tenor
    gb = auctions[auctions["bill_bond_ind"] == "bond"].copy()
    gb["Tenor"] = gb["issue_code"].map(tenor_bucket)
    gb["year"] = gb["auction_date"].dt.year
    gb = gb[gb["year"] >= pd.Timestamp.today().year - 12]
    gb_p = gb.pivot_table(index="year", columns="Tenor", values="total_amt_allot", aggfunc="sum").fillna(0) / 1000
    gb_p = gb_p[sorted(gb_p.columns, key=lambda x: _SG_LADDER_ORDER.get(x, 999))]
    fig_gross = go.Figure()
    for t in gb_p.columns:
        fig_gross.add_trace(go.Bar(x=gb_p.index.astype(str), y=gb_p[t], name=t))
    fig_gross.update_layout(**base_layout("Gross SGS Bond Issuance by Year & Original Tenor (S$B, current year YTD)"),
                            barmode="stack")

    bc_type = st.selectbox("Bid-to-cover: instrument type", sorted(auctions["bill_bond_ind"].dropna().unique()), key="sg_bc_type")
    bc_hist = auctions[(auctions["bill_bond_ind"] == bc_type) & auctions["bid_to_cover"].notna() &
                       (auctions["auction_date"] >= START) & (auctions["auction_date"] <= END)].sort_values("auction_date").copy()
    bc_hist["tenor_bucket"] = bc_hist["issue_code"].map(tenor_bucket)
    fig_btc = go.Figure()
    for term in sorted(bc_hist["tenor_bucket"].dropna().unique(), key=lambda t: _SG_LADDER_ORDER.get(t, 999)):
        term_df = bc_hist[bc_hist["tenor_bucket"] == term]
        fig_btc.add_trace(go.Scatter(x=term_df["auction_date"], y=term_df["bid_to_cover"], mode="lines+markers", name=term, marker=dict(size=4)))
    fig_btc.update_layout(**base_layout(f"Bid-to-Cover Ratio — {bc_type.title()}s"))

    render_two_col([
        ("Auction Concession", fig_conc, bonds_hist[["issue_code", "Tenor", "Concession (bps)"]] if not bonds_hist.empty else None),
        ("Auction Tails", fig_tail, bonds_hist[["issue_code", "Tenor", "Tail (bps)", "bid_to_cover"]] if not bonds_hist.empty else None),
        ("Outstanding SGS by Remaining Maturity", fig_sg_outstanding, sg_pivot.reset_index()),
        ("Net Issuance", fig_net_issuance, net_issuance_df),
        ("Gross Bond Issuance", fig_gross, gb_p.reset_index()),
        ("Bid-to-Cover Trend", fig_btc, bc_hist[["auction_date", "issue_code", "tenor_bucket", "bid_to_cover"]]),
    ])

    st.markdown('<div class="section-header">Upcoming SGS / T-Bill Issuance</div>', unsafe_allow_html=True)
    st.caption("MAS doesn't publish the offering size at announcement - only once the auction closes - so this "
               "shows tenor and dates only.")
    cal_up = load_sgs_issuance_calendar()
    days_ahead_bonds = st.select_slider("Forward-looking window (days)", options=[30, 60, 90, 180], value=90, key="sg_issuance_days")
    _today = pd.Timestamp.today().normalize()
    upcoming_cal = cal_up[(cal_up["auction_date"] >= _today) &
                          (cal_up["auction_date"] <= _today + pd.Timedelta(days=days_ahead_bonds))].sort_values("auction_date")
    st.dataframe(pd.DataFrame({
        "Auction Date": upcoming_cal["auction_date"].dt.strftime("%Y-%m-%d (%a)"),
        "Issue Date": upcoming_cal["issue_date"].dt.strftime("%Y-%m-%d"),
        "Issue Code": upcoming_cal["issue_code"], "Tenor": upcoming_cal["auction_tenor_formatted"],
        "Type": upcoming_cal["agency_custom_categories"]}), hide_index=True, use_container_width=True)
    csv_download(upcoming_cal, "sgs_upcoming_issuance")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Prices
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Prices")
    with st.spinner("Loading price data…"):
        cpi  = mom_yoy(fetch_singstat("M213751", "CPI"), "CPI", 12)
        core = mom_yoy(fetch_singstat("M213891", "Core"), "Core", 12)
        rsi  = mom_yoy(fetch_singstat("M602122", "Retail Sales"), "Retail Sales", 12)

        # CPI components - the 10 major expenditure groups SingStat publishes under the same
        # resourceId (M213751) already used for headline CPI above, just different series rows.
        CPI_GROUPS_SG = [
            ("Food",                          "1.0"),
            ("Clothing & Footwear",           "1.02"),
            ("Housing & Utilities",           "1.03"),
            ("Household Durables & Services", "1.04"),
            ("Health",                        "1.05"),
            ("Transport",                     "1.06"),
            ("Info & Communication",          "1.07"),
            ("Recreation, Sport & Culture",   "1.08"),
            ("Education",                     "1.09"),
            ("Miscellaneous",                 "1.10"),
        ]
        cpi_components_sg = {
            label: mom_yoy(fetch_singstat("M213751", label, series_no=sno), label, 12)
            for label, sno in CPI_GROUPS_SG
        }

    fig_cpi = go.Figure()
    for col in pd.concat([cpi, core], axis=1).columns:
        full = pd.concat([cpi, core], axis=1)[col]
        if "YoY" in col:
            zline(fig_cpi, full, col, freq="M", unit="%", width=1.8)
        else:
            src = clip(full.dropna())
            fig_cpi.add_trace(go.Scatter(x=src.index, y=src.values, name=col, mode="lines", yaxis="y2",
                                         line=dict(width=1, dash="dot")))
    fig_cpi.update_layout(**dual_axis_layout("Headline CPI vs MAS Core Inflation (hover: 36M z-score)", "YoY %", "MoM %"))

    # Momentum: 3M-average MoM annualised vs YoY. SingStat CPI is not seasonally adjusted, so
    # this is noisier than a US-style SA 3m/3m - read the direction, not single prints.
    fig_mom = go.Figure()
    mom_df = pd.DataFrame()
    if "CPI MoM %" in cpi and "Core MoM %" in core:
        mom_df = pd.DataFrame({
            "Core 3M ann.": ((1 + core["Core MoM %"] / 100).rolling(3).apply(np.prod, raw=True) ** 4 - 1) * 100,
            "Core 6M ann.": ((1 + core["Core MoM %"] / 100).rolling(6).apply(np.prod, raw=True) ** 2 - 1) * 100,
            "Core YoY": core["Core YoY %"],
            "Headline 3M ann.": ((1 + cpi["CPI MoM %"] / 100).rolling(3).apply(np.prod, raw=True) ** 4 - 1) * 100,
        })
        for c_, col, d_ in [("Core 3M ann.", "#ff9800", None), ("Core 6M ann.", "#26a69a", None),
                            ("Core YoY", "#e0e0e0", "dot"), ("Headline 3M ann.", "#ef5350", "dot")]:
            zline(fig_mom, mom_df[c_], c_, freq="M", color=col, unit="%", dash=d_)
    fig_mom.add_hline(y=2, line_dash="dash", line_color="#555", annotation_text="2%", annotation_position="top left")
    fig_mom.update_layout(**base_layout("Inflation Momentum — 3M / 6M Annualised vs YoY (NSA)"))
    fig_mom.update_yaxes(ticksuffix="%")

    fig_rsi = go.Figure()
    src = clip(rsi)
    for col in rsi.columns:
        ax = "y2" if "MoM" in col else "y"
        fig_rsi.add_trace(go.Scatter(x=src.index, y=src[col], name=col, mode="lines", yaxis=ax))
    fig_rsi.update_layout(**dual_axis_layout("Retail Sales Index (MoM & YoY)", "YoY %", "MoM %"))

    # CPI components - YoY % history (all 10 groups) + latest MoM/YoY snapshot, same pairing
    # as the US dashboard's CPI Components charts.
    fig_cpi_comp_hist = go.Figure()
    for label, _ in CPI_GROUPS_SG:
        df_c = clip(cpi_components_sg[label])
        col = f"{label} YoY %"
        if not df_c.empty and col in df_c.columns:
            fig_cpi_comp_hist.add_trace(go.Scatter(x=df_c.index, y=df_c[col], name=label, mode="lines"))
    fig_cpi_comp_hist.update_layout(**base_layout("CPI Components — YoY %"))
    fig_cpi_comp_hist.update_yaxes(ticksuffix="%")

    comp_rows_sg = []
    for label, _ in CPI_GROUPS_SG:
        df_c = cpi_components_sg[label]
        mom_col, yoy_col = f"{label} MoM %", f"{label} YoY %"
        if df_c.empty or mom_col not in df_c.columns:
            continue
        mom_s, yoy_s = df_c[mom_col].dropna(), df_c[yoy_col].dropna()
        if mom_s.empty or yoy_s.empty:
            continue
        comp_rows_sg.append({"Component": label, "MoM %": mom_s.iloc[-1], "YoY %": yoy_s.iloc[-1], "As Of": df_c.index[-1]})
    comp_df_sg = pd.DataFrame(comp_rows_sg).sort_values("YoY %", ascending=True)
    comp_latest_date_sg = comp_df_sg["As Of"].max().strftime("%b %Y") if not comp_df_sg.empty else ""
    comp_df_sg = comp_df_sg.drop(columns="As Of")

    fig_cpi_comp_snap = go.Figure()
    fig_cpi_comp_snap.add_trace(go.Bar(y=comp_df_sg["Component"], x=comp_df_sg["YoY %"], name="YoY %",
                                        orientation="h", marker_color="#ef5350"))
    fig_cpi_comp_snap.add_trace(go.Bar(y=comp_df_sg["Component"], x=comp_df_sg["MoM %"], name="MoM %",
                                        orientation="h", marker_color="#ff9800"))
    fig_cpi_comp_snap.update_layout(**base_layout(f"CPI Components — Latest MoM & YoY % ({comp_latest_date_sg})", height=420))
    fig_cpi_comp_snap.update_layout(barmode="group")
    fig_cpi_comp_snap.update_xaxes(ticksuffix="%")

    render_two_col([
        ("CPI vs Core Inflation", fig_cpi, clip(pd.concat([cpi, core], axis=1))),
        ("Inflation Momentum", fig_mom, clip(mom_df) if not mom_df.empty else None),
        ("Retail Sales Index", fig_rsi, clip(rsi)),
        ("CPI Components History", fig_cpi_comp_hist, pd.concat([cpi_components_sg[l] for l, _ in CPI_GROUPS_SG], axis=1)),
        ("CPI Components Snapshot", fig_cpi_comp_snap, comp_df_sg),
    ])

    st.markdown('<div class="section-header">CPI Breakdown — Price-Sensitive Drivers</div>', unsafe_allow_html=True)
    with st.spinner("Loading CPI components from SingStat…"):
        cpi_idx_sg = fetch_singstat_cpi_components()
    if cpi_idx_sg.empty or "1" not in cpi_idx_sg.columns:
        st.info("CPI component data unavailable right now.")
    else:
        cpi_all_sg = cpi_idx_sg["1"]
        cpi_yoy_sg = cpi_idx_sg.pct_change(12) * 100
        cpi_mom_sg = cpi_idx_sg.pct_change(1) * 100
        CONTRIB_FROM_SG = pd.Timestamp("2025-01-01")  # first month whose year-ago value is inside the 2024 base

        def _cpi_contrib_sg(code):
            # pp of headline YoY: weight x change in the item's index / prior-year headline index.
            c = CPI_W_SG[code] / 10000 * (cpi_idx_sg[code] - cpi_idx_sg[code].shift(12)) / cpi_all_sg.shift(12) * 100
            return c[c.index >= CONTRIB_FROM_SG]

        def _last_sg(s):
            s = s.dropna()
            return s.iloc[-1] if not s.empty else np.nan

        cpi_lab_sg = cpi_all_sg.dropna().index[-1].strftime("%b %Y")
        st.caption(f"Latest CPI month: {cpi_lab_sg}. Index levels from SingStat (2024=100); basket weights are SingStat's official 2024-base "
                   f"weights. Contribution = weight x change in the item's index / prior-year headline index - group contributions sum to "
                   f"headline YoY exactly. Contributions start Jan 2025, the first month whose year-ago value is inside the 2024 base "
                   f"(earlier history is chain-linked from the old basket). Imputed rentals are ~21% of the basket; MAS Core excludes "
                   f"Accommodation and Private Transport.")

        fig_sg_contrib = go.Figure()
        contrib_cols_sg = {}
        for i, g in enumerate(CPI_MAIN_SG):
            s_c = clip(_cpi_contrib_sg(g).dropna().round(3))
            contrib_cols_sg[f"{CPI_NAME_SG[g]} (pp)"] = s_c
            fig_sg_contrib.add_trace(go.Bar(x=s_c.index, y=s_c, name=f"{CPI_NAME_SG[g]} ({CPI_W_SG[g] / 100:.1f}%)",
                                             marker_color=CPI_GROUP_COLORS_SG[i]))
        h_c = clip(cpi_yoy_sg["1"].dropna().round(3))
        h_c = h_c[h_c.index >= CONTRIB_FROM_SG]
        contrib_cols_sg["Headline CPI YoY %"] = h_c
        fig_sg_contrib.add_trace(go.Scatter(x=h_c.index, y=h_c, name="Headline CPI YoY", mode="lines",
                                             line=dict(color="#ffffff", width=2)))
        fig_sg_contrib.update_layout(**base_layout("Contribution to CPI YoY by Division (pp)"))
        fig_sg_contrib.update_layout(barmode="relative")
        fig_sg_contrib.update_yaxes(ticksuffix="pp")

        sens_lines_sg = [("1.06.1.5", "#ba68c8"), ("1.03.2.3", "#ff9800"), ("1.06.3.1", "#4fc3f7"), ("1.06.1.1", "#ef5350"),
                         ("1.11.3", "#ffd54f"), ("1.08.7.3", "#f06292"), ("1.03.1.1", "#9ccc65")]
        fig_sg_sens = go.Figure()
        sens_cols_sg = {}
        for code, color in sens_lines_sg:
            s_y = clip(cpi_yoy_sg[code].dropna().round(3))
            sens_cols_sg[f"{CPI_NAME_SG[code]} YoY %"] = s_y
            fig_sg_sens.add_trace(go.Scatter(x=s_y.index, y=s_y, mode="lines", line=dict(color=color, width=1.6),
                                              name=f"{CPI_NAME_SG[code]} ({CPI_W_SG[code] / 100:.1f}%)"))
        fig_sg_sens.add_hline(y=0, line_dash="dot", line_color="#555")
        fig_sg_sens.update_layout(**base_layout("Price-Sensitive Areas — YoY %"))
        fig_sg_sens.update_yaxes(ticksuffix="%")

        # Diffusion: how broad-based inflation is across the >=1%-weight classes (unweighted share)
        diff_codes = [k for kids in CPI_KIDS_SG.values() for k in kids if k in cpi_yoy_sg.columns]
        cls_yoy = cpi_yoy_sg[diff_codes].dropna(how="all")
        diffusion = pd.DataFrame({
            "% of classes YoY > 2%": (cls_yoy > 2).sum(axis=1) / cls_yoy.notna().sum(axis=1) * 100,
            "% of classes YoY > 0%": (cls_yoy > 0).sum(axis=1) / cls_yoy.notna().sum(axis=1) * 100,
        }).dropna()
        fig_diff = go.Figure()
        zline(fig_diff, diffusion["% of classes YoY > 2%"], "% of classes YoY > 2%", freq="M", color="#ef5350", unit="%", fmt=".0f")
        zline(fig_diff, diffusion["% of classes YoY > 0%"], "% of classes YoY > 0%", freq="M", color="#26a69a", unit="%", fmt=".0f")
        fig_diff.update_layout(**base_layout(f"CPI Diffusion — Breadth Across {len(diff_codes)} Major Classes"))
        fig_diff.update_yaxes(ticksuffix="%", range=[0, 100])

        render_two_col([
            ("SG CPI Contribution by Division", fig_sg_contrib, pd.DataFrame(contrib_cols_sg)),
            ("SG CPI Price-Sensitive YoY", fig_sg_sens, pd.DataFrame(sens_cols_sg)),
            ("SG CPI Diffusion", fig_diff, clip(diffusion)),
        ])

        snap_rows_sg = []
        for g in CPI_MAIN_SG:
            snap_rows_sg.append({"Component": f"{CPI_NAME_SG[g].upper()} ({CPI_W_SG[g] / 100:.1f}%)",
                                 "YoY %": _last_sg(cpi_yoy_sg[g]), "MoM %": _last_sg(cpi_mom_sg[g]), "is_parent": True})
            for k in CPI_KIDS_SG[g]:
                snap_rows_sg.append({"Component": f"     {CPI_NAME_SG[k]} ({CPI_W_SG[k] / 100:.1f}%)",
                                     "YoY %": _last_sg(cpi_yoy_sg[k]), "MoM %": _last_sg(cpi_mom_sg[k]), "is_parent": False})
        snap_df_sg = pd.DataFrame(snap_rows_sg).iloc[::-1]
        fig_sg_snap = go.Figure()
        fig_sg_snap.add_trace(go.Bar(y=snap_df_sg["Component"], x=snap_df_sg["YoY %"].round(2), name="YoY %", orientation="h",
                                      marker_color=["#1f8f6b" if p else "#26a69a" for p in snap_df_sg["is_parent"]]))
        fig_sg_snap.add_trace(go.Bar(y=snap_df_sg["Component"], x=snap_df_sg["MoM %"].round(2), name="MoM %", orientation="h",
                                      marker_color=["#3568c9" if p else "#80cbc4" for p in snap_df_sg["is_parent"]]))
        fig_sg_snap.update_layout(**base_layout(f"CPI Components — Divisions + Classes ({cpi_lab_sg})", height=900))
        fig_sg_snap.update_layout(barmode="group")
        fig_sg_snap.update_xaxes(ticksuffix="%")
        fig_sg_snap.update_yaxes(tickfont=dict(size=10))

        drv_sg = pd.DataFrame({"code": CPI_SENSITIVE_SG})
        drv_sg["Item"] = drv_sg["code"].map(CPI_NAME_SG)
        drv_sg["Weight %"] = drv_sg["code"].map(lambda c: CPI_W_SG[c] / 100)
        drv_sg["YoY %"] = [_last_sg(cpi_yoy_sg[c]) for c in CPI_SENSITIVE_SG]
        drv_sg["YoY 3M ago %"] = [cpi_yoy_sg[c].dropna().iloc[-4] if len(cpi_yoy_sg[c].dropna()) > 3 else np.nan for c in CPI_SENSITIVE_SG]
        drv_sg["Contribution (pp)"] = [_last_sg(_cpi_contrib_sg(c)) for c in CPI_SENSITIVE_SG]
        drv_sg = drv_sg.sort_values("Contribution (pp)")
        fig_sg_drv = go.Figure(go.Bar(
            y=drv_sg["Item"], x=drv_sg["Contribution (pp)"].round(3), orientation="h",
            marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in drv_sg["Contribution (pp)"]],
            text=[f"{v:+.2f}pp" for v in drv_sg["Contribution (pp)"]], textposition="outside"))
        fig_sg_drv.update_layout(**base_layout(f"Price-Sensitive Drivers — Contribution ({cpi_lab_sg})", height=480))
        fig_sg_drv.update_xaxes(ticksuffix="pp")
        drv_table_sg = drv_sg.drop(columns="code").sort_values("Contribution (pp)", ascending=False).round(2)

        col_snap_sg, col_drv_sg = st.columns(2)
        with col_snap_sg:
            st.plotly_chart(fig_sg_snap, use_container_width=True, key="chart_sg_cpi_snapshot")
            csv_download(snap_df_sg.drop(columns="is_parent").iloc[::-1], "SG CPI Components Snapshot")
        with col_drv_sg:
            st.plotly_chart(fig_sg_drv, use_container_width=True, key="chart_sg_cpi_drivers")
            st.dataframe(drv_table_sg, hide_index=True, use_container_width=True)
            st.caption("Private Transport, Accommodation and their sub-items overlap (as do Hawker Centres / Restaurants "
                       "within food serving) - contributions here are not meant to be added together.")
            csv_download(drv_table_sg, "SG CPI Price-Sensitive Drivers")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — Growth & Labour
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Growth & Labour")
    with st.spinner("Loading growth & labour data…"):
        gdp_level = fetch_singstat("M014871", "GDP (S$M)")
        # series_no="2" = real GDP ("In Chained (2015) Dollars") - see get_summary_metrics
        # above for why series_no="1" (nominal, the default) is wrong for this chart.
        gdp_yoy   = fetch_singstat("M015631", "GDP YoY %", series_no="2")
        gdp_saar  = fetch_singstat("M015792", "GDP QoQ SAAR %")
        unemp     = fetch_singstat("M182342", "Unemployment Rate")
        emp_chg   = fetch_singstat("M183891", "Employment Change")
        job_vac   = fetch_singstat("M181641", "Job Vacancy Rate", series_no="3")

    # GDP level is quarterly (M014871), so the raw quarterly figure (~S$200B) reads far below
    # the ~S$650-790B annual GDP everyone actually knows. Trailing 4-quarter sum gives an
    # annualised run-rate instead - computed on the full series before clipping to the date
    # range, so the first visible quarter still has 3 real prior quarters behind it.
    gdp_level_annualized = gdp_level[["GDP (S$M)"]].rolling(4).sum().dropna()

    fig_gdp = go.Figure()
    g_level = clip(gdp_level_annualized)
    if not g_level.empty:
        fig_gdp.add_trace(go.Scatter(x=g_level.index, y=g_level["GDP (S$M)"] / 1000,
                                     name="GDP Level (S$B, Trailing 4Q)", line=dict(color="#26a69a"), yaxis="y",
                                     customdata=qlabels(g_level.index),
                                     hovertemplate="%{customdata}: S$%{y:.1f}B<extra></extra>"))
    g_yoy = clip(gdp_yoy)
    if not g_yoy.empty:
        fig_gdp.add_trace(go.Scatter(x=g_yoy.index, y=g_yoy["GDP YoY %"],
                                     name="YoY %", line=dict(color="#ff9800", dash="dot"), yaxis="y2",
                                     customdata=qlabels(g_yoy.index),
                                     hovertemplate="%{customdata}: %{y:.1f}%<extra></extra>"))
    fig_gdp.update_layout(**dual_axis_layout("GDP Level (Annualised) vs YoY Growth", "S$ Billion", "YoY %"))

    fig_saar = go.Figure()
    g_saar = clip(gdp_saar)
    if not g_saar.empty:
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in g_saar["GDP QoQ SAAR %"].fillna(0)]
        fig_saar.add_trace(go.Bar(x=g_saar.index, y=g_saar["GDP QoQ SAAR %"], marker_color=colors,
                                   customdata=qlabels(g_saar.index),
                                   hovertemplate="%{customdata}: %{y:.1f}%<extra></extra>"))
    fig_saar.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_saar.update_layout(**base_layout("GDP QoQ, Seasonally Adjusted Annualised Rate"))
    fig_saar.update_yaxes(ticksuffix="%")

    fig_unemp = go.Figure()
    u = clip(unemp)
    if not u.empty:
        fig_unemp.add_trace(go.Scatter(x=u.index, y=u["Unemployment Rate"], name="Unemployment Rate",
                                       line=dict(color="#ef5350"),
                                       customdata=qlabels(u.index),
                                       hovertemplate="%{customdata}: %{y:.1f}%<extra></extra>"))
    fig_unemp.update_layout(**base_layout("Unemployment Rate (Overall, Seasonally Adjusted)"))
    fig_unemp.update_yaxes(ticksuffix="%")

    fig_emp = go.Figure()
    e = clip(emp_chg)
    if not e.empty:
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in e["Employment Change"].fillna(0)]
        fig_emp.add_trace(go.Bar(x=e.index, y=e["Employment Change"], marker_color=colors,
                                  customdata=qlabels(e.index),
                                  hovertemplate="%{customdata}: %{y:,.0f}<extra></extra>"))
    fig_emp.update_layout(**base_layout("Total Employment Change (QoQ, persons)"))

    # Job Vacancy Rate - Singapore's JOLTS-openings-rate equivalent, from MOM via the same
    # Labour Market Statistics table SingStat already exposes the unemployment rate through.
    fig_job_vac = go.Figure()
    jv = clip(job_vac)
    if not jv.empty:
        fig_job_vac.add_trace(go.Scatter(x=jv.index, y=jv["Job Vacancy Rate"], name="Job Vacancy Rate",
                                         line=dict(color="#26a69a"),
                                         customdata=qlabels(jv.index),
                                         hovertemplate="%{customdata}: %{y:.1f}%<extra></extra>"))
    fig_job_vac.update_layout(**base_layout("Job Vacancy Rate"))
    fig_job_vac.update_yaxes(ticksuffix="%")

    # Contribution to real GDP YoY by industry (pp) - bars sum to headline real GDP growth
    GDP_CONTRIB_ROWS = {
        "1.1.1": "Manufacturing", "1.1.2": "Construction", "1.2.1": "Wholesale & Retail",
        "1.2.2": "Transport & Storage", "1.2.3": "Accom. & F&B", "1.2.4": "Info & Comms",
        "1.2.5": "Finance & Insurance", "1.2.6": "Real Estate & Prof. Svcs", "1.2.7": "Other Services",
        "1.1.3": "Utilities", "1.1.4": "Other Goods", "1.3": "Ownership of Dwellings", "1.4": "Taxes on Products",
    }
    gdp_contrib = fetch_singstat_multi("M015671", ("1",) + tuple(GDP_CONTRIB_ROWS))
    fig_gdp_contrib = go.Figure()
    if not gdp_contrib.empty:
        gc = clip(gdp_contrib)
        palette = ["#42a5f5", "#ff9800", "#26a69a", "#ab47bc", "#ffd54f", "#4fc3f7", "#ef5350",
                   "#9ccc65", "#8a94a6", "#5c6bc0", "#8d6e63", "#f06292", "#607d8b"]
        for (code, name), col in zip(GDP_CONTRIB_ROWS.items(), palette):
            if code in gc:
                fig_gdp_contrib.add_trace(go.Bar(x=gc.index, y=gc[code], name=name, marker_color=col,
                                                 customdata=qlabels(gc.index),
                                                 hovertemplate=f"%{{customdata}} {name}: %{{y:+.2f}}pp<extra></extra>"))
        if "1" in gc:
            fig_gdp_contrib.add_trace(go.Scatter(x=gc.index, y=gc["1"], name="Real GDP YoY", mode="lines+markers",
                                                 line=dict(color="white", width=2), customdata=qlabels(gc.index),
                                                 hovertemplate="%{customdata} GDP: %{y:.1f}%<extra></extra>"))
    fig_gdp_contrib.update_layout(**base_layout("Contribution to Real GDP YoY by Industry (pp)", height=520), barmode="relative")
    fig_gdp_contrib.update_yaxes(ticksuffix="pp")

    fig_unemp_z = go.Figure()
    if not unemp.empty:
        zline(fig_unemp_z, unemp["Unemployment Rate"], "Unemployment Rate", freq="Q", color="#ef5350", unit="%", label_fn=qlabels)
    if not job_vac.empty:
        zline(fig_unemp_z, job_vac["Job Vacancy Rate"], "Job Vacancy Rate", freq="Q", color="#26a69a", unit="%",
              label_fn=qlabels, yaxis="y2")
    fig_unemp_z.update_layout(**dual_axis_layout("Labour Market Tightness — Unemployment vs Vacancy Rate",
                                                 "Unemployment %", "Vacancy rate %"))

    st.plotly_chart(fig_gdp_contrib, use_container_width=True, key="chart_gdp_contrib")
    csv_download(gdp_contrib.rename(columns={"1": "Real GDP YoY", **GDP_CONTRIB_ROWS}), "sg_gdp_contribution")
    render_two_col([
        ("GDP Level vs YoY", fig_gdp, pd.concat([g_level, g_yoy], axis=1)),
        ("GDP QoQ SAAR", fig_saar, g_saar),
        ("Unemployment Rate", fig_unemp, u),
        ("Employment Change", fig_emp, e),
        ("Job Vacancy Rate", fig_job_vac, jv),
        ("Labour Market Tightness", fig_unemp_z, None),
    ])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — Trade & Activity
# ════════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Trade & Activity")
    st.caption("Singapore-specific external-facing indicators, plus property prices below — SingStat does "
               "publish real housing price indices (HDB resale + private residential), just at quarterly "
               "granularity rather than the US dashboard's monthly Case-Shiller.")
    with st.spinner("Loading trade & production data…"):
        nodx = mom_yoy(fetch_singstat("M451301", "NODX"), "NODX", 12)
        ipi  = mom_yoy(fetch_singstat("M355352", "IPI"), "IPI", 12)
        hdb_rpi = fetch_singstat("M212161", "HDB Resale (Public)")
        private_ppi = fetch_singstat("M212261", "Private Residential")

    fig_nodx = go.Figure()
    src = clip(nodx)
    for col in nodx.columns:
        ax = "y2" if "MoM" in col else "y"
        fig_nodx.add_trace(go.Scatter(x=src.index, y=src[col], name=col, mode="lines", yaxis=ax))
    fig_nodx.update_layout(**dual_axis_layout("Non-Oil Domestic Exports (NODX)", "YoY %", "MoM %"))

    fig_ipi = go.Figure()
    src = clip(ipi)
    for col in ipi.columns:
        ax = "y2" if "MoM" in col else "y"
        fig_ipi.add_trace(go.Scatter(x=src.index, y=src[col], name=col, mode="lines", yaxis=ax))
    fig_ipi.update_layout(**dual_axis_layout("Industrial Production Index", "YoY %", "MoM %"))

    # HDB resale vs private residential price index - both base 1Q2009=100, so they share one
    # axis honestly (same unit, same base period) rather than needing a dual-axis chart.
    fig_property = go.Figure()
    hdb_c = clip(hdb_rpi)
    priv_c = clip(private_ppi)
    if not hdb_c.empty:
        fig_property.add_trace(go.Scatter(x=hdb_c.index, y=hdb_c["HDB Resale (Public)"],
                                          name="HDB Resale (Public)", line=dict(color="#42a5f5"),
                                          customdata=qlabels(hdb_c.index),
                                          hovertemplate="%{customdata}: %{y:.1f}<extra></extra>"))
    if not priv_c.empty:
        fig_property.add_trace(go.Scatter(x=priv_c.index, y=priv_c["Private Residential"],
                                          name="Private Residential", line=dict(color="#ef5350"),
                                          customdata=qlabels(priv_c.index),
                                          hovertemplate="%{customdata}: %{y:.1f}<extra></extra>"))
    fig_property.update_layout(**base_layout("HDB Resale vs Private Residential Property Price Index (1Q2009 = 100)"))

    # NODX by destination - 3M-sum YoY smooths SG's very lumpy monthly (pharma/petchem) shipments
    NODX_MKTS = {"1.10": "China", "1.11": "US", "1.13": "EU", "1.9": "Taiwan", "1.8": "Korea",
                 "1.7": "Hong Kong", "1.6": "Japan", "1.3": "Malaysia", "1.2": "Indonesia", "1.5": "Thailand"}
    nodx_mkt = fetch_singstat_multi("M451301", tuple(NODX_MKTS))
    fig_nodx_mkt = go.Figure()
    fig_nodx_mkt_hist = go.Figure()
    nodx_mkt_yoy = pd.DataFrame()
    if not nodx_mkt.empty:
        nodx_mkt_yoy = (nodx_mkt.rolling(3).sum().pct_change(12) * 100).rename(columns=NODX_MKTS)
        latest_ = nodx_mkt_yoy.dropna(how="all").iloc[-1].dropna().sort_values()
        fig_nodx_mkt.add_trace(go.Bar(x=latest_.values, y=latest_.index, orientation="h",
                                      marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in latest_.values],
                                      text=[f"{v:+.1f}%" for v in latest_.values], textposition="outside"))
        fig_nodx_mkt.update_layout(**base_layout(
            f"NODX by Market — 3M YoY % ({nodx_mkt_yoy.dropna(how='all').index[-1]:%b %Y})", height=480))
        fig_nodx_mkt.update_xaxes(ticksuffix="%")
        for m_, col in [("China", "#ef5350"), ("US", "#42a5f5"), ("EU", "#ffd54f"), ("Taiwan", "#26a69a")]:
            zline(fig_nodx_mkt_hist, nodx_mkt_yoy[m_], m_, freq="M", color=col, unit="%", fmt=".1f")
    fig_nodx_mkt_hist.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_nodx_mkt_hist.update_layout(**base_layout("NODX to Key Markets — 3M YoY %"))
    fig_nodx_mkt_hist.update_yaxes(ticksuffix="%")

    fig_prop_yoy = go.Figure()
    prop_yoy = pd.DataFrame()
    if not hdb_rpi.empty and not private_ppi.empty:
        prop_yoy = pd.concat([hdb_rpi, private_ppi], axis=1).pct_change(4) * 100
        zline(fig_prop_yoy, prop_yoy["HDB Resale (Public)"], "HDB Resale YoY", freq="Q", color="#42a5f5", unit="%", label_fn=qlabels)
        zline(fig_prop_yoy, prop_yoy["Private Residential"], "Private Residential YoY", freq="Q", color="#ef5350", unit="%", label_fn=qlabels)
    fig_prop_yoy.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_prop_yoy.update_layout(**base_layout("Property Prices — YoY %"))
    fig_prop_yoy.update_yaxes(ticksuffix="%")

    visitors = fetch_singstat("M550001", "Visitor Arrivals")
    fig_vis = go.Figure()
    if not visitors.empty:
        vc = clip(visitors)
        fig_vis.add_trace(go.Bar(x=vc.index, y=vc["Visitor Arrivals"] / 1e6, name="Arrivals (M)", marker_color="rgba(144,164,212,0.5)"))
        v_yoy = visitors["Visitor Arrivals"].pct_change(12) * 100
        zline(fig_vis, v_yoy, "YoY %", freq="M", color="#ff9800", unit="%", fmt=".1f", yaxis="y2")
    fig_vis.update_layout(**dual_axis_layout("International Visitor Arrivals", "Millions", "YoY %"))

    render_two_col([
        ("NODX", fig_nodx, clip(nodx)),
        ("Industrial Production Index", fig_ipi, clip(ipi)),
        ("NODX by Market", fig_nodx_mkt, nodx_mkt_yoy.tail(1).T if not nodx_mkt_yoy.empty else None),
        ("NODX Key Markets History", fig_nodx_mkt_hist, clip(nodx_mkt_yoy) if not nodx_mkt_yoy.empty else None),
        ("Property Price Index", fig_property, pd.concat([hdb_c, priv_c], axis=1)),
        ("Property Price YoY", fig_prop_yoy, clip(prop_yoy) if not prop_yoy.empty else None),
        ("Visitor Arrivals", fig_vis, clip(visitors)),
    ])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 — Dividends
# ════════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Dividends")
    st.caption("Dividend seasonality for a 20-stock basket of SGX blue chips and major S-REITs — a different "
               "data domain from the rest of this dashboard (equities/corporate actions via yfinance, not "
               "SingStat or MAS).")

    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    div_years_back = st.slider("Years to include in average", min_value=1, max_value=15, value=10, key="sg_div_years")

    with st.spinner("Loading dividend history for 20 SGX stocks…"):
        div_payouts, missing_tickers = get_sg_dividend_payouts(years_back=div_years_back)

    if missing_tickers:
        st.warning(f"⚠️ No data came back for {len(missing_tickers)} of {len(SG_DIV_BASKET)} stocks this run "
                   f"(likely a transient yfinance/Yahoo Finance rate-limit — this happens more on cloud-hosted "
                   f"deployments than locally): **{', '.join(missing_tickers)}**. Charts below reflect only the "
                   f"stocks that loaded; a rerun in a bit should pick the rest back up (results are cached for "
                   f"just 1 hour specifically so this self-heals quickly).")

    if div_payouts.empty:
        st.warning("Could not load dividend data.")
    else:
        st.info("**Methodology caveat:** nominal payout = dividend/share x shares outstanding at the time. "
                "Historical share counts from yfinance only have dense coverage for some tickers (mostly the "
                "REITs, which issue new units often); the rest fall back to today's share count applied across "
                "their whole history. Treat magnitudes as directionally right, not to-the-dollar precise — "
                "especially older REIT payouts from before their unit count grew.")

        n_years = div_payouts["year"].nunique()
        month_avg = div_payouts.groupby("month")["payout_sgd_m"].sum().reindex(range(1, 13), fill_value=0) / n_years

        # Seasonality - average nominal payout per calendar month, across the whole basket
        fig_div_month = go.Figure()
        fig_div_month.add_trace(go.Bar(x=MONTH_NAMES, y=month_avg.values,
                                        marker_color="#ef5350",
                                        hovertemplate="%{x}: S$%{y:,.0f}M<extra></extra>"))
        fig_div_month.update_layout(**base_layout(f"Average Nominal Payout by Month (S$M/yr, {n_years}y basket average)"))
        fig_div_month.update_yaxes(title="S$ Million")

        # Same data by year - is the pattern stable, or drifting?
        heatmap_df = div_payouts.pivot_table(index="year", columns="month", values="payout_sgd_m", aggfunc="sum").fillna(0)
        heatmap_df = heatmap_df.reindex(columns=range(1, 13), fill_value=0)
        fig_div_heatmap = go.Figure(go.Heatmap(
            z=heatmap_df.values, x=MONTH_NAMES, y=[str(y) for y in heatmap_df.index],
            colorscale=[[0, PLOT_BG], [1, "#ef5350"]],
            text=heatmap_df.values, texttemplate="%{text:,.0f}", textfont=dict(size=10, color="#e0e0e0"),
            hovertemplate="%{y} %{x}: S$%{z:,.0f}M<extra></extra>",
            colorbar=dict(title="S$M"),
        ))
        fig_div_heatmap.update_layout(**base_layout("Nominal Payout by Month & Year (S$M)", height=380))

        # Which stocks actually drive the total
        by_ticker = (div_payouts.groupby("name")["payout_sgd_m"].sum() / n_years).sort_values(ascending=True)
        fig_div_ticker = go.Figure(go.Bar(
            x=by_ticker.values, y=by_ticker.index, orientation="h", marker_color="#90a4d4",
            hovertemplate="%{y}: S$%{x:,.0f}M<extra></extra>",
        ))
        fig_div_ticker.update_layout(**base_layout("Average Nominal Payout by Stock (S$M/yr)", height=460))
        fig_div_ticker.update_xaxes(title="S$ Million")

        render_two_col([
            ("Payout by Month", fig_div_month, month_avg.reset_index()),
            ("Payout by Month & Year", fig_div_heatmap, heatmap_df.reset_index()),
            ("Payout by Stock", fig_div_ticker, by_ticker.reset_index()),
        ])

        top3 = by_ticker.sort_values(ascending=False).head(3)
        top3_share = top3.sum() / by_ticker.sum() * 100
        st.caption(f"The three biggest payers ({', '.join(top3.index)}) account for {top3_share:.0f}% of total "
                   f"nominal payout across this basket — the May/August seasonality above is substantially a "
                   f"bank-earnings-calendar effect, not a broad market-wide pattern.")

    st.markdown('<div class="section-header">Basket Universe</div>', unsafe_allow_html=True)
    universe_df = pd.DataFrame(list(SG_DIV_BASKET.items()), columns=["Ticker", "Company"]).sort_values("Company")
    st.dataframe(universe_df, use_container_width=True, hide_index=True)
    csv_download(universe_df, "sg_dividend_universe")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 9 — Economic Calendar
# ════════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("Economic Calendar")
    st.caption("Live Investing.com economic calendar widget (same source as the US dashboard), times in GMT+8 "
               "(Singapore). Scrollable and date-navigable inside the widget itself.")

    # Investing.com country ids (verified live 2026-09-30 in the widget itself): 36 = Singapore,
    # 5 = United States, 37 = China. timeZone=113 = "(GMT +8:00) Singapore" (see usa_macro.py).
    # Investing.com tags almost every SG release as low importance, so a medium/high filter like
    # the US widget uses returns zero SGD rows - the SG-only view therefore shows all importance
    # levels, while the "SGD drivers" view adds US/China at medium/high only to avoid flooding.
    cal_view = st.radio("View", ["Singapore — all releases", "SGD drivers — SG + US + China (medium/high)"],
                        horizontal=True, key="sg_investing_view")
    countries, importance = ("36", "1,2,3") if cal_view.startswith("Singapore") else ("36,5,37", "2,3")
    INVESTING_CALENDAR_SRC = ("https://sslecal2.investing.com?"
        "columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous"
        f"&importance={importance}&features=datepicker,timezone&countries={countries}&calType=week&timeZone=113&lang=1")
    components.html(f"""
        <iframe src="{INVESTING_CALENDAR_SRC}" width="100%" height="700" frameborder="0"
                allowtransparency="true" marginwidth="0" marginheight="0"></iframe>
        <div style="font-family: Arial, Helvetica, sans-serif; text-align:right; margin-top:4px;">
            <span style="font-size: 11px; color: #888;">Real Time Economic Calendar provided by
            <a href="https://www.investing.com/" rel="nofollow" target="_blank"
               style="color:#06529D; font-weight:bold;">Investing.com</a>.</span>
        </div>
    """, height=740, scrolling=True)

    st.markdown('<div class="section-header">MAS Policy Statements & SGS Auctions (not covered by Investing.com)</div>',
                unsafe_allow_html=True)
    days_ahead = st.select_slider("Forward-looking window (days)", options=[30, 60, 90, 180], value=90, key="sg_econ_cal_days")
    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)

    with st.spinner("Loading MPS dates…"):
        mps_dates = [pd.Timestamp(d) for d in get_mps_dates()]
    future_mps = [d for d in mps_dates if today <= d <= cutoff]
    mps_df = pd.DataFrame([{"Date": d, "Event": "MAS Monetary Policy Statement", "Type": "MPS", "Detail": ""} for d in future_mps])
    if not future_mps:
        next_q_month = {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"}
        upcoming_qtrs = [m for m in [1, 4, 7, 10] if pd.Timestamp(year=today.year, month=m, day=1) >= today.replace(day=1)]
        est_month = next_q_month.get(upcoming_qtrs[0], "Jan") if upcoming_qtrs else "Jan"
        st.caption(f"No confirmed MPS date within the window — MAS meets quarterly (Jan/Apr/Jul/Oct) and only "
                   f"confirms the exact date ~2 weeks ahead; next is expected around {est_month} "
                   f"{today.year if upcoming_qtrs else today.year + 1} but no free source gives that exact day yet.")

    with st.spinner("Loading SGS auction calendar…"):
        cal = load_sgs_issuance_calendar()
    upcoming_auctions = cal[(cal["auction_date"] >= today) & (cal["auction_date"] <= cutoff)].copy()
    auction_df = pd.DataFrame([
        {"Date": row["auction_date"], "Event": f"{row['auction_tenor_formatted']} {row['agency_custom_categories'].split(',')[-1]} Auction",
         "Type": "SGS Auction", "Detail": row["issue_code"]}
        for _, row in upcoming_auctions.iterrows()
    ])

    calendar_df = pd.concat([mps_df, auction_df], ignore_index=True)
    if not calendar_df.empty:
        calendar_df = calendar_df.sort_values("Date")
        type_counts = calendar_df["Type"].value_counts()
        st.markdown(f"**{len(calendar_df)} events in the next {days_ahead} days** — "
                    + " | ".join(f"{t}: {c}" for t, c in type_counts.items()))
        display_df = calendar_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d (%a)")
        st.dataframe(display_df[["Date", "Event", "Type", "Detail"]], use_container_width=True, hide_index=True, height=500)
        csv_download(display_df, "sg_economic_calendar")
    else:
        st.info(f"No tracked MPS dates or SGS auctions in the next {days_ahead} days.")

    st.markdown('<div class="section-header">MPS History — Market Reaction & Realised S&#36;NEER Path</div>', unsafe_allow_html=True)
    def _day_move(s_, d, mult):
        s_ = s_.dropna()
        if s_.empty or d > s_.index[-1]:
            return np.nan
        before, after = s_.asof(d - pd.Timedelta(days=1)), s_[s_.index >= d]
        return (after.iloc[0] - before) * mult if (pd.notna(before) and not after.empty) else np.nan
    hist_rows = []
    sorted_mps = sorted(mps_dates)
    for i, d in enumerate(sorted_mps):
        nxt = sorted_mps[i + 1] if i + 1 < len(sorted_mps) else (neer_daily.index[-1] if not neer_daily.empty else d)
        n0, n1 = (neer_daily.asof(d), neer_daily.asof(nxt)) if not neer_daily.empty else (np.nan, np.nan)
        yrs = max((nxt - d).days, 1) / 365.25
        hist_rows.append({
            "MPS Date": d.strftime("%Y-%m-%d"),
            "USD/SGD Δ on day (%)": _day_move(usdsgd, d, 1) / usdsgd.asof(d) * 100 if not usdsgd.empty else np.nan,
            "2Y SGS Δ (bps)": _day_move(y2, d, 100), "10Y SGS Δ (bps)": _day_move(y10, d, 100),
            "S$NEER Δ to next MPS (% ann.)": ((n1 / n0) ** (1 / yrs) - 1) * 100 if pd.notna(n0) and pd.notna(n1) and n0 else np.nan,
        })
    hist_df = pd.DataFrame(hist_rows).iloc[::-1]
    st.dataframe(hist_df.style.format({c: "{:+.2f}" for c in hist_df.columns if c != "MPS Date"}, na_rep="—")
                 .map(_chgcolor, subset=[c for c in hist_df.columns if c != "MPS Date"]),
                 use_container_width=True, hide_index=True, height=400)
    st.caption("Day moves compare the first close on/after the MPS date with the prior close (MPS is released "
               "before the SG open, so this captures the reaction). \"S\\$NEER Δ to next MPS\" is the realised "
               "annualised appreciation until the following statement - a rough read of the slope MAS actually "
               "delivered in that window (uses the daily model where available, else the weekly MAS index).")
    csv_download(hist_df, "sg_mps_history")

st.markdown("---")
st.caption("Data: SingStat Table Builder · MAS (S\\$NEER, SORA, SGS benchmarks, bonds & bills) · US Treasury · "
           "NY Fed (SOFR) · yfinance (FX, dividends). Cache: 1hr FX & summary, 6hr MAS/SingStat/UST, 7d MPS dates & S\\$NEER.")
