import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import datetime
import io
import time
import re

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
            st.plotly_chart(item[1], use_container_width=True)
            if len(item) > 2 and item[2] is not None:
                csv_download(item[2], item[0])
            i += 1
        else:
            c1, c2 = st.columns(2)
            for col, item in [(c1, charts[i]), (c2, charts[i+1])]:
                with col:
                    st.plotly_chart(item[1], use_container_width=True)
                    if len(item) > 2 and item[2] is not None:
                        csv_download(item[2], item[0])
            i += 2

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

# ── Date range ────────────────────────────────────────────────────────────────
st.title("🇸🇬 SG Macro Dashboard")
st.caption("Data: SingStat Table Builder · MAS Bonds & Bills / Domestic Interest Rates · No NBER-style "
           "recession series exists for Singapore, so no recession shading here (unlike the US dashboard).")

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

# ── Summary bar ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Latest Readings</div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_summary_metrics():
    metrics = {}
    try:
        cpi = fetch_singstat("M213751", "CPI")["CPI"].dropna()
        val = cpi.pct_change(12).iloc[-1] * 100
        prev = cpi.pct_change(12).iloc[-2] * 100
        metrics["CPI YoY"] = (val, val - prev, "%")
    except Exception:
        metrics["CPI YoY"] = (None, None, "")
    try:
        core = fetch_singstat("M213891", "Core")["Core"].dropna()
        val = core.pct_change(12).iloc[-1] * 100
        prev = core.pct_change(12).iloc[-2] * 100
        metrics["Core Infl. YoY"] = (val, val - prev, "%")
    except Exception:
        metrics["Core Infl. YoY"] = (None, None, "")
    try:
        gdp = fetch_singstat("M015631", "GDP YoY")["GDP YoY"].dropna()
        metrics["GDP YoY"] = (float(gdp.iloc[-1]), float(gdp.iloc[-1] - gdp.iloc[-2]), "%")
    except Exception:
        metrics["GDP YoY"] = (None, None, "")
    try:
        u = fetch_singstat("M182342", "Unemployment")["Unemployment"].dropna()
        metrics["Unemp Rate"] = (float(u.iloc[-1]), float(u.iloc[-1] - u.iloc[-2]), "%")
    except Exception:
        metrics["Unemp Rate"] = (None, None, "")
    try:
        sora = fetch_sora(years_back=1)["SORA"].dropna()
        metrics["SORA"] = (float(sora.iloc[-1]), float(sora.iloc[-1] - sora.iloc[-2]), "%")
    except Exception:
        metrics["SORA"] = (None, None, "")
    try:
        y10 = fetch_sgs_yield("10", "10Y")["10Y"].dropna()
        metrics["10Y SGS"] = (float(y10.iloc[-1]), float(y10.iloc[-1] - y10.iloc[-2]), "%")
    except Exception:
        metrics["10Y SGS"] = (None, None, "")
    try:
        y2 = fetch_sgs_yield("2", "2Y")["2Y"].dropna()
        y5 = fetch_sgs_yield("5", "5Y")["5Y"].dropna()
        y10b = fetch_sgs_yield("10", "10Y")["10Y"].dropna()
        curve = pd.concat([y2, y5, y10b], axis=1, keys=["2Y", "5Y", "10Y"]).ffill().dropna()
        s2s10s = (curve["10Y"] - curve["2Y"]) * 100  # bps
        metrics["2s10s"] = (float(s2s10s.iloc[-1]), float(s2s10s.iloc[-1] - s2s10s.iloc[-2]), "bps")
        s2s5s10s = (2 * curve["5Y"] - curve["10Y"] - curve["2Y"]) * 100  # bps, butterfly
        metrics["2s5s10s"] = (float(s2s5s10s.iloc[-1]), float(s2s5s10s.iloc[-1] - s2s5s10s.iloc[-2]), "bps")
    except Exception:
        metrics["2s10s"] = (None, None, "")
        metrics["2s5s10s"] = (None, None, "")
    try:
        nodx = fetch_singstat("M451301", "NODX")["NODX"].dropna()
        val = nodx.pct_change(12).iloc[-1] * 100
        prev = nodx.pct_change(12).iloc[-2] * 100
        metrics["NODX YoY"] = (val, val - prev, "%")
    except Exception:
        metrics["NODX YoY"] = (None, None, "")
    try:
        rs = fetch_singstat("M602122", "Retail")["Retail"].dropna()
        val = rs.pct_change(12).iloc[-1] * 100
        prev = rs.pct_change(12).iloc[-2] * 100
        metrics["Retail Sales YoY"] = (val, val - prev, "%")
    except Exception:
        metrics["Retail Sales YoY"] = (None, None, "")
    return metrics

with st.spinner("Loading summary metrics…"):
    summary = get_summary_metrics()

items = list(summary.items())
ROW_SIZE = 5
for row_start in range(0, len(items), ROW_SIZE):
    row_items = items[row_start:row_start + ROW_SIZE]
    cols = st.columns(ROW_SIZE)
    for col, (name, (val, delta, unit)) in zip(cols, row_items):
        with col:
            if val is None:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{name}</div><div class="metric-value neutral">N/A</div></div>', unsafe_allow_html=True)
                continue
            val_str = f"{round(val, 3):.3f}{unit}"
            delta_str = f"{round(delta, 3):+.3f}{unit}" if delta is not None else ""
            delta_cls = "positive" if (delta or 0) > 0 else "negative" if (delta or 0) < 0 else "neutral"
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{name}</div>
              <div class="metric-value neutral">{val_str}</div>
              <div class="metric-delta {delta_cls}">{delta_str}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Prices",
    "Growth & Labour",
    "Trade & Production",
    "Monetary Policy",
    "Economic Calendar",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prices
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Prices")
    with st.spinner("Loading price data…"):
        cpi  = mom_yoy(fetch_singstat("M213751", "CPI"), "CPI", 12)
        core = mom_yoy(fetch_singstat("M213891", "Core"), "Core", 12)
        rsi  = mom_yoy(fetch_singstat("M602122", "Retail Sales"), "Retail Sales", 12)

    fig_cpi = go.Figure()
    for col in pd.concat([cpi, core], axis=1).columns:
        src = clip(pd.concat([cpi, core], axis=1))
        ax = "y2" if "MoM" in col else "y"
        fig_cpi.add_trace(go.Scatter(x=src.index, y=src[col], name=col, mode="lines",
                                     yaxis=ax, line=dict(width=1.5 if "YoY" in col else 1,
                                                          dash="solid" if "YoY" in col else "dot")))
    fig_cpi.update_layout(**dual_axis_layout("Headline CPI vs MAS Core Inflation", "YoY %", "MoM %"))

    fig_rsi = go.Figure()
    src = clip(rsi)
    for col in rsi.columns:
        ax = "y2" if "MoM" in col else "y"
        fig_rsi.add_trace(go.Scatter(x=src.index, y=src[col], name=col, mode="lines", yaxis=ax))
    fig_rsi.update_layout(**dual_axis_layout("Retail Sales Index (MoM & YoY)", "YoY %", "MoM %"))

    render_two_col([
        ("CPI vs Core Inflation", fig_cpi, clip(pd.concat([cpi, core], axis=1))),
        ("Retail Sales Index", fig_rsi, clip(rsi)),
    ])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Growth & Labour
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Growth & Labour")
    with st.spinner("Loading growth & labour data…"):
        gdp_level = fetch_singstat("M014871", "GDP (S$M)")
        gdp_yoy   = fetch_singstat("M015631", "GDP YoY %")
        gdp_saar  = fetch_singstat("M015792", "GDP QoQ SAAR %")
        unemp     = fetch_singstat("M182342", "Unemployment Rate")
        emp_chg   = fetch_singstat("M183891", "Employment Change")

    fig_gdp = go.Figure()
    g_level = clip(gdp_level)
    if not g_level.empty:
        fig_gdp.add_trace(go.Scatter(x=g_level.index, y=g_level["GDP (S$M)"] / 1000,
                                     name="GDP Level (S$B)", line=dict(color="#26a69a"), yaxis="y"))
    g_yoy = clip(gdp_yoy)
    if not g_yoy.empty:
        fig_gdp.add_trace(go.Scatter(x=g_yoy.index, y=g_yoy["GDP YoY %"],
                                     name="YoY %", line=dict(color="#ff9800", dash="dot"), yaxis="y2"))
    fig_gdp.update_layout(**dual_axis_layout("GDP Level vs YoY Growth", "S$ Billion", "YoY %"))

    fig_saar = go.Figure()
    g_saar = clip(gdp_saar)
    if not g_saar.empty:
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in g_saar["GDP QoQ SAAR %"].fillna(0)]
        fig_saar.add_trace(go.Bar(x=g_saar.index, y=g_saar["GDP QoQ SAAR %"], marker_color=colors))
    fig_saar.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_saar.update_layout(**base_layout("GDP QoQ, Seasonally Adjusted Annualised Rate"))
    fig_saar.update_yaxes(ticksuffix="%")

    fig_unemp = go.Figure()
    u = clip(unemp)
    if not u.empty:
        fig_unemp.add_trace(go.Scatter(x=u.index, y=u["Unemployment Rate"], name="Unemployment Rate",
                                       line=dict(color="#ef5350")))
    fig_unemp.update_layout(**base_layout("Unemployment Rate (Overall, Seasonally Adjusted)"))
    fig_unemp.update_yaxes(ticksuffix="%")

    fig_emp = go.Figure()
    e = clip(emp_chg)
    if not e.empty:
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in e["Employment Change"].fillna(0)]
        fig_emp.add_trace(go.Bar(x=e.index, y=e["Employment Change"], marker_color=colors))
    fig_emp.update_layout(**base_layout("Total Employment Change (QoQ, persons)"))

    render_two_col([
        ("GDP Level vs YoY", fig_gdp, pd.concat([g_level, g_yoy], axis=1)),
        ("GDP QoQ SAAR", fig_saar, g_saar),
        ("Unemployment Rate", fig_unemp, u),
        ("Employment Change", fig_emp, e),
    ])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trade & Production
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Trade & Production")
    st.caption("Singapore-specific external-facing indicators — the US dashboard's Housing tab has no direct "
               "equivalent here (no free, comparably granular SG property-transaction API was confirmed), so "
               "this slot covers trade & industrial activity instead.")
    with st.spinner("Loading trade & production data…"):
        nodx = mom_yoy(fetch_singstat("M451301", "NODX"), "NODX", 12)
        ipi  = mom_yoy(fetch_singstat("M355352", "IPI"), "IPI", 12)

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

    render_two_col([
        ("NODX", fig_nodx, clip(nodx)),
        ("Industrial Production Index", fig_ipi, clip(ipi)),
    ])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Monetary Policy
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Monetary Policy")
    st.info("No Fed-Funds-style hike/cut probability section here — MAS doesn't set a policy interest rate. "
             "It manages monetary policy via the S\\$NEER exchange-rate band, reviewed at scheduled Monetary "
             "Policy Statements (see the Economic Calendar tab), so there's no futures-implied-probability "
             "market to build that chart from. SORA below is the key overnight benchmark, not a policy lever. "
             "Note MAS does not disclose the band's width or slope - only the index level itself is published.")

    with st.spinner("Loading S\\$NEER…"):
        neer = fetch_sneer()
    n = clip(neer)
    fig_neer = go.Figure()
    if not n.empty:
        fig_neer.add_trace(go.Scatter(x=n.index, y=n["S$NEER"], name="S$NEER", line=dict(color="#ff9800")))
    fig_neer.update_layout(**base_layout("S$NEER — Nominal Effective Exchange Rate Index (Jan 1999 = 100, Weekly)"))

    neer_wow = pd.DataFrame(index=n.index)
    fig_neer_wow = go.Figure()
    if not n.empty:
        neer_wow["WoW %"] = (n["S$NEER"].pct_change() * 100).round(3)
        neer_wow["WoW % (4W MA)"] = neer_wow["WoW %"].rolling(4).mean().round(3)
        fig_neer_wow.add_trace(go.Bar(x=neer_wow.index, y=neer_wow["WoW %"],
                                       marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in neer_wow["WoW %"].fillna(0)],
                                       name="WoW %", opacity=0.6))
        fig_neer_wow.add_trace(go.Scatter(x=neer_wow.index, y=neer_wow["WoW % (4W MA)"],
                                           name="WoW % (4W MA)", line=dict(color="white", width=2)))
    fig_neer_wow.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_neer_wow.update_layout(**base_layout("S$NEER Week-on-Week % Change + 4W Moving Average"))
    fig_neer_wow.update_yaxes(ticksuffix="%")

    with st.spinner("Loading SORA…"):
        sora = fetch_sora(years_back=3)

    fig_sora = go.Figure()
    s = clip(sora)
    if not s.empty:
        fig_sora.add_trace(go.Scatter(x=s.index, y=s["SORA"], name="SORA", line=dict(color="#90a4d4")))
    fig_sora.update_layout(**base_layout("SORA — Singapore Overnight Rate Average"))
    fig_sora.update_yaxes(ticksuffix="%")

    TENORS = {"6M": "0.5", "1Y": "1", "2Y": "2", "5Y": "5", "10Y": "10", "15Y": "15", "20Y": "20", "30Y": "30", "50Y": "50"}
    with st.spinner("Loading SGS yields…"):
        yc = pd.concat([fetch_sgs_yield(t, label) for label, t in TENORS.items()], axis=1)
        yc = yc.ffill()

    fig_yc = go.Figure()
    if not yc.empty:
        snap_labels = {"Latest": 0, "1W Ago": -5, "1M Ago": -21, "3M Ago": -63}
        snap_colors = {"Latest": "cyan", "1W Ago": "orange", "1M Ago": "green", "3M Ago": "magenta"}
        for label, offset in snap_labels.items():
            idx = max(0, len(yc) - 1 + offset)
            snap_date = yc.index[idx]
            row_vals = yc.iloc[idx]
            fig_yc.add_trace(go.Scatter(
                x=list(TENORS.keys()), y=row_vals.values, mode="lines+markers",
                name=f"{label} ({snap_date.date()})",
                line=dict(color=snap_colors[label], width=2 if label == "Latest" else 1,
                          dash="solid" if label == "Latest" else "dash")))
    fig_yc.update_layout(**base_layout("SGS Yield Curve — Snapshots"))
    fig_yc.update_yaxes(ticksuffix="%", title="Yield")

    fig_yc_hist = go.Figure()
    for label in ["2Y", "5Y", "10Y"]:
        src = clip(yc[[label]]) if label in yc.columns else pd.DataFrame()
        if not src.empty:
            fig_yc_hist.add_trace(go.Scatter(x=src.index, y=src[label], name=label, mode="lines"))
    fig_yc_hist.update_layout(**base_layout("SGS 2Y / 5Y / 10Y Yields"))
    fig_yc_hist.update_yaxes(ticksuffix="%")

    # Spreads over time - 2s5s/2s10s/5s30s are simple slope spreads, 2s5s10s is the classic
    # butterfly (2x the belly minus both wings). All in bps, on one chart.
    fig_spreads = go.Figure()
    spread_curve = pd.DataFrame()
    if all(t in yc.columns for t in ["2Y", "5Y", "10Y", "30Y"]):
        spread_curve = yc[["2Y", "5Y", "10Y", "30Y"]].dropna()
        spreads_sg = pd.DataFrame(index=spread_curve.index)
        spreads_sg["2s5s"] = (spread_curve["5Y"] - spread_curve["2Y"]) * 100
        spreads_sg["2s10s"] = (spread_curve["10Y"] - spread_curve["2Y"]) * 100
        spreads_sg["2s5s10s"] = (2 * spread_curve["5Y"] - spread_curve["10Y"] - spread_curve["2Y"]) * 100
        spreads_sg["5s30s"] = (spread_curve["30Y"] - spread_curve["5Y"]) * 100
        spreads_sg = clip(spreads_sg)
        for col, color in [("2s5s", "#42a5f5"), ("2s10s", "#26a69a"), ("2s5s10s", "#ff9800"), ("5s30s", "#ab47bc")]:
            fig_spreads.add_trace(go.Scatter(x=spreads_sg.index, y=spreads_sg[col], name=col, mode="lines", line=dict(color=color)))
        fig_spreads.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_spreads.update_layout(**base_layout("SGS Curve Spreads — 2s5s / 2s10s / 2s5s10s / 5s30s"))
    fig_spreads.update_yaxes(ticksuffix=" bps")

    with st.spinner("Loading SGS auction results…"):
        auctions = load_sgs_auctions()

    # Outstanding by remaining maturity (nearest-tenor ladder, same buckets as the US
    # dashboard's version, extended 5Y apart past 30Y since SGS issues out to 50Y), stacked by
    # bond category, with the yield curve overlaid on a secondary axis.
    with st.spinner("Computing outstanding SGS by remaining maturity…"):
        _, sg_outstanding_summary = get_sg_outstanding_by_remaining_maturity(auctions)
    sg_ladder_labels = list(SG_MATURITY_LADDER.keys())
    sg_pivot = sg_outstanding_summary.pivot_table(index="maturity_bucket", columns="category", values="amt_bil", aggfunc="sum")
    sg_pivot = sg_pivot.reindex(sg_ladder_labels).fillna(0)

    fig_sg_outstanding = go.Figure()
    for cat in sg_pivot.columns:
        if sg_pivot[cat].sum() > 0:
            fig_sg_outstanding.add_trace(go.Bar(
                x=sg_ladder_labels, y=sg_pivot[cat], name=cat,
                marker_color=SG_CATEGORY_COLORS.get(cat, "#9e9e9e"), yaxis="y"))
    if not yc.empty:
        tenor_years = {label: float(code) for label, code in TENORS.items()}
        target_years = list(SG_MATURITY_LADDER.values())
        for label, offset in snap_labels.items():
            idx = max(0, len(yc) - 1 + offset)
            interp_yields = _interp_sg_yield_curve(yc.iloc[idx], tenor_years, target_years)
            fig_sg_outstanding.add_trace(go.Scatter(
                x=sg_ladder_labels, y=interp_yields, mode="lines+markers", name=f"Yield: {label}",
                yaxis="y2", line=dict(color=snap_colors[label], width=2 if label == "Latest" else 1,
                                       dash="solid" if label == "Latest" else "dash"),
                marker=dict(size=4)))
    total_sg_outstanding = sg_pivot.values.sum()
    fig_sg_outstanding.update_layout(**dual_axis_layout(
        f"Outstanding SGS & T-Bills by Remaining Maturity (S${total_sg_outstanding:,.0f}B)",
        "Outstanding (S$ Billion)", "Yield (%)"))
    fig_sg_outstanding.update_layout(barmode="stack", yaxis2=dict(ticksuffix="%"))

    bc_type = st.selectbox("Bid-to-cover: instrument type", sorted(auctions["bill_bond_ind"].dropna().unique()), key="sg_bc_type")
    bc_hist = auctions[(auctions["bill_bond_ind"] == bc_type) & auctions["bid_to_cover"].notna() &
                        (auctions["auction_date"] >= START) & (auctions["auction_date"] <= END)].sort_values("auction_date")
    fig_btc = go.Figure()
    if not bc_hist.empty:
        fig_btc.add_trace(go.Scatter(x=bc_hist["auction_date"], y=bc_hist["bid_to_cover"],
                                      mode="markers", marker=dict(size=4, color="#90a4d4")))
    fig_btc.update_layout(**base_layout(f"Bid-to-Cover Ratio — {bc_type.title()}s"))

    # Net issuance - recently issued (past Nd) vs upcoming maturities (next Nd), both real
    # settled amounts (see get_sg_net_issuance for why this is trailing rather than forward
    # like the US "Issuance vs Maturity" chart).
    net_issuance_days = st.select_slider("Net Issuance window (days)", options=[30, 60, 90, 180], value=90, key="sg_net_issuance_days")
    with st.spinner("Computing net issuance…"):
        net_issuance_df = get_sg_net_issuance(auctions, net_issuance_days)
    fig_net_issuance = go.Figure()
    if not net_issuance_df.empty:
        fig_net_issuance.add_trace(go.Bar(x=net_issuance_df["tenor_bucket"], y=net_issuance_df["Issuance"],
                                           name="Issued", marker_color="#26a69a"))
        fig_net_issuance.add_trace(go.Bar(x=net_issuance_df["tenor_bucket"], y=-net_issuance_df["Maturing"],
                                           name="Maturing", marker_color="#ef5350"))
        fig_net_issuance.add_trace(go.Scatter(x=net_issuance_df["tenor_bucket"], y=net_issuance_df["Net"],
                                               name="Net Issuance", mode="lines+markers", line=dict(color="#90a4d4", width=2)))
    total_net = net_issuance_df["Net"].sum() if not net_issuance_df.empty else 0
    fig_net_issuance.update_layout(**base_layout(
        f"Net Issuance by Original Tenor — Issued (past {net_issuance_days}d) vs Maturing (next {net_issuance_days}d), Net: S${total_net:,.1f}B"))
    fig_net_issuance.update_layout(barmode="relative")

    render_two_col([
        ("SGD NEER", fig_neer, n),
        ("SGD NEER WoW Change", fig_neer_wow, neer_wow),
        ("SORA", fig_sora, s),
        ("SGS Yield Curve Snapshots", fig_yc, yc.tail(1)),
        ("SGS 2Y/5Y/10Y Yields", fig_yc_hist, clip(yc[["2Y", "5Y", "10Y"]]) if not yc.empty else pd.DataFrame()),
        ("SGS Curve Spreads", fig_spreads, spread_curve),
        ("Outstanding SGS by Remaining Maturity", fig_sg_outstanding, sg_pivot.reset_index()),
        ("Net Issuance", fig_net_issuance, net_issuance_df),
        ("Bid-to-Cover Trend", fig_btc, bc_hist[["auction_date", "issue_code", "bid_to_cover"]]),
    ])

    st.markdown('<div class="section-header">Upcoming SGS / T-Bill Issuance</div>', unsafe_allow_html=True)
    st.caption("Unlike the US Treasury calendar, MAS does not publish the offering size at announcement — "
               "auction size is only disclosed once that auction has closed, so this table shows tenor and "
               "dates only.")
    with st.spinner("Loading issuance calendar…"):
        cal = load_sgs_issuance_calendar()
    days_ahead_bonds = st.select_slider("Forward-looking window (days)", options=[30, 60, 90, 180], value=90, key="sg_issuance_days")
    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead_bonds)
    upcoming_cal = cal[(cal["auction_date"] >= today) & (cal["auction_date"] <= cutoff)].sort_values("auction_date")
    fig_cal = go.Figure(go.Table(
        header=dict(values=["Auction Date", "Issue Date", "Issue Code", "Tenor", "Type"],
                    fill_color=PLOT_BG, font=dict(color="white", size=12), align="center", height=28),
        cells=dict(values=[upcoming_cal["auction_date"].dt.strftime("%Y-%m-%d"), upcoming_cal["issue_date"].dt.strftime("%Y-%m-%d"),
                            upcoming_cal["issue_code"], upcoming_cal["auction_tenor_formatted"], upcoming_cal["agency_custom_categories"]],
                   fill_color=PAPER_BG, font=dict(color="#e0e0e0", size=11), align="center")
    ))
    fig_cal.update_layout(**base_layout(f"Upcoming SGS/T-Bill Auctions — Next {days_ahead_bonds}d", height=420))
    st.plotly_chart(fig_cal, use_container_width=True)
    csv_download(upcoming_cal, "sgs_upcoming_issuance")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Economic Calendar
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Economic Calendar")
    st.caption("Monetary Policy Statement dates (MAS reviews the S\\$NEER policy band quarterly, not an interest "
               "rate) + upcoming SGS/T-Bill auctions. No SingStat release-date-schedule API was confirmed "
               "(unlike FRED's release/dates endpoint for the US dashboard), so this doesn't include a "
               "forward CPI/GDP release calendar — only auctions and MPS dates, which are pre-scheduled.")

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

    st.markdown('<div class="section-header">MPS History</div>', unsafe_allow_html=True)
    hist_df = pd.DataFrame({"Date": [d.strftime("%Y-%m-%d") for d in mps_dates]}).sort_values("Date", ascending=False)
    st.dataframe(hist_df, use_container_width=True, hide_index=True, height=250)

st.markdown("---")
st.caption("Data: SingStat Table Builder · MAS Bonds & Bills / Domestic Interest Rates · Refresh rate: 6hr cache "
           "(SORA/yields/auctions), 1hr (summary metrics), 24hr (outstanding SGS), 7d (MPS dates)")
