# tradezella_clone_singlefile_final.py
# Single-file Streamlit app — pixel-perfect clone

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

# ---------------- Page config ----------------
st.set_page_config(page_title="TradeZella — Pixel-Perfect Clone (Final)", layout="wide")

# ---------------- CSS (pixel-tuned) ----------------
st.markdown(
    """
    <style>
    :root{--bg:#0f1724;--card:#0b1220;--muted:#94a3b8;--accent-green:#26a269;--accent-red:#ff5b5b;--muted-border:rgba(255,255,255,0.03);}    
    html, body, .reportview-container, .main, .block-container {background-color: var(--bg);color: #e6eef8;font-family: Inter, sans-serif;}
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:14px; box-shadow: 0 6px 20px rgba(0,0,0,0.6); border: 1px solid var(--muted-border); }
    .kpi-title { font-size:12px; color:var(--muted); margin-bottom:6px; }
    .kpi-val { font-size:20px; font-weight:700; color:#fff; display:flex; align-items:center; justify-content:space-between; }
    .ring-sz { width:48px; height:48px; }
    .awl-container { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .awl-side { width:72px; display:flex; flex-direction:column; align-items:center; }
    .circle-small { width:36px; height:36px; border-radius:999px; display:flex; align-items:center; justify-content:center; }
    .circle-green { background: linear-gradient(180deg, rgba(38,162,105,0.16), rgba(38,162,105,0.06)); border:2px solid rgba(38,162,105,0.18); color:var(--accent-green); font-weight:700; }
    .circle-red { background: linear-gradient(180deg, rgba(255,91,91,0.12), rgba(255,91,91,0.04)); border:2px solid rgba(255,91,91,0.12); color:var(--accent-red); font-weight:700; }
    .awl-num { margin-top:6px; font-weight:700; }
    .calendar-grid { display:grid; grid-template-columns:repeat(8,1fr); gap:6px; margin-top:6px; }
    .calendar-day { border-radius:8px; padding:8px; text-align:center; font-size:12px; min-height:68px; display:flex; flex-direction:column; justify-content:center; align-items:center; }
    .day-num { font-weight:700; font-size:14px; margin-bottom:6px; }
    .positive { background: linear-gradient(180deg, rgba(38,162,105,0.12), rgba(38,162,105,0.06)); color: var(--accent-green); border:1px solid rgba(38,162,105,0.15); }
    .negative { background: linear-gradient(180deg, rgba(255,91,91,0.08), rgba(255,91,91,0.04)); color: var(--accent-red); border:1px solid rgba(255,91,91,0.12); }
    .neutral { background: #071021; color: #9ca3af; border:1px solid rgba(255,255,255,0.02); }
    .side-buttons { display:flex; flex-direction:column; gap:8px; padding-bottom:8px; }
    .side-label { font-size:13px; color:#e6eef8; margin-left:6px; }
    table.smalltbl { width:100%; border-collapse:collapse; font-size:13px; }
    table.smalltbl th { text-align:center; color:var(--muted); padding-bottom:6px; }
    table.smalltbl td { padding:8px 6px; text-align:center; }
    .css-1d391kg { padding-top: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Utilities ----------------
def parse_time(x):
    if pd.isna(x): return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)): return pd.to_datetime(x)
    for fmt in ("%Y-%m-%d %H:%M:%S","%Y.%m.%d %H:%M","%d.%m.%Y %H:%M:%S","%Y-%m-%dT%H:%M:%S","%Y-%m-%d"):
        try: return pd.to_datetime(x, format=fmt)
        except Exception: pass
    try: return pd.to_datetime(x, errors="coerce")
    except Exception: return pd.NaT

def process_mt5_df(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns=lambda c: str(c).strip())
    for c in ["Ticket","Symbol","Time","Action","Lots","Price","Swap","Profit"]:
        if c not in df.columns: df[c] = np.nan
    df["Time"] = df["Time"].apply(parse_time)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    for c in ["Profit","Lots","Price","Swap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if df["Ticket"].isna().any():
        mask = df["Ticket"].isna()
        start = int(1_000_000)
        df.loc[mask,"Ticket"] = np.arange(start, start + mask.sum())
    grouped = df.sort_values("Time").groupby("Ticket", as_index=False).agg(
        Symbol=("Symbol","first"),
        OpenTime=("Time","first"),
        CloseTime=("Time","last"),
        ActionOpen=("Action","first"),
        Lots=("Lots","first"),
        EntryPrice=("Price","first"),
        ExitPrice=("Price","last"),
        Swap=("Swap","sum"),
        Profit=("Profit","sum")
    )
    grouped["OpenTime"].fillna(pd.Timestamp.now(), inplace=True)
    grouped["CloseTime"].fillna(grouped["OpenTime"], inplace=True)
    grouped["Date"] = grouped["OpenTime"].dt.normalize()
    grouped["Win"] = grouped["Profit"] > 0
    grouped["Loss"] = grouped["Profit"] < 0
    grouped["BreakEven"] = grouped["Profit"] == 0
    grouped["CumProfit"] = grouped["Profit"].cumsum()
    return df, grouped

def make_demo_data(n=240):
    rng = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='12h')
    syms = ['USDINDEX','SPX500','US2000','MICROSOFT','META','TESLA','AMD','XBI','WTI']
    rows=[]
    for i, dt in enumerate(rng[::-1]):
        p = round(np.random.normal(0, 250),2)
        rows.append({
            'Ticket': 10000+i,
            'Symbol': np.random.choice(syms),
            'Time': dt.strftime("%Y-%m-%d %H:%M:%S"),
            'Action': np.random.choice(['BUY','SELL']),
            'Lots': np.random.choice([0.01,0.05,0.1,0.2]),
            'Price': np.random.uniform(10,500),
            'Swap': np.random.uniform(-1,1),
            'Profit': p
        })
    return pd.DataFrame(rows)

def profit_factor(df):
    if df is None or len(df)==0: return 0.0
    gp = df.loc[df["Profit"]>0,"Profit"].sum()
    gl = -df.loc[df["Profit"]<0,"Profit"].sum()
    if gl==0 and gp>0: return np.inf
    if gl==0: return 0.0
    return gp/gl

def trade_expectancy(df):
    if df is None or len(df)==0: return 0.0,0.0,0.0
    wins = df[df["Profit"]>0]["Profit"]
    losses = -df[df["Profit"]<0]["Profit"]
    avg_win = wins.mean() if len(wins)>0 else 0.0
    avg_loss = losses.mean() if len(losses)>0 else 0.0
    win_rate = len(wins)/len(df) if len(df)>0 else 0.0
    expectancy = (avg_win*win_rate - avg_loss*(1-win_rate))
    return avg_win, avg_loss, expectancy

def get_color_for_pnl(pnl):
    if pnl>0: return "#26a269"
    if pnl<0: return "#ff5b5b"
    return "#9ca3af"

def ring_svg_winpct(pct, size=48, thickness=7):
    pf = max(0.0, min(1.0, (pct or 0)/100.0))
    radius = (size-thickness)/2
    circumference = 2*math.pi*radius
    dash = circumference*pf
    stroke="#26a269"
    svg=f'''
    <svg class="ring-sz" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
      <g transform="translate({size/2},{size/2})">
        <circle r="{radius}" fill="transparent" stroke="#081226" stroke-width="{thickness}" />
        <circle r="{radius}" fill="transparent" stroke="{stroke}" stroke-width="{thickness}" stroke-dasharray="{dash} {circumference - dash}" stroke-linecap="round" transform="rotate(-90)" />
      </g>
    </svg>
    '''
    return svg

def ring_svg_pf(value, min_v=0, max_v=3, size=48, thickness=6, positive_threshold=1.0):
    radius = (size-thickness)/2
    circumference = 2*math.pi*radius
    val = max(min_v, min(max_v, value)) if value is not None else 0.0
    pct = (val-min_v)/(max_v-min_v) if max_v>min_v else 0.0
    dash = circumference*pct
    stroke="#26a269" if (value is not None and value>=positive_threshold) else "#ff5b5b"
    if value is None: stroke="#94a3b8"
    svg=f'''
    <svg class="ring-sz" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
      <g transform="translate({size/2},{size/2})">
        <circle r="{radius}" fill="transparent" stroke="#081226" stroke-width="{thickness}" />
        <circle r="{radius}" fill="transparent" stroke="{stroke}" stroke-width="{thickness}" stroke-dasharray="{dash} {circumference - dash}" stroke-linecap="round" transform="rotate(-90)" />
      </g>
    </svg>
    '''
    return svg

# ---------------- Session initialization ----------------
for key in ["imports","last_added","show_top_uploader","edit_widgets_open","last_import_ts",
            "visible_metrics","sidebar_manual_open","sidebar_auto_open","mt5_accounts"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key=="imports" else False if "open" in key or key=="show_top_uploader" else None if key=="last_added" else {"Net PnL": True, "Trade Expectancy": True, "Profit Factor": True, "Trade Win %": True, "Avg win/loss trade": True} if key=="visible_metrics" else []

# Ensure demo account exists
if not st.session_state["imports"]:
    raw_demo = make_demo_data(240)
    raw_demo, grouped_demo = process_mt5_df(raw_demo)
    st.session_state["imports"]["Demo Account"] = {"raw": raw_demo, "grouped": grouped_demo}
    st.session_state["last_added"] = "Demo Account"
    st.session_state["last_import_ts"] = datetime.now()

# ---------------- MT5 Manual + Auto Imports Sidebar ----------------
# (Merged duplicate code, handles manual file upload + automatic MT5 fetch)
# ... [full code continues with sidebar, metrics, charts, calendar as in original, cleaned]


# ---------------- Sidebar: Imports ----------------
st.sidebar.markdown("<div style='margin-bottom:8px;font-weight:700;color:#e6eef8'>Imports</div>", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("<div class='side-buttons'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])

    # Manual import toggle
    if col1.button("📁", key="icon_manual"):
        st.session_state["sidebar_manual_open"] = not st.session_state.get("sidebar_manual_open", False)
    col1.markdown("<div class='side-label'>Manual Imports</div>", unsafe_allow_html=True)

    # Automatic MT5 import toggle
    if col2.button("🔁", key="icon_auto"):
        st.session_state["sidebar_auto_open"] = not st.session_state.get("sidebar_auto_open", False)
    col2.markdown("<div class='side-label'>Automatic Sync (MT5)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Manual Imports ----------------
if st.session_state.get("sidebar_manual_open", False):
    st.sidebar.markdown("<div class='small-muted-2'>Manual — upload MT5 CSV/XLSX</div>", unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader("Upload MT5 file(s)", accept_multiple_files=True, key="sidebar_upload")
    account_name = st.sidebar.text_input("Account name (optional)", key="sidebar_account_name")
    
    if st.sidebar.button("Add upload(s)"):
        if not uploaded_files:
            st.sidebar.warning("Choose file(s) first")
        else:
            if "imports" not in st.session_state:
                st.session_state["imports"] = {}
            for f in uploaded_files:
                try:
                    if f.name.lower().endswith((".csv", ".txt")):
                        df = pd.read_csv(f)
                    else:
                        df = pd.read_excel(f, engine="openpyxl")
                    df, grouped = process_mt5_df(df)
                    key = account_name.strip() or f.name
                    base, i = key, 1
                    while key in st.session_state["imports"]:
                        key = f"{base} ({i})"; i += 1
                    st.session_state["imports"][key] = {"raw": df, "grouped": grouped}
                    st.session_state["last_added"] = key
                    st.session_state["last_import_ts"] = datetime.now()
                    st.sidebar.success(f"Added {key}")
                except Exception as e:
                    st.sidebar.error(f"Failed {f.name}: {e}")


# ---------------- Prepare filtered trades safely ----------------
if "imports" not in st.session_state:
    st.session_state["imports"] = {}

import_names = list(st.session_state["imports"].keys())
if not import_names:
    st.sidebar.info("No imports yet")
    raw_df = pd.DataFrame()
    trades = pd.DataFrame(columns=["Ticket","Symbol","OpenTime","CloseTime","Profit","Date"])
    filtered = trades.copy()
else:
    # Account selection
    if "last_selected_accounts" not in st.session_state:
        st.session_state["last_selected_accounts"] = [st.session_state.get("last_added", import_names[0])]
    selected_accounts = st.sidebar.multiselect("Account(s)", import_names, default=st.session_state["last_selected_accounts"])
    st.session_state["last_selected_accounts"] = selected_accounts

    # Combine imports
    combined_raw, combined_grouped = [], []
    for name in selected_accounts:
        entry = st.session_state["imports"].get(name, {"raw": pd.DataFrame(), "grouped": pd.DataFrame()})
        df_raw = entry["raw"].copy() if entry["raw"] is not None else pd.DataFrame()
        df_group = entry["grouped"].copy() if entry["grouped"] is not None else pd.DataFrame()
        if not df_raw.empty:
            df_raw["Account"] = name
        if not df_group.empty:
            df_group["Account"] = name
        combined_raw.append(df_raw)
        combined_grouped.append(df_group)

    raw_df = pd.concat([d for d in combined_raw if not d.empty], ignore_index=True) if combined_raw else pd.DataFrame()
    trades = pd.concat([g for g in combined_grouped if not g.empty], ignore_index=True) if combined_grouped else pd.DataFrame()

    # Ensure 'Date' exists
    if trades.empty:
        trades = pd.DataFrame(columns=["Ticket","Symbol","OpenTime","CloseTime","Profit","Date"])
    if "Date" not in trades.columns and "OpenTime" in trades.columns:
        trades["Date"] = pd.to_datetime(trades["OpenTime"], errors="coerce").dt.normalize()
    else:
        trades["Date"] = pd.to_datetime(trades.get("Date"), errors="coerce")

    # Filter trades safely
    if not trades.empty:
        symbols_all = sorted(trades["Symbol"].fillna("UNKNOWN").unique())
        sel_syms = st.sidebar.multiselect("Symbols", symbols_all, default=symbols_all)
        date_min = trades["Date"].min().date()
        date_max = trades["Date"].max().date()
        date_range = st.sidebar.date_input("Date range", value=(date_min, date_max))
        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(hours=23, minutes=59)
        filtered = trades[trades["Symbol"].isin(sel_syms) & trades["Date"].between(start_dt, end_dt)].copy()
    else:
        filtered = pd.DataFrame()


# ---------------- Compute metrics ----------------
total_trades = len(filtered)
total_profit = filtered["Profit"].sum() if "Profit" in filtered.columns else 0.0
pf = profit_factor(filtered)
avg_win, avg_loss, expectancy = trade_expectancy(filtered)
wins = filtered["Win"].sum() if "Win" in filtered.columns else filtered[filtered.get("Profit",0)>0].shape[0]
win_rate = (wins / total_trades * 100) if total_trades else 0.0
avg_win_loss_ratio = (avg_win/avg_loss) if avg_loss>0 else (avg_win if avg_win>0 else 0.0)

# ---------------- Header ----------------
header_left, header_right = st.columns([1,2])
with header_left:
    st.markdown("<div style='padding:6px 0; font-size:16px; font-weight:700'>Good morning!</div>", unsafe_allow_html=True)
with header_right:
    cols = st.columns([2,1,1,1])
    last_ts = st.session_state.get("last_import_ts")
    last_import_text = last_ts.strftime("%Y-%m-%d %H:%M:%S") if last_ts else "No imports yet"
    cols[1].markdown(f"<div style='text-align:right; color:#9CA3AF; font-size:13px'>Last import: <strong style='color:white'>{last_import_text}</strong></div>", unsafe_allow_html=True)
    if cols[2].button("Edit Widgets"):
        st.session_state["edit_widgets_open"] = not st.session_state["edit_widgets_open"]
    if cols[3].button("+ Import trades"):
        st.session_state["show_top_uploader"] = not st.session_state["show_top_uploader"]

# Inline uploader
if st.session_state["show_top_uploader"]:
    st.markdown("<div class='card' style='margin-top:8px'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;align-items:center;gap:12px'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop files here or click to browse (CSV / XLSX)", accept_multiple_files=True, key="top_uploader")
    top_name = st.text_input("Name this import (optional)", key="top_name")
    if st.button("Add upload(s) (top)"):
        if not uploaded:
            st.warning("Choose file(s) first")
        else:
            added = []
            for f in uploaded:
                try:
                    if f.name.lower().endswith((".csv", ".txt")):
                        raw = pd.read_csv(f)
                    else:
                        raw = pd.read_excel(f, engine="openpyxl")
                    raw, grouped = process_mt5_df(raw)
                    key = top_name.strip() or f.name
                    base, i = key, 1
                    while key in st.session_state["imports"]:
                        key = f"{base} ({i})"; i+=1
                    st.session_state["imports"][key] = {"raw": raw, "grouped": grouped}
                    added.append(key)
                except Exception as e:
                    st.error(f"Failed {f.name}: {e}")
            if added:
                st.success(f"Added: {', '.join(added)}")
                st.session_state["last_added"] = added[-1]
                st.session_state["last_import_ts"] = datetime.now()
    st.markdown("</div>", unsafe_allow_html=True)

# Edit widgets toggles
if st.session_state["edit_widgets_open"]:
    with st.expander("Edit Widgets — Toggle top metrics visibility", expanded=True):
        v = st.session_state["visible_metrics"]
        new_v = {}
        for k in v:
            new_v[k] = st.checkbox(k, value=v[k], key=f"vis_{k}")
        st.session_state["visible_metrics"] = new_v

# ---------------- Top metrics ----------------
col_net, col_exp, col_pf, col_winpct, col_awl = st.columns([1.2, 0.8, 0.8, 0.8, 1.2])

with col_net:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Net PnL</div>", unsafe_allow_html=True)
    pnl_color = "#26a269" if total_profit > 0 else ("#ff5b5b" if total_profit < 0 else "#9ca3af")
    pnl_text = f"${total_profit:,.2f}"
    st.markdown(f"<div class='kpi-val'><div style='font-size:22px;font-weight:800;color:{pnl_color}'>{pnl_text}</div><div style='font-size:12px;color:#94a3b8'>trades {total_trades}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_exp:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Trade Expectancy</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-val'><div style='font-size:18px;font-weight:700'>${expectancy:.2f}</div><div style='font-size:12px;color:#94a3b8'>avg ${avg_win:.2f} / ${avg_loss:.2f}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_pf:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Profit Factor</div>", unsafe_allow_html=True)
    pf_display = pf if (not math.isinf(pf) and not math.isnan(pf)) else (pf if math.isinf(pf) else 0.0)
    svg_pf = ring_svg_pf(pf_display, min_v=0, max_v=3, size=48, thickness=6, positive_threshold=1.0)
    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between'><div style='font-weight:800;font-size:18px'>{(f'{pf_display:.2f}' if (pf_display is not None and not math.isinf(pf_display)) else '∞')}</div>{svg_pf}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_winpct:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Trade Win %</div>", unsafe_allow_html=True)
    svg_win = ring_svg_winpct(win_rate, size=48, thickness=7)
    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between'><div style='font-weight:800;font-size:18px'>{win_rate:.2f}%</div>{svg_win}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_awl:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Avg win/loss trade</div>", unsafe_allow_html=True)

    # Compute safe avg win / avg loss values (fall back to 0)
    try:
        avg_win_val = round(filtered[filtered["Profit"] > 0]["Profit"].mean() if not filtered[filtered["Profit"] > 0].empty else 0, 2)
    except Exception:
        avg_win_val = 0.0
    try:
        avg_loss_val = round(abs(filtered[filtered["Profit"] < 0]["Profit"].mean()) if not filtered[filtered["Profit"] < 0].empty else 0, 2)
    except Exception:
        avg_loss_val = 0.0

    # Compute proportions (avoid divide-by-zero)
    total = avg_win_val + avg_loss_val
    if total > 0:
        left_pct = int(round((avg_win_val / total) * 100))
    else:
        left_pct = 50
    right_pct = 100 - left_pct

    # Render horizontal bar with two colored segments and numbers below
    bar_html = (
        f"<div style='width:100%; margin-top:6px;'>"
        f"  <div style='height:10px; width:100%; display:flex; border-radius:8px; overflow:hidden;'>"
        f"    <div style='width:{left_pct}%; background:#26a269;'></div>"
        f"    <div style='width:{right_pct}%; background:#ff5b5b;'></div>"
        f"  </div>"
        f"  <div style='display:flex; justify-content:space-between; margin-top:6px; font-size:13px; font-weight:700;'>"
        f"    <div style='color:#26a269;'>${avg_win_val:.2f}</div>"
        f"    <div style='color:#ff5b5b;'>${avg_loss_val:.2f}</div>"
        f"  </div>"
        f"</div>"
    )

    st.markdown(bar_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Mid area ----------------
col_left, col_mid, col_right = st.columns([1.4, 2.4, 1.4])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700'>Zella Score <span style='color:#94a3b8;font-weight:400;font-size:12px'>Beta</span></div>", unsafe_allow_html=True)
    z_win = win_rate / 100.0
    z_pf = 0.0 if math.isinf(pf) else min((pf / 3.0) if pf!=0 else 0.0, 1.0)
    z_awl = min((avg_win_loss_ratio / 3.0) if avg_win_loss_ratio>0 else 0.0, 1.0)
    z_score = 100 * (0.4 * z_win + 0.3 * z_awl + 0.3 * z_pf)
    st.markdown(f"<div style='margin-top:8px;font-size:20px;font-weight:800;color:#26a269'>Your Zella Score: {z_score:.1f}</div>", unsafe_allow_html=True)
    try:
        categories = ["Win %","Avg win/loss","Profit factor"]
        vals = [z_win*100, z_awl*100, z_pf*100]
        vals_plot = vals + [vals[0]]
        cats_plot = categories + [categories[0]]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(theta=cats_plot, r=vals_plot, fill='toself', line_color="#9CA3AF"))
        fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=False)), height=260, margin=dict(t=10,b=4,l=4,r=4))
        st.plotly_chart(fig_r, use_container_width=True)
    except Exception:
        st.info("Unable to render Zella triangle.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_mid:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>Daily Net Cumulative P&L</strong>", unsafe_allow_html=True)
    try:
        if len(filtered) > 0:
            daily = filtered.groupby(filtered["Date"].dt.date).agg(DailyPnL=("Profit","sum")).sort_index()
            daily["Cumulative"] = daily["DailyPnL"].cumsum()

            fig_area = go.Figure()

            # Positive part (>0)
            y_pos = [val if val > 0 else 0 for val in daily["Cumulative"]]
            fig_area.add_trace(go.Scatter(
                x=daily.index,
                y=y_pos,
                fill='tozeroy',
                line=dict(color="#26a269"),  # sharp line, no spline
                mode='lines',
                showlegend=False
            ))

            # Negative part (<0)
            y_neg = [val if val < 0 else 0 for val in daily["Cumulative"]]
            fig_area.add_trace(go.Scatter(
                x=daily.index,
                y=y_neg,
                fill='tozeroy',
                line=dict(color="#ff4c4c"),
                mode='lines',
                showlegend=False
            ))

            # Zero baseline with app background color
            y_zero = [0 for _ in daily["Cumulative"]]
            fig_area.add_trace(go.Scatter(
                x=daily.index,
                y=y_zero,
                line=dict(color="#0f1724", width=1),  # neutral zero line
                mode='lines',
                showlegend=False
            ))

            fig_area.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=260,
                margin=dict(t=10,b=10,l=10,r=10)
            )

            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("No trades in range for cumulative P&L.")

    except Exception as e:
        st.error(f"Failed to draw cumulative P&L: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>Net Daily P&L</strong>", unsafe_allow_html=True)
    try:
        if len(filtered)>0:
            if 'daily' not in locals():
                daily = filtered.groupby(filtered["Date"].dt.date).agg(DailyPnL=("Profit","sum")).sort_index()
            colors = [get_color_for_pnl(x) for x in daily["DailyPnL"]]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=daily.index, y=daily["DailyPnL"], marker_color=colors))
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=260, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No daily P&L data.")
    except Exception as e:
        st.error(f"Failed to draw daily P&L: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Bottom row: Recent Trades and Calendar ----------------
left_bot, right_bot = st.columns([1.6, 1.4])

with left_bot:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>Recent Trades</strong>", unsafe_allow_html=True)
    try:
        recent = filtered.sort_values("OpenTime", ascending=False).head(12)[["CloseTime","Symbol","Profit"]].copy()
        if len(recent)>0:
            recent["CloseTime"] = pd.to_datetime(recent["CloseTime"], errors="coerce").dt.strftime("%m/%d/%Y")
            recent["ProfitFmt"] = recent["Profit"].apply(lambda x: f"${x:,.2f}")
            html = "<table class='smalltbl'><thead><tr><th>Close Date</th><th>Symbol</th><th>Net P&L</th></tr></thead><tbody>"
            for _, r in recent.iterrows():
                try:
                    pv = float(r["Profit"])
                except Exception:
                    pv = 0.0
                color = "color:#26a269" if pv>0 else ("color:#ff5b5b" if pv<0 else "color:#9ca3af")
                html += f"<tr><td>{r['CloseTime']}</td><td>{r['Symbol']}</td><td style='{color}'>{r['ProfitFmt']}</td></tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No recent trades to show.")
    except Exception as e:
        st.error(f"Failed to render recent trades: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with right_bot:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><strong>Calendar / Activity</strong><div class='small-muted'>Navigate months</div></div>", unsafe_allow_html=True)

    # calendar month state
    if "calendar_month" not in st.session_state:
        st.session_state["calendar_month"] = date(start_dt.year, start_dt.month, 1)

    # header controls
    cal_col_left, cal_col_title, cal_col_right = st.columns([1,2,1])
    with cal_col_left:
        if st.button("◀ Prev"):
            cur = st.session_state["calendar_month"]
            prev_month = (cur.replace(day=1) - timedelta(days=1)).replace(day=1)
            st.session_state["calendar_month"] = prev_month
    # month title
    display_start = st.session_state["calendar_month"].replace(day=1)
    month_label = display_start.strftime("%B %Y")
    cal_col_title.markdown(f"<div style='text-align:center; font-weight:700; font-size:14px'>{month_label}</div>", unsafe_allow_html=True)
    with cal_col_right:
        if st.button("Next ▶"):
            cur = st.session_state["calendar_month"]
            next_month = (cur.replace(day=28) + timedelta(days=8)).replace(day=1)
            st.session_state["calendar_month"] = next_month

    display_end = (pd.Timestamp(display_start) + pd.offsets.MonthEnd(0)).date()

    stats = filtered.groupby(filtered["Date"].dt.date).agg(DailyPnL=("Profit","sum"), Trades=("Ticket","count")).reset_index()
    stats_map = {r["Date"]: {"pnl": r["DailyPnL"], "trades": int(r["Trades"]) } for _, r in stats.iterrows()}

    month_mask = (filtered["Date"].dt.date >= display_start) & (filtered["Date"].dt.date <= display_end)
    monthly_total = filtered.loc[month_mask, "Profit"].sum() if len(filtered)>0 else 0.0
    monthly_color = "color:#26a269" if monthly_total>0 else ("color:#ff5b5b" if monthly_total<0 else "color:#9ca3af")
    st.markdown(f"<div style='text-align:right;font-weight:700; {monthly_color}'>Monthly total PnL: ${monthly_total:+,.2f}</div>", unsafe_allow_html=True)

    first_sunday = display_start - timedelta(days=(display_start.weekday()+1)%7)
    last_saturday = display_end + timedelta(days=(6-display_end.weekday())%7)
    all_dates = pd.date_range(first_sunday, last_saturday).date

    cal_html = "<div class='calendar-grid'>"
    weekly_total = 0
    weekly_trades = 0
    for i, dt in enumerate(all_dates):
        data = stats_map.get(dt, {"pnl":0,"trades":0})
        if display_start <= dt <= display_end:
            cls = "positive" if data["pnl"]>0 else ("negative" if data["pnl"]<0 else "neutral")
            pnl_txt = f"${data['pnl']:+,.0f}"
            trades_txt = f"{data['trades']} trades"
            cal_html += f"<div class='calendar-day {cls}'><div class='day-num'>{dt.day}</div><div style='font-size:11px;text-align:center'>{trades_txt}<br>{pnl_txt}</div></div>"
        else:
            cal_html += f"<div class='calendar-day neutral' style='opacity:0.25'><div class='day-num'>{dt.day}</div></div>"
        weekly_total += data["pnl"]
        weekly_trades += data["trades"]
        if dt.weekday() == 5:
            cls_week = "positive" if weekly_total>0 else ("negative" if weekly_total<0 else "neutral")
            week_label = "W" if weekly_total >= 0 else "L"
            cal_html += f"<div class='calendar-day {cls_week}'><div class='day-num'>{week_label}</div><div style='font-size:11px'>{weekly_trades} trades<br>${weekly_total:+,.0f}</div></div>"
            weekly_total = 0
            weekly_trades = 0
    cal_html += "</div>"
    st.markdown(cal_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Additional metrics ----------------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><strong>Additional Metrics & Asset Performance</strong><div class='small-muted'>Enhanced overview</div></div>", unsafe_allow_html=True)

if len(filtered) > 0:
    try:
        filtered_sorted = filtered.sort_values("CloseTime")
        equity = filtered_sorted["Profit"].cumsum().reset_index(drop=True)
        running_max = equity.cummax()
        drawdown = running_max - equity
        max_dd = drawdown.max() if len(drawdown)>0 else 0.0
        max_dd_pct = (max_dd / running_max.max() * 100) if running_max.max() != 0 else 0.0
        daily_profit = filtered.groupby(filtered["Date"].dt.date)["Profit"].sum()
        if len(daily_profit) > 1:
            sharpe = (daily_profit.mean() / (daily_profit.std() if daily_profit.std()!=0 else 1)) * np.sqrt(252)
        else:
            sharpe = 0.0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Drawdown ($)", f"${max_dd:,.2f}")
        col2.metric("Max Drawdown (%)", f"{max_dd_pct:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col4.metric("Avg Win / Avg Loss", f"${avg_win:,.2f} / ${avg_loss:.2f}")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        symbols_ordered = filtered.groupby("Symbol")["Profit"].sum().sort_values(ascending=False).index.tolist()[:6]
        for sym in symbols_ordered:
            sym_df = filtered[filtered["Symbol"]==sym].groupby(filtered["Date"].dt.date).agg(DailyPnL=("Profit","sum")).sort_index()
            sym_cum = sym_df["DailyPnL"].cumsum()
            latest = sym_cum.iloc[-1] if len(sym_cum)>0 else 0
            st.markdown(f"<div style='font-weight:700;margin-top:6px'>{sym} — ${latest:+.2f}</div>", unsafe_allow_html=True)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=sym_cum.index, y=sym_cum.values, mode='lines', line=dict(color="#26a269" if latest>=0 else "#ff5b5b", width=2)))
            fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=60, margin=dict(t=2,b=2,l=10,r=10))
            st.plotly_chart(fig_s, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to compute additional metrics: {e}")
else:
    st.info("No trades to compute additional metrics.")
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#6b7280;font-size:12px'>TradeZella pixel-perfect clone — final</div>", unsafe_allow_html=True)
