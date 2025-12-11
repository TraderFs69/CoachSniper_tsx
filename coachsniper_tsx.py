import os, time, random, datetime as dt
from typing import Dict, Tuple, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ==============================
# Config Streamlit
# ==============================
st.set_page_config(
    page_title="Coach Sniper – TSX Composite",
    layout="wide"
)
st.title("🧭 Coach Sniper – TSX Composite (Yahoo Finance)")

# ==============================
# Heikin Ashi
# ==============================
def to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df["Low"],  ha_open, ha_close], axis=1).min(axis=1)
    out = df.copy()
    out["Open"], out["High"], out["Low"], out["Close"] = ha_open, ha_high, ha_low, ha_close
    return out

# ==============================
# Conversion tickers TSX → Yahoo Finance
# ==============================
def fix_tsx_ticker(ticker: str) -> str:
    t = ticker.strip().lower()

    # Cas simple : symbol -> symbol.TO
    if "." not in t:
        return t.upper() + ".TO"

    # Cas AP.un -> AP-UN.TO
    if ".un" in t:
        base = t.split(".")[0].upper()
        return f"{base}-UN.TO"

    # fallback par défaut
    return t.upper() + ".TO"

# ==============================
# Charger la liste TSX Composite
# ==============================
@st.cache_data(show_spinner=False, ttl=60*60)
def get_tsx_constituents() -> Tuple[pd.DataFrame, List[str]]:
    path = "tsxcomposite_constituents.xlsx"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier {path} introuvable.")

    df = pd.read_excel(path)

    if "Symbol" not in df.columns:
        raise ValueError("Le fichier doit contenir la colonne 'Symbol'.")

    df["Symbol"] = df["Symbol"].astype(str).str.strip()

    df["Company"] = df["Symbol"]        # Pas fourni → on copie
    df["Sector"]  = "Unknown"           # Pas fourni → inconnu

    df = df[df["Symbol"] != ""]
    tickers_raw = df["Symbol"].tolist()
    tickers_yahoo = [fix_tsx_ticker(t) for t in tickers_raw]

    return df, tickers_yahoo

# ==============================
# Indicateurs utilitaires
# ==============================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(0)

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

def cross_recent(cross: pd.Series, lookback: int = 3) -> pd.Series:
    out = cross.copy().astype(bool).fillna(False)
    for i in range(1, lookback+1):
        out |= cross.shift(i).fillna(False)
    return out

def ichimoku_components(high: pd.Series, low: pd.Series,
                        len_tenkan=9, len_kijun=26, len_senkou_b=52):
    tenkan = (high.rolling(len_tenkan).max() + low.rolling(len_tenkan).min())/2.0
    kijun  = (high.rolling(len_kijun).max()  + low.rolling(len_kijun).min()) /2.0
    spanA  = (tenkan + kijun)/2.0
    spanB  = (high.rolling(len_senkou_b).max() + low.rolling(len_senkou_b).min())/2.0
    return tenkan, kijun, spanA, spanB

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    denom = (hh - ll).replace(0, np.nan)
    wr = -100 * (hh - close) / denom
    return wr.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill")

def volume_oscillator(volume: pd.Series, fast=5, slow=20) -> pd.Series:
    ema_f = ema(volume, fast)
    ema_s = ema(volume, slow)
    with np.errstate(divide="ignore", invalid="ignore"):
        vo = (ema_f - ema_s) / ema_s * 100.0
    return pd.Series(np.where(np.isfinite(vo), vo, 0.0), index=volume.index).fillna(0)

# ==============================
# Logique Coach Swing (inchangée)
# ==============================
def coach_swing_signals(df: pd.DataFrame, mode: str = "Balanced", use_rsi50: bool = True):
    if df is None or df.empty:
        return False, False, {}, False, False

    data = to_heikin_ashi(df)

    if len(data) < 82:
        return False, False, {}, False, False

    h = data["High"].astype(float)
    l = data["Low"].astype(float)
    c = data["Close"].astype(float)
    v = data["Volume"].astype(float)

    tenkan, kijun, spanA, spanB = ichimoku_components(h, l)
    upperCloud = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lowerCloud = pd.concat([spanA, spanB], axis=1).min(axis=1)
    aboveCloud = c > upperCloud
    belowCloud = c < lowerCloud
    bullTK = tenkan > kijun
    bearTK = tenkan < kijun

    rsi14 = rsi_wilder(c, 14)
    rsiBullOK = (rsi14 > 50) if use_rsi50 else pd.Series(True, index=rsi14.index)
    rsiBearOK = (rsi14 < 50) if use_rsi50 else pd.Series(True, index=rsi14.index)

    wr = williams_r(h, l, c, 14)
    wr_cross_up_80 = crossover(wr, pd.Series(-80.0, index=wr.index))
    wr_cross_dn_20 = crossunder(wr, pd.Series(-20.0, index=wr.index))
    wr_up_turning  = (wr > -80) & (wr > wr.shift(1)) & (wr.shift(1) > wr.shift(2))
    wr_dn_turning  = (wr < -20) & (wr < wr.shift(1)) & (wr.shift(1) < wr.shift(2))
    wr_up_recent   = cross_recent(wr_cross_up_80, 14)
    wr_dn_recent   = cross_recent(wr_cross_dn_20, 14)

    vo = volume_oscillator(v)

    if mode == "Strict":
        longTrendOK  = aboveCloud & bullTK
        shortTrendOK = belowCloud & bearTK
        wrLongOK, wrShortOK = wr_up_recent, wr_dn_recent
        voLongOK, voShortOK = vo > 0, vo < 0
    elif mode == "Aggressive":
        longTrendOK  = c > kijun
        shortTrendOK = c < kijun
        wrLongOK     = (wr > -60) & (wr > wr.shift(1))
        wrShortOK    = (wr < -40) & (wr < wr.shift(1))
        voLongOK, voShortOK = vo >= -2, vo <= 2
    else:
        longTrendOK  = (c > kijun) & (aboveCloud | (spanA > spanB))
        shortTrendOK = (c < kijun) & (belowCloud | (spanA < spanB))
        wrLongOK     = wr_up_recent | wr_up_turning
        wrShortOK    = wr_dn_recent | wr_dn_turning
        voLongOK, voShortOK = vo >= -1, vo <= 1

    buyCond  = longTrendOK  & rsiBullOK & wrLongOK  & voLongOK
    sellCond = shortTrendOK & rsiBearOK & wrShortOK & voShortOK

    buy_now  = bool(buyCond.iloc[-1])
    sell_now = bool(sellCond.iloc[-1])

    if len(buyCond) >= 2:
        buy_new  = bool(buyCond.iloc[-1] and not buyCond.iloc[-2])
        sell_new = bool(sellCond.iloc[-1] and not sellCond.iloc[-2])
    else:
        buy_new = False
        sell_new = False

    ema9  = ema(c, 9)
    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    ema200= ema(c, 200)

    last = {
        "ema9":   float(ema9.iloc[-1]),
        "ema20":  float(ema20.iloc[-1]),
        "ema50":  float(ema50.iloc[-1]),
        "ema200": float(ema200.iloc[-1]),
        "RSI":    float(rsi14.iloc[-1]),
        "WR":     float(wr.iloc[-1]),
        "VO":     float(vo.iloc[-1]),
    }
    return buy_now, sell_now, last, buy_new, sell_new

# ==============================
# Download DAILY Yahoo Finance
# ==============================
@st.cache_data(show_spinner=True, ttl=60*60, max_entries=64)
def download_yahoo_bars(tickers: tuple[str, ...], years: int = 2):
    data = {}
    failed = []
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)

    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, interval="1d", progress=False)
            if df.empty:
                failed.append(t)
                continue

            df = df.rename(columns=str.capitalize)
            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            data[t] = df[keep]
        except Exception:
            failed.append(t)

        time.sleep(0.2 + random.random()*0.2)

    return data, failed

# ==============================
# UI – Filtres et chargement liste TSX
# ==============================
with st.spinner("Chargement de la liste TSX Composite…"):
    tsx_df_raw, tsx_yahoo_tickers = get_tsx_constituents()

search = st.text_input("Recherche (ticker)", "").strip().lower()

base_df = tsx_df_raw.copy()
if search:
    mask = base_df["Symbol"].str.lower().str.contains(search)
    base_df = base_df[mask]

tickers_to_scan = [
    fix_tsx_ticker(t) for t in base_df["Symbol"].tolist()
]

st.caption(f"📈 Tickers TSX filtrés : {len(tickers_to_scan)}")

mode = st.sidebar.selectbox("Mode stratégie", ["Balanced","Strict","Aggressive"])
use_rsi50 = st.sidebar.checkbox("Filtre RSI > 50", value=True)

go = st.button("▶️ Scanner (Yahoo Finance)")
if not go:
    st.stop()

with st.spinner("Téléchargement des données Yahoo Finance…"):
    bars, failed = download_yahoo_bars(tuple(tickers_to_scan))

valid = len(bars)
st.caption(f"✅ Jeux de données valides : {valid}/{len(tickers_to_scan)}")
if failed:
    st.warning(f"Tickers échoués : {failed[:10]}{'...' if len(failed)>10 else ''}")

if valid == 0:
    st.error("Aucune donnée valide. Arrêt.")
    st.stop()

# ==============================
# Calcul signaux
# ==============================
results = []
for sym_raw in base_df["Symbol"].tolist():
    yahoo_t = fix_tsx_ticker(sym_raw)
    dft = bars.get(yahoo_t)
    if dft is None or len(dft) < 82:
        continue

    buy_now, sell_now, last, buy_new, sell_new = coach_swing_signals(
        dft, mode=mode, use_rsi50=use_rsi50
    )

    results.append({
        "Symbol": sym_raw,
        "Yahoo":  yahoo_t,
        "Company": sym_raw,
        "Sector": "Unknown",
        "LastDate": dft.index[-1].date(),
        "Buy": buy_now,
        "Sell": sell_now,
        "BuyNew": buy_new,
        "SellNew": sell_new,
        "Close": float(dft["Close"].iloc[-1]),
        "RSI": last["RSI"],
        "WR": last["WR"],
        "VO": last["VO"],
    })

res_df = pd.DataFrame(results)
if res_df.empty:
    st.warning("Aucun résultat — aucun signal détecté.")
    st.stop()

# ==============================
# Affichage
# ==============================
colA, colB, colC = st.columns([1,1,2])
with colA:
    show = st.selectbox("Afficher", ["Tous","Buy seulement","Sell seulement","Buy nouveaux"])
with colB:
    sort_by = st.selectbox("Trier par",
        ["Buy","Sell","BuyNew","SellNew","Close","Symbol","LastDate"])
with colC:
    asc = st.checkbox("Tri ascendant", value=False)

if show == "Buy seulement":
    res_view = res_df[res_df["Buy"]]
elif show == "Sell seulement":
    res_view = res_df[res_df["Sell"]]
elif show == "Buy nouveaux":
    res_view = res_df[res_df["BuyNew"]]
else:
    res_view = res_df

res_view = res_view.sort_values(by=sort_by, ascending=asc)
st.dataframe(res_view, use_container_width=True)

csv = res_view.to_csv(index=False).encode("utf-8")
st.download_button("💾 Télécharger CSV", csv, "coach_swing_tsx.csv", mime="text/csv")
