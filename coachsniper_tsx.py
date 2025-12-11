import os, time, random, datetime as dt
from typing import Dict, Tuple, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ==============================
# Config Streamlit
# ==============================
st.set_page_config(page_title="Coach Sniper – TSX", layout="wide")
st.title("🧭 Coach Sniper – TSX Composite (Yahoo Finance)")

# ==============================
# 🔧 Conversion tickers TSX → Yahoo Finance
# ==============================
def fix_tsx_ticker(ticker: str) -> str:
    t = ticker.strip().lower()

    # Cas AP.un → AP-UN.TO
    if ".un" in t:
        base = t.split(".")[0].upper()
        return f"{base}-UN.TO"

    # Cas générique : CTC.A → CTC-A.TO
    if "." in t:
        base, suffix = t.split(".", 1)
        return f"{base.upper()}-{suffix.upper()}.TO"

    # Cas simple : RY → RY.TO
    return f"{t.upper()}.TO"

# ==============================
# 🔍 Charger liste TSX + correction automatique colonnes
# ==============================
@st.cache_data(show_spinner=True, ttl=3600)
def get_tsx_constituents():
    path = "tsxcomposite_constituents.xlsx"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable")

    df = pd.read_excel(path)

    # DEBUG
    st.write("🔍 Colonnes détectées :", df.columns.tolist())
    st.write(df.head())

    # Normalisation
    df.columns = [c.strip().lower() for c in df.columns]

    # Trouver colonne tickers
    colmap = {
        "symbol": "Symbol",
        "ticker": "Symbol",
        "tickers": "Symbol",
        "security": "Symbol"
    }

    renamed = False
    for c in df.columns:
        if c in colmap:
            df.rename(columns={c: colmap[c]}, inplace=True)
            renamed = True
            break

    if not renamed:
        st.error("Impossible de trouver la colonne contenant les tickers.")
        st.write("Colonnes disponibles :", df.columns.tolist())
        raise ValueError("Colonne tickers manquante")

    # Nettoyage
    df = df[df["Symbol"].notna()]
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df[df["Symbol"] != ""]

    df["Company"] = df["Symbol"]
    df["Sector"] = "Unknown"

    tickers = [fix_tsx_ticker(t) for t in df["Symbol"]]
    return df, tickers

# ==============================
# 📈 Indicateurs techniques
# ==============================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi_wilder(close, length=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def crossover(a, b): return (a > b) & (a.shift(1) <= b.shift(1))
def crossunder(a, b): return (a < b) & (a.shift(1) >= b.shift(1))

def cross_recent(cross, lookback=3):
    out = cross.copy()
    for i in range(1, lookback+1):
        out |= cross.shift(i).fillna(False)
    return out

def ichimoku_components(high, low, t=9, k=26, s=52):
    tenkan = (high.rolling(t).max() + low.rolling(t).min()) / 2
    kijun  = (high.rolling(k).max() + low.rolling(k).min()) / 2
    spanA  = (tenkan + kijun) / 2
    spanB  = (high.rolling(s).max() + low.rolling(s).min()) / 2
    return tenkan, kijun, spanA, spanB

def williams_r(high, low, close, length=14):
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll)

def volume_oscillator(volume, fast=5, slow=20):
    return (ema(volume, fast) - ema(volume, slow)) / ema(volume, slow) * 100

# ==============================
# 🔥 Heikin Ashi
# ==============================
def to_heikin_ashi(df):
    df = df.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    df["Open"] = ha_open
    df["Close"] = ha_close
    df["High"] = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    df["Low"]  = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)
    return df

# ==============================
# 📥 Télécharger données Yahoo
# ==============================
@st.cache_data(show_spinner=True, ttl=3600)
def download_yahoo_bars(tickers):
    out = {}
    failed = []
    end = dt.date.today()
    start = end - dt.timedelta(days=365*2)

    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, interval="1d", progress=False)
            if df.empty:
                failed.append(t)
                continue

            df = df.rename(columns=str.capitalize)
            df = df[["Open","High","Low","Close","Volume"]]
            out[t] = df

        except Exception:
            failed.append(t)

        time.sleep(0.15 + random.random()*0.15)

    return out, failed

# ==============================
# UI – Charger tickers
# ==============================
tsx_df, tickers_yahoo = get_tsx_constituents()

search = st.text_input("Recherche ticker", "").lower().strip()
if search:
    tsx_df = tsx_df[tsx_df["Symbol"].str.lower().str.contains(search)]

tickers = [fix_tsx_ticker(t) for t in tsx_df["Symbol"]]

st.subheader(f"📈 {len(tickers)} tickers à scanner")

mode = st.sidebar.selectbox("Mode stratégie", ["Balanced", "Strict", "Aggressive"])
use_rsi50 = st.sidebar.checkbox("Filtre RSI>50", True)

go = st.button("▶️ Scanner")
if not go:
    st.stop()

bars, failed = download_yahoo_bars(tuple(tickers))
st.write(f"Tickers valides : {len(bars)}")
if failed:
    st.warning(f"Échec : {failed[:10]}")

# ==============================
# 🔥 Calcul Signaux
# ==============================
results = []

for sym in tsx_df["Symbol"]:
    ysym = fix_tsx_ticker(sym)
    df = bars.get(ysym)

    if df is None or len(df) < 82:
        continue

    ha = to_heikin_ashi(df)

    h, l, c = ha["High"], ha["Low"], ha["Close"]
    v = ha["Volume"]

   # === Ichimoku components ===
    tenkan, kijun, spanA, spanB = ichimoku_components(h, l)

    # Force Series format
    c = pd.Series(c, index=c.index)
    tenkan = pd.Series(tenkan, index=tenkan.index)
    kijun = pd.Series(kijun, index=kijun.index)
    spanA = pd.Series(spanA, index=spanA.index)
    spanB = pd.Series(spanB, index=spanB.index)

    # Align all on common index
    idx = (
        c.index
        .intersection(tenkan.index)
        .intersection(kijun.index)
        .intersection(spanA.index)
        .intersection(spanB.index)
    )

    c = c.loc[idx]
    tenkan = tenkan.loc[idx]
    kijun = kijun.loc[idx]
    spanA = spanA.loc[idx]
    spanB = spanB.loc[idx]

    # Cloud
    upperCloud = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lowerCloud = pd.concat([spanA, spanB], axis=1).min(axis=1)

    # Align clouds as well
    upperCloud = upperCloud.loc[idx]
    lowerCloud = lowerCloud.loc[idx]

    # Drop NaN (MANDATORY)
    valid_idx = (
        c.dropna().index
        .intersection(upperCloud.dropna().index)
        .intersection(lowerCloud.dropna().index)
        .intersection(tenkan.dropna().index)
        .intersection(kijun.dropna().index)
    )

    c = c.loc[valid_idx]
    upperCloud = upperCloud.loc[valid_idx]
    lowerCloud = lowerCloud.loc[valid_idx]
    tenkan = tenkan.loc[valid_idx]
    kijun = kijun.loc[valid_idx]

    # === FINAL COMPARISONS (safe with numpy arrays) ===
    aboveCloud = c.values > upperCloud.values
    belowCloud = c.values < lowerCloud.values
    bullTK = tenkan.values > kijun.values
    bearTK = tenkan.values < kijun.values
    rsi14 = rsi_wilder(c)
    wr = williams_r(h, l, c)
    vo = volume_oscillator(v)

    wr_up = cross_recent(crossover(wr, pd.Series(-80, index=wr.index)), 14)
    wr_dn = cross_recent(crossunder(wr, pd.Series(-20, index=wr.index)), 14)

    if mode == "Strict":
        buyCond  = aboveCloud & bullTK & (rsi14>50) & wr_up & (vo>0)
        sellCond = belowCloud & bearTK & (rsi14<50) & wr_dn & (vo<0)
    else:
        buyCond  = (c>kijun) & (spanA>spanB) & ((rsi14>50) if use_rsi50 else True) & wr_up
        sellCond = (c<kijun) & (spanA<spanB) & ((rsi14<50) if use_rsi50 else True) & wr_dn

    buy_now  = bool(buyCond.iloc[-1])
    sell_now = bool(sellCond.iloc[-1])
    buy_new  = buy_now  and not bool(buyCond.iloc[-2])
    sell_new = sell_now and not bool(sellCond.iloc[-2])

    results.append({
        "Symbol": sym,
        "Yahoo": ysym,
        "Buy": buy_now,
        "Sell": sell_now,
        "BuyNew": buy_new,
        "SellNew": sell_new,
        "Close": float(df["Close"].iloc[-1]),
        "RSI": float(rsi14.iloc[-1]),
        "WR": float(wr.iloc[-1]),
        "VO": float(vo.iloc[-1]),
        "LastDate": df.index[-1].date(),
    })

res = pd.DataFrame(results)

# ==============================
# Affichage
# ==============================
show = st.selectbox("Afficher", ["Tous","Buy","Sell","BuyNew"])

if show=="Buy":    res = res[res["Buy"]]
if show=="Sell":   res = res[res["Sell"]]
if show=="BuyNew": res = res[res["BuyNew"]]

st.dataframe(res, use_container_width=True)
