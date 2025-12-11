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
# Conversion tickers TSX → Yahoo Finance
# ==============================
def fix_tsx_ticker(ticker: str) -> str:
    t = ticker.strip().lower()

    if "." not in t:
        return t.upper() + ".TO"

    if ".un" in t:
        base = t.split(".")[0].upper()
        return f"{base}-UN.TO"

    return t.upper() + ".TO"

# ==============================
# Charger liste TSX avec gestion robuste
# ==============================
@st.cache_data(show_spinner=True, ttl=3600)
def get_tsx_constituents():
    path = "tsxcomposite_constituents.xlsx"
    if not os.path.exists(path):
        st.error(f"❌ Fichier introuvable : {path}")
        raise FileNotFoundError(path)

    df = pd.read_excel(path)

    # === DEBUG affichage pour comprendre ce que Streamlit lit ===
    st.write("🔍 DEBUG – Colonnes détectées :", df.columns.tolist())
    st.write("🔍 DEBUG – Aperçu du fichier :")
    st.write(df.head())

    # === Normalisation automatique des colonnes ===
    # On remplace toutes les colonnes par une version nettoyée
    normalized_cols = [c.strip().lower() for c in df.columns]
    df.columns = normalized_cols

    # Gestion des variantes : symbol, ticker, tickers
    if "symbol" in df.columns:
        df.rename(columns={"symbol": "Symbol"}, inplace=True)
    elif "ticker" in df.columns:
        df.rename(columns={"ticker": "Symbol"}, inplace=True)
    elif "tickers" in df.columns:
        df.rename(columns={"tickers": "Symbol"}, inplace=True)
    else:
        st.error("❌ Impossible de trouver la colonne avec les tickers.")
        st.write("Colonnes disponibles :", df.columns.tolist())
        raise ValueError("Aucune colonne 'Symbol', 'Ticker' ou 'Tickers' trouvée.")

    # On supprime les lignes vides
    df = df[df["Symbol"].notna()]
    df = df[df["Symbol"].astype(str).str.strip() != ""]

    # Ajout Company & Sector si absents
    df["Company"] = df["Symbol"]
    df["Sector"] = "Unknown"

    # Conversion tickers en Yahoo format
    tickers_yahoo = [fix_tsx_ticker(t) for t in df["Symbol"].tolist()]

    return df, tickers_yahoo

# ==============================
# Indicateurs (inchangés)
# ==============================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def crossover(a, b): return (a > b) & (a.shift(1) <= b.shift(1))
def crossunder(a, b): return (a < b) & (a.shift(1) >= b.shift(1))

def cross_recent(cross, lookback=3):
    out = cross.copy().astype(bool)
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
# Heikin Ashi
# ==============================
def to_heikin_ashi(df):
    df = df.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    df["Open"], df["High"], df["Low"], df["Close"] = ha_open, \
        pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1), \
        pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1), \
        ha_close
    return df

# ==============================
# Télécharger OHLC Yahoo Finance
# ==============================
@st.cache_data(show_spinner=True, ttl=3600)
def download_yahoo(tickers):
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
            out[t] = df[["Open","High","Low","Close","Volume"]]
        except:
            failed.append(t)

        time.sleep(0.15 + random.random()*0.15)

    return out, failed

# ==============================
# UI – Chargement tickers
# ==============================
st.subheader("📥 Chargement des constituants TSX")
tsx_df, tickers_yahoo = get_tsx_constituents()

search = st.text_input("Recherche", "").lower().strip()
if search:
    tsx_df = tsx_df[tsx_df["Symbol"].str.lower().str.contains(search)]

tickers = [fix_tsx_ticker(t) for t in tsx_df["Symbol"].tolist()]

st.write(f"📈 {len(tickers)} tickers à scanner.")

mode = st.sidebar.selectbox("Mode stratégie", ["Balanced","Strict","Aggressive"])
use_rsi50 = st.sidebar.checkbox("Filtre RSI>50", True)

go = st.button("▶️ Scanner")
if not go:
    st.stop()

bars, failed = download_yahoo(tuple(tickers))
st.write(f"Valid tickers: {len(bars)}")
if failed:
    st.warning(f"Échec : {failed[:10]}")

# ==============================
# Calcul signaux
# ==============================
results = []
for symbol in tsx_df["Symbol"]:
    ysym = fix_tsx_ticker(symbol)
    df = bars.get(ysym)
    if df is None or len(df) < 82:
        continue

    data = to_heikin_ashi(df)

    h, l, c, v = data["High"], data["Low"], data["Close"], data["Volume"]
    tenkan, kijun, spanA, spanB = ichimoku_components(h, l)
    upperCloud = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lowerCloud = pd.concat([spanA, spanB], axis=1).min(axis=1)

    rsi14 = rsi_wilder(c)

    aboveCloud = c > upperCloud
    belowCloud = c < lowerCloud
    bullTK = tenkan > kijun
    bearTK = tenkan < kijun

    wr = williams_r(h, l, c)

    wr_up_recent = cross_recent(crossover(wr, pd.Series(-80, index=wr.index)), 14)
    wr_dn_recent = cross_recent(crossunder(wr, pd.Series(-20, index=wr.index)), 14)

    vo = volume_oscillator(v)

    if mode == "Strict":
        buyCond  = aboveCloud & bullTK & (rsi14>50) & wr_up_recent & (vo>0)
        sellCond = belowCloud & bearTK & (rsi14<50) & wr_dn_recent & (vo<0)
    else:
        buyCond  = (c>kijun) & (spanA>spanB) & (rsi14>50 if use_rsi50 else True) & wr_up_recent
        sellCond = (c<kijun) & (spanA<spanB) & (rsi14<50 if use_rsi50 else True) & wr_dn_recent

    buy_now = bool(buyCond.iloc[-1])
    sell_now = bool(sellCond.iloc[-1])
    buy_new = buy_now and not bool(buyCond.iloc[-2])
    sell_new = sell_now and not bool(sellCond.iloc[-2])

    results.append({
        "Symbol": symbol,
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

show = st.selectbox("Afficher", ["Tous","Buy","Sell","BuyNew"])
if show=="Buy":
    res = res[res["Buy"]]
elif show=="Sell":
    res = res[res["Sell"]]
elif show=="BuyNew":
    res = res[res["BuyNew"]]

st.dataframe(res, use_container_width=True)
