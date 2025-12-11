import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import math
import os

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(page_title="Coach Sniper – TSX Scanner", layout="wide")
st.title("🏒 Coach Sniper – Scanner TSX Composite (Yahoo Finance)")

# =========================================================
# LOAD TSX CONSTITUENTS
# =========================================================
@st.cache_data(ttl=3600)
def get_tsx_constituents():
    path = "tsxcomposite_constituents.xlsx"
    if not os.path.exists(path):
        raise FileNotFoundError("Le fichier tsxcomposite_constituents.xlsx est introuvable.")

    df = pd.read_excel(path)
    cols = [c.lower().strip() for c in df.columns]

    if "symbol" not in cols:
        raise ValueError(f"Colonnes trouvées : {df.columns.tolist()} — aucune colonne 'Symbol' détectée.")

    df.columns = cols
    df = df.rename(columns={"symbol": "Symbol"})
    df["Symbol"] = df["Symbol"].astype(str).str.strip()

    # Nettoyage minimal
    df = df[df["Symbol"] != ""]

    return df, df["Symbol"].tolist()

tsx_df, raw_symbols = get_tsx_constituents()

# =========================================================
# CONVERSION SYMBOLS → YAHOO FINANCE (.TO)
# TSX uniquement, pas TSXV
# =========================================================
def convert_to_yahoo(symbol: str) -> str:
    """Convertit un ticker TSX en format Yahoo Finance."""
    s = symbol.upper().strip()

    # Gestion .UN → -UN.TO
    if ".UN" in s:
        base = s.replace(".UN", "")
        return f"{base}-UN.TO"

    # Gestion .U, .K, etc : CTC.A → CTC-A.TO
    if "." in s:
        base, suffix = s.split(".", 1)
        return f"{base}-{suffix}.TO"

    # Cas normal
    return f"{s}.TO"

tsx_df["Yahoo"] = tsx_df["Symbol"].apply(convert_to_yahoo)
tickers = tsx_df["Yahoo"].tolist()

st.sidebar.write(f"📊 Nombre de tickers TSX : {len(tickers)}")

# =========================================================
# INDICATEURS TECHNIQUES
# =========================================================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean().replace(0, 1e-9)

    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    return out

def williams_r(high, low, close, length=14):
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    wr = -100 * (hh - close) / (hh - ll).replace(0, 1e-9)
    return wr

def ichimoku(df):
    high = df["High"]
    low = df["Low"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    spanA = ((tenkan + kijun) / 2).shift(26)
    spanB = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    return tenkan, kijun, spanA, spanB

# =========================================================
# DOWNLOAD DATA YAHOO
# =========================================================
@st.cache_data(ttl=3600)
def download_history(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.title)
        return df
    except Exception:
        return None

# =========================================================
# LOGIC BUY/SELL
# =========================================================
def compute_signals(df):

    if df is None or len(df) < 80:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    rsi14 = rsi(close, 14)
    wr = williams_r(high, low, close, 14)

    tenkan, kijun, spanA, spanB = ichimoku(df)

    # Cloud
    upper_cloud = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lower_cloud = pd.concat([spanA, spanB], axis=1).min(axis=1)

    above_cloud = close > upper_cloud
    below_cloud = close < lower_cloud
    bull_tk = tenkan > kijun
    bear_tk = tenkan < kijun

    # Extraction scalaires sécurisés
    try:
        r = float(rsi14.iloc[-1])
        w = float(wr.iloc[-1])
    except:
        return None

    if math.isnan(r) or math.isnan(w):
        return None

    # Conditions BUY
    buy = (
        above_cloud.iloc[-1] and
        bull_tk.iloc[-1] and
        r > 50 and
        w > -80
    )

    # Conditions SELL
    sell = (
        below_cloud.iloc[-1] and
        bear_tk.iloc[-1] and
        r < 50 and
        w < -20
    )

    # NEW SIGNALS
    buy_new = False
    sell_new = False

    if len(close) >= 2:
        buy_new = buy and not (
            above_cloud.iloc[-2] and bull_tk.iloc[-2] and
            float(rsi14.iloc[-2]) > 50 and float(wr.iloc[-2]) > -80
        )

        sell_new = sell and not (
            below_cloud.iloc[-2] and bear_tk.iloc[-2] and
            float(rsi14.iloc[-2]) < 50 and float(wr.iloc[-2]) < -20
        )

    return {
        "Buy": buy,
        "Sell": sell,
        "BuyNew": buy_new,
        "SellNew": sell_new,
        "Close": float(close.iloc[-1]),
        "RSI": r,
        "WR": w
    }

# =========================================================
# UI – SCAN BUTTON
# =========================================================
if st.button("🚀 Lancer le scan TSX"):
    st.write("Téléchargement des données…")

    results = []

    for i, t in enumerate(tickers):
        st.write(f"{i+1}/{len(tickers)} : {t}")
        df = download_history(t)
        sig = compute_signals(df)

        if sig is None:
            continue

        symbol = tsx_df.loc[tsx_df["Yahoo"] == t, "Symbol"].values[0]

        results.append({
            "Symbol": symbol,
            "Yahoo": t,
            "Close": sig["Close"],
            "Buy": sig["Buy"],
            "Sell": sig["Sell"],
            "BuyNew": sig["BuyNew"],
            "SellNew": sig["SellNew"],
            "RSI": sig["RSI"],
            "WR": sig["WR"]
        })

    if not results:
        st.error("Aucun résultat.")
        st.stop()

    res = pd.DataFrame(results)

    st.subheader("📊 Résultats du scan TSX")
    st.dataframe(res, use_container_width=True)

    csv = res.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger CSV", csv, "tsx_scan.csv", "text/csv")
