import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import os

# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Coach Sniper TSX Scanner", layout="wide")
st.title("🏒 Coach Sniper – TSX Composite Scanner (ANTI-FAIL VERSION)")

# ---------------------------------------------------------
# LOAD CONSTITUENTS
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_tsx():
    file = "tsxcomposite_constituents.xlsx"
    if not os.path.exists(file):
        st.error("Fichier tsxcomposite_constituents.xlsx introuvable.")
        st.stop()

    df = pd.read_excel(file)
    df.columns = [c.lower().strip() for c in df.columns]

    if "symbol" not in df.columns:
        st.error(f"Colonne 'Symbol' absente. Colonnes trouvées : {df.columns.tolist()}")
        st.stop()

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[df["symbol"] != ""]

    return df

tsx_df = load_tsx()

# ---------------------------------------------------------
# TICKER CONVERSION (AP.UN → AP-UN.TO, CTC.A → CTC-A.TO)
# ---------------------------------------------------------
def convert_to_yahoo(symbol: str) -> str:
    s = symbol.upper().strip()

    if ".UN" in s:
        base = s.replace(".UN", "")
        return f"{base}-UN.TO"

    if "." in s:
        base, suf = s.split(".", 1)
        return f"{base}-{suf}.TO"

    return f"{s}.TO"

tsx_df["yahoo"] = tsx_df["symbol"].apply(convert_to_yahoo)
tickers = tsx_df["yahoo"].tolist()

st.sidebar.write(f"📊 Tickers TSX détectés : {len(tickers)}")

# ---------------------------------------------------------
# YAHOO DOWNLOAD (ANTI-FAIL)
# ---------------------------------------------------------
def normalize_columns(df):
    rename_map = {}

    for c in df.columns:
        lc = c.lower()

        if "close" in lc and "adj" not in lc:
            rename_map[c] = "Close"
        elif "adj" in lc and "close" in lc:
            rename_map[c] = "Close"
        elif "high" in lc:
            rename_map[c] = "High"
        elif "low" in lc:
            rename_map[c] = "Low"

    df = df.rename(columns=rename_map)

    required = ["Close", "High", "Low"]
    for r in required:
        if r not in df.columns:
            return None

    return df


@st.cache_data(ttl=3600)
def get_history(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty:
            return None

        df = df.reset_index(drop=True)
        df = normalize_columns(df)

        return df
    except Exception:
        return None

# ---------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------
def ema(s, l):
    return s.ewm(span=l, adjust=False).mean()

def rsi(s, l=14):
    d = s.diff()
    up = d.clip(lower=0)
    down = (-d).clip(lower=0)
    avg_up = up.rolling(l).mean()
    avg_down = down.rolling(l).mean().replace(0, 1e-9)
    rs = avg_up / avg_down
    return 100 - 100 / (1 + rs)

def wr(high, low, close, l=14):
    hh = high.rolling(l).max()
    ll = low.rolling(l).min()
    return -100 * (hh - close) / (hh - ll).replace(0, 1e-9)

def ichimoku(df):
    high = df["High"]
    low = df["Low"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    spanA = ((tenkan + kijun) / 2).shift(26)
    spanB = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    return tenkan, kijun, spanA, spanB

# ---------------------------------------------------------
# SIGNAL ENGINE (ANTI-FAIL LOGIC)
# ---------------------------------------------------------
def compute_signals(df):
    if df is None or len(df) < 80:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Indicators ALWAYS align on df.index
    r = rsi(close).reindex(df.index)
    w = wr(high, low, close).reindex(df.index)
    tenkan, kijun, spanA, spanB = ichimoku(df)
    tenkan = tenkan.reindex(df.index)
    kijun = kijun.reindex(df.index)
    spanA = spanA.reindex(df.index)
    spanB = spanB.reindex(df.index)

    # Cloud
    upper = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lower = pd.concat([spanA, spanB], axis=1).min(axis=1)

    # Latest values (skip safely)
    try:
        c0 = float(close.iloc[-1])
        t0 = float(tenkan.iloc[-1])
        k0 = float(kijun.iloc[-1])
        u0 = float(upper.iloc[-1])
        l0 = float(lower.iloc[-1])
        r0 = float(r.iloc[-1])
        w0 = float(w.iloc[-1])
    except:
        return None

    # Avoid NaN issues
    if any(math.isnan(x) for x in [c0, t0, k0, u0, l0, r0, w0]):
        return None

    # Conditions
    above = c0 > u0
    below = c0 < l0
    bull_tk = t0 > k0
    bear_tk = t0 < k0

    # Previous
    try:
        above_prev = float(close.iloc[-2]) > float(upper.iloc[-2])
        below_prev = float(close.iloc[-2]) < float(lower.iloc[-2])
        bull_prev = float(tenkan.iloc[-2]) > float(kijun.iloc[-2])
        bear_prev = float(tenkan.iloc[-2]) < float(kijun.iloc[-2])
        r_prev = float(r.iloc[-2])
        w_prev = float(w.iloc[-2])
    except:
        return None

    # Signals
    buy = above and bull_tk and r0 > 50 and w0 > -80
    sell = below and bear_tk and r0 < 50 and w0 < -20

    buy_new = buy and not (above_prev and bull_prev and r_prev > 50 and w_prev > -80)
    sell_new = sell and not (below_prev and bear_prev and r_prev < 50 and w_prev < -20)

    return {
        "Close": c0,
        "Buy": buy,
        "Sell": sell,
        "BuyNew": buy_new,
        "SellNew": sell_new,
        "RSI": r0,
        "WR": w0
    }

# ---------------------------------------------------------
# RUN SCAN
# ---------------------------------------------------------
if st.button("🚀 Lancer le scan (ANTI-FAIL)"):
    results = []

    for i, ticker in enumerate(tickers):
        st.write(f"{i+1}/{len(tickers)} – {ticker}")

        df = get_history(ticker)
        sig = compute_signals(df)

        if sig is None:
            continue

        symbol = tsx_df.loc[tsx_df["yahoo"] == ticker, "symbol"].values[0]

        results.append({
            "Symbol": symbol,
            "Yahoo": ticker,
            "Close": sig["Close"],
            "Buy": sig["Buy"],
            "Sell": sig["Sell"],
            "BuyNew": sig["BuyNew"],
            "SellNew": sig["SellNew"],
            "RSI": sig["RSI"],
            "WR": sig["WR"]
        })

    if not results:
        st.error("❌ Aucun résultat. (Très improbable avec cette version)")

    df_res = pd.DataFrame(results)
    st.subheader("📊 Résultats du scan")
    st.dataframe(df_res, use_container_width=True)

    csv = df_res.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger CSV", csv, "tsx_scan.csv", "text/csv")
