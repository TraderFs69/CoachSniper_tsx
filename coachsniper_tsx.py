import os, time, random, datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Coach Sniper – TSX", layout="wide")
st.title("🏒 Coach Sniper – TSX Composite Scanner (Yahoo Finance)")

# ==============================
# 1) FIX TICKERS TSX → Yahoo
# ==============================
def fix_tsx_ticker(ticker: str) -> str:
    t = ticker.strip().lower()

    # AP.UN → AP-UN.TO
    if ".un" in t:
        base = t.split(".")[0].upper()
        return f"{base}-UN.TO"

    # CTC.A → CTC-A.TO
    if "." in t:
        base, suffix = t.split(".", 1)
        return f"{base.upper()}-{suffix.upper()}.TO"

    # RY → RY.TO
    return f"{t.upper()}.TO"


# ==============================
# 2) Charger fichier TSX
# ==============================
@st.cache_data(ttl=3600)
def load_tsx_constituents():
    path = "tsxcomposite_constituents.xlsx"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable dans le dossier Streamlit.")

    df = pd.read_excel(path)

    # Normalisation
    df.columns = [c.strip().lower() for c in df.columns]

    # Trouver la colonne tickers
    colmap = {"symbol": "Symbol", "ticker": "Symbol", "security": "Symbol"}
    renamed = False
    for col in df.columns:
        if col in colmap:
            df.rename(columns={col: "Symbol"}, inplace=True)
            renamed = True
            break

    if not renamed:
        st.error("Impossible de trouver la colonne des tickers (Symbol / Security).")
        st.stop()

    df = df[df["Symbol"].notna()]
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df[df["Symbol"] != ""]

    tickers = [fix_tsx_ticker(s) for s in df["Symbol"]]
    return df, tickers


# ==============================
# 3) Télécharger données Yahoo
# ==============================
@st.cache_data(ttl=3600)
def download_yahoo_bars(tickers):
    out = {}
    failed = []
    end = dt.date.today()
    start = end - dt.timedelta(days=730)

    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, interval="1d", progress=False)

            if df.empty:
                failed.append(t)
                continue

            df = df.rename(columns=str.capitalize)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            out[t] = df

        except Exception:
            failed.append(t)

        time.sleep(0.1)

    return out, failed


# ==============================
# 4) Heikin Ashi
# ==============================
def to_heikin_ashi(df):
    df = df.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2

    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    df["Open"] = ha_open
    df["Close"] = ha_close
    df["High"] = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    df["Low"] = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)
    return df


# ==============================
# 5) Ichimoku + Indicators
# ==============================
def ichimoku_components(high, low, t=9, k=26, s=52):
    tenkan = (high.rolling(t).max() + low.rolling(t).min()) / 2
    kijun  = (high.rolling(k).max() + low.rolling(k).min()) / 2
    spanA  = (tenkan + kijun) / 2
    spanB  = (high.rolling(s).max() + low.rolling(s).min()) / 2
    return tenkan, kijun, spanA, spanB

def rsi(close, length=14):
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def wr(high, low, close, length=14):
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll)


# ==============================
# UI
# ==============================
tsx_df, tickers = load_tsx_constituents()

search = st.text_input("Recherche ticker TSX :", "").lower().strip()
if search:
    tsx_df = tsx_df[tsx_df["Symbol"].str.lower().str.contains(search)]

tickers = [fix_tsx_ticker(t) for t in tsx_df["Symbol"]]

st.write(f"📈 {len(tickers)} tickers détectés.")

if st.button("▶️ Scanner maintenant"):
    bars, failed = download_yahoo_bars(tuple(tickers))

    st.write(f"Valides : {len(bars)} — Échecs : {len(failed)}")

    results = []

    for sym in tsx_df["Symbol"]:
        ysym = fix_tsx_ticker(sym)
        df = bars.get(ysym)

        if df is None or len(df) < 120:
            continue

        ha = to_heikin_ashi(df)

        # SAFE 1D
        c = ha["Close"].astype(float)
        h = ha["High"].astype(float)
        l = ha["Low"].astype(float)

        # Ichimoku
        tenkan, kijun, spanA, spanB = ichimoku_components(h, l)

        idx = (
            c.index
            .intersection(tenkan.index)
            .intersection(kijun.index)
            .intersection(spanA.index)
            .intersection(spanB.index)
        )

        if len(idx) < 50:
            continue

        c = c.loc[idx]
        tenkan = tenkan.loc[idx]
        kijun = kijun.loc[idx]
        spanA = spanA.loc[idx]
        spanB = spanB.loc[idx]

        upper = pd.concat([spanA, spanB], axis=1).max(axis=1).loc[idx]
        lower = pd.concat([spanA, spanB], axis=1).min(axis=1).loc[idx]

        # Numpy-safe comparisons
        above = np.asarray(c) > np.asarray(upper)
        below = np.asarray(c) < np.asarray(lower)
        bullTK = np.asarray(tenkan) > np.asarray(kijun)
        bearTK = np.asarray(tenkan) < np.asarray(kijun)

        # Indicators
        r = rsi(c).iloc[-1]
        w = wr(h, l, c).iloc[-1]

        buy = above[-1] and bullTK[-1] and r > 50 and w > -80
        sell = below[-1] and bearTK[-1] and r < 50 and w < -20

        results.append({
            "Symbol": sym,
            "Yahoo": ysym,
            "Buy": buy,
            "Sell": sell,
            "Close": float(df["Close"].iloc[-1]),
            "RSI": round(r,2),
            "WR": round(w,2),
        })

    st.subheader("📊 Résultats")
    st.dataframe(pd.DataFrame(results), use_container_width=True)
