import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Coach Sniper – TSX", layout="wide")
st.title("📈 Coach Sniper – TSX Scanner")

# ---------------------------------------------------------
# STRATÉGIE MODE (GLOBAL)
# ---------------------------------------------------------
st.sidebar.header("Stratégie")
strategy_mode = st.sidebar.selectbox(
    "Mode stratégie",
    ["Strict", "Balanced", "Balanced Permissif"]
)

# ---------------------------------------------------------
# LECTURE DU FICHIER TSX
# ---------------------------------------------------------
@st.cache_data
def load_tsx_list():
    df = pd.read_excel("tsxcomposite_constituents.xlsx")
    df.columns = df.columns.str.lower()

    # Chercher colonne de symboles
    symbol_col = None
    for c in ["symbol", "ticker", "security"]:
        if c in df.columns:
            symbol_col = c
            break

    if symbol_col is None:
        st.error("❌ Impossible de trouver une colonne 'Symbol', 'Ticker' ou 'Security'.")
        st.stop()

    df["symbol"] = df[symbol_col].astype(str).str.strip()

    # Conversion vers Yahoo
    def convert_to_yahoo(symbol):
        s = symbol.upper().replace(" ", "")

        # UNITÉS : X.UN → X-UN.TO
        if ".UN" in s:
            b = s.replace(".UN", "")
            return f"{b}-UN.TO"

        # CLASSES : CTC.A → CTC-A.TO
        if "." in s:
            a, b = s.split(".")
            return f"{a}-{b}.TO"

        # Normal : RY, SHOP → RY.TO
        return f"{s}.TO"

    df["yahoo"] = df["symbol"].apply(convert_to_yahoo)
    return df

tsx_df = load_tsx_list()
tickers = tsx_df["yahoo"].tolist()

st.write(f"Tickers TSX chargés : {len(tickers)}")


# ---------------------------------------------------------
# FONCTIONS INDICATEURS
# ---------------------------------------------------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length).mean()
    avg_loss = loss.ewm(alpha=1/length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def williams_r(high, low, close, length=14):
    highest = high.rolling(length).max()
    lowest  = low.rolling(length).min()
    return -100 * (highest - close) / (highest - lowest)

def ichimoku(high, low):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    spanA = ((tenkan + kijun) / 2).shift(26)
    spanB = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, spanA, spanB


# ---------------------------------------------------------
# TÉLÉCHARGEMENT YAHOO
# ---------------------------------------------------------
def get_history(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty:
            return None

        df = df.rename(columns=str.title)
        for col in ["Open","High","Low","Close","Volume"]:
            if col not in df.columns:
                return None

        return df
    except:
        return None


# ---------------------------------------------------------
# COMPUTE SIGNALS — VERSION ANTI-FAIL + MODE PERMISSIF
# ---------------------------------------------------------
def compute_signals(df):

    if df is None or len(df) < 120:
        return None

    df = df.copy()

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # Ichimoku
    tenkan, kijun, spanA, spanB = ichimoku(high, low)

    # Alignement anti-fail
    for x in [tenkan, kijun, spanA, spanB]:
        x.fillna(method="bfill", inplace=True)
        x.fillna(method="ffill", inplace=True)

    # RSI & WR
    r = rsi(close)
    wr = williams_r(high, low, close)

    # Volume oscillator
    vo = ema(df["Volume"], 5) - ema(df["Volume"], 20)

    # Remplissage NaN pour éviter crash
    r = r.fillna(50)
    wr = wr.fillna(-50)
    vo = vo.fillna(0)

    # ==========================================================
    # MODE STRICT
    # ==========================================================
    if strategy_mode == "Strict":
        buy = (close > spanA) & (tenkan > kijun) & (r > 50) & (wr > -80)
        sell = (close < spanB) & (tenkan < kijun) & (r < 50) & (wr < -20)

    # ==========================================================
    # MODE BALANCED (standard)
    # ==========================================================
    elif strategy_mode == "Balanced":
        buy = (close > kijun) & (r > 48) & (wr > -85)
        sell = (close < kijun) & (r < 52) & (wr < -15)

    # ==========================================================
    # MODE BALANCED PERMISSIF (NOUVEAU)
    # ==========================================================
    else:  # Balanced Permissif
        trend_long  = (close > kijun) & ((tenkan >= kijun) | (spanA > spanB))
        trend_short = (close < kijun) & ((tenkan <= kijun) | (spanA < spanB))

        rsi_long  = r > 45
        rsi_short = r < 55

        wr_long  = wr > -88
        wr_short = wr < -12

        vo_long  = vo > -5
        vo_short = vo < 5

        buy = trend_long & rsi_long & wr_long & vo_long
        sell = trend_short & rsi_short & wr_short & vo_short

    # Nouveaux signaux
    buy_new = buy & (~buy.shift(1).fillna(False))
    sell_new = sell & (~sell.shift(1).fillna(False))

    return {
        "Close": float(close.iloc[-1]),
        "Buy": bool(buy.iloc[-1]),
        "Sell": bool(sell.iloc[-1]),
        "BuyNew": bool(buy_new.iloc[-1]),
        "SellNew": bool(sell_new.iloc[-1]),
        "RSI": float(r.iloc[-1]),
        "WR": float(wr.iloc[-1])
    }


# ---------------------------------------------------------
# LANCEMENT DU SCAN
# ---------------------------------------------------------
st.subheader("🔍 Résultats du scan TSX")

results = []

if st.button("🚀 Lancer le scan"):
    for i, ticker in enumerate(tickers):
        st.write(f"{i+1}/{len(tickers)} — {ticker}")

        df = get_history(ticker)
        sig = compute_signals(df)

        symbol = tsx_df.loc[tsx_df["yahoo"] == ticker, "symbol"].values[0]

        if sig is None:
            results.append({
                "Symbol": symbol,
                "Yahoo": ticker,
                "Close": None,
                "Buy": False,
                "Sell": False,
                "BuyNew": False,
                "SellNew": False,
                "RSI": None,
                "WR": None
            })
            continue

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

    df_res = pd.DataFrame(results)
    st.dataframe(df_res, use_container_width=True)

    csv = df_res.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger CSV", csv, "tsx_scan.csv", "text/csv")
