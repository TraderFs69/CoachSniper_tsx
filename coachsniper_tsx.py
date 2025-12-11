import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Coach Sniper – TSX Scanner", layout="wide")
st.title("📈 Coach Sniper – TSX Composite Scanner")


# ==========================================================
# STRATÉGIE (GLOBAL SELECTOR)
# ==========================================================
st.sidebar.header("Stratégie de Trading")
strategy_mode = st.sidebar.selectbox(
    "Mode stratégie",
    ["Strict", "Balanced", "Balanced Permissif"]
)


# ==========================================================
# LECTURE DU FICHIER TSX ET ADAPTATION POUR YAHOO
# ==========================================================
@st.cache_data
def load_tsx_list():
    df = pd.read_excel("tsxcomposite_constituents.xlsx")
    df.columns = df.columns.str.lower()

    # identifier colonne symbol
    col = None
    for c in ["symbol", "ticker", "security"]:
        if c in df.columns:
            col = c
            break

    if col is None:
        st.error("❌ Aucun champ 'Symbol', 'Ticker' ou 'Security' trouvé dans le fichier.")
        st.stop()

    df["symbol"] = df[col].astype(str).str.strip()

    # Conversion Yahoo
    def to_yahoo(symbol):
        s = symbol.upper().replace(" ", "")

        # Fonds / unités : X.UN → X-UN.TO
        if ".UN" in s:
            base = s.replace(".UN", "")
            return f"{base}-UN.TO"

        # Classes d'action : CTC.A → CTC-A.TO
        if "." in s:
            a, b = s.split(".")
            return f"{a}-{b}.TO"

        # Sinon standard : RY → RY.TO
        return f"{s}.TO"

    df["yahoo"] = df["symbol"].apply(to_yahoo)
    return df


tsx_df = load_tsx_list()
tickers = tsx_df["yahoo"].tolist()

st.write(f"📌 {len(tickers)} tickers TSX détectés")


# ==========================================================
# INDICATEURS TECHNIQUES
# ==========================================================
def ema(s, length):
    return s.ewm(span=length, adjust=False).mean()

def rsi(close, length=14):
    d = close.diff()
    gain = d.where(d > 0, 0)
    loss = -d.where(d < 0, 0)
    avg_gain = gain.ewm(alpha=1/length).mean()
    avg_loss = loss.ewm(alpha=1/length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100/(1+rs))

def williams_r(high, low, close, length=14):
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll)

def ichimoku(high, low):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    spanA  = ((tenkan + kijun) / 2).shift(26)
    spanB  = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, spanA, spanB


# ==========================================================
# YAHOO FINANCE DOWNLOAD
# ==========================================================
def get_history(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty:
            return None

        df = df.rename(columns=str.title)

        # vérifier
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in df.columns:
                return None

        return df

    except Exception:
        return None


# ==========================================================
# SIGNALS (VERSION ANTI-FAIL + 3 MODES)
# ==========================================================
def compute_signals(df):

    if df is None or len(df) < 120:
        return None

    df = df.copy()

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # Ichimoku
    tenkan, kijun, spanA, spanB = ichimoku(high, low)

    # alignement anti-fail
    for x in [tenkan, kijun, spanA, spanB]:
        x.reindex(close.index)
        x.fillna(method="bfill", inplace=True)
        x.fillna(method="ffill", inplace=True)

    # RSI, Williams R & Volume
    r  = rsi(close).reindex(close.index).fillna(50)
    wr = williams_r(high, low, close).reindex(close.index).fillna(-50)
    vo = (ema(df["Volume"], 5) - ema(df["Volume"], 20)).reindex(close.index).fillna(0)

    # ======================================================
    # MODE STRICT
    # ======================================================
    if strategy_mode == "Strict":
        buy  = (close > spanA) & (tenkan > kijun) & (r > 50) & (wr > -80)
        sell = (close < spanB) & (tenkan < kijun) & (r < 50) & (wr < -20)

    # ======================================================
    # MODE BALANCED (standard)
    # ======================================================
    elif strategy_mode == "Balanced":
        buy  = (close > kijun) & (r > 48) & (wr > -85)
        sell = (close < kijun) & (r < 52) & (wr < -15)

    # ======================================================
    # MODE BALANCED PERMISSIF
    # ======================================================
    else:
        trend_long  = (close > kijun) & ((tenkan >= kijun) | (spanA > spanB))
        trend_short = (close < kijun) & ((tenkan <= kijun) | (spanA < spanB))

        rsi_long  = r > 45
        rsi_short = r < 55

        wr_long  = wr > -88
        wr_short = wr < -12

        vo_long  = vo > -5
        vo_short = vo < 5

        buy  = trend_long  & rsi_long  & wr_long  & vo_long
        sell = trend_short & rsi_short & wr_short & vo_short

    # nouveaux signaux
    buy_new  = buy & (~buy.shift(1).fillna(False))
    sell_new = sell & (~sell.shift(1).fillna(False))

    # ======================================================
    # PATCH ANTI-FAIL — convert Series → bool python
    # ======================================================
    def extract_bool(x):
        try:
            return bool(x.item())
        except:
            try:
                return bool(x)
            except:
                return False

    last_buy      = extract_bool(buy.iloc[-1])
    last_sell     = extract_bool(sell.iloc[-1])
    last_buy_new  = extract_bool(buy_new.iloc[-1])
    last_sell_new = extract_bool(sell_new.iloc[-1])

    return {
        "Close": float(close.iloc[-1]),
        "Buy": last_buy,
        "Sell": last_sell,
        "BuyNew": last_buy_new,
        "SellNew": last_sell_new,
        "RSI": float(r.iloc[-1]),
        "WR": float(wr.iloc[-1])
    }


# ==========================================================
# SCAN UI
# ==========================================================
st.subheader("🔍 Résultats du Scan TSX")

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
