# app.py
import os
import re
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Optional deps (graceful fallbacks)
try:
    from tensorflow.keras.models import load_model  # noqa: F401
    TF_OK = True
except Exception:
    TF_OK = False

try:
    # Use class-based API for best compatibility across ta versions
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    TA_OK = True
except Exception:
    TA_OK = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_OK = True
except Exception:
    NLTK_OK = False

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Your LSTM utilities (guarded import)
PREP_OK = True
try:
    from stock_predictor import prepare_data, train_model, predict_future  # noqa: F401
except Exception:
    PREP_OK = False

# ---------------------------
# Page Setup & Styling
# ---------------------------
st.set_page_config(
    page_title="Stock Price Prediction & Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FFFFFF; }
    .title {
        text-align: center; font-size: 2.2rem; font-weight: 700;
        color: #00BFFF; margin-bottom: 0.2rem;
    }
    .metric-box {
        background-color: #1E1E1E; border-radius: 12px; padding: 18px;
        text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .news-card {
        background-color: #1E1E1E; padding: 14px; border-radius: 10px;
        margin-bottom: 10px; line-height: 1.3;
    }
    .footer { text-align: center; color: #9aa0a6; font-size: 0.9em; margin-top: 24px; }
    a { color: #1E90FF; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📊 Stock Price Prediction & Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Primary Stock Symbol", "AAPL").upper().strip()
period = st.sidebar.selectbox("History Period", ["1y", "2y", "5y", "10y"], index=2)
interval = st.sidebar.selectbox("Price Interval", ["1d", "1wk", "1mo"], index=0)
# Keep the sidebar button if you like, but the tab button is what triggers predictions now
st.sidebar.button("🔮 Run Prediction (see tab)")

st.sidebar.markdown("---")
compare_toggle = st.sidebar.checkbox("🔁 Compare with another stock")
compare_ticker = ""
if compare_toggle:
    compare_ticker = st.sidebar.text_input("Comparison Symbol (optional)", "MSFT").upper().strip()

# ---------------------------
# Helpers
# ---------------------------
def valid_ticker(tkr):
    """Return (is_valid, info) using yfinance. Use .fast_info first, fallback to .info."""
    try:
        t = yf.Ticker(tkr)
        # fast_info is quicker and more reliable than .info
        finfo = getattr(t, "fast_info", None)
        if finfo and finfo.get("last_price") is not None:
            return True, {"longName": tkr, "regularMarketPrice": finfo.get("last_price"),
                          "marketCap": finfo.get("market_cap"), "volume": finfo.get("last_volume")}
        # fallback to .info (slower / sometimes empty)
        info = t.info
        if info and info.get("regularMarketPrice") is not None:
            return True, info
        return False, {}
    except Exception:
        return False, {}

@st.cache_data(show_spinner=False, ttl=600)
def fetch_history(tkr, period_val, interval_val):
    try:
        df = yf.download(tkr, period=period_val, interval=interval_val, progress=False, auto_adjust=False)
        # Ensure standard columns exist
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for col in needed:
            if col not in df.columns:
                df[col] = np.nan
        return df
    except Exception:
        return pd.DataFrame()

def add_indicators(df):
    """Add SMA/EMA/RSI/MACD/Bollinger Bands if ta is available."""
    if df.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        return df
    if not TA_OK:
        return df

    out = df.copy()
    close = out["Close"].astype(float)

    try:
        out["SMA20"] = SMAIndicator(close=close, window=20).sma_indicator()
        out["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator()
        out["RSI14"] = RSIIndicator(close=close, window=14).rsi()

        macd = MACD(close=close)
        out["MACD"] = macd.macd()
        out["MACD_signal"] = macd.macd_signal()

        bb = BollingerBands(close=close, window=20, window_dev=2)
        out["BB_high"] = bb.bollinger_hband()
        out["BB_low"] = bb.bollinger_lband()
    except Exception:
        # If any indicator fails due to NaNs/short series, just return what we have
        return out

    return out

def plot_price_with_indicators(df, symbol):
    if df.empty or "Close" not in df.columns:
        st.warning("No price data to chart.")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxis=True, vertical_spacing=0.05)

    # Candlesticks or line
    if {"Open", "High", "Low", "Close"}.issubset(df.columns) and df[["Open","High","Low","Close"]].notna().all().all():
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"), row=1, col=1)

    # Overlays
    for col, name in [("SMA20", "SMA 20"), ("EMA20", "EMA 20"), ("BB_high", "BB High"), ("BB_low", "BB Low")]:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, line=dict(width=1)), row=1, col=1)

    # RSI
    if "RSI14" in df.columns and df["RSI14"].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI(14)"), row=2, col=1)
        try:
            fig.add_hline(y=70, line_width=1, line_dash="dot", row=2, col=1)
            fig.add_hline(y=30, line_width=1, line_dash="dot", row=2, col=1)
        except Exception:
            # Older plotly versions might not support row/col in add_hline
            pass

    fig.update_layout(
        title=f"{symbol} — Price & Indicators",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

def get_news_titles(tkr, limit=6):
    """Fetch headlines from Yahoo Finance quote page (best-effort)."""
    try:
        url = f"https://finance.yahoo.com/quote/{tkr}?p={tkr}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        titles = re.findall(r'"title":"(.*?)"', r.text)
        # de-dup & clean
        cleaned, seen = [], set()
        for t in titles:
            if t and (t not in seen):
                seen.add(t)
                cleaned.append(t)
        return cleaned[:limit]
    except Exception:
        return []

def sentiment_for_titles(titles):
    """Return (rows, avg) where rows = [(title, score, label), ...]."""
    if not titles or not NLTK_OK:
        return [], 0.0
    try:
        nltk.download("vader_lexicon", quiet=True)
        analyzer = SentimentIntensityAnalyzer()
    except Exception:
        return [], 0.0

    rows, scores = [], []
    for t in titles:
        s = analyzer.polarity_scores(t)["compound"]
        label = "🟢 Positive" if s > 0.05 else ("🔴 Negative" if s < -0.05 else "🟡 Neutral")
        rows.append((t, s, label))
        scores.append(s)
    avg = float(np.mean(scores)) if scores else 0.0
    return rows, avg

def company_header(info, symbol):
    long_name = info.get("longName") or symbol
    mkt_price = info.get("regularMarketPrice")
    mkt_cap = info.get("marketCap")
    pe = info.get("trailingPE")
    volume = info.get("volume") or info.get("regularMarketVolume")

    # Safe formatting
    price_str = f"${mkt_price:.2f}" if isinstance(mkt_price, (int, float)) else "—"
    cap_str = f"${int(mkt_cap):,}" if isinstance(mkt_cap, (int, float)) else "—"
    vol_str = f"{int(volume):,}" if isinstance(volume, (int, float)) else "—"
    pe_str = "—" if (pe is None or not isinstance(pe, (int, float))) else f"{pe:.2f}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-box'><h4>{long_name} ({symbol})</h4><h2>{price_str}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><h4>Market Cap</h4><h2>{cap_str}</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><h4>P/E Ratio</h4><h2>{pe_str}</h2></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-box'><h4>Volume</h4><h2>{vol_str}</h2></div>", unsafe_allow_html=True)

# ---------------------------
# Date
# ---------------------------
today = date.today()
st.write(f"**Date:** {today.strftime('%B %d, %Y')}")

# ---------------------------
# Validate Primary Ticker
# ---------------------------
ok, info = valid_ticker(ticker)
if not ok:
    st.error(f"⚠️ '{ticker}' is not a valid stock symbol. Try AAPL, MSFT, TSLA, RELIANCE.NS, TCS.NS, INFY.NS, etc.")
    st.stop()

company_header(info, ticker)

# ---------------------------
# Tabs
# ---------------------------
tab_overview, tab_predict, tab_news, tab_compare = st.tabs(["📋 Overview", "🔮 Prediction", "🗞️ News & Sentiment", "🔁 Compare"])

# --- Overview Tab ---
with tab_overview:
    st.info(f"Fetching historical data for **{ticker}** …")
    df = fetch_history(ticker, period, interval)
    if df.empty:
        st.error("No historical data returned. Try a different period/interval.")
    else:
        df_ind = add_indicators(df)
        plot_price_with_indicators(df_ind, ticker)

        # MACD quick panel
        if {"MACD", "MACD_signal"}.issubset(df_ind.columns) and df_ind[["MACD", "MACD_signal"]].notna().any().any():
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"], name="MACD"))
            fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_signal"], name="Signal"))
            fig2.update_layout(template="plotly_dark", height=300, title="MACD")
            st.plotly_chart(fig2, use_container_width=True)

# --- Prediction Tab ---
with tab_predict:
    if not (TF_OK and PREP_OK):
        missing = []
        if not TF_OK:
            missing.append("TensorFlow/Keras")
        if not PREP_OK:
            missing.append("stock_predictor module")
        st.warning("Prediction is unavailable because: " + ", ".join(missing) + ".")
        st.info("You can still use all other tabs.")
    else:
        if st.button("Train / Load Model & Predict", type="primary"):
            with st.spinner("Preparing data & training model (if needed)…"):
                df_full = fetch_history(ticker, period, "1d")
                if df_full.empty or "Close" not in df_full.columns or df_full["Close"].dropna().empty:
                    st.error("⚠️ No valid 'Close' prices found to train. Try another symbol/period.")
                    st.stop()

                close_df = df_full[["Close"]].dropna().copy()

                # Prepare data
                scaler, train_data, test_data, x_train, y_train = prepare_data(close_df)

                # Load or train
                model_path = "lstm_model.keras"
                try:
                    if os.path.exists(model_path):
                        from tensorflow.keras.models import load_model
                        model = load_model(model_path)
                    else:
                        model = train_model(x_train, y_train)
                        model.save(model_path)
                except Exception as e:
                    st.error(f"Model load/train failed: {e}")
                    st.stop()

                # Predict next close
                try:
                    last_price = float(close_df["Close"].iloc[-1])
                    predicted_price = float(predict_future(model, test_data, scaler))
                    change_pct = ((predicted_price - last_price) / max(last_price, 1e-9)) * 100.0
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            st.success("✅ Prediction Complete")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-box'><h4>Last Close</h4><h2>${last_price:.2f}</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><h4>Predicted Next Close</h4><h2>${predicted_price:.2f}</h2></div>", unsafe_allow_html=True)
            color = "lime" if change_pct >= 0 else "red"
            c3.markdown(f"<div class='metric-box'><h4>Change</h4><h2 style='color:{color}'>{change_pct:.2f}%</h2></div>", unsafe_allow_html=True)

            # Actual vs predicted (walk-forward over test window)
            try:
                total = np.concatenate((train_data, test_data), axis=0)
                inputs = total[len(total) - len(test_data) - 60:]
                X_test, y_test = [], []
                for i in range(60, len(inputs)):
                    X_test.append(inputs[i-60:i, 0])
                    y_test.append(inputs[i, 0])
                X_test, y_test = np.array(X_test), np.array(y_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                pred_seq = model.predict(X_test)
                from sklearn.preprocessing import MinMaxScaler  # in case scaler is sklearn (commonly)
                # We assume 'scaler' has inverse_transform; if not, this will throw which we catch
                pred_seq = scaler.inverse_transform(pred_seq)
                y_seq = scaler.inverse_transform(y_test.reshape(-1, 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_seq.flatten()[-100:], mode="lines", name="Actual"))
                fig.add_trace(go.Scatter(y=pred_seq.flatten()[-100:], mode="lines", name="Predicted"))
                fig.update_layout(template="plotly_dark", height=400, title="Actual vs Predicted (Last 100 points)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Could not render Actual vs Predicted chart (insufficient data or scaler mismatch).")

        else:
            st.info("Click the button above to run the prediction.")

# --- News & Sentiment Tab ---
with tab_news:
    st.subheader("🗞️ Latest Headlines")
    titles = get_news_titles(ticker, limit=8)
    rows, avg = sentiment_for_titles(titles)

    if not rows:
        st.warning("No recent headlines available right now or sentiment analyzer unavailable.")
    else:
        for t, score, label in rows:
            st.markdown(f"<div class='news-card'>📰 {t}<br><b>Sentiment:</b> {label} &nbsp;(<i>{score:+.2f}</i>)</div>", unsafe_allow_html=True)

        st.markdown("---")
        if avg > 0.05:
            st.success("🟢 **Overall Market Sentiment: Positive** — Investors appear optimistic.")
        elif avg < -0.05:
            st.error("🔴 **Overall Market Sentiment: Negative** — Investors appear cautious.")
        else:
            st.warning("🟡 **Overall Market Sentiment: Neutral** — Mixed signals in the news.")

# --- Compare Tab ---
with tab_compare:
    if not compare_toggle or not compare_ticker:
        st.info("Enable **Compare with another stock** in the sidebar to use this tab.")
    else:
        ok2, info2 = valid_ticker(compare_ticker)
        if not ok2:
            st.error(f"'{compare_ticker}' is not a valid comparison symbol.")
        else:
            st.subheader(f"Comparing **{ticker}** vs **{compare_ticker}**")

            # Quick metrics
            m1 = info.get("regularMarketPrice", None)
            m2 = info2.get("regularMarketPrice", None)
            m1_str = f"${m1:.2f}" if isinstance(m1, (int, float)) else "—"
            m2_str = f"${m2:.2f}" if isinstance(m2, (int, float)) else "—"
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='metric-box'><h4>{ticker} Price</h4><h2>{m1_str}</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><h4>{compare_ticker} Price</h4><h2>{m2_str}</h2></div>", unsafe_allow_html=True)

            # Price charts normalized to 100
            df1 = fetch_history(ticker, period, interval)
            df2 = fetch_history(compare_ticker, period, interval)
            if df1.empty or df2.empty or "Close" not in df1.columns or "Close" not in df2.columns:
                st.warning("Not enough data to compare.")
            else:
                common_idx = df1.index.intersection(df2.index)
                if common_idx.empty:
                    st.warning("No overlapping dates to compare.")
                else:
                    s1 = (df1.loc[common_idx, "Close"] / df1.loc[common_idx, "Close"].iloc[0]) * 100
                    s2 = (df2.loc[common_idx, "Close"] / df2.loc[common_idx, "Close"].iloc[0]) * 100
                    figc = go.Figure()
                    figc.add_trace(go.Scatter(x=common_idx, y=s1, name=ticker))
                    figc.add_trace(go.Scatter(x=common_idx, y=s2, name=compare_ticker))
                    figc.update_layout(template="plotly_dark", height=450, title="Normalized Performance (Start = 100)")
                    st.plotly_chart(figc, use_container_width=True)

st.markdown("<div class='footer'>Made using Streamlit, Plotly, Yahoo Finance, and optional ML/Sentiment extras</div>", unsafe_allow_html=True)
