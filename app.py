# app.py
import os
import re
from datetime import date
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from stock_predictor import prepare_data, train_model, predict_future
from tensorflow.keras.models import load_model

# 🧠 Page Setup
st.set_page_config(page_title="Stock Price Prediction Dashboard", page_icon="📈", layout="wide")

# 🎨 Page Style
st.markdown("""
    <style>
        .main { background-color: #0E1117; color: #FFFFFF; }
        .title { text-align: center; font-size: 2.4em; font-weight: bold; color: #00BFFF; }
        .metric-box { background-color: #1E1E1E; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }
        .news-card { background-color: #1E1E1E; padding: 12px; border-radius: 8px; margin-bottom: 8px; }
        .footer { text-align: center; font-size: 0.9em; color: gray; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📊 Stock Price Prediction & Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper().strip()
period = st.sidebar.selectbox("Select Period", ["1y", "2y", "5y"], index=1)
run_prediction = st.sidebar.button("🔮 Run Prediction")
compare_toggle = st.sidebar.checkbox("🔁 Compare another stock")
compare_ticker = st.sidebar.text_input("Comparison Symbol", "MSFT" if compare_toggle else "").upper().strip() if compare_toggle else ""

# Date Display
st.write(f"**Date:** {date.today().strftime('%B %d, %Y')}")

# Validate Ticker
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    if "regularMarketPrice" not in info:
        st.error("⚠️ Invalid ticker. Try AAPL, TSLA, RELIANCE.NS, etc.")
        st.stop()
except:
    st.error("⚠️ Could not fetch data. Check your connection.")
    st.stop()

# Company Info
st.subheader(f"{info.get('longName', ticker)} ({ticker})")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Market Price", f"${info.get('regularMarketPrice', 0):,.2f}")
col2.metric("🏢 Market Cap", f"${info.get('marketCap', 0):,}")
col3.metric("📊 P/E Ratio", f"{info.get('trailingPE', '—')}")
col4.metric("🔊 Volume", f"{info.get('volume', 0):,}")

st.markdown("---")

# Function to add indicators
def add_indicators(df):
    df["SMA20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI14"] = ta.momentum.rsi(df["Close"], window=14)
    return df

# Historical Data
st.subheader("📈 Historical Close Prices & Indicators")
df = yf.download(ticker, period=period)
if df.empty:
    st.error("⚠️ No data available.")
    st.stop()

df = add_indicators(df)
st.line_chart(df[["Close", "SMA20", "EMA20"]])

# RSI Chart
fig, ax = plt.subplots()
ax.plot(df.index, df["RSI14"], label="RSI(14)", color="orange")
ax.axhline(70, color="red", linestyle="--", linewidth=1)
ax.axhline(30, color="green", linestyle="--", linewidth=1)
ax.set_title("RSI (Relative Strength Index)")
st.pyplot(fig)

# Prediction Section
if run_prediction:
    st.subheader("🔮 Stock Price Prediction (LSTM Model)")
    with st.spinner("Training model..."):
        df_close = df[["Close"]]
        scaler, train_data, test_data, x_train, y_train = prepare_data(df_close)

        if os.path.exists("lstm_model.keras"):
            model = load_model("lstm_model.keras")
        else:
            model = train_model(x_train, y_train)
            model.save("lstm_model.keras")

        last_price = df_close["Close"].iloc[-1]
        predicted_price = predict_future(model, test_data, scaler)
        change = ((predicted_price - last_price) / last_price) * 100

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-box'><h4>Last Price</h4><h2>${last_price:.2f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'><h4>Predicted Next</h4><h2>${predicted_price:.2f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'><h4>Change</h4><h2 style='color:{'lime' if change>0 else 'red'}'>{change:.2f}%</h2></div>", unsafe_allow_html=True)

    # Plot Actual vs Predicted
    total_data = np.concatenate((train_data, test_data), axis=0)
    inputs = total_data[len(total_data) - len(test_data) - 60:]
    X_test, y_test = [], []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
        y_test.append(inputs[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.plot(y_test_scaled[-100:], label="Actual", color="white")
    ax.plot(predicted[-100:], label="Predicted", color="cyan")
    ax.legend()
    ax.set_title("Actual vs Predicted Prices")
    st.pyplot(fig)

# Compare Stocks
if compare_toggle and compare_ticker:
    st.subheader(f"🔁 Comparing {ticker} vs {compare_ticker}")
    df1 = yf.download(ticker, period=period)
    df2 = yf.download(compare_ticker, period=period)
    if not df1.empty and not df2.empty:
        common_idx = df1.index.intersection(df2.index)
        norm1 = (df1.loc[common_idx, "Close"] / df1.loc[common_idx, "Close"].iloc[0]) * 100
        norm2 = (df2.loc[common_idx, "Close"] / df2.loc[common_idx, "Close"].iloc[0]) * 100
        fig, ax = plt.subplots()
        ax.plot(common_idx, norm1, label=ticker, color="cyan")
        ax.plot(common_idx, norm2, label=compare_ticker, color="orange")
        ax.legend()
        ax.set_title("Normalized Performance (Start = 100)")
        st.pyplot(fig)

# News & Sentiment
st.subheader("🗞️ Latest News & Sentiment")
try:
    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
    r = requests.get(url)
    titles = re.findall(r'"title":"(.*?)"', r.text)
    unique_titles = list(dict.fromkeys(titles))[:5]
    sentiments = []
    for t in unique_titles:
        score = analyzer.polarity_scores(t)["compound"]
        label = "🟢 Positive" if score > 0.05 else "🔴 Negative" if score < -0.05 else "🟡 Neutral"
        sentiments.append(score)
        st.markdown(f"<div class='news-card'>📰 {t}<br><b>Sentiment:</b> {label}</div>", unsafe_allow_html=True)
    avg = np.mean(sentiments)
    st.markdown("---")
    if avg > 0.05:
        st.success("🟢 Overall Sentiment: Positive")
    elif avg < -0.05:
        st.error("🔴 Overall Sentiment: Negative")
    else:
        st.warning("🟡 Overall Sentiment: Neutral")
except Exception:
    st.warning("⚠️ Could not fetch news or sentiment data.")

st.markdown("<div class='footer'>Made with using Streamlit, TensorFlow, and Yahoo Finance API</div>", unsafe_allow_html=True)
