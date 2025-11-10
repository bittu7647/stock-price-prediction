# app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from stock_predictor import *
from tensorflow.keras.models import load_model
import os
import requests

# 🧠 Page setup
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# 🎨 Styling
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #00BFFF;
            margin-bottom: 10px;
        }
        .metric-box {
            background-color: #1E1E1E;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 0.9em;
            margin-top: 30px;
        }
        .news-card {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        a {
            color: #1E90FF;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# 🧭 Sidebar
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
period = st.sidebar.selectbox("Select Period", ["1y", "2y", "5y", "10y"], index=2)
run_prediction = st.sidebar.button("🔮 Run Prediction")

# 🧠 Dashboard Title
st.markdown("<div class='title'>📊 Stock Price Prediction & Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("---")

# 🕒 Display date
today = date.today()
st.write(f"**Date:** {today.strftime('%B %d, %Y')}")

# 📈 Fetch stock info
try:
    stock = yf.Ticker(ticker)
    info = stock.info

    logo_url = info.get("logo_url", "")
    long_name = info.get("longName", ticker)
    market_price = info.get("regularMarketPrice", 0)
    previous_close = info.get("regularMarketPreviousClose", 0)
    market_cap = info.get("marketCap", 0)
    pe_ratio = info.get("trailingPE", None)
    volume = info.get("volume", 0)

    # 🧩 Header info with logo
    col1, col2 = st.columns([1, 4])
    with col1:
        if logo_url:
            st.image(logo_url, width=90)
    with col2:
        st.subheader(f"{long_name} ({ticker.upper()})")
        st.write(f"💰 **Market Price:** ${market_price}")
        st.write(f"🏢 **Market Cap:** ${market_cap:,}")
        if pe_ratio:
            st.write(f"📊 **P/E Ratio:** {pe_ratio}")
        st.write(f"🔊 **Volume:** {volume:,}")

    st.markdown("---")

except Exception as e:
    st.warning("⚠️ Unable to fetch stock data. Please check the symbol.")

# 🧩 Run Prediction
if run_prediction:
    st.info(f"Fetching historical data for **{ticker}**...")

    with st.spinner("Downloading data & training model..."):
        df = yf.download(ticker, period=period)
        df = df[['Close']]
        st.subheader("📈 Historical Close Prices")
        st.line_chart(df['Close'])

        scaler, train_data, test_data, x_train, y_train = prepare_data(df)

        if os.path.exists("lstm_model.keras"):
            model = load_model("lstm_model.keras")
        else:
            model = train_model(x_train, y_train)
            model.save("lstm_model.keras")

        predicted_price = predict_future(model, test_data, scaler)
        last_price = df['Close'][-1]
        change = ((predicted_price - last_price) / last_price) * 100

    st.success("✅ Prediction Complete!")

    # 📊 Display metrics
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-box'><h4>Last Closing Price</h4><h2>${last_price:.2f}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'><h4>Predicted Next Price</h4><h2>${predicted_price:.2f}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-box'><h4>Change (%)</h4><h2 style='color:{'lime' if change>0 else 'red'}'>{change:.2f}%</h2></div>", unsafe_allow_html=True)

    # 📈 Plot Actual vs Predicted
    st.subheader("📉 Actual vs Predicted Prices")
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

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test_scaled[-100:], label="Actual", linewidth=2)
    ax.plot(predicted[-100:], label="Predicted", linewidth=2)
    ax.legend()
    st.pyplot(fig)

# 📰 Stock News Section
st.subheader("🗞️ Latest News")
try:
    url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
    response = requests.get(url)
    if response.status_code == 200:
        import re
        titles = re.findall(r'"title":"(.*?)"', response.text)
        unique_titles = list(dict.fromkeys(titles))[:5]
        for t in unique_titles:
            st.markdown(f"<div class='news-card'>📰 {t}</div>", unsafe_allow_html=True)
    else:
        st.warning("No news found.")
except:
    st.warning("⚠️ Could not fetch news at this time.")

st.markdown("<div class='footer'>Made with ❤️ using Streamlit, TensorFlow & Yahoo Finance API</div>", unsafe_allow_html=True)
