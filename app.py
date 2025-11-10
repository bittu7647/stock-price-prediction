# app.py
import streamlit as st
import matplotlib.pyplot as plt
from stock_predictor import *

st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict future stock prices using deep learning and Yahoo Finance data.")

ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, RELIANCE.NS)", "AAPL")

if st.button("Predict"):
    st.write(f"Fetching data for {ticker}...")
    df = get_stock_data(ticker)
    st.line_chart(df['Close'])
    
    st.write("Training model...")
    scaler, train_data, test_data, x_train, y_train = prepare_data(df)
    model = train_model(x_train, y_train)
    
    predicted_price = predict_future(model, test_data, scaler)
    
    st.success(f"Predicted Next Day Closing Price for {ticker}: **${predicted_price:.2f}**")

    # Show actual vs predicted for last 100 days
    st.write("📊 Actual vs Predicted (Last 100 Days)")
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
    ax.plot(y_test_scaled[-100:], label="Actual Price", linewidth=2)
    ax.plot(predicted[-100:], label="Predicted Price", linewidth=2)
    ax.legend()
    st.pyplot(fig)
