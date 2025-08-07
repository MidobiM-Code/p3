import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from predictor.model import load_model, predict_price
from predictor.utils import fetch_tgju_price, save_prediction_to_db, init_db, load_prediction_history
from datetime import datetime

# Initialize DB
init_db()

# UI
st.set_page_config(page_title="پیش‌بینی قیمت دلار", layout="centered")
st.title("📈 پیش‌بینی قیمت دلار با LSTM")
st.markdown("منبع قیمت: [tgju.org](https://www.tgju.org/)")

# Fetch current price
current_price = fetch_tgju_price()
if current_price:
    st.metric("قیمت لحظه‌ای دلار", f"{current_price:,.0f} تومان")
else:
    st.error("عدم دریافت قیمت لحظه‌ای!")

# User input for days
n_days = st.slider("تعداد روزهای آینده برای پیش‌بینی", 1, 30, 7)

# Predict
if st.button("پیش‌بینی کن"):
    model, scaler = load_model()
    future_dates, predicted_prices = predict_price(model, scaler, n_days)
    df_pred = pd.DataFrame({
        "date": future_dates,
        "predicted_price": predicted_prices
    })
    st.subheader("نتیجه پیش‌بینی")
    fig = px.line(df_pred, x="date", y="predicted_price", title="پیش‌بینی قیمت دلار")
    st.plotly_chart(fig)

    # Save predictions
    for d, p in zip(future_dates, predicted_prices):
        save_prediction_to_db(d.strftime("%Y-%m-%d"), float(p), "LSTM")

    st.success("پیش‌بینی‌ها ذخیره شدند.")

# Show history
if st.checkbox("نمایش تاریخچه پیش‌بینی‌ها"):
    df_history = load_prediction_history()
    st.dataframe(df_history)
    fig2 = px.line(df_history, x="date", y="predicted_price", color="model_name", title="تاریخچه پیش‌بینی‌ها")
    st.plotly_chart(fig2)