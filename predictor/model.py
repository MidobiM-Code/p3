import numpy as np
import joblib
from keras.models import load_model as keras_load_model
from datetime import datetime, timedelta

def load_model():
    model = keras_load_model("predictor/lstm_model.h5")
    scaler = joblib.load("predictor/scaler.gz")
    return model, scaler

def predict_price(model, scaler, n_days):
    # Fake last sequence
    last_sequence = np.random.rand(1, 60, 1)
    predictions = []
    future_dates = []
    current_date = datetime.today()

    for _ in range(n_days):
        pred = model.predict(last_sequence)[0][0]
        predictions.append(scaler.inverse_transform([[pred]])[0][0])
        new_sequence = np.append(last_sequence[:, 1:, :], [[[pred]]], axis=1)
        last_sequence = new_sequence
        future_dates.append(current_date)
        current_date += timedelta(days=1)

    return future_dates, predictions