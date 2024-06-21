import streamlit as st
import requests
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import os
import pickle
from PIL import Image

# MLflow setup (assuming you've set the tracking URI somewhere else)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ARIMAmodel_v2")

def fetch_historical_data(api_key, limit=2000):
    url = f"https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': limit,
        'api_key': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()['Data']['Data']
    return pd.DataFrame(data)

def run_arima_experiment(api_key):
    with mlflow.start_run():
        mlflow.log_param('API_KEY', api_key)

        if not api_key:
            raise ValueError("API key is missing or invalid.")

        # Fetch historical data
        df = fetch_historical_data(api_key)
        df = df.drop(columns=['conversionType', 'conversionSymbol'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df.drop(df.index[-1])

        # Perform Augmented Dickey-Fuller test
        result = adfuller(df['close'])
        if result[1] > 0.05:
            df['close_diff'] = df['close'].diff().dropna()
            result_diff = adfuller(df['close_diff'].dropna())
            if result_diff[1] < 0.05:
                print("Data is stationary after differencing.")
                d = 1
            else:
                d = 2
        else:
            print("Data is already stationary.")
            d = 0

        mlflow.log_param('d', d)

        # ARIMA parameters
        p = 1
        q = 1

        mlflow.log_param('p', p)
        mlflow.log_param('q', q)

        # Split data
        train_size = int(len(df) * 0.8)
        train, test = df[:train_size], df[train_size:]

        # Fit ARIMA model
        model = ARIMA(df['close'], order=(p, d, q))
        model_fit = model.fit()

        # Predictions
        start_index = test.index[0]
        end_index = test.index[-1]
        predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')

        # Evaluation metrics
        mae = mean_absolute_error(test['close'], predictions)
        rmse = np.sqrt(mean_squared_error(test['close'], predictions))
        r2 = r2_score(test['close'], predictions)

        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('R2', r2)

        # Log model artifact
        model_name = 'trained_model.pkl'
        model_path = model_name
        pickle.dump(model_fit, open(model_path, 'wb'))
        mlflow.log_artifact(model_path, artifact_path='models')
        os.remove(model_path)  # Clean up local model artifact

        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'R² Score: {r2}')

        return predictions[-1]

# Streamlit interface
st.title("Bitcoin Price Prediction")

bitcoin = Image.open('Bitcoin.png')
st.image(bitcoin, width=200)

api_key = st.text_input("Enter your CryptoCompare API key", type="password")

if st.button("Predict Next Hour Bitcoin Price"):
    if api_key:
        with st.spinner("Fetching data and running prediction..."):
            try:
                next_hour_prediction = run_arima_experiment(api_key)
                st.success(f"The predicted Bitcoin price for the next hour is: ${next_hour_prediction:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid API key.")
