import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
# from keras.models import load_model
# from tensorflow.keras.metrics import MeanSquaredError
# from tensorflow.keras.saving import register_keras_serializable

# @register_keras_serializable()
# class CustomMSE(MeanSquaredError):
#     pass

# custom_objects = {'CustomMSE': CustomMSE}

# Paths for model directories
arima_models_folder = 'models/arima_Models'
lstm_models_folder = 'models/lstm_Models'

# Load the merged dataset
merged_data = pd.read_csv('./data/merged_data.csv')

# Prepare weekly sales data for top 10 products
top_10_products = merged_data.groupby('StockCode')['Quantity'].sum().nlargest(10).index
weekly_sales = merged_data.groupby(['StockCode', 'Year', 'Week'])['Quantity'].sum().reset_index()

# Load product codes (as an example, you can modify this to load from your source)
product_codes = list(top_10_products)

# Title of the app
st.title("Sales Forecasting with ARIMA and LSTM")

# Model selection dropdown
model_option = st.selectbox("Select Model", ["ARIMA", "LSTM"])

# Product selection dropdown
selected_product = st.selectbox("Select Product", product_codes)

def preprocess_data_for_lstm(data, n_lags=4):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    return X.reshape(-1, n_lags, 1), y

# Forecasting button
if st.button("Forecast"):
    if model_option == "ARIMA":
        # Load ARIMA model
        model_filename = os.path.join(arima_models_folder, f'arima_model_product_{selected_product}.pkl')
        with open(model_filename, 'rb') as file:
            arima_model = pickle.load(file)

        # Forecasting with ARIMA
        arima_forecast = arima_model.forecast(steps=15)
        st.write(f"ARIMA Forecast for Product {selected_product}:")
        st.write(arima_forecast)

        # Plot ARIMA results
        plt.figure(figsize=(10, 6))
        plt.plot(arima_forecast, label='ARIMA Forecast', color='blue')
        plt.title(f'ARIMA Forecast for Product {selected_product}')
        plt.xlabel('Weeks')
        plt.ylabel('Sales Quantity')
        plt.legend()
        st.pyplot()

    elif model_option == "LSTM":
        # Load LSTM model
        lstm_model_filename = os.path.join(lstm_models_folder, f'lstm_model_product_{selected_product}.h5')
        lstm_model = tf.keras.models.load_model(lstm_model_filename)

        # Prepare input data for LSTM
        product_sales = weekly_sales[weekly_sales['StockCode'] == selected_product].set_index('Week')['Quantity']
        product_sales = product_sales.fillna(0)  # Fill missing values with 0

        # Prepare data for LSTM
        n_lags = 4  # Example lag value; adjust based on your model
        X, _ = preprocess_data_for_lstm(product_sales.values, n_lags=n_lags)

        # Make predictions using LSTM
        lstm_forecast = lstm_model.predict(X[-1].reshape(1, n_lags, 1))  # Predict using the last input
        lstm_forecast = np.concatenate([product_sales.values[-n_lags:], lstm_forecast.flatten()])  # Combine for plotting

        st.write(f"LSTM Forecast for Product {selected_product}:")
        st.write(lstm_forecast)

        # Plot LSTM results
        plt.figure(figsize=(10, 6))
        plt.plot(lstm_forecast, label='LSTM Forecast', color='orange')
        plt.title(f'LSTM Forecast for Product {selected_product}')
        plt.xlabel('Weeks')
        plt.ylabel('Sales Quantity')
        plt.legend()
        st.pyplot()

    # Comparison button
    if st.button("Compare Both Models"):
        plt.figure(figsize=(10, 6))
        plt.plot(arima_forecast, label='ARIMA Forecast', color='blue')
        plt.plot(lstm_forecast[-15:], label='LSTM Forecast', color='orange')  # Only last 15 for comparison
        plt.title(f'Comparison of Forecasts for Product {selected_product}')
        plt.xlabel('Weeks')
        plt.ylabel('Sales Quantity')
        plt.legend()
        st.pyplot()
