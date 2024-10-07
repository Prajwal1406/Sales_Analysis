import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

st.set_option('deprecation.showPyplotGlobalUse', False)

# Paths for model directories
arima_models_folder = 'models/arima_Models'
dt_models_folder = 'models/decision_tree_Models'
xgb_models_folder = 'models/xgb_Models'
lstm_models_folder = 'models/lstm_Models'

# Load and prepare data
merged_data = pd.read_csv('./data/merged_data.csv')

# Prepare weekly sales data for top 10 products
top_10_products = merged_data.groupby('StockCode')['Quantity'].sum().nlargest(10).index
weekly_sales = merged_data.groupby(['StockCode', 'Year', 'Week'])['Quantity'].sum().reset_index()

# Load product codes
product_codes = ['85123A', '85099B', '22197', '84879', '23084',
                 '21181', '22423', '21212', '20713', '21915']

# Title of the app
st.title("Sales Forecasting with ARIMA, Decision Tree, XGBoost, and LSTM")

# Model selection dropdown
model_option = st.selectbox("Select Model", ["ARIMA", "Decision Tree", "XGBoost", "LSTM"])

# Product selection dropdown
selected_product = st.selectbox("Select Product", product_codes)

# Initialize session state to store forecasts
if 'arima_forecast' not in st.session_state:
    st.session_state.arima_forecast = None
if 'dt_forecast' not in st.session_state:
    st.session_state.dt_forecast = None
if 'xgb_forecast' not in st.session_state:
    st.session_state.xgb_forecast = None
if 'lstm_forecast' not in st.session_state:
    st.session_state.lstm_forecast = None

# Helper function to calculate error metrics
def calculate_metrics(actual, predicted):
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, mape

# Forecasting button
if st.button("Forecast"):
    # Prepare the historical data for comparison
    product_sales_series = weekly_sales[weekly_sales['StockCode'] == selected_product].set_index(['Year', 'Week'])['Quantity']
    train_data = product_sales_series[:-15]  # Use all data except the last 15 weeks for training
    test_data = product_sales_series[-15:]   # Use the last 15 weeks for testing

    if model_option == "ARIMA":
        # Load ARIMA model
        model_filename = os.path.join(arima_models_folder, f'arima_model_product_{selected_product}.pkl')
        with open(model_filename, 'rb') as file:
            arima_model = pickle.load(file)

        # Forecasting with ARIMA
        st.session_state.arima_forecast = arima_model.forecast(steps=15)

        # Calculate and display error metrics
        rmse, mae, mape = calculate_metrics(test_data.values, st.session_state.arima_forecast)
        st.write(f"ARIMA Forecast for Product {selected_product}:")
        st.write(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}%")

        # Plot ARIMA results
        fig, ax = plt.subplots()
        ax.plot(test_data.values, label='Actual', color='black')
        ax.plot(st.session_state.arima_forecast, label='ARIMA Forecast', color='blue')
        ax.set_title(f'ARIMA Forecast for Product {selected_product}')
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Sales Quantity')
        ax.legend()
        st.pyplot(fig)

    elif model_option == "Decision Tree":
        # Load Decision Tree model
        dt_model_filename = os.path.join(dt_models_folder, f'decision_tree_model_product_{selected_product}.pkl')
        with open(dt_model_filename, 'rb') as file:
            dt_model = pickle.load(file)

        # Prepare input data for Decision Tree
        future_weeks = np.array([[2023, week] for week in range(40, 55)])  # Future weeks for prediction
        st.session_state.dt_forecast = dt_model.predict(future_weeks)

        # Calculate and display error metrics
        rmse, mae, mape = calculate_metrics(test_data.values, st.session_state.dt_forecast[:15])
        st.write(f"Decision Tree Forecast for Product {selected_product}:")
        st.write(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}%")

        # Plot Decision Tree results
        fig, ax = plt.subplots()
        ax.plot(test_data.values, label='Actual', color='black')
        ax.plot(st.session_state.dt_forecast[:15], label='Decision Tree Forecast', color='green')
        ax.set_title(f'Decision Tree Forecast for Product {selected_product}')
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Sales Quantity')
        ax.legend()
        st.pyplot(fig)

    elif model_option == "XGBoost":
        # Load XGBoost model
        xgb_model_filename = os.path.join(xgb_models_folder, f'xgb_model_product_{selected_product}.pkl')
        with open(xgb_model_filename, 'rb') as file:
            xgb_model = pickle.load(file)

        # Prepare input data for XGBoost
        future_weeks = np.array([[2023, week] for week in range(40, 55)])  # Future weeks for prediction
        st.session_state.xgb_forecast = xgb_model.predict(future_weeks)

        # Calculate and display error metrics
        rmse, mae, mape = calculate_metrics(test_data.values, st.session_state.xgb_forecast[:15])
        st.write(f"XGBoost Forecast for Product {selected_product}:")
        st.write(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}%")

        # Plot XGBoost results
        fig, ax = plt.subplots()
        ax.plot(test_data.values, label='Actual', color='black')
        ax.plot(st.session_state.xgb_forecast[:15], label='XGBoost Forecast', color='orange')
        ax.set_title(f'XGBoost Forecast for Product {selected_product}')
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Sales Quantity')
        ax.legend()
        st.pyplot(fig)

    elif model_option == "LSTM":
        # Load LSTM model
        lstm_model_filename = os.path.join(lstm_models_folder, f'lstm_model_product_{selected_product}.h5')
        lstm_model = load_model(lstm_model_filename)

        # Prepare input data for LSTM
        product_sales_series = weekly_sales[weekly_sales['StockCode'] == selected_product]['Quantity'].values
        product_sales_series = product_sales_series.reshape((len(product_sales_series), 1))
        
        # Normalize or scale your data as needed for LSTM
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(product_sales_series)

        # Prepare input for LSTM
        X_test = []
        for i in range(len(scaled_data) - 15):
            X_test.append(scaled_data[i:i + 15])
        X_test = np.array(X_test)

        # Make predictions
        st.session_state.lstm_forecast = lstm_model.predict(X_test[-1].reshape(1, X_test.shape[1], 1))

        # Inverse transform to get actual values
        st.session_state.lstm_forecast = scaler.inverse_transform(st.session_state.lstm_forecast)

        # Calculate and display error metrics
        rmse, mae, mape = calculate_metrics(test_data.values, st.session_state.lstm_forecast.flatten())
        st.write(f"LSTM Forecast for Product {selected_product}:")
        st.write(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}%")

        # Plot LSTM results
        fig, ax = plt.subplots()
        ax.plot(test_data.values, label='Actual', color='black')
        ax.plot(st.session_state.lstm_forecast.flatten(), label='LSTM Forecast', color='purple')
        ax.set_title(f'LSTM Forecast for Product {selected_product}')
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Sales Quantity')
        ax.legend()
        st.pyplot(fig)
