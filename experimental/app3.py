import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from keras.models import load_model  # Import for LSTM
st.set_option('deprecation.showPyplotGlobalUse', False)

# Paths for model directories
arima_models_folder = 'models/arima_Models'
dt_models_folder = 'models/decision_tree_Models'
xgb_models_folder = 'models/xgb_Models'
lstm_models_folder = 'models/lstm_Models'  # New folder for LSTM

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

# Forecasting button
if st.button("Forecast"):
    if model_option == "ARIMA":
        # Load ARIMA model
        model_filename = os.path.join(arima_models_folder, f'arima_model_product_{selected_product}.pkl')
        with open(model_filename, 'rb') as file:
            arima_model = pickle.load(file)

        # Forecasting with ARIMA
        st.session_state.arima_forecast = arima_model.forecast(steps=15)
        st.write(f"ARIMA Forecast for Product {selected_product}:")
        st.write(st.session_state.arima_forecast)

        # Plot ARIMA results
        plt.figure(figsize=(10, 6))
        plt.plot(st.session_state.arima_forecast, label='ARIMA Forecast', color='blue')
        plt.title(f'ARIMA Forecast for Product {selected_product}')
        plt.xlabel('Weeks')
        plt.ylabel('Sales Quantity')
        plt.legend()
        st.pyplot()

    elif model_option == "Decision Tree":
        # Load Decision Tree model
        dt_model_filename = os.path.join(dt_models_folder, f'decision_tree_model_product_{selected_product}.pkl')
        with open(dt_model_filename, 'rb') as file:
            dt_model = pickle.load(file)

        # Prepare input data for Decision Tree
        future_weeks = np.array([[2023, week] for week in range(40, 55)])  # Future weeks for prediction
        st.session_state.dt_forecast = dt_model.predict(future_weeks)

        st.write(f"Decision Tree Forecast for Product {selected_product}:")
        st.write(st.session_state.dt_forecast)

        # Plot Decision Tree results
        plt.figure(figsize=(10, 6))
        plt.plot(st.session_state.dt_forecast, label='Decision Tree Forecast', color='green')
        plt.title(f'Decision Tree Forecast for Product {selected_product}')
        plt.xlabel('Weeks')
        plt.ylabel('Sales Quantity')
        plt.legend()
        st.pyplot()

    elif model_option == "XGBoost":
        # Load XGBoost model
        xgb_model_filename = os.path.join(xgb_models_folder, f'xgb_model_product_{selected_product}.pkl')
        with open(xgb_model_filename, 'rb') as file:
            xgb_model = pickle.load(file)

        # Prepare input data for XGBoost
        future_weeks = np.array([[2023, week] for week in range(40, 55)])  # Future weeks for prediction
        st.session_state.xgb_forecast = xgb_model.predict(future_weeks)

        st.write(f"XGBoost Forecast for Product {selected_product}:")
        st.write(st.session_state.xgb_forecast)

        # Plot XGBoost results
        plt.figure(figsize=(10, 6))
        plt.plot(st.session_state.xgb_forecast, label='XGBoost Forecast', color='orange')
        plt.title(f'XGBoost Forecast for Product {selected_product}')
        plt.xlabel('Weeks')
        plt.ylabel('Sales Quantity')
        plt.legend()
        st.pyplot()

    elif model_option == "LSTM":
        # Load LSTM model
        lstm_model_filename = os.path.join(lstm_models_folder, f'lstm_model_product_{selected_product}.h5')
        lstm_model = load_model(lstm_model_filename)

        # Prepare input data for LSTM
        # You may need to reshape your data as per LSTM input requirements
        # Here, assuming your sales data is in a suitable format:
        product_sales_series = weekly_sales[weekly_sales['StockCode'] == selected_product]['Quantity'].values
        product_sales_series = product_sales_series.reshape((len(product_sales_series), 1))
        
        # Normalize or scale your data as needed for LSTM
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(product_sales_series)

        # Prepare input for LSTM
        X_test = []
        for i in range(len(scaled_data) - 15):  # 15 timesteps
            X_test.append(scaled_data[i:i + 15])
        X_test = np.array(X_test)

        # Make predictions
        st.session_state.lstm_forecast = lstm_model.predict(X_test[-1].reshape(1, X_test.shape[1], 1))

        # Inverse transform to get actual values
        st.session_state.lstm_forecast = scaler.inverse_transform(st.session_state.lstm_forecast)

        st.write(f"LSTM Forecast for Product {selected_product}:")
        st.write(st.session_state.lstm_forecast.flatten())

        # Plot LSTM results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(st.session_state.lstm_forecast.flatten(), label='LSTM Forecast', color='purple')
        ax.set_title(f'LSTM Forecast for Product {selected_product}')
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Sales Quantity')
        ax.legend()

        st.pyplot(fig)

# Comparison button
if st.button("Compare Models"):
    # Load and forecast with ARIMA
    model_filename_arima = os.path.join(arima_models_folder, f'arima_model_product_{selected_product}.pkl')
    with open(model_filename_arima, 'rb') as file:
        arima_model = pickle.load(file)

    # Prepare input data for ARIMA
    product_sales_series = weekly_sales[weekly_sales['StockCode'] == selected_product].set_index(['Year', 'Week'])['Quantity']
    arima_result = ARIMA(product_sales_series, order=(1, 1, 1)).fit()
    st.session_state.arima_forecast = arima_result.forecast(steps=15)

    # Load and forecast with Decision Tree
    dt_model_filename = os.path.join(dt_models_folder, f'decision_tree_model_product_{selected_product}.pkl')
    with open(dt_model_filename, 'rb') as file:
            dt_model = pickle.load(file)

    # Prepare input data for Decision Tree
    future_data = np.array([[2023, week] for week in range(40, 55)])  # Adjust weeks based on your need
    st.session_state.dt_forecast = dt_model.predict(future_data)

    # Load and forecast with XGBoost
    xgb_model_filename = os.path.join(xgb_models_folder, f'xgb_model_product_{selected_product}.pkl')
    with open(xgb_model_filename, 'rb') as file:
            xgb_model = pickle.load(file)

    # Forecast using XGBoost
    st.session_state.xgb_forecast = xgb_model.predict(future_data)

    # Load and forecast with LSTM
    lstm_model_filename = os.path.join(lstm_models_folder, f'lstm_model_product_{selected_product}.h5')
    lstm_model = load_model(lstm_model_filename)

    # Prepare input data for LSTM
    product_sales_series = weekly_sales[weekly_sales['StockCode'] == selected_product]['Quantity'].values
    product_sales_series = product_sales_series.reshape((len(product_sales_series), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(product_sales_series)

    X_test = []
    for i in range(len(scaled_data) - 15):
        X_test.append(scaled_data[i:i + 15])
    X_test = np.array(X_test)

    # Make predictions
    lstm_forecast = lstm_model.predict(X_test[-1].reshape(1, X_test.shape[1], 1))
    lstm_forecast = scaler.inverse_transform(lstm_forecast)

    st.session_state.lstm_forecast = lstm_forecast.flatten()

    # Now plot all the forecasts
    plt.figure(figsize=(10, 6))

    # ARIMA forecast
    if st.session_state.arima_forecast is not None:
        plt.plot(st.session_state.arima_forecast, label='ARIMA Forecast', color='blue')

    # Decision Tree forecast
    if st.session_state.dt_forecast is not None:
        plt.plot(st.session_state.dt_forecast, label='Decision Tree Forecast', color='green')

    # XGBoost forecast
    if st.session_state.xgb_forecast is not None:
        plt.plot(st.session_state.xgb_forecast, label='XGBoost Forecast', color='orange')

    # LSTM forecast
    if st.session_state.lstm_forecast is not None:
        plt.plot(st.session_state.lstm_forecast, label='LSTM Forecast', color='purple')

    plt.title(f'Comparison of Forecasts for Product {selected_product}')
    plt.xlabel('Weeks')
    plt.ylabel('Sales Quantity')
    plt.legend()
    st.pyplot()
