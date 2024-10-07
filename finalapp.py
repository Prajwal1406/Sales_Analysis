import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

st.set_option('deprecation.showPyplotGlobalUse', False)

# Paths for model directories
arima_models_folder = 'models/arima_Models'
dt_models_folder = 'models/decision_tree_Models'
xgb_models_folder = 'models/xgb_Models'

# Load and prepare data
merged_data = pd.read_csv('./data/merged_data.csv')

# Prepare weekly sales data for top 10 products
top_10_products = merged_data.groupby('StockCode')['Quantity'].sum().nlargest(10).index
weekly_sales = merged_data.groupby(['StockCode', 'Year', 'Week'])['Quantity'].sum().reset_index()

# Load product codes
product_codes = ['85123A', '85099B', '22197', '84879', '23084',
                 '21181', '22423', '21212', '20713', '21915']

# Title of the app
st.title("Sales Forecasting with ARIMA, Decision Tree, and XGBoost")

# Model selection dropdown
model_option = st.selectbox("Select Model", ["ARIMA", "Decision Tree", "XGBoost"])

# Product selection dropdown
selected_product = st.selectbox("Select Product", product_codes)

# Initialize session state to store forecasts
if 'arima_forecast' not in st.session_state:
    st.session_state.arima_forecast = None
if 'dt_forecast' not in st.session_state:
    st.session_state.dt_forecast = None
if 'xgb_forecast' not in st.session_state:
    st.session_state.xgb_forecast = None

# Helper function to calculate error metrics
def calculate_metrics(actual, predicted):
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, mape

# Function to plot ACF and PACF
def plot_acf_pacf(series):
    try:
        # Ensure the series has enough data points for the desired number of lags
        if len(series) < 20:
            st.write("The series is too short to generate ACF and PACF plots with 20 lags.")
            return  # Skip plotting

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot ACF and PACF with error handling for shape mismatch
        plot_acf(series, lags=20, ax=ax[0])
        plot_pacf(series, lags=20, ax=ax[1])
        
        # Set titles for the plots
        ax[0].set_title('Auto-Correlation Function (ACF)')
        ax[1].set_title('Partial Auto-Correlation Function (PACF)')

        # Display the plots using Streamlit
        st.pyplot(fig)
        
    except ValueError as e:
        st.error(f"An error occurred while plotting ACF and PACF: {e}")
    
def calculate_metrics2(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse
# Forecasting button
if st.button("Forecast"):
    # Prepare the historical data for comparison
    product_sales_series = weekly_sales[weekly_sales['StockCode'] == selected_product].set_index(['Year', 'Week'])['Quantity']
    train_data = product_sales_series[:-15]  # Use all data except the last 15 weeks for training
    test_data = product_sales_series[-15:]   # Use the last 15 weeks for testing

    # Plot ACF and PACF
    plot_acf_pacf(train_data)

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
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(test_data.values, label='Actual', color='black')
        ax[1].plot(st.session_state.arima_forecast, label='ARIMA Forecast', color='blue')
        ax[0].set_title(f'ARIMA Actual Forecast for Product {selected_product}')
        ax[0].set_xlabel('Weeks')
        ax[0].set_ylabel('Sales Quantity')
        ax[0].legend()
        ax[1].set_title(f'ARIMA Forecast for Product {selected_product}')
        ax[1].set_xlabel('Weeks')
        ax[1].set_ylabel('Sales Quantity')
        ax[1].legend()
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

    # Create subplots for each model's forecast
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot ARIMA forecast
    if st.session_state.arima_forecast is not None:
        axs[0, 0].plot(st.session_state.arima_forecast, label='ARIMA Forecast', color='blue')
        axs[0, 0].set_title('ARIMA Forecast')
        axs[0, 0].set_xlabel('Weeks')
        axs[0, 0].set_ylabel('Sales Quantity')
        axs[0, 0].legend()

        # Calculate and display ARIMA metrics
        arima_mae, arima_rmse = calculate_metrics2(product_sales_series[-15:], st.session_state.arima_forecast)
        st.write(f'**ARIMA Model Metrics:**\n- MAE: {arima_mae:.2f}\n- RMSE: {arima_rmse:.2f}')

    # Plot Decision Tree forecast
    if st.session_state.dt_forecast is not None:
        axs[0, 1].plot(st.session_state.dt_forecast, label='Decision Tree Forecast', color='green')
        axs[0, 1].set_title('Decision Tree Forecast')
        axs[0, 1].set_xlabel('Weeks')
        axs[0, 1].set_ylabel('Sales Quantity')
        axs[0, 1].legend()

        # Calculate and display Decision Tree metrics
        dt_mae, dt_rmse = calculate_metrics2(product_sales_series[-15:], st.session_state.dt_forecast)
        st.write(f'**Decision Tree Model Metrics:**\n- MAE: {dt_mae:.2f}\n- RMSE: {dt_rmse:.2f}')

    # Plot XGBoost forecast
    if st.session_state.xgb_forecast is not None:
        axs[1, 0].plot(st.session_state.xgb_forecast, label='XGBoost Forecast', color='orange')
        axs[1, 0].set_title('XGBoost Forecast')
        axs[1, 0].set_xlabel('Weeks')
        axs[1, 0].set_ylabel('Sales Quantity')
        axs[1, 0].legend()

        # Calculate and display XGBoost metrics
        xgb_mae, xgb_rmse = calculate_metrics2(product_sales_series[-15:], st.session_state.xgb_forecast)
        st.write(f'**XGBoost Model Metrics:**\n- MAE: {xgb_mae:.2f}\n- RMSE: {xgb_rmse:.2f}')

    # Hide the last subplot (bottom right) if not needed
    fig.delaxes(axs[1, 1])

    st.pyplot(fig)
