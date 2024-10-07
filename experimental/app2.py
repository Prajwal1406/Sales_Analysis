import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Paths for model directories
arima_models_folder = 'models/arima_Models'
dt_models_folder = 'models/decision_tree_Models'
xgb_models_folder = 'models/xgb_Models'

merged_data = pd.read_csv('./data/merged_data.csv')

# Prepare weekly sales data for top 10 products
top_10_products = merged_data.groupby('StockCode')['Quantity'].sum().nlargest(10).index
weekly_sales = merged_data.groupby(['StockCode', 'Year', 'Week'])['Quantity'].sum().reset_index()

# Load product codes (as an example, you can modify this to load from your source)
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
        fig, ax = plt.subplots(figsize=(10, 6))
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

        st.write(f"Decision Tree Forecast for Product {selected_product}:")
        st.write(st.session_state.dt_forecast)

        # Plot Decision Tree results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(st.session_state.dt_forecast, label='Decision Tree Forecast', color='green')
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

        st.write(f"XGBoost Forecast for Product {selected_product}:")
        st.write(st.session_state.xgb_forecast)

        # Plot XGBoost results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(st.session_state.xgb_forecast, label='XGBoost Forecast', color='orange')
        ax.set_title(f'XGBoost Forecast for Product {selected_product}')
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

    # Prepare input data for ARIMA (you may need to adjust this based on your data structure)
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

    # Now plot all the forecasts
    fig, ax = plt.subplots(figsize=(10, 6))

    # ARIMA forecast
    if st.session_state.arima_forecast is not None:
        ax.plot(st.session_state.arima_forecast, label='ARIMA Forecast', color='blue')
    
    # Decision Tree forecast
    if st.session_state.dt_forecast is not None:
        ax.plot(st.session_state.dt_forecast, label='Decision Tree Forecast', color='green')
    
    # XGBoost forecast
    if st.session_state.xgb_forecast is not None:
        ax.plot(st.session_state.xgb_forecast, label='XGBoost Forecast', color='orange')
    
    ax.set_title(f'Comparison of Forecasts for Product {selected_product}')
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Sales Quantity')
    ax.legend()
    
    st.pyplot(fig)

