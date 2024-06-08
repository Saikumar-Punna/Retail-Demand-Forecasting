# Streamlit code for Demand Forecasting Model with file upload

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Function to preprocess data
def preprocess_data(df_train, df_test):
    # Convert date columns to datetime
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    # Convert item and store columns to categorical
    df_train['item'] = df_train['item'].astype('category')
    df_train['store'] = df_train['store'].astype('category')
    df_test['item'] = df_test['item'].astype('category')
    df_test['store'] = df_test['store'].astype('category')
    
    # Extract date features
    for df in [df_train, df_test]:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Add lag features
    for lag in range(1, 8):
        df_train[f'sales_lag_{lag}'] = df_train['sales'].shift(lag)
    
    df_train.fillna(df_train.median(), inplace=True)
    
    return df_train, df_test

# Function to train models
def train_models(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_train_rf = rf_model.predict(X_train)
    train_rmse_rf = mean_squared_error(y_train, y_pred_train_rf, squared=False)
    
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_train_gb = gb_model.predict(X_train)
    train_rmse_gb = mean_squared_error(y_train, y_pred_train_gb, squared=False)
    
    return rf_model, gb_model, train_rmse_rf, train_rmse_gb

# Streamlit App
st.title("Demand Forecasting for Retail Store Chain")

st.header("1. Upload Train Data")
train_file = st.file_uploader("Upload CSV or Excel file for training data", type=["csv", "xlsx"])
if train_file is not None:
    if train_file.name.endswith('.csv'):
        df_train = pd.read_csv(train_file)
    else:
        df_train = pd.read_excel(train_file)
    st.write("Train Data", df_train.head())

st.header("2. Upload Test Data")
test_file = st.file_uploader("Upload CSV or Excel file for test data", type=["csv", "xlsx"])
if test_file is not None:
    if test_file.name.endswith('.csv'):
        df_test = pd.read_csv(test_file)
    else:
        df_test = pd.read_excel(test_file)
    st.write("Test Data", df_test.head())

if st.button('Preprocess Data') and train_file is not None and test_file is not None:
    df_train, df_test = preprocess_data(df_train, df_test)
    st.write("Preprocessed Train Data", df_train.head())
    st.write("Preprocessed Test Data", df_test.head())

if st.button('Train Models') and train_file is not None and test_file is not None:
    # Prepare features and target variable for training
    y_train = df_train['sales'].copy(deep=True)
    X_train = df_train.drop(['sales', 'date'], axis=1).copy(deep=True)
    
    rf_model, gb_model, train_rmse_rf, train_rmse_gb = train_models(X_train, y_train)
    
    st.write(f'Random Forest Train RMSE: {train_rmse_rf}')
    st.write(f'Gradient Boosting Machines Train RMSE: {train_rmse_gb}')

    # Prepare test data
    X_test = df_test.drop(['date'], axis=1).copy(deep=True)
    df_test['sales'] = rf_model.predict(X_test)
    
    st.write("Predicted Test Data", df_test.head())

if st.button('Show Visualizations') and train_file is not None:
    sns.set_style('whitegrid')

    # Histogram for sales
    plt.figure(figsize=(20, 15))
    df_train['sales'].hist(bins=30)
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # Time series plot for sales
    plt.figure(figsize=(12, 6))
    df_train_vis = df_train.copy()
    df_train_vis.set_index('date', inplace=True)
    sns.lineplot(data=df_train_vis, x='date', y='sales')
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    st.pyplot(plt)

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    corr_matrix = df_train.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

if st.button('Generate Submission') and train_file is not None and test_file is not None:
    df_test[['id', 'sales']].to_csv('submission.csv', index=False)
    st.write("Submission file created successfully!")

st.sidebar.title("About")
st.sidebar.info("This app is built for demand forecasting using Random Forest and Gradient Boosting models.")
