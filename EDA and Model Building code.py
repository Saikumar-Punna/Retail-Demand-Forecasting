# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:52:31 2024

@author: sai kumar
"""
'''
Business problem: Develop a demand forecasting model for a retail store chain 
that can predict future sales for each product category and store location. 
The model should incorporate factors like past sales data, promotions, 
holidays, and demographic information.

Business Objective: To predict future sales for each category and store location.

Business Constraint: Maximize the sales for each category and store location

Success Criteria
Business Success Criteria: 
    Increased sales, improved customer satisfaction, and optimized inventory management.
Machine Learning Success Criteria: 
    High model accuracy, robustness, and scalability.
Economic Success Criteria: 
    Positive return on investment (ROI) and reduced inventory costs.
    
Data Information:
    Columns: 913000 Rows
1   date: 913000 object (2013 to 2017)
2   store: 913000 int64 (There are 10 stores)
3   item: 913000 int64 (There 50 categories of items)
4   sales: 913000 int64

'''

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
import warnings

df_train = pd.read_csv(r"C:\Documents\demand-forecasting-kernels-only\train.csv")
df_test = pd.read_csv(r"C:\Documents\demand-forecasting-kernels-only\test.csv")

df_train.head(10)

df_train.tail(10)

df_train.dtypes

df_train.info()

df_train.describe()

df_train['store'].value_counts()
df_train['item'].value_counts()

unique_items: list = df_train['item'].unique()

print(f"There is {len(unique_items)} diferent types of items.\n")
print(f"This items are: {unique_items}")

unique_stores: list = df_train['store'].unique()

print(f"There is {len(unique_stores)} diferent types of stores.\n")
print(f"This stores are: {unique_stores}")

# Converting the date field to date type (currently it is an object)
df_train['date'] = pd.to_datetime(df_train['date']) #train
df_test['date'] = pd.to_datetime(df_test['date']) #test
# Converting the item and store fields to categorical type
# train
df_train['item'] = df_train['item'].astype('category')
df_train['store'] = df_train['store'].astype('category')

# test
df_test['item'] = df_test['item'].astype('category')
df_test['store'] = df_test['store'].astype('category')

# Check the updated DataFrame
df_train.head()

#Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Histogram for each feature
df_train['date'].hist(bins=30, figsize=(20, 15))
plt.show()

#Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Histogram for each feature
df_train['item'].hist(bins=30, figsize=(20, 15))
plt.show()

df_train.head(5)

# Box plot for each feature
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train['item'])
plt.xticks(rotation=90)
plt.show()

#Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Histogram for each feature
df_train['store'].hist(bins=30, figsize=(20, 15))
plt.show()

# Box plot for each feature
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train['store'])
plt.xticks(rotation=90)
plt.show()

#Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Histogram for each feature
df_train['sales'].hist(bins=30, figsize=(20, 15))
plt.show()

# Box plot for each feature
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train['sales'])
plt.xticks(rotation=90)
plt.show()

# Extract train date features
df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['day_of_week'] = df_train['date'].dt.dayofweek  # Monday=0, Sunday=6
df_train['day_of_year'] = df_train['date'].dt.dayofyear
df_train['week_of_year'] = df_train['date'].dt.isocalendar().week


# Extrat test date features
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['day_of_week'] = df_test['date'].dt.dayofweek  # Monday=0, Sunday=6
df_test['day_of_year'] = df_test['date'].dt.dayofyear
df_test['week_of_year'] = df_test['date'].dt.isocalendar().week

# Add lagged train features
for lag in range(1, 8):
    df_train[f'sales_lag_{lag}'] = df_train['sales'].shift(lag)

# replace nan values generated within median value
df_train["sales_lag_1"].fillna(df_train['sales_lag_1'].median(), inplace=True)

df_train["sales_lag_2"].fillna(df_train['sales_lag_2'].median(), inplace=True)

df_train["sales_lag_3"].fillna(df_train['sales_lag_3'].median(), inplace=True)

df_train["sales_lag_4"].fillna(df_train['sales_lag_4'].median(), inplace=True)

df_train["sales_lag_5"].fillna(df_train['sales_lag_5'].median(), inplace=True)

df_train["sales_lag_6"].fillna(df_train['sales_lag_6'].median(), inplace=True)

df_train["sales_lag_7"].fillna(df_train['sales_lag_7'].median(), inplace=True)

# Add lagged test features
y_test = pd.read_csv(r"C:\Documents\demand-forecasting-kernels-only\sample_submission.csv")
for lag in range(1, 8):
    y_test[f'sales_lag_{lag}'] = y_test['sales'].shift(lag)
    
# replace nan values generated within median value
y_test["sales_lag_1"].fillna(y_test['sales_lag_1'].median(), inplace=True)

y_test["sales_lag_2"].fillna(y_test['sales_lag_2'].median(), inplace=True)

y_test["sales_lag_3"].fillna(y_test['sales_lag_3'].median(), inplace=True)

y_test["sales_lag_4"].fillna(y_test['sales_lag_4'].median(), inplace=True)

y_test["sales_lag_5"].fillna(y_test['sales_lag_5'].median(), inplace=True)

y_test["sales_lag_6"].fillna(y_test['sales_lag_6'].median(), inplace=True)

y_test["sales_lag_7"].fillna(y_test['sales_lag_7'].median(), inplace=True)

y_test.head()

# Check the updated DataFrame
y_test.info()

#Set the 'Date' column as the index
df_train_vis = df_train.copy()

df_train_vis.set_index('date', inplace=True)

# Plotting the time series
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_train_vis, x='date', y='sales')

# Adding title and labels
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Sales')

# Display the plot
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = df_train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# One-hot encoding for train categorical variables
df_train = pd.get_dummies(df_train, drop_first=True)

# One-hot encoding for test categorical variables
df_test = pd.get_dummies(df_test, drop_first=True)

# Inner join on the 'id' column
df_test = pd.merge(df_test, y_test, on='id', how='inner')

# Reorder the test DataFrame columns to match the train DataFrame
df_test = df_test[df_train.columns]

# Prepare features and target variable for training
y_test = df_test['sales'].copy(deep=True)
y_train = df_train['sales'].copy(deep=True)

X_train = df_train.drop('sales', axis=1).copy(deep=True)
X_test = df_test.drop('sales', axis=1).copy(deep=True)

X_train = X_train.drop('date', axis=1)
X_test = X_test.drop('date', axis=1)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)
train_rmse_rf = mean_squared_error(y_train, y_pred_train_rf, squared=False)
test_rmse_rf = mean_squared_error(y_test, y_pred_test_rf, squared=False)
print(f'Random Forest Train RMSE: {train_rmse_rf}')
print(f'Random Forest Test RMSE: {test_rmse_rf}')

# Gradient Boosting Machines
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_train_gb = gb_model.predict(X_train)
y_pred_test_gb = gb_model.predict(X_test)
train_rmse_gb = mean_squared_error(y_train, y_pred_train_gb, squared=False)
test_rmse_gb = mean_squared_error(y_test, y_pred_test_gb, squared=False)
print(f'Gradient Boosting Machines Train RMSE: {train_rmse_gb}')
print(f'Gradient Boosting Machines Test RMSE: {test_rmse_gb}')


submission = pd.DataFrame()
submission['sales'] = y_pred_test_rf.copy()
submission.insert(0, 'id', range(0, len(submission)))
submission

submission.to_csv('/kaggle/working/submission.csv', index=False)
