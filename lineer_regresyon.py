# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:13:50 2024

@author: iozer
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'D:/bilimsel/veri/bazalt_ersin_v1.xlsx'  # Update this path
data = pd.read_excel(file_path)

# Split the data into features (X) and targets (Y)
X = data.iloc[:, :4]
Y = data.iloc[:, 4:]

# Apply log transformation to the features, ensuring all values are positive
X_transformed = np.log(X + 1)  # Adding 1 to handle zero values in the data

# Initialize a scaler for the transformed features and targets
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

# Normalize the transformed features
X_scaled = scaler_X.fit_transform(X_transformed)

# Normalize the targets
Y_scaled = scaler_Y.fit_transform(Y)

# Define the regression model with Linear Regression
linear_model = MultiOutputRegressor(LinearRegression())

# Fit the model to the scaled data
linear_model.fit(X_scaled, Y_scaled)

# Predict the targets
y_pred_cv_scaled = linear_model.predict(X_scaled)

# Function to calculate VAF
def calculate_vaf(y_true, y_pred):
    return 100 * (1 - np.var(y_true - y_pred) / np.var(y_true))

# Calculate and print RMSE, MAE, R2, and VAF for each target using the normalized predicted and actual values
metrics = pd.DataFrame(index=Y.columns, columns=['RMSE', 'MAE', 'R2', 'VAF'])
for i, column in enumerate(Y.columns):
    rmse = np.sqrt(mean_squared_error(Y_scaled[:, i], y_pred_cv_scaled[:, i]))
    mae = mean_absolute_error(Y_scaled[:, i], y_pred_cv_scaled[:, i])
    r2 = r2_score(Y_scaled[:, i], y_pred_cv_scaled[:, i])
    vaf = calculate_vaf(Y_scaled[:, i], y_pred_cv_scaled[:, i])
    metrics.loc[column] = [rmse, mae, r2, vaf]

print("Metrics for normalized values:")
print(metrics)

# Plotting the actual vs. predicted values separately for each target variable using denormalized values
fig, axs = plt.subplots(Y.shape[1], figsize=(10, 15))
for i in range(Y.shape[1]):
    # Denormalize the actual and predicted values
    actual_values = scaler_Y.inverse_transform(Y_scaled)[:, i]
    predicted_values = scaler_Y.inverse_transform(y_pred_cv_scaled)[:, i]
    actual_sorted_index = np.argsort(actual_values)
    pred_sorted_index = np.argsort(predicted_values)
    
    axs[i].plot(np.arange(1, len(Y) + 1), actual_values[actual_sorted_index], color='green', label='Actual Values', marker='o', linestyle='None', markersize=5)
    axs[i].plot(np.arange(1, len(Y) + 1), predicted_values[pred_sorted_index], color='red', label='Predicted Values', marker='x', linestyle='None', markersize=5)
    
    axs[i].set_title(f'{Y.columns[i]}: Actual vs. Predicted Values')
    axs[i].set_xlabel('Observation Number')
    axs[i].set_ylabel('Value')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()
