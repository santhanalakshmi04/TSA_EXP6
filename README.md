# NAME : Santhana Lakshmi k
# REGISTER NUMBER : 212222240091
# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:
To apply the Holt-Winters Method for time series forecasting on power consumption dataset, decompose the series, fit the model, and make future predictions while evaluating model accuracy.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original powerconsumption data and the predictions
### PROGRAM:
```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
# Load the dataset
file_path = '/content/powerconsumption.csv'  # Path to the uploaded dataset
data = pd.read_csv(file_path)
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
monthly_data = data['PowerConsumption_Zone1'].resample('ME').sum()
# Plot the time series data
plt.figure(figsize=(10, 5))
plt.plot(monthly_data, label='Monthly Power Consumption Zone 1')
plt.title('Monthly Power Consumption Zone 1')
plt.xlabel('Date')
plt.ylabel('Power Consumption (kWh)')
plt.legend()
plt.show()
# Split data into training and testing sets (80% for training, 20% for testing)
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data[:train_size], monthly_data[train_size:]
# Check if there's enough data to fit a seasonal model
if len(monthly_data) < 24:
    print("Not enough data to fit a seasonal model. Using Simple Exponential Smoothing.")
    model = ExponentialSmoothing(train, trend=None, seasonal=None)
else:
    # Fit the Holt-Winters model on training data
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)

fit = model.fit()
predictions = fit.forecast(len(test))
# Calculate RMSE for the test set predictions
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'Test RMSE: {rmse}')
# Fit Holt-Winters model on the entire dataset for future forecasting
if len(monthly_data) < 24:
    final_model = ExponentialSmoothing(monthly_data, trend=None, seasonal=None)
else:
    final_model = ExponentialSmoothing(monthly_data, trend="add", seasonal="add", seasonal_periods=12)

final_fit = final_model.fit()
# Make future predictions (for 12 months)
future_steps = 12
final_forecast = final_fit.forecast(steps=future_steps)
# Plotting Test Predictions and Final Predictions
plt.figure(figsize=(12, 6))

# Plot Test Predictions
plt.subplot(1, 2, 1)
plt.plot(monthly_data.index[:train_size], train, label='Training Data', color='blue')
plt.plot(monthly_data.index[train_size:], test, label='Test Data', color='green')
plt.plot(monthly_data.index[train_size:], predictions, label='Predictions', color='orange')
plt.title('Test Predictions for Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power Consumption (kWh)')
plt.legend(loc='upper left')  # Move legend to upper left
plt.xticks(rotation=45, ha='right')  # Rotate and align x-axis labels
plt.grid(True)  # Add gridlines
plt.tight_layout()
# Plot Final Predictions
plt.subplot(1, 2, 2)
plt.plot(monthly_data.index, monthly_data, label='Original Power Consumption', color='blue')

# Plot future forecast
plt.plot(pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=future_steps, freq='ME'), 
         final_forecast, label='Final Forecast', color='orange')

plt.title('Final Predictions for Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power Consumption (kWh)')
plt.legend(loc='upper left')  # Move legend to upper left
plt.grid(True) 
```

### OUTPUT:
ORIGINAL POWER CONSUMPTION DATA:

![image](https://github.com/user-attachments/assets/4a6512f6-a62d-49fa-9e4c-da73a82e4d6f)


TEST_PREDICTION AND FINAL_PREDICTION:

![image](https://github.com/user-attachments/assets/edb4bab7-ba58-47bd-a7d7-d37a7daaf4e6)


### RESULT:
Thus the program based on the Holt Winters Method model for power consumption is implemented successfully.
