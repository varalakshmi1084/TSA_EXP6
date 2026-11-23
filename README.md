# Ex.No: 6               HOLT WINTERS METHOD
### Date: 23.11.2026




### AIM:


To implement the Holt-Winters Exponential Smoothing method in Python for time series forecasting by analyzing trend and seasonal patterns in historical data and predicting future values.


##DATASET::
cars_dataset


##SOFTWARE REQUIRED:



google colab




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
9. You plot the original sales data and the predictions
### PROGRAM:


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 1. Load the dataset (using correct path)
file_path = '/content/cars_dataset.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find the dataset at {file_path}. Please make sure it is uploaded.")

data = pd.read_csv(file_path)
print("Dataset loaded successfully!")
display(data.head())

# 2. Group data by year and calculate average price
data_yearly = data.groupby('year')['price'].mean()
data_yearly.index = pd.to_datetime(data_yearly.index, format='%Y')
data_yearly = data_yearly.asfreq('YS')
data_yearly = data_yearly.interpolate()

# Plot yearly average car prices
plt.figure(figsize=(10, 4))
data_yearly.plot()
plt.title('Average Car Price by Year')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()

# 3. Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_yearly.values.reshape(-1, 1)).flatten(),
    index=data_yearly.index
)

plt.figure(figsize=(10, 4))
scaled_data.plot()
plt.title('Scaled Car Price Data')
plt.grid(True)
plt.show()

# 4. Split train and test data
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Create Holt-Winters model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal=None
).fit()

# Predict test data
test_predictions_add = model_add.forecast(steps=len(test_data))

# Visual evaluation
fig, ax = plt.subplots(figsize=(10, 4))
train_data.plot(ax=ax, label='Train Data')
test_predictions_add.plot(ax=ax, label='Test Predictions (Add)')
test_data.plot(ax=ax, label='Test Data')
ax.legend()
ax.set_title('Visual Evaluation')
plt.grid(True)
plt.show()

# 5. Model performance evaluation
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"RMSE: {rmse}")
print(f"Standard deviation: {np.sqrt(scaled_data.var())}")
print(f"Mean: {scaled_data.mean()}")

# 6. Create final model using all scaled data
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal=None
).fit()

# Predict future car prices for the next 5 years
final_predictions = final_model.forecast(steps=5)

# Convert predictions back to original price scale
final_predictions_rescaled = scaler.inverse_transform(
    final_predictions.values.reshape(-1, 1)
).flatten()

final_predictions_series = pd.Series(
    final_predictions_rescaled,
    index=pd.date_range(
        start=data_yearly.index[-1] + pd.DateOffset(years=1),
        periods=5,
        freq='YS'
    )
)

# Plot final prediction
fig, ax = plt.subplots(figsize=(10, 5))
data_yearly.plot(ax=ax, label='Historical Average Price')
final_predictions_series.plot(ax=ax, label='5-Year Forecast', color='orange', marker='o')

ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Average Car Price')
ax.set_title('Car Price Forecast (Next 5 Years)')
plt.grid(True)
plt.show()


```


### OUTPUT:



<img width="891" height="562" alt="tsa ex6 screenshot 1" src="https://github.com/user-attachments/assets/594fd0db-0d75-4649-8e9e-165e5d3bbed0" />



<img width="570" height="266" alt="tsa ex6 screenshot 2" src="https://github.com/user-attachments/assets/b1cd9e2c-7113-4ef4-aa33-170e0bae30d4" />




<img width="593" height="298" alt="tsa ex6 screenshot 3" src="https://github.com/user-attachments/assets/a1220e0a-48fc-4d8b-8b50-0a724f11ba57" />





<img width="617" height="322" alt="tsa ex6 screenshot 4" src="https://github.com/user-attachments/assets/e446ea53-3ab6-42cf-898f-b3b613a82099" />




### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
