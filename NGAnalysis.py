import pandas as pd
import matplotlib.pyplot as plt
from config import FILE_PATH
from statsmodels.tsa.arima.model import ARIMA

# load data
file_path = FILE_PATH
data = pd.read_csv(file_path)

# converting dates column to datetime format
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')
print(data.head())

# '''---- Estimating Historical Prices and Extrapolating Future Prices ----'''

# # Set the index to the Dates column for time series modeling
# data.set_index('Dates', inplace=True)

# # Fit the ARIMA model (order can be tuned)
# model = ARIMA(data['Prices'], order=(5, 1, 0))
# model_fit = model.fit()

# # Forecast the next 12 months
# forecast = model_fit.forecast(steps=12)
# forecast_dates = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]

# # Combine the historical data with the forecast
# forecast_series = pd.Series(forecast, index=forecast_dates)

# # Plot the historical data and the forecast
# plt.figure(figsize=(12, 6))
# plt.plot(data.index, data['Prices'], marker='o', label='Historical Prices')
# plt.plot(forecast_series.index, forecast_series, marker='o', linestyle='--', label='Forecasted Prices')
# plt.title('Monthly Natural Gas Prices with Forecast')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.show()


# data plot to visualise trends
# plt.figure(figsize=(12, 6))
# plt.plot(data['Dates'], data['Prices'], marker='o')
# plt.title('18-Month Natural Gas Prices')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.grid(True)
# plt.show()
