import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import itertools
from config import FILE_PATH

# Load the data
file_path = FILE_PATH
data = pd.read_csv(file_path)
freq = 'M'
# Convert the Dates column to datetime
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')
data.set_index('Dates', inplace=True)


# Function to find the best ARIMA parameters
def optimize_arima(data, p_values, d_values, q_values):
    best_aic = float("inf")
    best_order = None
    best_model = None
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(data, order=(p, d, q), freq=freq)
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_model = model_fit
        except:
            continue
    return best_order, best_model


# Define the range of p, d, q values
p_values = range(0, 6)
d_values = range(0, 3)
q_values = range(0, 6)

# Optimise ARIMA model
best_order, best_model = optimize_arima(data['Prices'], p_values, d_values, q_values)

# Forecast the next 12 months using the best ARIMA model
forecast_arima = best_model.forecast(steps=12)
forecast_dates_arima = pd.date_range(start=data.index[-1], periods=13, freq=freq)[1:]
forecast_series_arima = pd.Series(forecast_arima, index=forecast_dates_arima)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Prices'], marker='o', label='Historical Prices')
plt.plot(forecast_series_arima.index, forecast_series_arima, marker='o', linestyle='--', label='ARIMA Forecasted Prices')
plt.title('Monthly Natural Gas Prices with ARIMA Forecasts')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# Function to estimate price for a given date
def estimate_price(date_str):
    date = pd.to_datetime(date_str, format='%m/%d/%Y')
    if date in data.index:
        price = data.loc[date, 'Prices']
    elif date in forecast_series_arima.index:
        price = forecast_series_arima.loc[date]
    else:
        return "Date is too far in the future for the current model."
    return price


# Example usage of the function
date_input = '12/31/2024'
estimated_price = estimate_price(date_input)
print(f"\nEstimated price on {date_input}: {estimated_price}\n")
