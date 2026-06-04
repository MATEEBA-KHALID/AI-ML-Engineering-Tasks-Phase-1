# Task 2: Predict Future Stock Prices 

# Import Libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.line--ar_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Download Stock Data

# Choose a stock ticker
ticker = "AAPL"  # Apple
# ticker = "TSLA"  # Tesla

# Download historical data
data = yf.download(ticker, start="2022-01-01", end="2025-01-01")

print("First 5 Rows:")
print(data.head())

# Step 2: Create Target Variable

# Predict next day's closing price
data["Next_Close"] = data["Close"].shift(-1)

# Remove last row (contains NaN target)
data.dropna(inplace=True)

# Step 3: Select Features and Target

X = data[["Open", "High", "Low", "Volume"]]
y = data["Next_Close"]

# Step 4: Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Step 5: Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions

y_pred = model.predict(X_test)

# Step 7: Evaluate Model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Step 8: Plot Actual vs Predicted

plt.figure(figsize=(12, 6))
plt.plot(
    y_test.values,
    label="Actual Closing Price"
)
plt.plot(
    y_pred,
    label="Predicted Closing Price"
)
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Predict Next Day Price

latest_data = X.iloc[-1:].values
next_day_prediction = model.predict(latest_data)
print("\nPredicted Next Day Closing Price:")
print(f"${next_day_prediction[0]:.2f}")