import sys
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Get stock symbol from command line arguments
if len(sys.argv) != 2:
    print("Usage: python train_model.py <STOCK_SYMBOL>")
    sys.exit(1)

symbol = sys.argv[1]

# Fetch historical stock data (5 years of data)
try:
    data = yf.Ticker(symbol).history(period="5y")
    if data.empty:
        print(f"No data available for {symbol}. Please check the symbol and try again.")
        sys.exit(1)
    prices = data["Close"].values.reshape(-1, 1)
except Exception as e:
    print(f"Error fetching data for {symbol}: {e}")
    sys.exit(1)

# Scaling the data (Normalization)
scaler = MinMaxScaler(feature_range=(0, 1))  # Normalizes data to the range [0, 1]
prices_scaled = scaler.fit_transform(prices)

# Prepare data for LSTM (sequence of 60 days as features)
X, y = [], []
for i in range(60, len(prices_scaled)):
    X.append(prices_scaled[i-60:i, 0])  # Use the past 60 days as features
    y.append(prices_scaled[i, 0])  # The next day's closing price as the label

X, y = np.array(X), np.array(y)

# Reshaping X to be 3D [samples, time steps, features] for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))  # LSTM layer 1
model.add(LSTM(units=50))  # LSTM layer 2
model.add(Dense(units=1))  # Output layer
model.compile(optimizer="adam", loss="mean_squared_error")

# Train Model
print(f"Training model for stock symbol: {symbol}")
model.fit(X, y, epochs=20, batch_size=32)

# Save the trained model and the scaler for future use
model_path = r"C:\Users\deepa\OneDrive\Desktop\StockMarketPredictionSystem\model\stock_lstm_model.keras"
scaler_path = r"C:\Users\deepa\OneDrive\Desktop\StockMarketPredictionSystem\model\scaler.pkl"
model.save(model_path)  # Save the LSTM model
joblib.dump(scaler, scaler_path)  # Save the scaler

print(f"Training completed for symbol: {symbol}")
print(f"Model and scaler saved successfully to {model_path} and {scaler_path}")
