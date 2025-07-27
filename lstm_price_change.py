import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set random seeds
import random
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("🎯 Price Change Prediction Model")

data = pd.read_csv("aapl.csv")
data = data[["Close"]]

# Calculate price changes (percentage change)
data['Price_Change'] = data['Close'].pct_change()
data = data.dropna()

# Use price changes as target, but include actual prices as features
price_changes = data['Price_Change'].values
closing_prices = data['Close'].values

# Create features: [price, volume indicators, moving averages, changes]
# For now, let's use price + recent changes
sequence_length = 60

# Prepare the data
X = []
y = []
for i in range(sequence_length, len(data)):
    # Features: last 60 days of [closing prices normalized, price changes]
    price_seq = closing_prices[i-sequence_length:i]
    change_seq = price_changes[i-sequence_length:i]
    
    # Normalize prices for this sequence
    price_normalized = (price_seq - price_seq.mean()) / price_seq.std()
    
    # Combine features: normalized prices + actual changes
    combined_features = np.column_stack([price_normalized, change_seq])
    X.append(combined_features)
    
    # Target: next day's price change
    y.append(price_changes[i])

X = np.array(X)
y = np.array(y)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build model for price change prediction
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    verbose=1,
    validation_split=0.15,
    callbacks=[early_stopping]
)

# Make predictions
train_change_pred = model.predict(X_train, verbose=0)
test_change_pred = model.predict(X_test, verbose=0)

# Convert price changes back to actual prices
def changes_to_prices(start_price, changes):
    """Convert price changes back to actual prices"""
    prices = [start_price]
    for change in changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    return np.array(prices[1:])  # Remove start price

# Get starting prices for reconstruction
train_start_idx = sequence_length
test_start_idx = train_size + sequence_length

train_start_price = closing_prices[train_start_idx]
test_start_price = closing_prices[test_start_idx]

# Reconstruct actual prices from changes
train_actual_prices = changes_to_prices(train_start_price, y_train)
test_actual_prices = changes_to_prices(test_start_price, y_test)

train_pred_prices = changes_to_prices(train_start_price, train_change_pred.flatten())
test_pred_prices = changes_to_prices(test_start_price, test_change_pred.flatten())

# Calculate metrics
test_mse = mean_squared_error(test_actual_prices, test_pred_prices)
test_mae = mean_absolute_error(test_actual_prices, test_pred_prices)

print(f"Testing MSE: ${test_mse:.2f}")
print(f"Testing MAE: ${test_mae:.2f}")

# Direction accuracy
actual_directions = np.sign(y_test)
pred_directions = np.sign(test_change_pred.flatten())
direction_accuracy = np.mean(actual_directions == pred_directions)
print(f"Direction Accuracy: {direction_accuracy:.2%}")

# Save the price change model
model.save("lstm_price_change_model.h5")
print("✅ Price change model saved as 'lstm_price_change_model.h5'")

# Save the scalers and other necessary data for the web app
import pickle

# We need to save important data for web app
web_app_data = {
    'sequence_length': sequence_length,
    'train_start_price': train_start_price,
    'model_type': 'price_change'
}

with open('price_change_model_data.pkl', 'wb') as f:
    pickle.dump(web_app_data, f)

print("✅ Model data saved for web app integration")

# Plotting
plt.figure(figsize=(15, 10))

# Plot 1: Price Change Predictions
plt.subplot(3, 1, 1)
plt.plot(y_test[:100], label="Actual Changes", color='blue', alpha=0.7)
plt.plot(test_change_pred.flatten()[:100], label="Predicted Changes", color='red', alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Price Change")
plt.title("Price Change Predictions (First 100 test days)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Reconstructed Prices (Test Set)
plt.subplot(3, 1, 2)
plt.plot(test_actual_prices, label="Actual Test Prices", color='blue', linewidth=2)
plt.plot(test_pred_prices, label="Predicted Test Prices", color='red', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.title("Reconstructed Price Predictions - Test Set")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Recent performance zoom
plt.subplot(3, 1, 3)
recent_days = 50
plt.plot(test_actual_prices[-recent_days:], label="Actual Prices", color='blue', linewidth=2, marker='o', markersize=3)
plt.plot(test_pred_prices[-recent_days:], label="Predicted Prices", color='red', linewidth=2, marker='s', markersize=3)
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.title(f"Recent {recent_days} Days - Detailed View")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)
plt.savefig('price_change_prediction_results.png', dpi=300, bbox_inches='tight')
