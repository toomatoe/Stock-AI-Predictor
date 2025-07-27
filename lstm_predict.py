import matplotlib
matplotlib.use("TkAgg")  # Try different backend
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set random seeds for reproducible results
import random
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

data = pd.read_csv("aapl.csv")
data = data[["Close"]]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 60
x = []
y = []
for i in range(sequence_length, len(scaled_data)):
    x.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])
X = np.array(x)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"📊 Total sequences: {len(X)}")

# Split data into train and test sets (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"📚 Training sequences: {len(X_train)}")
print(f"🧪 Testing sequences: {len(X_test)}")
print(f"📅 Training period: First {train_size} sequences")
print(f"📅 Testing period: Last {len(X_test)} sequences")

#now we gotta build the model with BALANCED performance

model = Sequential()
# Sweet spot: Not too big, not too small
model.add(LSTM(units=40, return_sequences=False, input_shape=(X.shape[1], 1)))
# Moderate dropout - not too aggressive
model.add(Dropout(0.25))  # Balanced regularization
model.add(Dense(units=1))

# Slightly higher learning rate for better pattern learning
optimizer = Adam(learning_rate=0.002)  # A bit more aggressive than 0.001
model.compile(optimizer=optimizer, loss='mean_squared_error')

#train ts
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Back to 3 epochs - give it more time to learn
    restore_best_weights=True,
    verbose=1,
    min_delta=0.0005  # Less strict improvement threshold
)

history = model.fit(
    X_train, y_train, 
    epochs=12,  # Slightly more epochs for better learning
    batch_size=32, 
    verbose=1, 
    validation_split=0.15,  # Balanced validation split
    callbacks=[early_stopping]
)

# Check training history for overfitting signs
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

if final_val_loss/final_train_loss < 1.3:
    print("✅ Good balance achieved")
elif final_val_loss/final_train_loss > 2.0:
    print("⚠️ Model might be overfitting")
else:
    print("🎯 Reasonable balance achieved")

#now we predict ts
train_predictions = model.predict(X_train, verbose=0)
test_predictions = model.predict(X_test, verbose=0)

# Convert back to real prices
train_pred_prices = scaler.inverse_transform(train_predictions)
test_pred_prices = scaler.inverse_transform(test_predictions)
train_actual_prices = scaler.inverse_transform(y_train.reshape(-1, 1))
test_actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate performance metrics
train_mse = mean_squared_error(train_actual_prices, train_pred_prices)
test_mse = mean_squared_error(test_actual_prices, test_pred_prices)
train_mae = mean_absolute_error(train_actual_prices, train_pred_prices)
test_mae = mean_absolute_error(test_actual_prices, test_pred_prices)

print("\n📈 MODEL PERFORMANCE:")
print(f"🎯 Training MSE: ${train_mse:.2f}")
print(f"🎯 Testing MSE: ${test_mse:.2f}")
print(f"📊 Training MAE: ${train_mae:.2f}")
print(f"📊 Testing MAE: ${test_mae:.2f}")
print(f"🔍 RMSE (Root MSE): ${np.sqrt(test_mse):.2f}")

if test_mse < train_mse * 1.5:
    print("✅ Model generalizes well!")
else:
    print("⚠️ Model might be overfitting")

print(f"Test actual prices shape: {test_actual_prices.shape}")
print(f"Test predicted prices shape: {test_pred_prices.shape}")

# Save the model
model.save("lstm_stock_model.h5")
model.save("lstm_stock_model_web.h5", save_format='h5')

# Create a prediction function for web apps
def predict_next_price(model, scaler, last_60_days):
    """
    Predict next day's stock price
    Args:
        model: Trained LSTM model
        scaler: MinMaxScaler used for training
        last_60_days: Array of last 60 closing prices
    Returns:
        Predicted next day price
    """
    # Scale the input
    scaled_input = scaler.transform(last_60_days.reshape(-1, 1))
    
    # Reshape for LSTM
    X_input = scaled_input.reshape(1, 60, 1)
    
    # Make prediction
    scaled_prediction = model.predict(X_input, verbose=0)
    
    # Convert back to real price
    prediction = scaler.inverse_transform(scaled_prediction)
    
    return prediction[0][0]

# Test the prediction function
test_input = data['Close'].values[-60:]  # Last 60 days
predicted_price = predict_next_price(model, scaler, test_input)
print(f"Next day prediction: ${predicted_price:.2f}")
print(f"Current price: ${data['Close'].values[-1]:.2f}")
print(f"Predicted change: ${predicted_price - data['Close'].values[-1]:.2f}")

# Enable interactive mode
plt.ion()

#now plot ts - Show test results (more honest evaluation)
plt.figure(figsize=(12, 8))

# Plot 1: Test predictions vs actual
plt.subplot(2, 1, 1)
plt.plot(test_actual_prices, label="Actual Test Prices", color='blue', linewidth=2)
plt.plot(test_pred_prices, label="Predicted Test Prices", color='red', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.title("LSTM Stock Price Prediction - Test Set Performance")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Recent training vs test (last 200 points)
plt.subplot(2, 1, 2)
recent_train_actual = train_actual_prices[-100:]
recent_train_pred = train_pred_prices[-100:]
plt.plot(range(len(recent_train_actual)), recent_train_actual, label="Train Actual", color='lightblue', alpha=0.7)
plt.plot(range(len(recent_train_pred)), recent_train_pred, label="Train Predicted", color='orange', alpha=0.7)
plt.plot(range(len(recent_train_actual), len(recent_train_actual) + len(test_actual_prices)), 
         test_actual_prices, label="Test Actual", color='blue', linewidth=2)
plt.plot(range(len(recent_train_actual), len(recent_train_actual) + len(test_pred_prices)), 
         test_pred_prices, label="Test Predicted", color='red', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.title("Train vs Test Performance (Recent Data)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=len(recent_train_actual), color='black', linestyle='--', alpha=0.5, label='Train/Test Split')

# display ts
plt.tight_layout()
plt.show(block=True)
plt.savefig('lstm_prediction_results.png', dpi=300, bbox_inches='tight')

