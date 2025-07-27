import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv("aapl.csv")

data["NextClose"] = data["Close"].shift(-1)

data.dropna(inplace=True)

X = data[["Open", "High", "Low", "Close", "Volume"]] #often a matrix/table that represents features or inputs
y = data["NextClose"]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42)

model = LinearRegression() #rn this is training the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

#Now we gotta evaluate ts
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

print(f"\nFirst 5 Predictions vs Actual:")
for pred, actual in zip(predictions[:5], y_test[:5]):
    print(f"Predicted: ${pred:.2f}, Actual: ${actual:.2f}")

#Plot ts
print("✅ plt is loaded:", plt)
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label="Actual")
plt.plot(predictions[:50], label="Predicted")
plt.title("🔍 Linear Regression Predictions (First 50 Test Points)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()