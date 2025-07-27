import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the price change model 

def load_price_change_model():
    try:
        model = load_model('lstm_price_change_model.h5')
        model_type = 'price_change'
        st.success("✅ Using LSTM Price Change Model")
    except:
        st.error("❌ No trained model found! Please train the model first.")
        return None, None, None
    
    # Load training data to recreate scaler
    data = pd.read_csv("aapl.csv")
    scaler = MinMaxScaler()
    scaler.fit(data[["Close"]])
    
    return model, scaler, model_type

model, scaler, model_type = load_price_change_model()

# Price change prediction function
def predict_price_change(last_60_days, model, scaler):
    """Predict using price change model"""
    prices = last_60_days
    price_changes = np.diff(prices) / prices[:-1]  # 59 changes from 60 prices
    
    # Normalize prices for this sequence
    price_normalized = (prices - prices.mean()) / prices.std()
    
    # Combine features: normalized prices + price changes
    price_changes_padded = np.concatenate([[0], price_changes])  # Add 0 for first day
    combined_features = np.column_stack([price_normalized, price_changes_padded])
    
    # Reshape for LSTM input
    X_input = combined_features.reshape(1, 60, 2)
    
    # Predict the price change
    predicted_change = model.predict(X_input, verbose=0)[0][0]
    
    # Convert change back to actual price
    current_price = last_60_days[-1]
    predicted_price = current_price * (1 + predicted_change)
    
    return predicted_price, predicted_change

# Streamlit interface

st.title("Stock Price Predictor")
st.write("This app predicts the next day's stock price based on the last 60 days of closing prices using an LSTM model.")

uploaded_file = st.file_uploader("Upload stock CSV", type="csv")
if uploaded_file and model is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data:", df.tail())

    if 'Close' not in df.columns:
         st.error("CSV must have a 'Close' column.")
    elif len(df) < 60:
        st.error("You need at least 60 data points to predict.")
    else:
        last_60 = df["Close"].values[-60:]
        
        predicted_price, predicted_change = predict_price_change(last_60, model, scaler)
        st.success(f"Predicted Next Close Price: ${predicted_price:.2f}")
        st.info(f"Predicted Price Change: {predicted_change:.4f} ({predicted_change*100:.2f}%)")

        # Show last 60 + predicted point
        extended = np.append(last_60, predicted_price)
        plt.figure(figsize=(10, 4))
        plt.plot(range(60), last_60, label="Last 60 Days")
        plt.plot(60, predicted_price, 'ro', label="Predicted")
        plt.legend()
        plt.title("Price Prediction")
        st.pyplot(plt)
elif uploaded_file and model is None:
    st.error("Please train a model first! Run 'python lstm_price_change.py' to train the model.")