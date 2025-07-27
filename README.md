# Stock-AI-Predictor
Idk bruh i made sum
This project is a web-based application for predicting next-day stock prices using an LSTM (Long Short-Term Memory) neural network. Built with Streamlit, it allows users to upload a CSV file containing historical stock data and generates a prediction for the next closing price based on the most recent 60 days of data.

The application handles data preprocessing, including normalization using MinMaxScaler, reshaping the data to fit the LSTM input requirements, and feeding it into a pre-trained model. The result is displayed alongside a chart that visualizes recent price trends and the predicted value.

Technologies used in this project include Streamlit for the web interface, TensorFlow/Keras for the deep learning model, Pandas and NumPy for data processing, Matplotlib for data visualization, and scikit-learn for scaling.

To run the app, install the required dependencies listed in requirements.txt and start the app with the command: streamlit run app.py. Once launched, upload a CSV file with a "Close" column and at least 60 rows of data. The app will output the predicted next-day price along with a plotted chart of recent data.

This project demonstrates how deep learning can be integrated into user-friendly web applications for real-time financial analysis and forecasting.
