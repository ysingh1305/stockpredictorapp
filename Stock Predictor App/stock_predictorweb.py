import streamlit as st
import pandas as pd
import numpy as np
from keras import models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)
google_data = pd.DataFrame(google_data)

model = models.load_model("stock_model.keras")

st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
closing_data = google_data['Close']
x_test = pd.DataFrame(closing_data[splitting_len:])

# Define a function for plotting graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset, 'green')
    plt.xlabel("Date")
    plt.ylabel("Price")
    return fig

# Plot Moving Averages
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
st.subheader('Original Close Price and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
st.subheader('Original Close Price and MA for 200 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
st.subheader('Original Close Price and MA for 100 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Prepare the data for the model
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predict with the model
predictions = model.predict(x_data)

# Inverse transform the predictions and actual values
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare DataFrame for plotting
plotting_data = pd.DataFrame({
    'Original Test Data': inv_y_test.flatten(),
    'Predictions': inv_pre.flatten()
}, index=google_data.index[splitting_len + 100:])

# Display predictions
st.subheader("Original values vs Predicted values")
st.write(plotting_data)

# Plot original vs predicted prices
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data['Close'][:splitting_len + 100], label='Data not used')
plt.plot(plotting_data['Original Test Data'], label='Original Test Data')
plt.plot(plotting_data['Predictions'], label='Predicted Test Data')
plt.legend()
st.pyplot(fig)