import streamlit as st
import pandas as pd
import numpy as np
from keras import models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

st.title("Stock Price Predictor App")

# Input for stock symbol
stock_symbol = st.text_input("Enter the Stock ID", "GOOG").strip()

def load_and_prepare_data(stock):
    try:
        end = datetime.now()
        start = datetime(end.year - 20, end.month, end.day)
        
        # Download stock data
        data = yf.download(stock, start, end)
        if data.empty:
            st.error(f"No data found for stock symbol: {stock}")
            return None, None, None, None
        
        data = pd.DataFrame(data)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        closing_data = data['Close']
        splitting_len = int(len(data) * 0.7)
        x_test = pd.DataFrame(closing_data[splitting_len:])
        scaled_data = scaler.fit_transform(x_test[['Close']])
        
        # Prepare the data for the model
        x_data = []
        y_data = []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])
        x_data, y_data = np.array(x_data), np.array(y_data)
        
        return data, x_data, y_data, scaler
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None, None

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset, 'green')
    plt.xlabel("Date")
    plt.ylabel("Price")
    return fig

if stock_symbol:
    st.subheader(f"Stock Data for {stock_symbol}")
    
    # Load and prepare data
    stock_data, x_data, y_data, scaler = load_and_prepare_data(stock_symbol)
    if stock_data is not None:
        st.write(stock_data)
        
        # Plot Moving Averages
        stock_data['MA_for_250_days'] = stock_data['Close'].rolling(250).mean()
        st.subheader(f'Original Close Price and MA for 250 days for {stock_symbol}')
        st.pyplot(plot_graph((15, 6), stock_data['MA_for_250_days'], stock_data))
        
        stock_data['MA_for_200_days'] = stock_data['Close'].rolling(200).mean()
        st.subheader(f'Original Close Price and MA for 200 days for {stock_symbol}')
        st.pyplot(plot_graph((15, 6), stock_data['MA_for_200_days'], stock_data))
        
        stock_data['MA_for_100_days'] = stock_data['Close'].rolling(100).mean()
        st.subheader(f'Original Close Price and MA for 100 days for {stock_symbol}')
        st.pyplot(plot_graph((15, 6), stock_data['MA_for_100_days'], stock_data))
        
        st.subheader(f'Original Close Price and MA for 100 days and MA for 250 days for {stock_symbol}')
        st.pyplot(plot_graph((15, 6), stock_data['MA_for_100_days'], stock_data, 1, stock_data['MA_for_250_days']))
        
        # Predict with the model
        model = models.load_model("stock_model.keras")
        predictions = model.predict(x_data)
        
        # Inverse transform the predictions and actual values
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)
        
        # Prepare DataFrame for plotting
        plotting_data = pd.DataFrame({
            'Original Test Data': inv_y_test.flatten(),
            'Predictions': inv_pre.flatten()
        }, index=stock_data.index[int(len(stock_data)*0.7) + 100:])
        
        # Display predictions
        st.subheader(f"Original values vs Predicted values for {stock_symbol}")
        st.write(plotting_data)
        
        # Plot original vs predicted prices
        st.subheader(f'Original Close Price vs Predicted Close Price for {stock_symbol}')
        fig = plt.figure(figsize=(15, 6))
        plt.plot(stock_data['Close'][:int(len(stock_data)*0.7) + 100], label='Data not used')
        plt.plot(plotting_data['Original Test Data'], label='Original Test Data')
        plt.plot(plotting_data['Predictions'], label='Predicted Test Data')
        plt.legend()
        st.pyplot(fig)