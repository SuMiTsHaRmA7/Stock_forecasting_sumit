import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array

st.title('Stock trend prediction and forecasting')

# Fetch data from Yahoo Finance using yfinance
symbol = st.text_input('Enter Stock Ticker','AAPL')
try:
    df = yf.download(symbol, start='2015-01-01', end='2023-01-01')  # Adjust the date range as needed
    st.write("Data loaded successfully.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Display the raw data
# st.subheader("Raw Data")
# st.write(df)

# Display summary statistics using describe()
st.subheader("Summary Statistics")
st.write(df.describe())

# You can add more sections to display different aspects of the data as needed.
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100 & 200 MA")
m_100 = df.Close.rolling(100).mean()
m_200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(m_100, 'r', label = '100 Days Moving Average')
plt.plot(m_200, 'g', label = '200 Days Moving Average')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# LET'S DO SOME GADBAD FROM HERE

df1 = df.reset_index()['Close']
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
train_size = int(len(df1)*0.65)
test_size = len(df1)-train_size
train_data = df1[0:train_size,:]
test_data = df1[train_size:len(df1),:1]

def create_dataset(dataset, time_step=1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = load_model('keras_model.h5')

train_predict  = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

math.sqrt(mean_squared_error(y_train,train_predict))
math.sqrt(mean_squared_error(ytest,test_predict))

st.subheader('Plotting of Model on Graph')
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,  :]= train_predict
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
fig = plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot, 'r', label = 'Traning Data')
plt.plot(testPredictPlot, 'g', label = 'Testing Data')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

x = len(test_data)-100
x_input = test_data[x:].reshape(1,-1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()
# print(len(test_data))
# from numpy import array

first_output = []
n_steps = 100
i = 0
while (i < 30):

    if (len(temp_input) > 100):
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        first_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        i = i + 1

day_new = np.arange(1,101)
day_pred = np.arange(102,131)
df3 = df1.tolist()
df3.extend(first_output)

y = len(df1)-100
st.subheader("Forecasting the graph of next 30 days")
st.write("DISCLAIMER: This is the predicated graph of next 30 days please don't risk your asset by following this graph")
fig = plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[y:]), label = 'Original Data')
plt.plot(day_pred,scaler.inverse_transform(first_output), label = 'Predicated Data')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

z = len(df1)-200
fig = plt.figure(figsize=(12,6))
df3 = df1.tolist()
df3.extend(first_output)
plt.plot(df3[z:])
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)