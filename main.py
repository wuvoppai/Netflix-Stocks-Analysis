import pandas as pd
import yfinance as yf
import numpy as np
import datetime
from datetime import date,timedelta
import streamlit as st
today=date.today()
d1=today.strftime("%Y-%m-%d")
end_date=d1
d2=date.today() - datetime.timedelta(days=5000)
d2=d2.strftime("%Y-%m-%d") 
start_date=d2
data=yf.download('NFLX',start=start_date,end=end_date,progress=False)
data["Date"]=data.index
data=data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True,inplace=True)
st.write(data.tail())

import plotly.graph_objects as pg
fig=pg.Figure(data=[pg.Candlestick(x=data["Date"],open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"])])
st.plotly_chart(fig.update_layout(title="Netflix stock Price Analysis",xaxis_rangeslider_visible=False))


x = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
st.write(model.fit(xtrain, ytrain, batch_size=1, epochs=30))
