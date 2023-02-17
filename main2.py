#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

'''payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_cryptocurrencies')
df_wiki = payload[0]
Tickers = df_wiki.loc[(df_wiki['Currency'] == 'Bitcoin')]
ticker_list = Tickers.values.tolist()
for Ticker in ticker_list:'''
df = yf.download('^FCHI', start='2012-01-01', end='2022-01-01')

df = df['Adj Close'].values


df = df.reshape(-1, 1)
print(df)
dataset_train = np.array(df[:int(df.shape[0]*.8)])
dataset_test = np.array(df[int(df.shape[0]*.8):])
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test =scaler.fit_transform(dataset_test)

def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i - 50:i, 0])
        y.append(df[i, 0])

    x = np.array(x)
    y = np.array(y)

    return x,y


x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)



model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1],1 )))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.save('stock_prediction')

model = load_model('stock_prediction')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions2=np.mean(predictions,axis=1)


fig,ax=plt.subplots(figsize=(18,10))
plt.plot(y_test_scaled, color='blue', label='Original price')
plt.plot(predictions2, color='green', label='Predicted price')
plt.show()