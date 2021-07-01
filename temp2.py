#hi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from stock import Dataprocess
from stock.Dataprocess import Dataset, DMinMaxScaler
from stock.Dataprocess import load_stock, make_ma, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer,PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow import keras
stock_data = load_stock(filename=r'E:\StocksData\FPT.csv')
#data = stock_data.iloc[:,[4,5]]
ds = Dataset(stock_data)

x = ds.create_dataset()
data = np.asarray(x).T
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


def create_data_train(data, lookback = 5):
    LEN = len(data)
    resultX = []
    resultY = []
    for i in range(LEN-lookback-1-1):
        resultX.append(data[i:i+lookback,:])
        resultY.append(max(data[i+lookback:i+lookback+1,2]))
    return np.asarray(resultX), np.asarray(resultY)
        
    
X, Y = create_data_train(data_scaled,lookback=20)
X_train = X[:-1000]
Y_train = Y[:-1000]
X_test = X[-1000:]
Y_test = Y[-1000:]
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

model = Sequential()
model.add(layers.LSTM(81,return_sequences=True, input_shape=(20,9)))
model.add(layers.LSTM(32,return_state=False))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(loss='mse', optimizer='adam')

model.summary()

his = model.fit(X_train,Y_train, epochs=100, batch_size=64,validation_data=(X_test,Y_test), callbacks=[callback] )

plt.plot(his.history['val_loss'], label='train')
plt.plot(his.history['accuracy'], label='train')

y_pre = model.predict(X_test)
y_acc = Y_test

plt.plot(y_acc[:100], 'ro', markersize=2)
plt.plot(y_pre[:], color = 'black')
plt.plot(y_acc[:], color = 'red')

model.evaluate(X_test,Y_test)
