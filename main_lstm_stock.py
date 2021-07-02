import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from stock import Dataprocess
from stock.Dataprocess import Dataset, DMinMaxScaler
from stock.Dataprocess import load_stock, make_ma, create_dataset, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras import layers

stock_data = load_stock(filename=r'E:\StocksData\ACB.csv')
#data = stock_data.iloc[:,[4,5]]
ds = Dataset(stock_data)

x = ds.create_dataset()

START_TRAIN = -2000
NUM_TRAIN  = 300
END_TRAIN = START_TRAIN + NUM_TRAIN

START_TEST = END_TRAIN
NUM_TEST = 20
END_TEST = END_TRAIN + NUM_TEST

x_ma = ds.create_moving_average('Close',5)[START_TRAIN:END_TRAIN]
myScale = DMinMaxScaler()
myScale.fit(x_ma.dropna().values)
x_ma_scaled = myScale.transform()

x_ma_vol = ds.create_moving_average('Volume',1,20)[START_TRAIN:END_TRAIN]
myScale_vol = DMinMaxScaler()
myScale_vol.fit(x_ma_vol.dropna().values)
x_ma_vol_scaled = myScale_vol.transform()

x_ma_vol_scaled_shift = x_ma_vol_scaled[1:,0], x_ma_vol_scaled[0:-1,1]
x_ma_vol_scaled_shift = np.asarray(x_ma_vol_scaled_shift).T

x_scaled = np.concatenate((x_ma_scaled[:-1],x_ma_vol_scaled_shift),axis=1)

data = pd.DataFrame(x_scaled)

x_serial, y_serial = ds.create_dataset(data,5)

y_serial_max10 = [np.max(y_serial[i:i+20]) for i in range(len(y_serial))]
y_serial_max10 = np.asarray(y_serial_max10)

x_train = x_serial[:200]
y_train = y_serial_max10[:200].reshape((-1,1))

x_test = x_serial[200:]
y_test = y_serial_max10[200:].reshape((-1,1))

# design network
np.random.seed(0)
model = Sequential()
model.add(layers.LSTM(3,return_sequences=True, input_shape=(5, 2)))
#model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.2))
#model.add(layers.LSTM(6,return_sequences=True))
#model.add(layers.TimeDistributed(layers.Dense(1)))
#model.add(layers.Flatten())
model.add(layers.Dense(1,activation='linear'))
#model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.2))
#model.add(layers.LSTM(4,return_sequences=True))
#model.add(layers.BatchNormalization())
#model.add(layers.LSTM(32,return_sequences=False))
#model.add(layers.BatchNormalization())
#model.add(layers.LSTM(32,return_sequences=False))
#model.add(layers.Dropout(0.2))
#model.add(layers.Dropout(0.2))

#model.add(layers.Dense(1,activation='linear'))
model.compile(loss='mse', optimizer='adam')

model.summary()
model.output_shape

for layer in model.layers:
    print("layer-------" + layer.name)
    print("input " , layer.input_shape)
    print("Output " , layer.output_shape)

history = model.fit(x_train, y_train, epochs=20, batch_size=32,validation_data = (x_test,y_test), verbose=1, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


y_hat = model.predict(x_train)
rmse = (mean_absolute_error(y_train, y_hat))

y_hat = model.predict(x_test)
(mean_absolute_error(y_test, y_hat))

math.sqrt(rmse)
#print('Test RMSE: %.3f' % rmse)

(mean_absolute_error(y_test[0:-10], y_test[1:-9]))

math.sqrt(model.evaluate(x_train,y_train))
math.sqrt(model.evaluate(x_test,y_test))

y_hat = model.predict(x_test)
plt.plot(y_hat,color='red')
plt.plot(y_test)

plt.plot(y_test[-20:-10])
plt.plot(y_hat[-20:-10])

y_hat = model.predict(x_train)
plt.plot(y_hat,color='red')
plt.plot(y_train)

plt.plot(y_train[:-1])
plt.plot(y_train[1:])

